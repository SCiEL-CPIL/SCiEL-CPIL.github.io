# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_START_TIME = time.time()
_DEADLINE = _START_TIME + 610.0

p = 7
K = 3
N_TOTAL = p**4
N_DIRS = p*p + 1 # 50

def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))

def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb

def _precompute_lines():
    lines = np.zeros((N_DIRS * p * p, p * p), dtype=np.int32)
    line_idx = 0
    # Vertical
    for xa in range(p):
        for xb in range(p):
            for k in range(p * p):
                ya = k // p
                yb = k % p
                lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
            line_idx += 1
    # Slopes
    for ma in range(p):
        for mb in range(p):
            for ca in range(p):
                for cb in range(p):
                    for k in range(p * p):
                        xa = k // p
                        xb = k % p
                        ya = (ca + ma * xa + 3 * mb * xb) % p
                        yb = (cb + ma * xb + mb * xa) % p
                        lines[line_idx, k] = _point_to_idx(xa, xb, ya, yb)
                    line_idx += 1
    return lines

def nikodym_set() -> np.ndarray:
    start = time.time()
    lines = _precompute_lines()
    n_lines = lines.shape[0]
    
    # Build point_lines: for each point, list of line indices
    point_lines = [[] for _ in range(N_TOTAL)]
    for i in range(n_lines):
        for idx in lines[i]:
            point_lines[idx].append(i)
            
    # Spectral Embedding for Initial Guess
    # Build adjacency matrix of the union graph
    # This is dense, so use sparse
    rows = []
    cols = []
    for i in range(n_lines):
        pts = lines[i]
        for u in range(len(pts)):
            for v in range(u+1, len(pts)):
                rows.append(pts[u])
                cols.append(pts[v])
                rows.append(pts[v])
                cols.append(pts[u])
                
    data = np.ones(len(rows))
    A = sp.csr_matrix((data, (rows, cols)), shape=(N_TOTAL, N_TOTAL))
    L = sp.diags(np.asarray(A.sum(axis=1)).flatten()) - A
    
    # Get Fiedler vector
    try:
        # Smallest eigenvalues
        vals, vecs = spla.eigsh(L, k=2, which='SM', return_eigenvectors=True)
        fiedler = vecs[:, 1]
    except Exception:
        fiedler = np.random.randn(N_TOTAL)
        
    # Initial S: Points with highest Fiedler value (one side of the cut)
    # Or just random?
    # Let's try a greedy addition based on spectral score
    in_S = np.zeros(N_TOTAL, dtype=np.int8)
    
    # Score for each point: number of clean directions initially
    # Initially all directions are clean.
    # We want to add points.
    # Risk of adding c: it "pollutes" directions for other points.
    
    # Use a priority queue based on spectral score
    scores = fiedler
    candidates = np.argsort(-scores) # Descending
    
    # Track line_s_counts
    line_s_counts = np.zeros(n_lines, dtype=np.int32)
    
    # Bad count: for each point, how many directions are "blocked" by S?
    # A direction v is blocked for x if there is y in S on L_{x,v}, y != x.
    # We need at least 1 unblocked direction for x in S.
    # So if x is added to S, we need to ensure it has an unblocked direction.
    # And adding x blocks directions for others.
    
    # blocked[x, v] = 1 if direction v is blocked for x by S.
    # We can't store 2401x50 easily? 120k ints. Fine.
    blocked = np.zeros((N_TOTAL, N_DIRS), dtype=np.int8)
    
    # Map line index to (point, direction)
    # Line i corresponds to direction d(i) and point on line?
    # Actually, a line is defined by direction and offset.
    # We just need to know which direction a line is.
    
    dir_of_line = np.zeros(n_lines, dtype=np.int32)
    # Vertical is 0
    # Slopes are 1..49
    # line_idx 0..48 are vertical? No.
    # In _precompute_lines:
    # 49 vertical lines.
    # Then 49*49=2401 slope lines.
    # So lines 0-48 are vertical (dir 0).
    # Lines 49-2449 are slopes.
    # Line 49 + 49*ma + 49*mb + ... 
    # Wait, the loop structure in _precompute_lines:
    # ma, mb, ca, cb.
    # So line index for slope (ma,mb) is 49 + 49*49*ma + 49*49*mb?
    # No, 49 lines per (ma,mb).
    # Total slope lines = 49*49*49 = 117649? No.
    # p=7.
    # Vertical: 7*7 = 49 lines.
    # Slopes: 7*7 (m values) * 7*7 (offsets) = 49*49 = 2401 lines.
    # Total 2450 lines.
    
    for i in range(49):
        dir_of_line[i] = 0
    for ma in range(7):
        for mb in range(7):
            base = 49 + (ma*7 + mb)*49
            for _ in range(49):
                dir_of_line[base] = 1 + ma*7 + mb
                base += 1
                
    # Greedy Add
    added = 0
    for c_idx in candidates:
        if time.time() > _DEADLINE - 60:
            break
            
        if in_S[c_idx]:
            continue
            
        # Check if c_idx has a clean direction
        clean_dirs = np.where(blocked[c_idx] == 0)[0]
        if len(clean_dirs) == 0:
            continue
            
        # If we add c_idx, it will block directions for other points on lines through c_idx
        # We need to pick a "witness" direction for c_idx?
        # No, c_idx just needs SOME clean direction.
        # But adding c_idx might block the ONLY clean direction of some other point y in S.
        # So we must check all y in S.
        
        # Optimization: Only check y in S that are on lines through c_idx.
        affected_y = set()
        for l_idx in point_lines[c_idx]:
            d = dir_of_line[l_idx]
            # If d is clean for c_idx, adding c_idx blocks d for others on this line.
            # Actually, adding c_idx blocks direction d for ALL other points on L_{c_idx, d}.
            # If any of those points y in S has d as their ONLY clean direction, y becomes invalid.
            
            # We don't know which direction c_idx will use as witness.
            # But c_idx can use ANY clean direction.
            # To be safe, we assume c_idx will use the direction that causes LEAST damage.
            # Or, we just check if adding c_idx makes any y in S invalid, REGARDLESS of c_idx's witness.
            # If y becomes invalid, we can't add c_idx (unless we remove y).
            
            # Let's just check: Does adding c_idx block a direction d such that some y in S has ONLY d clean?
            
            # This is expensive.
            # Simpler: Just add c_idx if it has a clean direction.
            # Then fix the set by removing invalid points.
            
            pass
            
        # For speed, let's just add it and mark blocked
        in_S[c_idx] = 1
        added += 1
        
        for l_idx in point_lines[c_idx]:
            d = dir_of_line[l_idx]
            # c_idx is on line l_idx with direction d.
            # This blocks direction d for all other points on l_idx.
            line_pts = lines[l_idx]
            for y in line_pts:
                if y != c_idx:
                    blocked[y, d] = 1
                    
    # Local Search: Remove invalid points
    # A point x in S is invalid if blocked[x] has no 0s.
    changed = True
    while changed:
        if time.time() > _DEADLINE - 30:
            break
        changed = False
        for x in range(N_TOTAL):
            if in_S[x]:
                if np.all(blocked[x] == 1):
                    # Remove x
                    in_S[x] = 0
                    changed = True
                    # Unblock directions for others
                    for l_idx in point_lines[x]:
                        d = dir_of_line[l_idx]
                        line_pts = lines[l_idx]
                        for y in line_pts:
                            if y != x:
                                # Check if any other S point blocks d for y
                                # If not, unblock
                                # This is O(49) per point.
                                # We can just recompute blocked for y?
                                # Or maintain a count of blockers.
                                # Let's just recompute blocked for affected y.
                                pass
                                
    # Rebuild blocked from scratch for correctness after removals
    blocked = np.zeros((N_TOTAL, N_DIRS), dtype=np.int8)
    for x in range(N_TOTAL):
        if in_S[x]:
            for l_idx in point_lines[x]:
                d = dir_of_line[l_idx]
                line_pts = lines[l_idx]
                for y in line_pts:
                    if y != x:
                        blocked[y, d] = 1
                        
    # Final validation and cleanup
    for x in range(N_TOTAL):
        if in_S[x]:
            if np.all(blocked[x] == 1):
                in_S[x] = 0
                # Repeat cleanup if needed, but we assume removal makes it valid
                # Actually removal of x unblocks others, so it might make them valid.
                # But we just need S to be valid.
                
    # Build result
    points = []
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
