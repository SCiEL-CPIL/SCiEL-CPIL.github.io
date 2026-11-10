# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 610.0  # safety margin before 630s kill

p = 7
K = 3  # alpha^2 = 3
N_TOTAL = p**4  # 2401
N_DIRS = p * p + 1  # 50 directions: (0,1) and (1, m)

def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))

def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb

def _precompute_lines():
    """
    Precompute all lines in F_49^2.
    Returns:
      lines: list of arrays, each of shape (49,), containing point indices on that line.
      line_dirs: list of direction indices for each line.
      point_to_lines: for each point, list of line indices it belongs to.
    """
    lines = []
    line_dirs = []
    point_to_lines = [[] for _ in range(N_TOTAL)]
    
    # Direction 0: v = (0, 1)
    # Lines parameterized by (xa, xb)
    # c_a, c_b are effectively ya, yb for the "base" but since x is fixed, 
    # the line is just all y for a fixed x.
    # Actually, for v=(0,1), x is constant. So line is determined by (xa, xb).
    for xa in range(p):
        for xb in range(p):
            line_idx = len(lines)
            line_points = np.empty(49, dtype=np.int32)
            for i, ya in enumerate(range(p)):
                for j, yb in enumerate(range(p)):
                    idx = _point_to_idx(xa, xb, ya, yb)
                    line_points[i * p + j] = idx
                    point_to_lines[idx].append(line_idx)
            lines.append(line_points)
            line_dirs.append(0)
            
    # Directions 1..49: v = (1, m) for m = ma + mb*alpha
    # Line determined by (c_a, c_b) where:
    # c_a = ya - ma*xa - 3*mb*xb
    # c_b = yb - ma*xb - mb*xa
    for ma in range(p):
        for mb in range(p):
            dir_idx = 1 + ma * p + mb
            for ca in range(p):
                for cb in range(p):
                    line_idx = len(lines)
                    line_points = np.empty(49, dtype=np.int32)
                    for i, xa in enumerate(range(p)):
                        for j, xb in enumerate(range(p)):
                            ya = (ca + ma * xa + 3 * mb * xb) % p
                            yb = (cb + ma * xb + mb * xa) % p
                            idx = _point_to_idx(xa, xb, ya, yb)
                            line_points[i * p + j] = idx
                            point_to_lines[idx].append(line_idx)
                    lines.append(line_points)
                    line_dirs.append(dir_idx)
    
    return lines, line_dirs, point_to_lines

def nikodym_set() -> np.ndarray:
    start = time.time()
    
    # Precomputation
    lines, line_dirs, point_to_lines = _precompute_lines()
    n_lines = len(lines) # 2450
    
    # Convert lines to a 2D array for vectorization
    # lines_arr: shape (n_lines, 49)
    lines_arr = np.array(lines, dtype=np.int32)
    
    # Strategy: 
    # Model the problem as finding a large set S (complement of N) such that for every point x,
    # there is at least one direction v where the line L(x, v) contains no points of S other than possibly x.
    # Equivalently, for every x, there exists v such that count_S(L(x,v)) <= (1 if x in S else 0).
    
    # We will use a local search approach starting from a random sparse S.
    # We try to add points to S greedily if they don't violate the condition for any existing point in S or any point in N.
    # Actually, the condition must hold for ALL points x in F_49^2.
    
    # Let's define the "safety" of a point x.
    # x is "safe" if there exists a line L through x such that count_S(L) <= (1 if x in S else 0).
    # If x is not safe, it is "bad".
    
    # We want to maximize |S| such that no point is bad.
    
    # Let's start with a random S of small size and grow it.
    # Or start with S = empty and add points.
    
    # Let's try a simulated annealing or hill-climbing approach.
    # State: S (boolean array of size 2401).
    # Objective: Maximize |S| subject to all points being safe.
    
    # We can use a greedy algorithm:
    # 1. Initialize S = empty.
    # 2. While we can add a point:
    #    For each point c not in S:
    #       Check if adding c keeps all points safe.
    #       This is expensive to check for all points.
    
    # Let's use a different heuristic.
    # Start with a random S of size ~1000.
    # Check for bad points.
    # If there are bad points, remove points from S to fix them.
    # This is similar to the current program but let's try a different initialization and removal strategy.
    
    # Let's try a "random restart" strategy with a smarter removal.
    
    best_S_size = 0
    best_S = None
    
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() > _DEADLINE - 10:
            break
        
        # Random initialization: choose ~20% of points randomly
        density = 0.2
        in_S = np.random.random(N_TOTAL) < density
        count_S_on_line = np.zeros(n_lines, dtype=np.int32)
        
        # Compute count_S_on_line
        for L in range(n_lines):
            count_S_on_line[L] = np.sum(in_S[lines_arr[L]])
        
        def compute_bad():
            bad = np.zeros(N_TOTAL, dtype=np.int32)
            for L in range(n_lines):
                c = count_S_on_line[L]
                if c == 0:
                    continue
                pts = lines_arr[L]
                thresholds = in_S[pts].astype(np.int32)
                conditions = c > thresholds
                if np.any(conditions):
                    affected_pts = pts[conditions]
                    bad[affected_pts] += 1
            return bad
        
        bad = compute_bad()
        
        # Local search: try to remove points to resolve bad points.
        # If a point x is bad (bad[x] == 50), it means for ALL lines L through x, count_S(L) > (1 if x in S else 0).
        # To fix x, we must find a line L through x such that we can reduce count_S(L) to <= (1 if x in S else 0).
        # This requires removing all S points on L except possibly x.
        
        # We can try to remove points that are "responsible" for the most bad points.
        
        max_iters = 1000
        for _ in range(max_iters):
            if time.time() > _DEADLINE - 5:
                break
            
            bad = compute_bad()
            bad_pts = np.where(bad == 50)[0]
            if len(bad_pts) == 0:
                break
            
            # Pick a random bad point
            x = bad_pts[np.random.randint(len(bad_pts))]
            
            # Pick a random line through x
            lines_through_x = point_to_lines[x]
            L = lines_through_x[np.random.randint(len(lines_through_x))]
            
            # Determine which points to remove from L
            pts_on_L = lines_arr[L]
            if in_S[x]:
                # Remove all y in S on L, y != x
                to_remove = pts_on_L[in_S[pts_on_L]]
                to_remove = to_remove[to_remove != x]
            else:
                # Remove all y in S on L
                to_remove = pts_on_L[in_S[pts_on_L]]
            
            for y in to_remove:
                if in_S[y]:
                    in_S[y] = False
                    for Ly in point_to_lines[y]:
                        count_S_on_line[Ly] -= 1
        
        # Check if valid
        bad = compute_bad()
        if np.any(bad == 50):
            # Invalid. Try to fix more aggressively or skip.
            # Let's try to remove points until valid.
            for _ in range(1000):
                if time.time() > _DEADLINE - 5:
                    break
                bad = compute_bad()
                bad_pts = np.where(bad == 50)[0]
                if len(bad_pts) == 0:
                    break
                x = bad_pts[np.random.randint(len(bad_pts))]
                lines_through_x = point_to_lines[x]
                L = lines_through_x[np.random.randint(len(lines_through_x))]
                pts_on_L = lines_arr[L]
                if in_S[x]:
                    to_remove = pts_on_L[in_S[pts_on_L]]
                    to_remove = to_remove[to_remove != x]
                else:
                    to_remove = pts_on_L[in_S[pts_on_L]]
                for y in to_remove:
                    if in_S[y]:
                        in_S[y] = False
                        for Ly in point_to_lines[y]:
                            count_S_on_line[Ly] -= 1
        
        bad = compute_bad()
        if not np.any(bad == 50):
            # Valid!
            S_size = np.sum(in_S)
            if S_size > best_S_size:
                best_S_size = S_size
                best_S = in_S.copy()
    
    if best_S is None:
        # Fallback: return a small valid set (e.g., empty S -> N = all points)
        in_S = np.zeros(N_TOTAL, dtype=np.bool_)
    else:
        in_S = best_S
    
    # Build result: N = complement of S
    points = []
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
