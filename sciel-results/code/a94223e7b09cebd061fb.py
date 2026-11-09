import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k_field = 3
    
    def to_int(a, b):
        return a + b * p

    def to_pair(x):
        return (x % p, (x // p) % p)

    # Precompute field arithmetic
    f_add_table = np.zeros((q, q), dtype=np.int32)
    f_mul_table = np.zeros((q, q), dtype=np.int32)
    f_neg_table = np.zeros(q, dtype=np.int32)
    
    for x in range(q):
        xa, xb = to_pair(x)
        f_neg_table[x] = to_int((-xa) % p, (-xb) % p)
        for y in range(q):
            ya, yb = to_pair(y)
            f_add_table[x, y] = to_int((xa + ya) % p, (xb + yb) % p)
            real = (xa * ya + xb * yb * k_field) % p
            imag = (xa * yb + xb * ya) % p
            f_mul_table[x, y] = to_int(real, imag)

    directions = [(1, m) for m in range(q)] + [(0, 1)]
    n_dir = len(directions)
    
    # Precompute lines for each direction and point
    # line_points[p_idx][d_idx] = list of point indices on the line through p_idx in direction d_idx
    # Actually, we just need to update counts.
    # For a point Q removed from C, we need to update cnt[P][d] for all P on line(Q, d).
    # Let's precompute for each Q and d, the list of P such that Q is on line(P, d).
    # This is the same as line(Q, d).
    
    # To save memory, we can compute on the fly or store the line members.
    # line_members[Q][d] is a list of P indices.
    line_members = [[[0 for _ in range(q)] for _ in range(n_dir)] for _ in range(q*q)]
    # This is 2401 * 50 * 49 ints. ~6M ints. 24MB. OK.
    
    # Actually, we can just store the indices.
    # Let's precompute the neighbors for each point and direction.
    # neighbors[Q][d] = list of P such that Q is on line(P, d).
    # Note: line(P, d) = line(Q, d) if Q is on it.
    # So for a fixed d, the lines partition the plane.
    # For each d, for each line L, for each Q in L, neighbors[Q][d] = L.
    
    # Let's build neighbors[Q][d]
    # neighbors is a list of lists of lists.
    # neighbors[Q][d] is a numpy array of point indices.
    
    neighbors = [None] * (q * q)
    for Q in range(q * q):
        neighbors[Q] = [None] * n_dir
    
    for d_idx, (vx, vy) in enumerate(directions):
        # Find all lines in this direction
        # A line is determined by a point not on the "base" line?
        # Simpler: for each P, the line is P + t*v.
        # We can just iterate P and mark the line.
        # But we need to group them.
        
        # For a given direction v, the lines are parallel.
        # We can identify a line by its "base point" which is the point on the line with minimum index?
        # Or just iterate all P, compute the line, and add P to the line's list.
        
        # Let's use a dictionary: line_id -> list of points
        # line_id can be the index of the point on the line with smallest x?
        # Or just use the set of points as the key? No, too slow.
        
        # Better:
        # For each P, the line is unique.
        # We can compute the "offset" of the line.
        # If v=(1, m), line is y - m x = c.
        # If v=(0, 1), line is x = c.
        
        # Let's compute the line ID for each point for each direction.
        # Then group points by line ID.
        
        line_ids = [0] * (q * q)
        for P in range(q * q):
            px, py = to_pair(P)
            if vx == 1:
                # y - m x = c => c = y - m x
                # c is an element of F_q
                # c = py - f_mul(m, px)
                c = f_neg_table[f_mul_table[vy, px]]
                c = f_add_table[py, c]
                line_ids[P] = c
            else: # vx == 0, vy == 1
                # x = c
                line_ids[P] = px # px is x_a + x_b * p? No, px is the index.
                # Wait, px is the index of x in F_q.
                # The line is x = const.
                # So line ID is the x-coordinate.
                line_ids[P] = px
        
        # Now group points by line ID
        # line_id can be 0..q-1
        lines = [[] for _ in range(q)]
        for P in range(q * q):
            lines[line_ids[P]].append(P)
            
        # For each line, for each point Q in the line, neighbors[Q][d_idx] = line
        for line_pts in lines:
            for Q in line_pts:
                neighbors[Q][d_idx] = line_pts # List of ints

    # Function to check if C is valid
    # cnt[P][d] = number of points in C on line(P, d)
    # Condition:
    # if P in C: exists d, cnt[P][d] == 1
    # if P not in C: exists d, cnt[P][d] == 0
    
    def is_valid(C_set, cnt):
        for P in range(q * q):
            if P in C_set:
                if not any(cnt[P][d] == 1 for d in range(n_dir)):
                    return False
            else:
                if not any(cnt[P][d] == 0 for d in range(n_dir)):
                    return False
        return True

    def compute_cnt(C_set):
        cnt = [[0] * n_dir for _ in range(q * q)]
        for Q in C_set:
            for d in range(n_dir):
                for P in neighbors[Q][d]:
                    cnt[P][d] += 1
        return cnt

    best_N_size = 0
    best_C = None
    
    deadline = time.time() + 590
    
    while time.time() < deadline:
        # Random initialization
        # Start with a line, or random points
        # Let's try random set of size q
        size_C = q
        C_list = random.sample(range(q * q), size_C)
        C_set = set(C_list)
        
        cnt = compute_cnt(C_set)
        
        if is_valid(C_set, cnt):
            # Try to remove points
            changed = True
            while changed and time.time() < deadline:
                changed = False
                # Shuffle to avoid bias
                random.shuffle(C_list)
                for Q in C_list:
                    if Q not in C_set:
                        continue
                    
                    # Tentatively remove Q
                    C_set.remove(Q)
                    
                    # Update cnt
                    # For each d, for each P in neighbors[Q][d], cnt[P][d] -= 1
                    # We need to check validity after update
                    # To be fast, we can just recompute or update and check.
                    # Update is O(q^2). Check is O(q^2).
                    
                    for d in range(n_dir):
                        for P in neighbors[Q][d]:
                            cnt[P][d] -= 1
                    
                    if is_valid(C_set, cnt):
                        changed = True
                        # Keep removed
                    else:
                        # Restore
                        C_set.add(Q)
                        for d in range(n_dir):
                            for P in neighbors[Q][d]:
                                cnt[P][d] += 1
            
            # Update best
            if len(C_set) < best_N_size or best_N_size == 0:
                # Note: we want to MAXIMIZE N, so MINIMIZE C.
                # best_N_size should track min C size?
                # Let's track min_C_size
                pass
                
            if best_C is None or len(C_set) < len(best_C):
                best_C = set(C_set)
                
        # Try smaller initial sizes?
        # If q fails, try q+1? No, we want small C.
        # If random q works, we prune it.
        # Maybe try q, q+1, ... to find a valid one if q is too hard?
        # But random q is likely valid.

    if best_C is None:
        # Fallback: return a line (size q)
        best_C = set(range(q)) # y=0?
        # Let's construct y=0
        best_C = set()
        for x in range(q):
            y = 0
            idx = x * q + y
            best_C.add(idx)
            
    # Construct N
    N = []
    for i in range(q * q):
        if i not in best_C:
            x_idx, y_idx = to_pair(i)
            # Wait, i is index in 0..q^2-1.
            # x_idx = i // q, y_idx = i % q
            x_idx = i // q
            y_idx = i % q
            xa, xb = to_pair(x_idx)
            ya, yb = to_pair(y_idx)
            N.append([[xa, xb], [ya, yb]])
            
    return np.array(N, dtype=np.int64)
