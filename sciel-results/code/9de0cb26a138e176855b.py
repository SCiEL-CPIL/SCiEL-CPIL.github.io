# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using greedy with proper bookkeeping.
    """
    start_time = time.time()
    deadline = start_time + 560  # safety margin before 630s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    n_points = q * q  # 2401
    
    # Build field operations
    idx_to_ab = np.zeros((q, 2), dtype=np.int64)
    ab_to_idx = np.full((p, p), -1, dtype=np.int64)
    
    for i in range(p):
        for j in range(p):
            idx = i * p + j
            idx_to_ab[idx] = [i, j]
            ab_to_idx[i, j] = idx
    
    # Precompute addition and multiplication tables
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    
    for a1 in range(q):
        a1a, a1b = idx_to_ab[a1]
        for b1 in range(q):
            b1a, b1b = idx_to_ab[b1]
            sum_a = (a1a + b1a) % p
            sum_b = (a1b + b1b) % p
            add_table[a1, b1] = ab_to_idx[sum_a, sum_b]
            
            prod_a = (a1a * b1a + a1b * b1b * k) % p
            prod_b = (a1a * b1b + a1b * b1a) % p
            mul_table[a1, b1] = ab_to_idx[prod_a, prod_b]
    
    def point_to_idx(xa, xb, ya, yb):
        return (xa * p + xb) * q + (ya * p + yb)
    
    # Precompute lines
    point_line_id = np.zeros((50, n_points), dtype=np.int64)
    lines = [None] * 50
    
    for d in range(50):
        lines[d] = [[] for _ in range(q)]
        
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        x_idx = xa * p + xb
                        y_idx = ya * p + yb
                        pt_idx = x_idx * q + y_idx
                        
                        if d < 49:
                            m_idx = d
                            mx = mul_table[m_idx, x_idx]
                            mxa, mxb = idx_to_ab[mx]
                            neg_mx = ab_to_idx[(-mxa) % p, (-mxb) % p]
                            c_idx = add_table[y_idx, neg_mx]
                            line_id = c_idx
                        else:
                            line_id = x_idx
                        
                        point_line_id[d, pt_idx] = line_id
                        lines[d][line_id].append(pt_idx)
    
    # Convert lines to numpy arrays for speed
    lines_arr = []
    for d in range(50):
        d_lines = []
        for lid in range(q):
            d_lines.append(np.array(lines[d][lid], dtype=np.int32))
        lines_arr.append(d_lines)
    
    def greedy_search(seed_offset=0):
        """Greedy: try to add each point to excluded set S if validity maintained."""
        rng = np.random.default_rng(seed_offset)
        
        in_S = np.zeros(n_points, dtype=bool)
        clean_count = np.full(n_points, 50, dtype=np.int32)
        
        order = rng.permutation(n_points)
        
        for pt_idx in order:
            if in_S[pt_idx]:
                continue
            
            # Tentatively add pt_idx to S
            in_S[pt_idx] = True
            
            # Track affected points and their old clean counts
            affected_pts = []
            for d in range(50):
                lid = point_line_id[d, pt_idx]
                line_pts = lines_arr[d][lid]
                for x in line_pts:
                    if x != pt_idx:
                        clean_count[x] -= 1
                        affected_pts.append(x)
            
            # Check validity: no affected point should have clean_count == 0
            valid = True
            for x in affected_pts:
                if clean_count[x] == 0:
                    valid = False
                    break
            
            if not valid:
                in_S[pt_idx] = False
                for x in affected_pts:
                    clean_count[x] += 1
            
            if time.time() > deadline - 30:
                break
        
        return in_S
    
    # Try multiple random orderings, keep best
    best_S = None
    best_complement = 0
    
    n_tries = 15
    for i in range(n_tries):
        S_candidate = greedy_search(seed_offset=i)
        complement = np.count_nonzero(S_candidate)
        if complement > best_complement:
            best_complement = complement
            best_S = S_candidate
        
        if time.time() > deadline - 60:
            break
    
    if best_S is None:
        # Fallback: exclude nothing (N = all points, valid but score 0)
        best_S = np.zeros(n_points, dtype=bool)
    
    # Build the result: N = all points NOT in S
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = point_to_idx(xa, xb, ya, yb)
                    if not best_S[idx]:
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
