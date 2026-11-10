# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy algorithm with efficient line tracking.
    
    Strategy:
    - Work with the excluded set S = F_49^2 \ N
    - For each point x, track how many "clean" lines pass through x (lines where all other 48 points are in N)
    - Greedily add points to S that cause the least damage to clean line counts
    - Use randomized ordering for diversity
    """
    start_time = time.time()
    deadline = start_time + 570  # safety margin before 630s limit
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
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
    
    # Point indexing: point (x, y) where x, y in F_49 -> idx = x * q + y
    # x = (xa, xb), y = (ya, yb)
    # x_idx = xa * p + xb, y_idx = ya * p + yb
    # point_idx = x_idx * q + y_idx
    
    def point_to_idx(xa, xb, ya, yb):
        return (xa * p + xb) * q + (ya * p + yb)
    
    # Precompute all 50 directions and for each direction, the 49 parallel lines
    # Direction (1, m) for m in F_49, and direction (0, 1)
    
    # For each direction, compute the line assignment for each point
    # line_id for direction d and point (x,y) determines which parallel line it's on
    
    # For direction (1, m): line through (x0, y0) has equation y - m*x = y0 - m*x0
    # For direction (0, 1): line through (x0, y0) has equation x = x0
    
    # Precompute for each direction d (0..49), for each point, the line_id
    # and for each (d, line_id), the set of points on that line
    
    n_points = q * q  # 2401
    
    # For efficiency, precompute line membership
    # lines[d][line_id] = list of point indices on that line
    
    # Direction 0..48: v = (1, m) where m_idx = d
    # Direction 49: v = (0, 1)
    
    # For direction (1, m): 
    #   Line through (x0, y0): points (x0 + t, y0 + m*t) for t in F_49
    #   The line is determined by the "intercept" c = y0 - m*x0
    #   For a given point (x, y), the line_id is c = y - m*x
    
    # For direction (0, 1):
    #   Line through (x0, y0): points (x0, y0 + t) for t in F_49
    #   The line is determined by x = x0
    #   For a given point (x, y), the line_id is x
    
    # Precompute: for each direction d, for each point, the line_id
    point_line_id = np.zeros((50, n_points), dtype=np.int64)
    
    # For each direction and line_id, precompute the list of points
    # We'll store as a list of lists: lines[d][line_id] = [point_idx1, point_idx2, ...]
    
    lines = [None] * 50
    
    # Precompute m*t for all m, t in F_49
    # mul_table[m, t] already gives this
    
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
                            # Direction (1, m) where m_idx = d
                            m_idx = d
                            # c = y - m*x in F_49
                            # c_idx = add_table[y_idx, neg(m*x)]
                            # m*x = mul_table[m_idx, x_idx]
                            mx = mul_table[m_idx, x_idx]
                            # neg(mx)
                            mxa, mxb = idx_to_ab[mx]
                            neg_mx = ab_to_idx[(-mxa) % p, (-mxb) % p]
                            c_idx = add_table[y_idx, neg_mx]
                            line_id = c_idx
                        else:
                            # Direction (0, 1)
                            line_id = x_idx
                        
                        point_line_id[d, pt_idx] = line_id
                        lines[d][line_id].append(pt_idx)
    
    # Now we have all lines precomputed.
    # For each point, the 50 lines through it are:
    #   for d in 0..49: line_id = point_line_id[d, pt_idx]
    #   the line is lines[d][line_id]
    
    # Greedy algorithm:
    # 1. Start with S = empty (all points in N)
    # 2. For each point, clean_count[pt] = 50 (all 50 lines are clean)
    # 3. For each point p, compute the "damage" of adding p to S:
    #    For each direction d, let L = lines[d][point_line_id[d, p]]
    #    For each other point x in L, clean_count[x] decreases by 1
    #    If any x would have clean_count[x] = 0, we can't add p (or it's bad)
    # 4. Greedily add points with minimum damage
    
    # But this is O(n_points * 50 * 49) per addition = O(2401 * 50 * 49) ≈ 5.9M per step
    # With 2401 steps, that's ~14 billion operations - too slow.
    
    # Better approach: use a priority queue and lazy updates.
    # Or: use a different strategy.
    
    # Alternative: random greedy with occasional validation.
    # 1. Randomly order points
    # 2. For each point, tentatively add to S
    # 3. Check if validity is maintained (all points have at least one clean line)
    # 4. If valid, keep; else, remove
    
    # Validation: for each point, check if clean_count > 0
    # This is O(n_points) per check, so O(n_points^2) total = ~5.8M - feasible!
    
    # Let's implement this:
    
    def try_greedy(seed_offset=0):
        """Greedy algorithm with random ordering."""
        rng = np.random.default_rng(seed_offset)
        
        # Initialize
        S = np.zeros(n_points, dtype=bool)  # True if in excluded set
        clean_count = np.full(n_points, 50, dtype=np.int32)
        
        # Random order
        order = rng.permutation(n_points)
        
        for pt_idx in order:
            if S[pt_idx]:
                continue
            
            # Tentatively add pt_idx to S
            S[pt_idx] = True
            
            # Update clean_count for all other points on lines through pt_idx
            affected = []
            for d in range(50):
                line_id = point_line_id[d, pt_idx]
                line_points = lines[d][line_id]
                for x in line_points:
                    if x != pt_idx:
                        clean_count[x] -= 1
                        affected.append(x)
            
            # Check if any point has clean_count == 0
            # Only need to check affected points
            valid = True
            for x in affected:
                if clean_count[x] == 0:
                    valid = False
                    break
            
            if not valid:
                # Undo
                S[pt_idx] = False
                for x in affected:
                    clean_count[x] += 1
            
            # Check time
            if time.time() > deadline - 30:
                break
        
        return np.count_nonzero(S)
    
    # Try multiple random seeds
    best_size = 0
    best_S = None
    
    n_tries = 20
    for i in range(n_tries):
        size = try_greedy(seed_offset=i)
        if size > best_size:
            best_size = size
            if size > 0:
                # We need to reconstruct S, but we didn't save it
                # Let's modify try_greedy to return S as well
                pass
        
        if time.time() > deadline - 60:
            break
    
    # The above approach doesn't save S. Let me redo with proper tracking.
    
    def try_greedy_save(seed_offset=0):
        """Greedy algorithm that saves the final S."""
        rng = np.random.default_rng(seed_offset)
        
        S = np.zeros(n_points, dtype=bool)
        clean_count = np.full(n_points, 50, dtype=np.int32)
        
        order = rng.permutation(n_points)
        
        for pt_idx in order:
            if S[pt_idx]:
                continue
            
            S[pt_idx] = True
            
            affected = []
            for d in range(50):
                line_id = point_line_id[d, pt_idx]
                line_points = lines[d][line_id]
                for x in line_points:
                    if x != pt_idx:
                        clean_count[x] -= 1
                        affected.append(x)
            
            valid = True
            for x in affected:
                if clean_count[x] == 0:
                    valid = False
                    break
            
            if not valid:
                S[pt_idx] = False
                for x in affected:
                    clean_count[x] += 1
            
            if time.time() > deadline - 30:
                break
        
        return S
    
    best_S = None
    best_size = 0
    
    for i in range(10):  # Try 10 random orderings
        S_candidate = try_greedy_save(seed_offset=i)
        size = np.count_nonzero(S_candidate)
        if size > best_size:
            best_size = size
            best_S = S_candidate
        
        if time.time() > deadline - 60:
            break
    
    if best_S is None:
        # Fallback: empty S (all points in N)
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
