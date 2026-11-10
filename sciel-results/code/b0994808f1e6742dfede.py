# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    
    Strategy:
    1. Use the structural construction from Inspiration 1 as a high-quality seed.
       This excludes points from the Hermitian unital U that do not have a 'special' y-coordinate.
    2. Use the greedy local search from Inspiration 3 to refine the set.
       We attempt to add more points to the excluded set S if doing so does not
       cause any point in F_49^2 to lose its last 'clear' line.
    
    This combines the strong geometric prior with local optimization to maximize
    the complement size (2401 - |N|).
    """
    p = 7
    q = p * p
    total_points = q * q
    
    # Precompute field operations
    # F_49 = F_7[alpha], alpha^2 = 3
    k = 3
    
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = (a0*b0 + a1*b1*k) + (a0*b1 + a1*b0)*alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    # Generate all points
    # Index mapping: (xa, xb, ya, yb) -> index
    # We use a flat index for numpy operations
    points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) for ya in range(p) for yb in range(p)]
    point_index = {pt: i for i, pt in enumerate(points)}
    
    # Precompute lines
    # Directions: (1, m) for m in F_49, and (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append((1, ma, 1, mb)) # (1, m)
    directions.append((0, 1, 0, 1)) # (0, 1)
    
    # For each point, store the list of other points on each line through it
    # lines_through[i] = list of 50 lists, each list contains 49 point indices
    lines_through = []
    for i, pt in enumerate(points):
        pt_lines = []
        for d in directions:
            line_pts = []
            for t in range(q):
                if t == 0:
                    continue
                # x + t*v
                # v = (vx, vy), x = (x, y)
                # t is in F_49
                ta, tb = t // p, t % p
                t_val = (ta, tb)
                
                # t * v
                tvx = mul(t_val, (d[0], d[1]))
                tvy = mul(t_val, (d[2], d[3]))
                
                # x + t*v
                nx = add((pt[0], pt[1]), tvx)
                ny = add((pt[2], pt[3]), tvy)
                
                npt = (nx[0], nx[1], ny[0], ny[1])
                line_pts.append(point_index[npt])
            pt_lines.append(line_pts)
        lines_through.append(pt_lines)
        
    # --- Step 1: Structural Seed (Inspiration 1) ---
    # Identify the initial excluded set S_seed
    # In Inspiration 1:
    # p = 7
    # k = 3 (smallest quadratic non-residue)
    # y_special = {(u^2 mod p, v) : u, v in F_p}
    # Excluded if in_unital and not in_y_special
    # in_unital: 2*xa + ya*ya - yb*yb*k == 0 (mod p)
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
            
    S_seed = set()
    for i, pt in enumerate(points):
        xa, xb, ya, yb = pt
        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
        in_y_special = (ya, yb) in y_special
        if in_unital and not in_y_special:
            S_seed.add(i)
            
    # --- Step 2: Greedy Refinement (Inspiration 3) ---
    # Initialize clear_lines
    # clear_lines[i] = number of lines through i that contain no points of S \ {i}
    clear_lines = np.full(total_points, 50, dtype=np.int32)
    
    # Apply the seed S to clear_lines
    # For each s in S_seed, for each line through s, for each x on that line (x != s):
    #   clear_lines[x] -= 1
    
    # Optimization: Precompute the list of neighbors for each point on each line?
    # We already have lines_through.
    
    for s in S_seed:
        for line_pts in lines_through[s]:
            for x in line_pts:
                clear_lines[x] -= 1
                
    # Now try to add more points to S
    # We iterate through all points not in S_seed.
    # Order doesn't strictly matter for the greedy check, but we can try a few orders if time permits.
    # Given the small size, a single pass or a few passes should suffice.
    
    S = set(S_seed)
    
    # We will try to add points one by one.
    # To be safe and potentially find more, we can do a few passes or just one full pass.
    # One pass is O(N * 50 * 49) which is fast.
    
    # Let's do a few passes to ensure we catch dependencies (though the greedy condition 
    # "if clear_lines[x] == 1 then conflict" is local, adding one point might open up 
    # possibilities? No, adding points to S only DECREASES clear_lines for others.
    # So if a point is invalid now, it will remain invalid later.
    # Thus, a single pass in any order is sufficient to find the maximal set reachable 
    # by this specific greedy rule from the seed? 
    # Actually, the greedy rule is: add s if no neighbor has clear_lines == 1.
    # If we skip s, clear_lines of neighbors don't change. 
    # If we add s, clear_lines of neighbors decrease.
    # This is a monotonic process. The final set S depends on the order of processing.
    # However, since we start from a strong seed, the remaining "gaps" might be fillable 
    # regardless of order, or order might matter. 
    # Let's do a few random shuffles of the remaining candidates to be robust.
    
    remaining_candidates = [i for i in range(total_points) if i not in S]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s margin for output generation
    
    best_S = set(S)
    
    # Try the natural order first
    candidates = list(remaining_candidates)
    current_S = set(S_seed)
    current_clear = clear_lines.copy()
    
    # Helper to run greedy on a given candidate list starting from initial state
    def run_greedy(cands, init_S, init_clear):
        S_local = set(init_S)
        clear_local = init_clear.copy()
        
        for s_idx in cands:
            if s_idx in S_local:
                continue
            # Check if adding s_idx causes any conflict
            conflict = False
            for line_pts in lines_through[s_idx]:
                for x_idx in line_pts:
                    if clear_local[x_idx] == 1:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                S_local.add(s_idx)
                for line_pts in lines_through[s_idx]:
                    for x_idx in line_pts:
                        clear_local[x_idx] -= 1
        return S_local

    # Run 1: Natural order
    S1 = run_greedy(candidates, S_seed, clear_lines)
    if len(S1) > len(best_S):
        best_S = S1
        
    # Run a few random shuffles if time permits
    import random
    random.seed(42)
    
    num_tries = 10
    for _ in range(num_tries):
        if time.time() > deadline:
            break
        random.shuffle(candidates)
        S_try = run_greedy(candidates, S_seed, clear_lines)
        if len(S_try) > len(best_S):
            best_S = S_try
            
    # Construct N
    # N is the complement of best_S
    N = []
    for i, pt in enumerate(points):
        if i not in best_S:
            N.append([[pt[0], pt[1]], [pt[2], pt[3]]])
    
    # Ensure we return a valid array
    if not N:
        # Fallback, should not happen
        return np.empty((0, 2, 2), dtype=np.int64)
        
    return np.array(N, dtype=np.int64)

# EVOLVE-BLOCK-END
