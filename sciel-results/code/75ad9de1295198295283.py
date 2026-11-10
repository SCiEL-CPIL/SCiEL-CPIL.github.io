import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 570.0  # safety margin before 630s kill

    p = 7
    k = 3  # alpha^2 = 3, non-residue mod 7
    total_points = p * p * p * p  # 2401
    num_dirs = p * p + 1  # 50 directions

    def point_flat(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    # Precompute line membership: for each direction d and point x,
    # line_pts[d][x] = array of 49 point indices (including x) on L(d, x)
    # line_other[d][x] = array of 48 point indices (excluding x)
    
    # For efficient verification, we need:
    # For direction d, point x is "covered" if all other points on L(d,x) are in N.
    # i.e., in_N[line_other[d][x]] are all True.

    # Let's precompute line_other as a 3D array: line_other[d][x] = array of 48 indices
    # Shape: (50, 2401, 48)
    
    line_other = np.zeros((num_dirs, total_points, p*p - 1), dtype=np.int32)
    
    # Direction 0: v = (0, 1)
    # x + t*(0,1) = (x_x, x_y + t) where t is in F_49
    # Point (xa, xb, ya, yb) + t*(0,1) = (xa, xb, ya+ta, yb+tb) where t = (ta, tb)
    d = 0
    idx = 0
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    x_idx = point_flat(xa, xb, ya, yb)
                    t_idx = 0
                    for ta in range(p):
                        for tb in range(p):
                            if ta == 0 and tb == 0:
                                continue
                            nya = (ya + ta) % p
                            nyb = (yb + tb) % p
                            line_other[d, x_idx, t_idx] = point_flat(xa, xb, nya, nyb)
                            t_idx += 1
                    idx += 1

    # Directions 1..49: v = (1, m) for m in F_49
    d = 1
    for ma in range(p):
        for mb in range(p):
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            x_idx = point_flat(xa, xb, ya, yb)
                            t_idx = 0
                            for ta in range(p):
                                for tb in range(p):
                                    if ta == 0 and tb == 0:
                                        continue
                                    # new_x = (xa, xb) + (ta, tb)
                                    nxa = (xa + ta) % p
                                    nxb = (xb + tb) % p
                                    # new_y = (ya, yb) + (ta, tb)*(ma, mb)
                                    # t*m = (ta*ma + tb*mb*k) + (ta*mb + tb*ma)*alpha
                                    tma = (ta * ma + tb * mb * k) % p
                                    tmb = (ta * mb + tb * ma) % p
                                    nya = (ya + tma) % p
                                    nyb = (yb + tmb) % p
                                    line_other[d, x_idx, t_idx] = point_flat(nxa, nxb, nya, nyb)
                                    t_idx += 1
            d += 1

    # Now line_other has shape (50, 2401, 48)
    
    # Verification function using vectorized operations
    def verify_nikodym(in_N):
        """
        in_N: boolean array of shape (total_points,)
        Returns True if Nikodym property holds.
        """
        if time.time() - start_time > deadline:
            return False
        
        # For each direction d, compute which points x have all other points on L(d,x) in N.
        # line_other[d] has shape (2401, 48)
        # in_N[line_other[d]] has shape (2401, 48)
        # all_in[d] = np.all(in_N[line_other[d]], axis=1) has shape (2401,)
        
        # A point x is valid if there exists a direction d such that all_in[d][x] is True.
        # So we need: for each x, OR over all d of all_in[d][x].
        
        # Compute all_in for all directions at once
        # in_N[line_other] has shape (50, 2401, 48)
        all_in = np.all(in_N[line_other], axis=2)  # shape (50, 2401)
        
        # A point is valid if any direction covers it
        valid_points = np.any(all_in, axis=0)  # shape (2401,)
        
        return np.all(valid_points)

    # Start with a known valid construction:
    # Hermitian unital: U = {(xa, xb, ya, yb) : 2*xa + ya^2 - 3*yb^2 = 0 mod 7}
    # Complement S = {points in U where y_a is a square mod 7}
    # N = F_49^2 \ S

    in_S = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        # Check if ya is a square mod 7: squares are {0, 1, 2, 4}
                        if ya in {0, 1, 2, 4}:
                            in_S[point_flat(xa, xb, ya, yb)] = True

    in_N = ~in_S

    # Verify initial construction
    if not verify_nikodym(in_N):
        # Fallback: return a trivial valid set (all points)
        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    # Initial complement size
    initial_S_size = np.sum(in_S)
    
    # Greedy expansion: try to add more points to S (remove from N)
    # A point can be added to S if removing it from N doesn't break the Nikodym property.
    
    # For efficiency, we'll use a randomized local search approach:
    # 1. Start with the initial valid set
    # 2. Try to add random points to S
    # 3. If it remains valid, keep it; otherwise revert
    # 4. Also try to remove points from S (add to N) to find better configurations
    
    # Actually, we want to MINIMIZE |N|, so we want to MAXIMIZE |S|.
    # So we should try to add points to S.
    
    # Let's use a simulated annealing approach:
    # - State: in_S (boolean array)
    # - Energy: -|S| (we want to maximize |S|, so minimize -|S|)
    # - But we also need validity constraint
    
    # Simpler approach: greedy with random restarts
    # Try to add points to S in random order, keeping only valid additions
    
    # Precompute: for each point p, which points x have p on L(d,x) for some direction d
    # This is expensive, so let's just do direct verification
    
    # Randomized greedy: try all points in random order, add if valid
    best_in_S = in_S.copy()
    best_S_size = initial_S_size
    
    # Try multiple random orders
    np.random.seed(42)
    
    for restart in range(50):  # 50 restarts
        if time.time() - start_time > deadline:
            break
        
        current_in_S = in_S.copy()
        current_S_size = initial_S_size
        
        # Create random order of all points
        all_point_indices = np.arange(total_points)
        np.random.shuffle(all_point_indices)
        
        # Try to add each point to S
        for p_idx in all_point_indices:
            if time.time() - start_time > deadline:
                break
            
            if current_in_S[p_idx]:
                continue
            
            # Try adding p_idx to S
            current_in_S[p_idx] = True
            current_in_N = ~current_in_S
            
            if verify_nikodym(current_in_N):
                current_S_size += 1
                # Keep this addition
            else:
                # Revert
                current_in_S[p_idx] = False
        
        # Update best
        if current_S_size > best_S_size:
            best_S_size = current_S_size
            best_in_S = current_in_S.copy()
    
    # Also try a more sophisticated approach: local search with swaps
    # Try to find a point in S that can be removed while adding another point to S
    # This allows us to explore different configurations
    
    # Actually, let's try a different strategy:
    # Instead of just adding points, let's try to find a minimal blocking set
    
    # For now, let's use the best result from the greedy approach
    
    # Build output from best_in_S
    in_N_final = ~best_in_S
    
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if not best_in_S[point_flat(xa, xb, ya, yb)]:
                        points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)
