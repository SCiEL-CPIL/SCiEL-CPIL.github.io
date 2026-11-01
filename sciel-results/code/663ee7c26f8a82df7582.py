"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    def _score(pos):
        """
        Compute the actual packing score: min(min_pair_dist/2, min_wall_dist).
        pos: (N, 3) array
        """
        if pos.shape != (N, 3):
            return -1.0
        if not np.all(np.isfinite(pos)):
            return -1.0
        
        # Pairwise distances
        # Using broadcasting for N=100 is efficient enough (100x100x3)
        d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid zero division, though diagonal is 0
        min_pair_dist = np.sqrt(np.min(dist_sq))
        
        # Wall distances
        # Distance to nearest wall for each coordinate
        # Walls are at -0.5 and 0.5
        # dist = min(0.5 - x, x + 0.5) = 0.5 - |x|
        wall_dists = 0.5 - np.abs(pos)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def _softmin_objective(flat_pos, beta):
        """
        Objective to MINIMIZE.
        We want to MAXIMIZE the soft-min of distances.
        So we minimize -softmin(distances).
        
        Distances considered:
        1. Pairwise distances / 2 (radius constraint)
        2. Wall distances
        
        Softmin(x) = -1/beta * log(sum(exp(-beta * x_i)))
        We want to maximize min(d_i). 
        Approx: max softmin(d_i) ~= min(d_i) as beta -> inf.
        So we minimize -softmin(d_i).
        
        However, computing exp(-beta * x) can overflow if x is large and beta is large.
        Standard trick: softmin(x) = -1/beta * log(sum(exp(-beta * (x - max_x)))) - max_x
        Wait, let's stick to the standard log-sum-exp stability.
        
        Let D be the vector of all relevant distances (pair/2 and wall).
        We want to maximize min(D).
        Let f(D) = -1/beta * log(sum(exp(-beta * D_i))).
        As beta -> inf, f(D) -> -min(D).
        So minimizing -f(D) is equivalent to maximizing min(D) in the limit.
        
        Actually, let's define the objective as:
        J = -1/beta * log( sum_{i} exp(-beta * d_i) )
        We want to MINIMIZE J? No.
        If d_i are all large, exp(-beta*d_i) is small, sum is small, log is very negative, J is positive and large?
        Let's check limits.
        If d_min is small, one term exp(-beta*d_min) dominates.
        sum ~= exp(-beta*d_min).
        log(sum) ~= -beta*d_min.
        J ~= -1/beta * (-beta*d_min) = d_min.
        So J approximates d_min.
        We want to MAXIMIZE d_min.
        So we should MINIMIZE -J.
        
        Let's verify gradient direction.
        We want to increase d_min.
        If we minimize -J, we are maximizing J.
        J increases as d_min increases.
        So yes, minimize -J.
        
        Wait, is J always positive?
        If d_i > 0, exp(-beta d_i) < 1. Sum < N. Log < log(N).
        If beta is large, J ~ d_min.
        
        Let's implement:
        objective = -J = 1/beta * log( sum(exp(-beta * D)) )
        We minimize this.
        
        Stability:
        Let m = max(D).
        exp(-beta * D_i) = exp(-beta * (D_i - m)) * exp(-beta * m)
        sum = exp(-beta * m) * sum(exp(-beta * (D_i - m)))
        log(sum) = -beta * m + log(sum(exp(-beta * (D_i - m))))
        
        Objective = 1/beta * ( -beta * m + log(sum(exp(-beta * (D_i - m)))) )
        Objective = -m + 1/beta * log(sum(exp(-beta * (D_i - m))))
        
        Note: D_i - m <= 0. So -beta * (D_i - m) >= 0.
        exp of positive numbers can overflow if beta*(m-D_i) is large.
        But m is the max, so m-D_i >= 0.
        If the spread of D is large, some terms might overflow.
        However, for packing, D values are close to each other near the optimum.
        To be safe, we can use a smaller beta or handle it carefully.
        
        Alternative: Use the "soft-max" formulation directly on the negative distances?
        Let y_i = -d_i. We want to minimize max(y_i) (since max(-d) = -min(d)).
        Softmax(y) = 1/beta * log(sum(exp(beta * y_i))).
        As beta -> inf, Softmax(y) -> max(y).
        So we minimize Softmax(-d).
        
        Let's use this:
        y = -D (vector of negative distances)
        y_max = max(y)
        terms = exp(beta * (y - y_max))
        sum_terms = sum(terms)
        obj = y_max + 1/beta * log(sum_terms)
        
        This is stable because y - y_max <= 0, so beta*(y-y_max) <= 0, exp <= 1.
        No overflow.
        """
        pos = flat_pos.reshape(N, 3)
        
        # Check bounds (L-BFGS-B handles bounds, but just in case)
        # The objective should return a large penalty if out of bounds, 
        # but L-BFGS-B respects bounds strictly.
        
        # Calculate all distances D
        # 1. Pairwise
        d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(d**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        pair_dists = np.sqrt(dist_sq) / 2.0
        
        # Only take upper triangle to avoid duplicates? 
        # Summing over all i!=j is fine, just doubles the weight of pairs.
        # To be consistent with "min", duplicates don't change the min.
        # For log-sum-exp, duplicates change the magnitude of the sum, 
        # but the dominant term (the min distance) is still determined by the smallest d.
        # So it's safe to include all i != j.
        
        # 2. Wall
        wall_dists = 0.5 - np.abs(pos)
        
        # Flatten all distances into one vector D
        # Exclude diagonal (dist 0)
        mask = ~np.eye(N, dtype=bool)
        D_pairs = pair_dists[mask]
        
        D_walls = wall_dists.flatten()
        
        D = np.concatenate([D_pairs, D_walls])
        
        # Ensure D is positive. If any D is <= 0, it's invalid.
        # L-BFGS-B should keep us inside, but numerical errors might push slightly out.
        # If D <= 0, the "distance" is effectively 0 or negative.
        # We want to penalize this heavily.
        # In the softmax formulation, if d is negative, y=-d is positive.
        # It will dominate the sum. This is good.
        
        y = -D
        y_max = np.max(y)
        
        # beta is passed via closure or global? 
        # L-BFGS-B takes a function f(x). We need beta to change.
        # We will define a factory or use a mutable container.
        # For simplicity in this structure, we'll define the objective to use a global beta variable 
        # that we update in the loop.
        
        beta = _softmin_beta[0]
        
        terms = np.exp(beta * (y - y_max))
        sum_terms = np.sum(terms)
        
        # Handle case where sum_terms is 0 (shouldn't happen as exp(0)=1 for max term)
        if sum_terms == 0:
            return 1e10
            
        obj = y_max + (1.0 / beta) * np.log(sum_terms)
        
        return obj

    def _init_grid():
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def _init_bcc():
        """
        Body-Centered Cubic lattice initialization.
        This is a known good dense packing structure.
        We place points on a BCC lattice clipped to the box.
        """
        # BCC has 2 points per unit cell.
        # We need 100 points. 50 unit cells.
        # 50 is not a perfect cube (3^3=27, 4^3=64).
        # Let's use a grid of side 4. 4^3 = 64 cells. 128 points.
        # We can take the first 100 or optimize the scale.
        # Side length 4 means spacing 1/4 = 0.25.
        # BCC points in a unit cell [0,1]^3: (0,0,0) and (0.5,0.5,0.5).
        
        m = 4
        scale = 1.0 / m
        points = []
        
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    # Corner point
                    x, y, z = i * scale, j * scale, k * scale
                    # Shift to center of box [-0.5, 0.5]
                    # Box is side 1. Grid is m * scale = 1.
                    # Points are at 0, 0.25, 0.5, 0.75.
                    # Centered: -0.5, -0.25, 0.0, 0.25.
                    # Wait, if m=4, points are at 0, 0.25, 0.5, 0.75.
                    # Range is 0.75. Box is 1.0.
                    # Let's center the grid.
                    # Grid extent: (m-1)*scale to m*scale? No, corners are at i*scale.
                    # If i in 0..m-1, max is (m-1)*scale.
                    # Let's just generate and center.
                    
                    # Corner
                    c1 = np.array([i, j, k]) * scale
                    # Center
                    c2 = (np.array([i, j, k]) + 0.5) * scale
                    
                    # Shift to center of [-0.5, 0.5]
                    # Current range: [0, (m-1)*scale] approx.
                    # Let's assume the grid fills [0, 1] roughly.
                    # Center shift: -0.5
                    # But BCC centers are at 0.5*scale offset.
                    
                    # Simpler: Generate raw coordinates in [0, m]^3, then scale and shift.
                    pass
        
        # Let's do it properly.
        # Generate lattice points in [0, m]^3.
        pts = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    pts.append([i, j, k])
                    pts.append([i + 0.5, j + 0.5, k + 0.5])
        
        pts = np.array(pts)
        # Scale to fit in [0, 1]
        # Max coord is m-0.5? No, i goes to m-1.
        # Corner max: m-1. Center max: m-0.5.
        # Let's scale by 1/m.
        pts /= m
        # Now pts are in [0, 1].
        # Shift to [-0.5, 0.5]
        pts -= 0.5
        
        # We have m^3 * 2 points. For m=4, 128 points.
        # We need 100.
        # Take the first 100? This might be biased towards one corner.
        # Better: Randomly sample 100 from the 128?
        # Or just take a subset that is more central?
        # For initialization, taking the first N is fine, we will optimize.
        # But to be safe, let's sort by distance to center and take closest?
        # Or just random selection to break symmetry.
        
        idx = np.random.choice(len(pts), N, replace=False)
        return pts[idx]

    # Mutable container for beta to allow changing it in the loop without redefining objective
    _softmin_beta = [10.0]

    # 1. Initialize
    # Try multiple initializations: Grid and BCC
    inits = [
        _init_grid(),
        _init_bcc(),
        np.random.uniform(-0.45, 0.45, (N, 3)) # Random
    ]
    
    best_pos = None
    best_score = -1.0
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
            
        # 2. L-BFGS-B Optimization with increasing beta
        current_pos = init_pos.copy().flatten()
        
        # Bounds: [-0.5, 0.5] for each coordinate
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Schedule for beta
        betas = [5.0, 10.0, 20.0, 50.0, 100.0]
        
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
                
            _softmin_beta[0] = beta
            
            # Run L-BFGS-B
            # Max iterations depends on time. 
            # L-BFGS-B is fast for N=100.
            res = minimize(
                _softmin_objective, 
                current_pos, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            current_pos = res.x
            
            # Check validity
            pos_check = current_pos.reshape(N, 3)
            if not np.all(np.isfinite(pos_check)):
                break
                
        # 3. Local Polishing (Coordinate Ascent on the exact score)
        # L-BFGS-B optimizes the soft-min. The soft-min is close to the min, but not exact.
        # We can do a few steps of coordinate ascent to push the *actual* min distance up.
        
        pos = current_pos.reshape(N, 3)
        current_score = _score(pos)
        
        if current_score > best_score:
            best_score = current_score
            best_pos = pos.copy()
            
        # Coordinate Ascent
        # Try moving each point in +/- direction of the vector that increases the min distance?
        # Or just random steps?
        # Simple hill climbing:
        # For each point, try moving it slightly in 6 directions ( +/- x, +/- y, +/- z ).
        # If score improves, keep it.
        
        step_size = 0.005
        polish_iters = 100
        
        for _ in range(polish_iters):
            if time.time() > deadline - 5.0:
                break
                
            improved = False
            for i in range(N):
                # Directions: +x, -x, +y, -y, +z, -z
                for d in range(3):
                    for sign in [1, -1]:
                        new_pos = pos.copy()
                        new_pos[i, d] += sign * step_size
                        new_pos[i, d] = np.clip(new_pos[i, d], -0.5, 0.5)
                        
                        new_score = _score(new_pos)
                        if new_score > current_score + 1e-9:
                            pos = new_pos
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
            if not improved:
                # Try smaller step
                step_size *= 0.5
                if step_size < 1e-5:
                    break

        if current_score > best_score:
            best_score = current_score
            best_pos = pos.copy()

    if best_pos is None:
        return _init_grid()
        
    return best_pos
# EVOLVE-BLOCK-END
