import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    n = 8
    # Fix P0 at (0,0) and P4 at (2,0)
    # Free indices: 1, 2, 3, 5, 6, 7
    fixed_idx = [0, 4]
    free_idx = [i for i in range(n) if i not in fixed_idx]
    n_free = len(free_idx)  # 6
    n_dims = n_free * 2

    def unpack(x):
        p = np.zeros((n, 2))
        p[0] = [0.0, 0.0]
        p[4] = [2.0, 0.0]
        for i, idx in enumerate(free_idx):
            p[idx] = [x[2 * i], x[2 * i + 1]]
        return p

    def objective(x):
        """
        Minimize negative log product of distances.
        """
        p = unpack(x)
        # Compute all pairwise distances
        diffs = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        iu = np.triu_indices(n, k=1)
        d = np.sqrt(np.maximum(dists_sq[iu], 1e-15))
        
        # Sum of logs
        sum_log_d = np.sum(np.log(d))
        return -sum_log_d

    # Constraints for SLSQP
    # Each free point must be within distance 2 of (0,0) and (2,0)
    def make_constraint(i_idx, fixed_point):
        def constraint(x):
            idx = free_idx[i_idx]
            # Get coordinates of the free point
            px = x[2 * i_idx]
            py = x[2 * i_idx + 1]
            # Distance squared to fixed_point
            d_sq = (px - fixed_point[0])**2 + (py - fixed_point[1])**2
            return 4.0 - d_sq # Must be >= 0
        return constraint

    constraints = []
    for i_idx in range(n_free):
        # Constraint 1: dist to P0 (0,0) <= 2
        constraints.append({'type': 'ineq', 'fun': make_constraint(i_idx, [0.0, 0.0])})
        # Constraint 2: dist to P4 (2,0) <= 2
        constraints.append({'type': 'ineq', 'fun': make_constraint(i_idx, [2.0, 0.0])})

    # Bounds for free coordinates
    # x in [-2, 4], y in [-2, 2] is a safe box containing the lens
    bounds = [(-2.0, 4.0), (-2.0, 2.0)] * n_free

    # Initial guesses
    starts = []

    # 1. Regular Octagon
    # Center (1,0), radius 1.
    # Vertices: (1 + cos(2*pi*k/8), sin(2*pi*k/8))
    # P0 should be (0,0) -> k=4 (cos(pi)=-1 -> 1-1=0, sin(pi)=0)
    # P4 should be (2,0) -> k=0 (cos(0)=1 -> 1+1=2, sin(0)=0)
    # So in the standard octagon, index 4 is (0,0) and index 0 is (2,0).
    # Our fixed frame: P0=(0,0), P4=(2,0).
    # So we map standard_octagon[4] -> P0, standard_octagon[0] -> P4.
    # The other points are just the standard octagon points shifted so that standard_octagon[4] is at origin.
    # Since standard_octagon[4] is (0,0), no shift needed.
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts_oct = np.column_stack((1.0 + np.cos(angles), np.sin(angles)))
    
    x0_oct = np.zeros(n_dims)
    for i, idx in enumerate(free_idx):
        # idx is the index in the final array.
        # We need to find which standard octagon point corresponds to this index.
        # This mapping is tricky because we fixed specific indices.
        # Let's just create the full array and extract.
        pass
    
    # Easier: Just construct the full octagon, identify which points are (0,0) and (2,0),
    # and assign the rest to the free slots in order.
    # But the order of free slots is 1,2,3,5,6,7.
    # The standard octagon points are symmetric.
    # Let's just use the coordinates directly.
    # Standard Octagon points:
    # k=0: (2,0) -> This is our P4.
    # k=1: (1+cos(pi/4), sin(pi/4)) = (1+0.707, 0.707)
    # k=2: (1, 1)
    # k=3: (1-0.707, 0.707)
    # k=4: (0,0) -> This is our P0.
    # k=5: (1-0.707, -0.707)
    # k=6: (1, -1)
    # k=7: (1+0.707, -0.707)
    
    # We need to fill free_idx = [1, 2, 3, 5, 6, 7] with the remaining 6 points.
    # The remaining points are k=1,2,3,5,6,7.
    # Let's map them arbitrarily but consistently.
    # x0_oct[0:2] = pts_oct[1] (for free_idx[0]=1)
    # x0_oct[2:4] = pts_oct[2] (for free_idx[1]=2)
    # x0_oct[4:6] = pts_oct[3] (for free_idx[2]=3)
    # x0_oct[6:8] = pts_oct[5] (for free_idx[3]=5)
    # x0_oct[8:10] = pts_oct[6] (for free_idx[4]=6)
    # x0_oct[10:12] = pts_oct[7] (for free_idx[5]=7)
    
    x0_oct = np.zeros(n_dims)
    free_to_oct = {1: 1, 2: 2, 3: 3, 5: 5, 6: 6, 7: 7}
    for i, idx in enumerate(free_idx):
        k = free_to_oct[idx]
        x0_oct[2*i] = pts_oct[k][0]
        x0_oct[2*i+1] = pts_oct[k][1]
    starts.append(x0_oct)

    # 2. "Two Squares" Configuration
    # A known good configuration for n=8 is often close to two squares.
    # Let's try a perturbed version of the octagon or a specific guess.
    # For now, let's add random starts.
    np.random.seed(42)
    for _ in range(20):
        # Sample random points in the box [-2, 4] x [-2, 2]
        x_rand = np.random.uniform(-2, 4, n_dims)
        starts.append(x_rand)

    # Add some structured random starts near the octagon
    for scale in [0.1, 0.5, 1.0]:
        for _ in range(5):
            x_pert = x0_oct + np.random.normal(0, scale, n_dims)
            starts.append(x_pert)

    best_x = starts[0]
    best_obj = objective(best_x)

    # Optimization Loop
    for x_start in starts:
        if time.time() > deadline - 10:
            break
        
        # Check if start is valid (within constraints)
        # SLSQP can handle slightly invalid starts, but better to check
        # We'll just let SLSQP try.
        
        try:
            res = minimize(objective, x_start, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': 200, 'ftol': 1e-10, 'disp': False})
            
            if res.success or res.fun < best_obj:
                # Verify constraints are met
                valid = True
                p = unpack(res.x)
                for i in range(n):
                    for j in [0, 4]:
                        if i != j:
                            d = np.linalg.norm(p[i] - p[j])
                            if d > 2.0 + 1e-6:
                                valid = False
                                break
                    if not valid: break
                
                if valid:
                    if res.fun < best_obj:
                        best_obj = res.fun
                        best_x = res.x.copy()
        except Exception:
            continue

    # Polish the best result
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_x, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    p = unpack(best_x)

    # Ensure distinct points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(p[i] - p[j]) < 1e-8:
                p[j] += np.array([1e-6, 0])

    # Scale to ensure diameter is exactly 2
    # Although we constrained it, numerical errors might make it slightly off.
    max_d = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(p[i] - p[j])
            if d > max_d:
                max_d = d

    if max_d > 0 and abs(max_d - 2.0) > 1e-8:
        scale_factor = 2.0 / max_d
        p = p * scale_factor

    return [[float(p[i][0]), float(p[i][1])] for i in range(n)]
