import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    def is_valid(params, tol=1e-9):
        """Check if the configuration is valid (no overlaps, inside square)."""
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        # Radii must be positive
        if np.any(r < -tol):
            return False
        
        # Boundary check
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False
        
        # Overlap check
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
            
        return True

    def get_score(params):
        """Return the sum of radii."""
        n = N
        r = params[2*n:]
        return np.sum(r)

    def push_apart(params, n_iters=200):
        """
        Iteratively push overlapping circles apart and clamp to boundaries.
        This ensures the output is a valid packing.
        """
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        
        for _ in range(n_iters):
            moved = False
            # Pairwise repulsion
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist_sq = dx*dx + dy*dy
                    min_dist = r[i] + r[j]
                    
                    if dist_sq < min_dist**2:
                        dist = np.sqrt(dist_sq)
                        if dist < 1e-12:
                            # Random nudge if centers coincide
                            angle = np.random.uniform(0, 2*np.pi)
                            xy[i, 0] -= 0.01 * np.cos(angle)
                            xy[i, 1] -= 0.01 * np.sin(angle)
                            xy[j, 0] += 0.01 * np.cos(angle)
                            xy[j, 1] += 0.01 * np.sin(angle)
                            moved = True
                        else:
                            overlap = min_dist - dist
                            # Push apart proportionally to radius (larger circles move less)
                            total_r = r[i] + r[j]
                            wi = r[i] / total_r if total_r > 1e-15 else 0.5
                            wj = 1 - wi
                            
                            push_x = (dx / dist) * overlap
                            push_y = (dy / dist) * overlap
                            
                            xy[i, 0] -= push_x * wi
                            xy[i, 1] -= push_y * wi
                            xy[j, 0] += push_x * wj
                            xy[j, 1] += push_y * wj
                            moved = True
            
            # Boundary clamping
            for i in range(n):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]
                    moved = True
                if xy[i, 0] > 1 - r[i]:
                    xy[i, 0] = 1 - r[i]
                    moved = True
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]
                    moved = True
                if xy[i, 1] > 1 - r[i]:
                    xy[i, 1] = 1 - r[i]
                    moved = True
            
            if not moved:
                break

        return np.concatenate([xy.flatten(), r])

    def constraints(params):
        """
        Define inequality constraints for SLSQP.
        We want: dist(i,j) - (r_i + r_j) >= 0
                x_i - r_i >= 0
                1 - x_i - r_i >= 0
                y_i - r_i >= 0
                1 - y_i - r_i >= 0
        """
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]
        
        cons = []
        
        # Boundary constraints
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, idx=0: p[2*idx] - p[2*n+idx]})
        # 1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, idx=0: 1 - p[2*idx] - p[2*n+idx]})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, idx=0: p[2*idx+1] - p[2*n+idx]})
        # 1 - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, idx=0: 1 - p[2*idx+1] - p[2*n+idx]})
        
        # Overlap constraints
        # dist(i,j) - (r_i + r_j) >= 0
        # Vectorized calculation for speed
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlaps = dist[iu] - min_dist[iu]
        
        # SLSQP expects a single function returning an array of constraint values
        # We wrap this in a lambda that reconstructs the array from p
        def overlap_constraints(p):
            n = N
            xy = p[:2*n].reshape(n, 2)
            r = p[2*n:]
            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r[:, None] + r[None, :]
            iu = np.triu_indices(n, k=1)
            return dist[iu] - min_dist[iu]
            
        cons.append({'type': 'ineq', 'fun': overlap_constraints})
        
        # Add boundary constraints for all circles
        # Note: SLSQP can handle vectorized constraints, but the lambda above only handles one index.
        # Let's create vectorized boundary constraints for efficiency.
        
        def boundary_constraints(p):
            n = N
            xy = p[:2*n].reshape(n, 2)
            r = p[2*n:]
            # Returns array of: [x_i-r_i, 1-x_i-r_i, y_i-r_i, 1-y_i-r_i] for all i
            return np.concatenate([
                xy[:, 0] - r,
                1 - xy[:, 0] - r,
                xy[:, 1] - r,
                1 - xy[:, 1] - r
            ])
            
        # Replace individual boundary constraints with the vectorized one
        # Remove the 4 lambdas added above
        cons = [cons[-1]] # Keep overlap
        cons.append({'type': 'ineq', 'fun': boundary_constraints})
        
        return cons

    def objective(params):
        """Objective to minimize: -sum(radii)."""
        n = N
        r = params[2*n:]
        return -np.sum(r)

    def get_initial_config(restart, rng):
        n = N
        side = int(np.ceil(np.sqrt(N)))
        
        if restart % 4 == 0:
            # Grid packing
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart % 4 == 1:
            # Hexagonal packing approximation
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            # 6x6 grid with offset
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
        elif restart % 4 == 2:
            # Random uniform with larger radii
            xy = rng.uniform(0.1, 0.9, size=(n, 2))
            r = rng.uniform(0.08, 0.12, size=n)
        else:
            # 8x4 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                xy[i] = [(col + 0.5)/8, (row + 0.5)/4]
                r[i] = 0.5/8 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))

        params = np.concatenate([xy.flatten(), r])
        return params

    def optimize_from_start(start_params, max_iter=300):
        """Run SLSQP from a given start point."""
        # Ensure start is valid
        start_params = push_apart(start_params, n_iters=100)
        
        # Bounds for SLSQP
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
        for i in range(N):
            bounds.append((0, 0.5)) # r
            
        try:
            res = minimize(
                objective, 
                start_params, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints(start_params), # Note: constraints function needs to be callable without args or with p
                options={'maxiter': max_iter, 'ftol': 1e-12}
            )
            return res.x
        except Exception:
            return start_params

    # Fix: The constraints function defined above returns a list of dicts. 
    # SLSQP expects the constraints argument to be a list of dicts or a dict.
    # The 'fun' inside the dict takes 'p'. This is correct.
    # However, I defined `constraints(params)` which computes values for a specific params.
    # I need a function that returns the *structure* of constraints, where the functions inside take p.
    
    def get_constraints_struct():
        n = N
        iu = np.triu_indices(n, k=1)
        
        def overlap_constraints(p):
            xy = p[:2*n].reshape(n, 2)
            r = p[2*n:]
            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r[:, None] + r[None, :]
            return dist[iu] - min_dist[iu]
            
        def boundary_constraints(p):
            xy = p[:2*n].reshape(n, 2)
            r = p[2*n:]
            return np.concatenate([
                xy[:, 0] - r,
                1 - xy[:, 0] - r,
                xy[:, 1] - r,
                1 - xy[:, 1] - r
            ])
            
        return [
            {'type': 'ineq', 'fun': overlap_constraints},
            {'type': 'ineq', 'fun': boundary_constraints}
        ]

    # Update optimize_from_start to use the struct
    def optimize_from_start_v2(start_params, max_iter=300):
        start_params = push_apart(start_params, n_iters=100)
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
        for i in range(N):
            bounds.append((0, 0.5)) # r
            
        try:
            res = minimize(
                objective, 
                start_params, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=get_constraints_struct(),
                options={'maxiter': max_iter, 'ftol': 1e-12}
            )
            return res.x
        except Exception:
            return start_params

    best_result = None
    best_score = -np.inf
    rng = np.random.RandomState(12345)

    # Phase 1: Multiple restarts
    n_restarts = 40
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        params = get_initial_config(restart, rng)
        
        # Push apart to ensure validity before optimization
        params = push_apart(params, n_iters=150)
        
        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()
                
        # Optimize
        if time.time() < deadline - 5:
            optimized = optimize_from_start_v2(params, max_iter=200)
            
            if is_valid(optimized):
                score = get_score(optimized)
                if score > best_score:
                    best_score = score
                    best_result = optimized.copy()
            else:
                # If optimization broke validity, try to repair
                repaired = push_apart(optimized, n_iters=100)
                if is_valid(repaired):
                    score = get_score(repaired)
                    if score > best_score:
                        best_score = score
                        best_result = repaired.copy()

    # Phase 2: High-precision polish on the best result
    if best_result is not None and time.time() < deadline - 10:
        # Try a few perturbations and re-optimizations
        for polish_iter in range(5):
            if time.time() > deadline - 5:
                break
                
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            
            # Small random perturbation
            xy_p += rng.uniform(-0.005, 0.005, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            
            # Push apart to restore validity
            perturbed = push_apart(perturbed, n_iters=100)
            
            # Optimize with higher precision
            if time.time() < deadline - 5:
                optimized = optimize_from_start_v2(perturbed, max_iter=300)
                
                if is_valid(optimized):
                    score = get_score(optimized)
                    if score > best_score:
                        best_score = score
                        best_result = optimized.copy()
                else:
                    repaired = push_apart(optimized, n_iters=100)
                    if is_valid(repaired):
                        score = get_score(repaired)
                        if score > best_score:
                            best_score = score
                            best_result = repaired.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2).copy()
        r = best_result[2*n:].copy()
        # Final clamp to ensure strict validity
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        
        if is_valid(final_params):
            return np.column_stack([xy, r])

    # Fallback: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

