import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 32
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_solution = None
    
    # Helper to check time
    def time_remaining():
        return TIME_LIMIT - (time.time() - start_time)

    def get_xyz(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        return x, y, r

    def calc_constraints(z):
        x, y, r = get_xyz(z)
        
        c_bounds = np.concatenate([
            x - r,
            1.0 - x - r,
            y - r,
            1.0 - y - r
        ])
        
        # Vectorized pairwise constraints
        # dist_sq[i,j] = (x[i]-x[j])^2 + (y[i]-y[j])^2
        dist_mat = cdist(np.column_stack([x, y]), np.column_stack([x, y]))
        rad_sum_mat = r[:, None] + r[None, :]
        c_pairwise_mat = dist_mat**2 - rad_sum_mat**2
        
        # Get upper triangle (i < j)
        iu = np.triu_indices(N, k=1)
        c_pairwise = c_pairwise_mat[iu]
        
        return np.concatenate([c_bounds, c_pairwise])

    def obj_func(z):
        r = z[2::3]
        return -np.sum(r)

    def cons_func(z):
        return calc_constraints(z)

    bounds_int = []
    for _ in range(N):
        bounds_int.append((0, 1)) # x
        bounds_int.append((0, 1)) # y
        bounds_int.append((0, 0.5)) # r

    constraints = [
        {'type': 'ineq', 'fun': cons_func}
    ]

    def run_optimizer(z0, maxiter=500, ftol=1e-12):
        res = minimize(
            obj_func, 
            z0, 
            method='SLSQP', 
            bounds=bounds_int, 
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': ftol}
        )
        return res

    # Penalty-based approach for L-BFGS-B
    def penalty_objective(z, lam=100.0):
        r = z[2::3]
        base_obj = -np.sum(r)
        c = calc_constraints(z)
        # Penalty for constraint violations (c >= 0 required)
        violations = np.maximum(0, -c)
        penalty = lam * np.sum(violations**2)
        return base_obj + penalty

    def penalty_grad(z, lam=100.0):
        x, y, r = get_xyz(z)
        grad = np.zeros(3*N)
        
        # Base gradient: d(-sum(r))/dz
        grad[2::3] = -1.0
        
        # Penalty gradient
        c = calc_constraints(z)
        violations = np.maximum(0, -c)
        
        # Gradient of penalty term
        # For each constraint c_k, if c_k < 0, gradient contribution is -2*lam*c_k * dc_k/dz
        # This is complex to vectorize fully, so we approximate
        
        # Simpler: use finite differences or just rely on L-BFGS-B's internal gradient estimation
        # Actually, let's just use the objective without explicit gradient for L-BFGS-B
        return None  # Let L-BFGS-B estimate gradient

    def run_penalty_optimizer(z0, maxiter=2000, lam=100.0):
        res = minimize(
            penalty_objective, 
            z0, 
            method='L-BFGS-B', 
            bounds=bounds_int,
            args=(lam,),
            options={'maxiter': maxiter, 'ftol': 1e-15}
        )
        return res

    # Initialization strategies
    def init_grid():
        side = int(np.ceil(np.sqrt(N)))
        xs, ys = np.meshgrid(np.linspace(0.5/side, 1-0.5/side, side), np.linspace(0.5/side, 1-0.5/side, side))
        x = xs.flatten()
        y = ys.flatten()
        x = x[:N]
        y = y[:N]
        r = np.full(N, 0.08)
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        return z

    def init_random(seed=42):
        np.random.seed(seed)
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.05)
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        return z

    def init_hexagonal():
        x = []
        y = []
        for i in range(4):
            offset = 0 if i % 2 == 0 else 0.0625
            for j in range(8):
                x.append(offset + j * 0.125 + 0.0625)
                y.append(0.15 + i * 0.216)
        x = np.array(x)
        y = np.array(y)
        r = np.full(N, 0.06)
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        return z

    def init_perturbed_grid(seed=42):
        np.random.seed(seed)
        z = init_grid()
        noise = np.random.uniform(-0.05, 0.05, 3*N)
        z += noise
        z = np.clip(z, 0, 1)
        z[2::3] = np.maximum(z[2::3], 0.01)
        return z

    def init_circular():
        # Place circles in a circular pattern
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii_pos = 0.35
        x = 0.5 + radii_pos * np.cos(angles)
        y = 0.5 + radii_pos * np.sin(angles)
        r = np.full(N, 0.06)
        z = np.zeros(3*N)
        z[0::3] = x
        z[1::3] = y
        z[2::3] = r
        return z

    initializations = [
        init_grid,
        init_random,
        init_hexagonal,
        init_perturbed_grid,
        init_circular,
    ]

    # Run multiple restarts with different methods
    for i, init_func in enumerate(initializations):
        if time_remaining() < 15:
            break
            
        z0 = init_func()
        
        # Try SLSQP first
        try:
            res = run_optimizer(z0, maxiter=300, ftol=1e-10)
            if res.success:
                score = -res.fun
                # Verify constraints are satisfied
                c = calc_constraints(res.x)
                if np.min(c) > -1e-6:
                    if score > best_score:
                        best_score = score
                        best_solution = res.x
        except Exception:
            pass

        # Try penalty-based L-BFGS-B if time remains
        if time_remaining() > 10:
            try:
                res = run_penalty_optimizer(z0, maxiter=1000, lam=50.0)
                score = -np.sum(res.x[2::3])
                c = calc_constraints(res.x)
                if np.min(c) > -1e-5:
                    if score > best_score:
                        best_score = score
                        best_solution = res.x
            except Exception:
                pass

    # High-precision polish on best solution
    if best_solution is not None and time_remaining() > 10:
        try:
            res = run_optimizer(best_solution, maxiter=1000, ftol=1e-15)
            if res.success:
                score = -res.fun
                c = calc_constraints(res.x)
                if np.min(c) > -1e-8:
                    if score > best_score:
                        best_score = score
                        best_solution = res.x
        except Exception:
            pass

    # Additional polish with penalty method
    if best_solution is not None and time_remaining() > 5:
        try:
            res = run_penalty_optimizer(best_solution, maxiter=2000, lam=200.0)
            score = -np.sum(res.x[2::3])
            c = calc_constraints(res.x)
            if np.min(c) > -1e-7:
                if score > best_score:
                    best_score = score
                    best_solution = res.x
        except Exception:
            pass

    # Format output
    if best_solution is None:
        z0 = init_grid()
        best_solution = z0
        
    x = best_solution[0::3]
    y = best_solution[1::3]
    r = best_solution[2::3]
    
    # Ensure validity: clip to bounds and enforce non-overlap
    # Clip positions and radii to valid range
    r = np.clip(r, 0, 0.5)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    
    # If any circles still overlap, shrink radii
    for iter in range(10):
        c = calc_constraints(np.column_stack([x, y, r]).reshape(-1))
        if np.min(c) > -1e-8:
            break
        # Find most violated constraint and shrink involved radii
        min_idx = np.argmin(c)
        # Determine which constraint this is
        n_bounds = 4 * N
        if min_idx < n_bounds:
            # Boundary violation - adjust position
            if min_idx < N:
                # x - r < 0
                i = min_idx
                x[i] = r[i] + 1e-6
            elif min_idx < 2*N:
                # 1 - x - r < 0
                i = min_idx - N
                x[i] = 1 - r[i] - 1e-6
            elif min_idx < 3*N:
                # y - r < 0
                i = min_idx - 2*N
                y[i] = r[i] + 1e-6
            else:
                # 1 - y - r < 0
                i = min_idx - 3*N
                y[i] = 1 - r[i] - 1e-6
        else:
            # Pairwise violation
            pair_idx = min_idx - n_bounds
            # Convert linear index to (i, j)
            count = 0
            found = False
            for ii in range(N):
                for jj in range(ii+1, N):
                    if count == pair_idx:
                        # Shrink both radii slightly
                        r[ii] *= 0.99
                        r[jj] *= 0.99
                        found = True
                        break
                    count += 1
                if found:
                    break
            if not found:
                break
    
    result = np.column_stack((x, y, r))
    return result
