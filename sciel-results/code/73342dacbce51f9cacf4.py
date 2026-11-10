import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit

    def calc_score(x):
        """Compute exact minimum radius for given centers."""
        if not np.all(np.isfinite(x)):
            return 0.0
        if np.any(x < -0.5) or np.any(x > 0.5):
            return 0.0
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        wall_dists = 0.5 - np.abs(x)
        min_wall = np.min(wall_dists)
        
        return min(min_pair, min_wall)

    def soft_min_objective_and_grad(x_flat, beta):
        """
        Vectorized soft-min objective with analytical gradient.
        Uses log-sum-exp for numerical stability.
        """
        x = x_flat.reshape(N, 3)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=2)
        dists = np.sqrt(dists_sq + 1e-15)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        pair_dists = dists[mask] / 2.0  # radius contribution
        
        # Wall distances
        wall_dists = 0.5 - np.abs(x)
        min_wall_dists = np.min(wall_dists, axis=1)
        
        # Combine all distances
        all_dists = np.concatenate([pair_dists, min_wall_dists])
        
        # Log-sum-exp for soft-min
        vals = -beta * all_dists
        max_val = np.max(vals)
        log_sum_exp = max_val + np.log(np.sum(np.exp(vals - max_val)))
        
        obj = -log_sum_exp / beta
        
        # Gradient computation
        exp_vals = np.exp(vals - max_val)
        S = np.sum(exp_vals)
        
        grad = np.zeros_like(x)
        
        # Pair distance gradients
        pair_exp = exp_vals[:N*(N-1)]
        W = np.zeros((N, N))
        W[mask] = pair_exp / S
        
        # unit_vecs[i,j] = (x_i - x_j) / |x_i - x_j|
        unit_vecs = diff / (dists[:, :, np.newaxis] + 1e-15)
        
        # grad_pair[i] = sum_j W[i,j] * unit_vecs[i,j]
        grad_pair = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
        
        # Wall distance gradients
        wall_exp = exp_vals[N*(N-1):]
        weights_wall = wall_exp / S
        
        grad_wall = np.zeros_like(x)
        min_dims = np.argmin(wall_dists, axis=1)
        for i in range(N):
            k = min_dims[i]
            grad_wall[i, k] = weights_wall[i] * (-np.sign(x[i, k]))
        
        grad = grad_pair + grad_wall
        
        return obj, grad

    def run_optimization(init_x, beta, maxiter=200):
        """Run L-BFGS-B optimization with given beta."""
        x0 = init_x.flatten()
        bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)
        
        def obj_wrapper(x_flat):
            return soft_min_objective_and_grad(x_flat, beta)
        
        res = minimize(
            obj_wrapper, 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        return res.x.reshape(N, 3)

    def resolve_overlaps(x, r_target, steps=80):
        """Resolve overlaps using force-based relaxation."""
        x = x.copy()
        for _ in range(steps):
            forces = np.zeros_like(x)
            # Pairwise repulsion
            for i in range(N):
                for j in range(i+1, N):
                    d = x[i] - x[j]
                    dist = np.linalg.norm(d)
                    if dist < 2 * r_target and dist > 1e-9:
                        f = (2 * r_target - dist) / dist * d
                        forces[i] += f
                        forces[j] -= f
            # Wall constraints
            for i in range(N):
                for k in range(3):
                    d_left = x[i, k] + 0.5
                    if d_left < r_target:
                        forces[i, k] += (r_target - d_left) * (-1)
                    d_right = 0.5 - x[i, k]
                    if d_right < r_target:
                        forces[i, k] += (r_target - d_right) * (1)
            x += forces * 0.1
            x = np.clip(x, -0.5 + 1e-9, 0.5 - 1e-9)
        return x

    def generate_lattice_init(seed, lattice_type='fcc'):
        """Generate lattice-based initial configuration."""
        np.random.seed(seed)
        points = []
        
        if lattice_type == 'fcc':
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        points.append([i*0.5, j*0.5, k*0.5])
                        points.append([i*0.5+0.25, j*0.5+0.25, k*0.5])
                        points.append([i*0.5+0.25, j*0.5, k*0.5+0.25])
                        points.append([i*0.5, j*0.5+0.25, k*0.5+0.25])
        else:  # bcc
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        points.append([i*0.5, j*0.5, k*0.5])
                        points.append([i*0.5+0.25, j*0.5+0.25, k*0.5+0.25])
        
        points = np.array(points)
        dists_from_center = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(dists_from_center)
        x_init = points[sorted_indices[:N]].copy()
        
        max_coord = np.max(np.abs(x_init))
        if max_coord > 0.4:
            x_init *= (0.4 / max_coord)
            
        x_init += np.random.normal(0, 0.01, x_init.shape)
        return x_init

    def generate_random_init(seed, scale=0.1):
        """Generate random initial configuration."""
        np.random.seed(seed)
        x_init = np.random.uniform(-0.5 + scale, 0.5 - scale, (N, 3))
        return x_init

    def polish_exact(x, iterations=40):
        """
        High-precision polish using exact min-distance with subgradient.
        """
        x = x.copy()
        for it in range(iterations):
            if time.time() > deadline - 5:
                break
                
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            wall_dists = 0.5 - np.abs(x)
            min_wall_dists = np.min(wall_dists, axis=1)
            
            min_pair_dist = np.min(dists)
            min_wall = np.min(min_wall_dists)
            
            r = min(min_pair_dist / 2.0, min_wall)
            
            forces = np.zeros_like(x)
            
            # If pair distance is limiting
            if min_pair_dist / 2.0 <= min_wall:
                for i in range(N):
                    for j in range(i+1, N):
                        d = x[i] - x[j]
                        dist = np.linalg.norm(d)
                        if dist < min_pair_dist + 1e-10:
                            if dist > 1e-9:
                                f = d / dist * 0.005
                                forces[i] += f
                                forces[j] -= f
            
            # If wall distance is limiting
            if min_wall <= min_pair_dist / 2.0:
                for i in range(N):
                    min_dim = np.argmin(wall_dists[i])
                    k = min_dim
                    if wall_dists[i, k] < min_wall + 1e-10:
                        if x[i, k] > 0:
                            forces[i, k] -= 0.005
                        else:
                            forces[i, k] += 0.005
            
            x_new = x + forces
            x_new = np.clip(x_new, -0.5 + 1e-9, 0.5 - 1e-9)
            
            new_r = calc_score(x_new)
            if new_r > r:
                x = x_new
            else:
                # Try smaller step
                x_new = x + forces * 0.5
                x_new = np.clip(x_new, -0.5 + 1e-9, 0.5 - 1e-9)
                new_r = calc_score(x_new)
                if new_r > r:
                    x = x_new
        
        return x

    best_x = None
    best_score = -np.inf
    
    # Multi-stage optimization function
    def optimize_from_init(x_init, beta_schedule=None):
        if beta_schedule is None:
            beta_schedule = [(20, 150), (50, 150), (100, 100), (200, 80)]
        
        x = x_init.copy()
        for beta, maxiter in beta_schedule:
            if time.time() > deadline - 10:
                break
            x = run_optimization(x, beta, maxiter)
        
        x = polish_exact(x, iterations=30)
        return x

    # Strategy 1: FCC Lattice + Multi-stage optimization
    for seed in range(6):
        if time.time() > deadline - 15:
            break
        
        x_init = generate_lattice_init(seed, 'fcc')
        x_resolved = resolve_overlaps(x_init, 0.05, steps=60)
        
        x_opt = optimize_from_init(x_resolved)
        
        score = calc_score(x_opt)
        if score > best_score:
            best_score = score
            best_x = x_opt.copy()

    # Strategy 2: BCC Lattice Init
    for seed in range(6, 12):
        if time.time() > deadline - 15:
            break
        
        x_init = generate_lattice_init(seed, 'bcc')
        x_resolved = resolve_overlaps(x_init, 0.05, steps=60)
        
        x_opt = optimize_from_init(x_resolved)
        
        score = calc_score(x_opt)
        if score > best_score:
            best_score = score
            best_x = x_opt.copy()

    # Strategy 3: Random + Overlap Resolution
    for seed in range(12, 18):
        if time.time() > deadline - 15:
            break
        
        scale = 0.08 + 0.02 * (seed - 12)
        x_init = generate_random_init(seed, scale)
        x_resolved = resolve_overlaps(x_init, 0.05, steps=60)
        
        x_opt = optimize_from_init(x_resolved)
        
        score = calc_score(x_opt)
        if score > best_score:
            best_score = score
            best_x = x_opt.copy()

    # Strategy 4: Final polish on best solution
    if best_x is not None and time.time() < deadline - 15:
        best_x = polish_exact(best_x, iterations=50)
        best_score = calc_score(best_x)

    # Strategy 5: Try to improve with different beta schedules
    if best_x is not None and time.time() < deadline - 20:
        alt_schedule = [(15, 200), (40, 200), (80, 150), (150, 100), (300, 80)]
        x_alt = optimize_from_init(best_x, beta_schedule=alt_schedule)
        score_alt = calc_score(x_alt)
        if score_alt > best_score:
            best_score = score_alt
            best_x = x_alt.copy()

    if best_x is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_x = grid[:N]
        
    return best_x
