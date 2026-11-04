"""
Optimization of 100 equal spheres in a unit cube.
Combines simulated annealing exploration with gradient-based local refinement
to maximize the minimum radius.
"""
N = 100
CUBE_HALF = 0.5

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(c):
    """
    Compute the maximum radius for a given set of centers.
    r = min(min_pairwise_dist / 2, min_boundary_dist)
    """
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    
    # Pairwise distances
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    abs_c = np.abs(c)
    boundary_dists = CUBE_HALF - np.max(abs_c, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    if min_boundary_dist < 0:
        return 0.0
        
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _initial_grid(nx, ny, nz):
    """Create a structured grid initialization."""
    xs = np.linspace(-CUBE_HALF + CUBE_HALF/nx, CUBE_HALF - CUBE_HALF/nx, nx)
    ys = np.linspace(-CUBE_HALF + CUBE_HALF/ny, CUBE_HALF - CUBE_HALF/ny, ny)
    zs = np.linspace(-CUBE_HALF + CUBE_HALF/nz, CUBE_HALF - CUBE_HALF/nz, nz)
    
    grid = []
    for z in zs:
        for y in ys:
            for x in xs:
                grid.append([x, y, z])
    return np.array(grid)

def _initial_random(rng):
    """Create a random initial placement."""
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _simulated_annealing(c, time_limit=30.0, T_start=0.012, T_end=1e-8, rng=None):
    """
    Simulated Annealing with adaptive step size and neighbor-aware moves.
    """
    if rng is None:
        rng = np.random.RandomState()
    if isinstance(rng, np.random.RandomState):
        def rand():
            return rng.uniform()
        def randn():
            return rng.randn()
        def randint(a, b):
            return rng.randint(a, b+1)
    else:
        def rand():
            return rng.random()
        def randn():
            return rng.normal()
        def randint(a, b):
            return rng.randint(a, b+1)
        
    start_time = time.time()
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    T = T_start
    T_min = T_end
    base_step = 0.025
    iter_count = 0
    
    # Move type probabilities
    p_single = 0.6
    p_double = 0.3
    p_neighbor = 0.1
    
    while time.time() - start_time < time_limit:
        iter_count += 1
        
        # Adaptive cooling
        tau = 2000
        T = max(T_start * np.exp(-iter_count / tau), T_min)
        
        # Step size scales with temperature and current quality
        step_scale = max(0.4, min(1.2, current_score / 0.1))
        step_size = base_step * step_scale * (T / T_start + 0.1)
        step_size = max(0.003, min(0.06, step_size))
        
        r_move = rand()
        c_new = c.copy()
        
        if r_move < p_single:
            # Move one point
            idx = randint(0, N - 1)
            delta = randn(3) * step_size
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -CUBE_HALF, CUBE_HALF)
            
        elif r_move < p_single + p_double:
            # Move two points
            i1 = randint(0, N - 1)
            i2 = randint(0, N - 1)
            if i2 == i1:
                i2 = (i2 + 1) % N
            delta1 = randn(3) * step_size
            delta2 = randn(3) * step_size
            c_new[i1] = c_new[i1] + delta1
            c_new[i2] = c_new[i2] + delta2
            c_new[i1] = np.clip(c_new[i1], -CUBE_HALF, CUBE_HALF)
            c_new[i2] = np.clip(c_new[i2], -CUBE_HALF, CUBE_HALF)
            
        elif r_move < p_single + p_double + p_neighbor:
            # Neighbor-aware: move away from nearest neighbor
            idx = randint(0, N - 1)
            other = randint(0, N - 1)
            if other == idx:
                other = (other + 1) % N
            dir_vec = c[idx] - c[other]
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-10:
                dir_vec = dir_vec / norm
            else:
                dir_vec = randn(3)
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
            
            delta = dir_vec * step_size * 0.6 + randn(3) * step_size * 0.4
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -CUBE_HALF, CUBE_HALF)
            
        else:
            # Large jump
            idx = randint(0, N - 1)
            c_new[idx] = rng.uniform(-0.45, 0.45, size=3)
            c_new[idx] = np.clip(c_new[idx], -CUBE_HALF, CUBE_HALF)
        
        new_score = _score(c_new)
        
        if new_score >= current_score:
            c = c_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        else:
            delta_score = new_score - current_score
            T_safe = max(T, 1e-10)
            prob = np.exp(delta_score / T_safe)
            if rand() < prob:
                c = c_new
                current_score = new_score
        
    return best_c, best_score

def _energy_and_gradient(x, R):
    """
    Calculates repulsion energy and gradient for target radius R.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    v_ij = np.maximum(0, 2.0 * R - d_ij)
    
    if np.any(v_ij > 0):
        energy += np.sum(v_ij**2)
        
        for k in range(len(v_ij)):
            if v_ij[k] <= 0:
                continue
            i = i_idx[k]
            j = j_idx[k]
            v = v_ij[k]
            d = d_ij[k]
            if d < 1e-10:
                continue
            factor = -2.0 * v / d
            vec = centers[i] - centers[j]
            force = factor * vec
            grad[i] += force
            grad[j] -= force

    # Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = CUBE_HALF - max_abs
    v_wall = np.maximum(0, R - wall_dists)
    
    if np.any(v_wall > 0):
        energy += np.sum(v_wall**2)
        
        for i in range(n):
            if v_wall[i] <= 0:
                continue
            v = v_wall[i]
            max_val = max_abs[i]
            grad_wall_i = np.zeros(3)
            for k in range(3):
                if np.isclose(abs_centers[i, k], max_val, atol=1e-10):
                    grad_wall_i[k] = np.sign(centers[i, k])
                    break
            
            force_wall = 2.0 * v * grad_wall_i
            grad[i] += force_wall
            
    return energy, grad.flatten()

def _gradient_polish(centers, time_budget=10.0):
    """
    Gradient-based polishing using L-BFGS-B to maximize minimum radius.
    Uses a soft-min surrogate for smoothness.
    """
    start_time = time.time()
    current_r = _score(centers)
    
    def objective_and_grad(x):
        centers_x = x.reshape(-1, 3)
        
        # Pairwise distances
        diff = centers_x[:, np.newaxis, :] - centers_x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
        np.fill_diagonal(dists, np.inf)
        
        i_idx, j_idx = np.triu_indices(len(centers_x), k=1)
        pair_dists = dists[i_idx, j_idx] / 2.0
        
        # Wall distances
        max_abs = np.max(np.abs(centers_x), axis=1)
        wall_dists = CUBE_HALF - max_abs
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Soft min
        beta = 80.0
        min_d = np.min(all_dists)
        if min_d < 1e-10:
            return 0.0, np.zeros_like(x)
        
        shifted = all_dists - min_d
        log_sum = np.log(np.sum(np.exp(-beta * shifted)))
        soft_min = min_d - np.log(np.sum(np.exp(-beta * shifted))) / beta
        
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            
            # Recompute soft min for perturbed
            cp = x_plus.reshape(-1, 3)
            cm = x_minus.reshape(-1, 3)
            
            dp = cp[:, np.newaxis, :] - cp[np.newaxis, :, :]
            dists_p = np.sqrt(np.sum(dp**2, axis=2) + 1e-15)
            np.fill_diagonal(dists_p, np.inf)
            ip, jp = np.triu_indices(len(cp), k=1)
            pair_p = dists_p[ip, jp] / 2.0
            wall_p = CUBE_HALF - np.max(np.abs(cp), axis=1)
            all_p = np.concatenate([pair_p, wall_p])
            min_p = np.min(all_p)
            if min_p < 1e-10:
                val_p = 0.0
            else:
                sh_p = all_p - min_p
                ls_p = np.log(np.sum(np.exp(-beta * sh_p)))
                val_p = min_p - ls_p / beta
            
            dm = cm[:, np.newaxis, :] - cm[np.newaxis, :, :]
            dists_m = np.sqrt(np.sum(dm**2, axis=2) + 1e-15)
            np.fill_diagonal(dists_m, np.inf)
            im, jm = np.triu_indices(len(cm), k=1)
            pair_m = dists_m[im, jm] / 2.0
            wall_m = CUBE_HALF - np.max(np.abs(cm), axis=1)
            all_m = np.concatenate([pair_m, wall_m])
            min_m = np.min(all_m)
            if min_m < 1e-10:
                val_m = 0.0
            else:
                sh_m = all_m - min_m
                ls_m = np.log(np.sum(np.exp(-beta * sh_m)))
                val_m = min_m - ls_m / beta
            
            grad[i] = (val_p - val_m) / (2 * eps)
        
        return -soft_min, grad

    bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 15000}
        )
        
        if time.time() - start_time > time_budget:
            pass
            
        polished = res.x.reshape(-1, 3)
        polished = np.clip(polished, -0.499999, 0.499999)
        return polished
    except Exception:
        return centers

def _local_refine(centers, time_limit=5.0):
    """
    Local refinement: try small moves in 26 directions for each point.
    """
    start_time = time.time()
    c = centers.copy()
    current_score = _score(c)
    
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                directions.append(np.array([dx, dy, dz], dtype=float))
    directions = np.array(directions)
    
    step = 0.002
    
    while time.time() - start_time < time_limit:
        improved = False
        # Shuffle point order
        order = np.random.permutation(N)
        
        for idx in order:
            if time.time() - start_time > time_limit:
                break
            
            for d in directions:
                c_new = c.copy()
                c_new[idx] = c[idx] + d * step
                c_new[idx] = np.clip(c_new[idx], -CUBE_HALF, CUBE_HALF)
                
                new_score = _score(c_new)
                if new_score > current_score:
                    c = c_new
                    current_score = new_score
                    improved = True
                    break
        
        if not improved:
            # Try smaller step
            step *= 0.5
            if step < 1e-5:
                break
                
    return c, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_radius = 0.0
    best_centers = None
    
    rng = np.random.RandomState(42)
    
    # Create initial configurations
    inits = []
    
    # Grid configurations
    grid_configs = [
        (5, 4, 5),
        (4, 5, 5),
        (5, 5, 4),
    ]
    
    for nx, ny, nz in grid_configs:
        if nx * ny * nz == N:
            grid = _initial_grid(nx, ny, nz)
            # Add small jitter
            jittered = grid + rng.normal(0, 0.01, size=(N, 3))
            jittered = np.clip(jittered, -0.45, 0.45)
            inits.append(jittered)
    
    # Random inits
    for i in range(6):
        rc = _initial_random(rng)
        inits.append(rc)
    
    # Structured random
    base_grid = _initial_grid(5, 4, 5)
    for i in range(3):
        noise_scale = 0.03 * (i + 1)
        noisy = base_grid + rng.normal(0, noise_scale, size=(N, 3))
        noisy = np.clip(noisy, -0.45, 0.45)
        inits.append(noisy)
    
    # Process each initial configuration
    for init in inits:
        if time.time() > deadline - 25.0:
            break
            
        r_est = _score(init)
        
        # Phase 1: Simulated Annealing
        sa_time = min(20.0, max(5.0, deadline - 30.0 - time.time()))
        sa_centers, sa_score = _simulated_annealing(init, time_limit=sa_time, rng=rng)
        
        if sa_score > best_radius:
            best_radius = sa_score
            best_centers = sa_centers.copy()
        
        # Phase 2: Local refinement
        if time.time() < deadline - 15.0:
            lr_time = min(5.0, deadline - 15.0 - time.time())
            lr_centers, lr_score = _local_refine(sa_centers, time_limit=lr_time)
            
            if lr_score > best_radius:
                best_radius = lr_score
                best_centers = lr_centers.copy()
        
        # Phase 3: Gradient polish
        if time.time() < deadline - 10.0:
            gp_time = min(8.0, deadline - 10.0 - time.time())
            gp_centers = _gradient_polish(best_centers, time_budget=gp_time)
            gp_score = _score(gp_centers)
            
            if gp_score > best_radius:
                best_radius = gp_score
                best_centers = gp_centers.copy()
        
        # Phase 4: Final local polish
        if time.time() < deadline - 5.0:
            fp_time = min(4.0, deadline - 5.0 - time.time())
            fp_centers, fp_score = _local_refine(best_centers, time_limit=fp_time)
            
            if fp_score > best_radius:
                best_radius = fp_score
                best_centers = fp_centers.copy()
    
    # Final safety check
    if best_centers is None:
        best_centers = _initial_grid(5, 4, 5)
    else:
        best_centers = np.clip(best_centers, -0.49999, 0.49999)
        if not np.all(np.isfinite(best_centers)):
            best_centers = _initial_grid(5, 4, 5)
        
    return best_centers
# EVOLVE-BLOCK-END
