import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective(centers_flat, beta, iu):
    """
    Objective function for L-BFGS-B.
    Minimizes the soft-min approximation of the radius.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    sum_exp = np.sum(np.exp(neg_beta_x - max_val))
    obj = (max_val + np.log(sum_exp)) / beta
    
    return obj

def _soft_min_gradient(centers_flat, beta, iu):
    """
    Gradient of the soft-min objective for L-BFGS-B.
    """
    centers = centers_flat.reshape(-1, 3)
    
    # 1. Wall distances
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Soft-min gradient
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp  # softmax weights
    
    # Gradient with respect to all_vals: -weights
    grad_all = -weights
    
    # Map back to centers
    n_wall = N * 3
    n_pair = len(iu[0])
    
    grad_centers = np.zeros_like(centers)
    
    # Wall contributions: d(0.5 - |c|)/dc = -sign(c)
    wall_grad = grad_all[:n_wall].reshape(N, 3)
    signs = np.sign(centers)
    signs[signs == 0] = 1.0  # avoid zero
    grad_centers -= wall_grad * signs
    
    # Pair contributions
    pair_grad = grad_all[n_wall:]
    for idx in range(n_pair):
        i, j = iu[0][idx], iu[1][idx]
        d = pair_grad[idx]
        diff = centers[i] - centers[j]
        dist = np.linalg.norm(diff)
        if dist > 1e-15:
            unit = diff / dist
            grad_centers[i] += d * unit
            grad_centers[j] -= d * unit
    
    return grad_centers.flatten()

def _build_initial_grid(nx, ny, nz, noise_scale=0.005):
    """Build a grid initialization with small noise to break symmetry."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    rng = np.random.default_rng(42)
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_bcc_init(n=10, noise_scale=0.005):
    """
    Build a BCC (Body-Centered Cubic) inspired initialization.
    """
    nx, ny, nz = 10, 10, 1
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    all_centers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                all_centers.append([xs[i], ys[j], zs[k]])
                if i < nx-1 and j < ny-1 and k < nz-1:
                    all_centers.append([(xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, (zs[k]+zs[k+1])/2])
                elif i < nx-1 and j < ny-1:
                    all_centers.append([(xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, zs[k]])
                elif i < nx-1:
                    all_centers.append([(xs[i]+xs[i+1])/2, ys[j], zs[k]])
                elif j < ny-1:
                    all_centers.append([xs[i], (ys[j]+ys[j+1])/2, zs[k]])
    
    all_centers = np.array(all_centers)
    rng = np.random.default_rng(42)
    
    if len(all_centers) >= N:
        idx = rng.choice(len(all_centers), N, replace=False)
        centers = all_centers[idx]
    else:
        centers = all_centers
        while len(centers) < N:
            centers = np.vstack([centers, centers[0]])
    
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(n=100, seed=42):
    """
    Build a Halton sequence based initialization.
    """
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 3))
    for i in range(1, n+1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _anneal(centers, deadline, T_start=0.01, T_end=0.0001, n_steps=5000):
    """
    Simulated annealing with incremental radius updates.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    current_centers = centers.copy()
    
    rng = np.random.default_rng(42)
    
    # Precompute distance matrix
    dist_mat = cdist(current_centers, current_centers)
    np.fill_diagonal(dist_mat, np.inf)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(1, n_steps - 1)
        T = T_start * (T_end / T_start) ** progress
        
        idx = rng.integers(N)
        delta = rng.normal(0, 0.01, 3)
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
        
        # Incremental update: only recompute distances from idx
        new_dists_to_idx = np.linalg.norm(current_centers - new_centers[idx], axis=1)
        new_dists_to_idx[idx] = np.inf
        
        # Update dist_mat row/col for idx
        old_min = np.min(dist_mat)
        dist_mat[idx, :] = new_dists_to_idx
        dist_mat[:, idx] = new_dists_to_idx
        
        new_min_pair = np.min(dist_mat)
        
        # Wall distance for moved point
        wall_dists = 0.5 - np.abs(new_centers)
        new_min_wall = np.min(wall_dists)
        
        new_radius = min(new_min_wall, new_min_pair / 2.0)
        
        if new_radius >= current_radius:
            current_centers = new_centers
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        else:
            delta_E = current_radius - new_radius
            if delta_E < 1e-15:
                accept = True
            else:
                accept = np.log(rng.uniform()) > -delta_E / T
            if accept:
                current_centers = new_centers
                current_radius = new_radius
        
        # Restore dist_mat if not accepted
        if not (new_radius >= current_radius or accept):
            dist_mat[idx, :] = np.linalg.norm(current_centers - current_centers[idx], axis=1)
            dist_mat[idx, idx] = np.inf
            dist_mat[:, idx] = np.linalg.norm(current_centers - current_centers[idx], axis=1)
            dist_mat[idx, :] = np.linalg.norm(current_centers - current_centers[idx], axis=1)
            dist_mat[:, idx] = np.linalg.norm(current_centers - current_centers[idx], axis=1)
            dist_mat[idx, idx] = np.inf
    
    return best_centers, best_radius

def _coordinate_descent_polish(centers, deadline, max_rounds=10):
    """
    Coordinate descent: move each point to maximize the minimum constraint.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            # Try small moves in each direction
            best_local = best_centers.copy()
            best_local_r = best_radius
            
            for dim in range(3):
                for step_size in [0.001, 0.0005, 0.0001]:
                    for direction in [-1, 1]:
                        trial = best_local.copy()
                        trial[idx, dim] += direction * step_size
                        trial[idx, dim] = np.clip(trial[idx, dim], -0.5, 0.5)
                        
                        r = _compute_radius(trial)
                        if r > best_local_r + 1e-12:
                            best_local = trial
                            best_local_r = r
                            improved = True
            
            if np.any(best_local != best_centers):
                best_centers = best_local
                if best_local_r > best_radius:
                    best_radius = best_local_r
        
        if not improved:
            break
    
    return best_centers, best_radius

def _optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=6):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                jac=_soft_min_gradient,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        if time.time() > deadline - 3.0:
            break
    
    return best_centers, best_radius

def _polish(centers, deadline):
    """
    Final high-precision polish: L-BFGS-B with high beta + coordinate descent.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    
    # Multi-scale beta polish
    for beta in [10000.0, 50000.0, 200000.0]:
        if time.time() > deadline - 2.0:
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                jac=_soft_min_gradient,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    # Coordinate descent polish
    if time.time() < deadline - 1.0:
        cd_centers, cd_radius = _coordinate_descent_polish(best_centers, deadline)
        if cd_radius > best_radius:
            best_radius = cd_radius
            best_centers = cd_centers
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Generate diverse initial configurations
    inits = []
    
    grid_configs = [(5, 5, 4), (4, 5, 5), (5, 4, 5)]
    for nx, ny, nz in grid_configs:
        for noise in [0.003, 0.005, 0.01]:
            c = _build_initial_grid(nx, ny, nz, noise)
            inits.append(c)
    
    inits.append(_build_bcc_init(noise_scale=0.005))
    inits.append(_build_halton_init())
    
    # Phase 2: Optimize each init
    # Sort inits by initial radius (descending) to prioritize good starts
    init_radii = []
    for c in inits:
        r = _compute_radius(c)
        init_radii.append((r, c))
    init_radii.sort(key=lambda x: -x[0])
    
    for i, (r_init, init_centers) in enumerate(init_radii):
        if time.time() > deadline - 10.0:
            break
        
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init_centers.copy()
        
        # Run annealing with time budget
        if time.time() > deadline - 8.0:
            break
        
        anneal_deadline = min(deadline - 5.0, start_time + 12.0)
        annealed_centers, anneal_radius = _anneal(
            init_centers, anneal_deadline,
            T_start=0.005, T_end=0.0001, n_steps=3000
        )
        
        if anneal_radius > best_radius:
            best_radius = anneal_radius
            best_centers = annealed_centers.copy()
        
        # Run L-BFGS-B from annealed result
        if time.time() > deadline - 3.0:
            break
        
        opt_centers, opt_radius = _optimize_from_initial(
            annealed_centers, deadline,
            beta_start=50.0, beta_end=10000.0, n_beta_steps=8
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    # Phase 3: Final polish with extra time
    if best_centers is not None and time.time() < deadline - 1.0:
        polished_centers, polished_radius = _polish(best_centers, deadline)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centers = polished_centers
    
    # Fallback
    if best_centers is None:
        best_centers = _build_initial_grid(5, 5, 4, 0.01)
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    assert best_centers.shape == (N, 3)
    assert np.all(np.isfinite(best_centers))
    
    return best_centers

# EVOLVE-BLOCK-END
