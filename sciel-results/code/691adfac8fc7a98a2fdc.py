import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_wall_dist, min_pair_dist / 2.0)


def _soft_min_objective(x, beta, iu):
    """
    Objective: soft-min approximation of the radius constraints.
    Minimizes G(x) = (1/beta) * ln(sum_k exp(-beta * c_k(x)))
    As beta -> inf, G(x) -> -min_k c_k(x), so minimizing G maximizes min c_k.
    """
    c = x.reshape(-1, 3)
    
    # Wall constraints: c_w = 0.5 - |c_i|
    wall_vals = (0.5 - np.abs(c)).flatten()
    
    # Pair constraints: c_p = dist_ij / 2
    dists = cdist(c, c, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = (dists / 2.0)[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Compute soft-min: (1/beta) * ln(sum exp(-beta * c_k))
    neg_beta_c = -beta * all_vals
    max_neg_beta_c = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - max_neg_beta_c)
    S = np.sum(exp_vals)
    obj = (max_neg_beta_c + np.log(S)) / beta
    
    return obj


def _soft_min_grad(x, beta, iu):
    """Gradient of the soft-min objective."""
    c = x.reshape(-1, 3)
    
    wall_vals = (0.5 - np.abs(c)).flatten()
    dists = cdist(c, c, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = (dists / 2.0)[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    neg_beta_c = -beta * all_vals
    max_neg_beta_c = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - max_neg_beta_c)
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    grad = np.zeros_like(c)
    
    # Wall gradient: dG/dc[i,k] = w_{i,k} * sign(c[i,k])
    wall_weights = weights[:3*N].reshape(-1, 3)
    grad += wall_weights * np.sign(c)
    
    # Pair gradient: dG/dc_i = -sum_j w_{ij} * (c_i - c_j) / dist_ij
    pair_weights = weights[3*N:]
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    mask = dists > 1e-12
    inv_dists = np.zeros_like(dists)
    inv_dists[mask] = 1.0 / dists[mask]
    unit_diff = diff * inv_dists[:, :, np.newaxis]
    
    w_mat = np.zeros((N, N))
    w_mat[iu] = pair_weights
    w_mat[iu[1], iu[0]] = pair_weights
    
    grad_pairs = -np.sum(w_mat[:, :, np.newaxis] * unit_diff, axis=1)
    grad += grad_pairs
    
    return grad.ravel()


def _soft_min_objective_and_grad(x, beta, iu):
    """Combined objective and gradient for L-BFGS-B."""
    obj = _soft_min_objective(x, beta, iu)
    grad = _soft_min_grad(x, beta, iu)
    return obj, grad


def _anneal(centers, deadline, T_start=0.005, T_end=0.00001, n_steps=8000):
    """
    Simulated annealing to explore the configuration space.
    Uses Metropolis-Hastings with energy = -radius (we want to maximize radius).
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    current_centers = centers.copy()
    
    rng = np.random.default_rng(777)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Annealing schedule
        progress = step / max(1, n_steps - 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Random perturbation
        idx = rng.integers(N)
        delta = rng.normal(0, 0.008, 3)
        
        new_centers = current_centers.copy()
        new_centers[idx] += delta
        new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
        
        # Compute new radius
        new_radius = _compute_radius(new_centers)
        
        # Metropolis criterion
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
    
    return best_centers, best_radius


def _optimize_soft_min(x0, deadline, beta_start=20.0, beta_end=20000.0, n_steps=8):
    """
    Run L-BFGS-B optimization with increasing beta.
    """
    best_x = x0.copy()
    best_r = _compute_radius(x0.reshape(-1, 3))
    iu = np.triu_indices(N, k=1)
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _soft_min_objective_and_grad,
                best_x,
                args=(beta, iu),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            new_x = np.clip(res.x, -0.5, 0.5)
            new_r = _compute_radius(new_x.reshape(-1, 3))
            
            if new_r > best_r:
                best_r = new_r
                best_x = new_x
        except Exception:
            pass
    
    return best_x, best_r


def _targeted_polish(centers, deadline, n_iters=200):
    """
    Targeted polishing: identify the most constrained pairs and walls,
    then apply local moves to relieve those constraints.
    """
    c = centers.copy()
    r_current = _compute_radius(c)
    rng = np.random.default_rng(999)
    
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Identify bottleneck constraints
        wall_dists = 0.5 - np.abs(c)
        dists = cdist(c, c, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        # Find the most constrained wall and pair
        min_wall_idx = np.argmin(wall_dists)
        min_pair_idx = np.argmin(pair_dists)
        
        # Choose which to target (randomly or based on severity)
        target_wall = min_wall_dists = wall_dists.flatten()[min_wall_idx]
        target_pair = min_pair_dists = pair_dists.flatten()[min_pair_idx]
        
        # Apply targeted moves
        # Move the most constrained sphere away from its bottleneck
        if target_wall < target_pair:
            # Wall constraint is tighter - move sphere away from wall
            i = min_wall_idx // 3
            k = min_wall_idx % 3
            direction = -np.sign(c[i, k])
            step_size = 0.002
            new_c = c.copy()
            new_c[i, k] += direction * step_size
            new_c[i, k] = np.clip(new_c[i, k], -0.5, 0.5)
        else:
            # Pair constraint is tighter - separate the two spheres
            iu = np.triu_indices(N, k=1)
            flat_idx = min_pair_idx
            i, j = iu[0][flat_idx], iu[1][flat_idx]
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                direction = diff / dist
                step_size = 0.002
                new_c = c.copy()
                new_c[i] += direction * step_size
                new_c[j] -= direction * step_size
                new_c = np.clip(new_c, -0.5, 0.5)
            else:
                new_c = c
        
        # Also try random small moves
        if rng.random() < 0.3:
            idx = rng.integers(N)
            delta = rng.normal(0, 0.003, 3)
            new_c[idx] += delta
            new_c[idx] = np.clip(new_c[idx], -0.5, 0.5)
        
        r_new = _compute_radius(new_c)
        if r_new > r_current:
            c = new_c
            r_current = r_new
    
    return c, r_current


def _init_grid(nx, ny, nz, rng):
    """Create a perturbed grid initialization."""
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))[:N]
    noise_scale = 0.006
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_bcc_like(rng):
    """BCC-like initialization for better space-filling."""
    bcc_offsets = [
        [0, 0, 0], [0.5, 0.5, 0.5]
    ]
    
    nx = ny = nz = 5
    sub_size = 1.0 / nx
    
    centers = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                base = np.array([-0.5 + ix * sub_size, -0.5 + iy * sub_size, -0.5 + iz * sub_size])
                for offset in bcc_offsets:
                    pos = base + np.array(offset) * sub_size
                    if -0.5 <= pos[0] <= 0.5 and -0.5 <= pos[1] <= 0.5 and -0.5 <= pos[2] <= 0.5:
                        centers.append(pos)
    
    centers = np.array(centers[:N])
    if len(centers) < N:
        remaining = N - len(centers)
        random_pts = rng.uniform(-0.45, 0.45, size=(remaining, 3))
        centers = np.vstack([centers, random_pts])
    
    centers += rng.normal(0, 0.004, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_halton(rng):
    """Halton sequence initialization."""
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    
    centers = np.zeros((N, 3))
    for i in range(1, N + 1):
        centers[i - 1, 0] = -0.5 + halton(i, 2)
        centers[i - 1, 1] = -0.5 + halton(i, 3)
        centers[i - 1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.004, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def _init_random(rng):
    """Random initialization."""
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def _init_fcc_like(rng):
    """FCC-like initialization."""
    fcc_offsets = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0, 0, 0.5], [0.5, 0.5, 0.5], [0.5, 0, 0], [0, 0.5, 0]
    ]
    
    nx = ny = nz = 4
    sub_size = 1.0 / nx
    
    centers = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                base = np.array([-0.5 + ix * sub_size, -0.5 + iy * sub_size, -0.5 + iz * sub_size])
                for offset in fcc_offsets:
                    pos = base + np.array(offset) * sub_size
                    if -0.5 <= pos[0] <= 0.5 and -0.5 <= pos[1] <= 0.5 and -0.5 <= pos[2] <= 0.5:
                        centers.append(pos)
    
    centers = np.array(centers[:N])
    if len(centers) < N:
        remaining = N - len(centers)
        random_pts = rng.uniform(-0.45, 0.45, size=(remaining, 3))
        centers = np.vstack([centers, random_pts])
    
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    return centers


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    rng = np.random.default_rng(12345)
    
    best_centers = None
    best_radius = -np.inf
    
    # Generate diverse initializations
    inits = [
        _init_grid(5, 5, 4, rng),
        _init_grid(4, 5, 5, rng),
        _init_grid(5, 4, 5, rng),
        _init_bcc_like(rng),
        _init_halton(rng),
        _init_fcc_like(rng),
        _init_random(rng),
    ]
    
    # Phase 1: Annealing + soft-min optimization for each initialization
    for init_idx, init_centers in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        c = init_centers.copy()
        
        # Simulated annealing for basin exploration
        c, r_anneal = _anneal(c, deadline - 15.0, T_start=0.004, T_end=0.00001, n_steps=5000)
        
        if r_anneal > best_radius:
            best_radius = r_anneal
            best_centers = c.copy()
        
        # Soft-min optimization
        if time.time() > deadline - 10.0:
            break
        
        opt_centers, opt_r = _optimize_soft_min(c.flatten(), deadline - 5.0, beta_start=30.0, beta_end=30000.0, n_steps=7)
        opt_centers = opt_centers.reshape(-1, 3)
        
        if opt_r > best_radius:
            best_radius = opt_r
            best_centers = opt_centers.copy()
    
    # Phase 2: Targeted polishing on the best result
    if best_centers is not None and time.time() < deadline - 15.0:
        c = best_centers.copy()
        c, r_polished = _targeted_polish(c, deadline - 10.0, n_iters=300)
        
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = c
        
        # Final soft-min optimization with high beta
        if time.time() < deadline - 5.0:
            opt_centers, opt_r = _optimize_soft_min(c.flatten(), deadline - 2.0, beta_start=1000.0, beta_end=100000.0, n_steps=4)
            opt_centers = opt_centers.reshape(-1, 3)
            if opt_r > best_radius:
                best_radius = opt_r
                best_centers = opt_centers
    
    # Phase 3: One more anneal from the best result if time permits
    if best_centers is not None and time.time() < deadline - 5.0:
        c = best_centers.copy()
        c, r_final = _anneal(c, deadline - 2.0, T_start=0.002, T_end=0.000005, n_steps=3000)
        
        if r_final > best_radius:
            best_radius = r_final
            best_centers = c
    
    # Fallback
    if best_centers is None:
        best_centers = _init_grid(5, 5, 4, rng)
    
    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

# EVOLVE-BLOCK-END
