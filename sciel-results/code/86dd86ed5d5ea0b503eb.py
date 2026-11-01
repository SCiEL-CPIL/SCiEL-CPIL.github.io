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
    BCC is known to be a good lattice for sphere packing.
    We create a 10x10x1 grid of BCC unit cells, giving 2*10*10*1 = 200 points,
    then randomly select 100.
    """
    # BCC unit cell has points at corners and center
    # We'll create a grid of BCC cells
    nx, ny, nz = 10, 10, 1
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    all_centers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Corner points
                all_centers.append([xs[i], ys[j], zs[k]])
                # Body center
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
        # Pad if needed
        while len(centers) < N:
            centers = np.vstack([centers, centers[0]])
    
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _build_halton_init(n=100, seed=42):
    """
    Build a Halton sequence based initialization for good space-filling properties.
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
    Simulated annealing with efficient incremental radius updates.
    Only recomputes distances involving the moved point.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    current_radius = best_radius
    current_centers = centers.copy()
    
    rng = np.random.default_rng(42)
    
    # Precompute full distance matrix for incremental updates
    dist_mat = cdist(current_centers, current_centers)
    np.fill_diagonal(dist_mat, np.inf)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Annealing schedule
        progress = step / max(1, n_steps - 1)
        T = T_start * (T_end / T_start) ** progress
        
        # Random perturbation
        idx = rng.integers(N)
        delta = rng.normal(0, 0.008, 3)
        
        new_pos = np.clip(current_centers[idx] + delta, -0.5, 0.5)
        
        # Incremental radius update: only need distances from idx to all others
        new_dists = cdist(new_pos.reshape(1, 3), current_centers, metric='euclidean')[0]
        new_dists[idx] = np.inf
        
        # Wall distance for the moved point
        new_wall = np.min(0.5 - np.abs(new_pos))
        
        # Find current min pair dist and min wall dist
        min_pair = np.min(dist_mat)
        min_wall = np.min(0.5 - np.abs(current_centers))
        current_r = min(min_wall, min_pair / 2.0)
        
        # New min pair dist: min of (min of old dist_mat excluding row/col idx, new_dists)
        old_min_excl = np.inf
        if N > 2:
            # Get min of all pairs not involving idx
            mask = np.ones(N, dtype=bool)
            mask[idx] = False
            sub = dist_mat[np.ix_(mask, mask)]
            old_min_excl = np.min(sub)
        
        new_min_pair = min(old_min_excl, np.min(new_dists))
        new_r = min(new_wall, new_min_pair / 2.0)
        
        # Metropolis criterion
        if new_r >= current_r:
            # Accept: update dist_mat
            dist_mat[idx, :] = new_dists
            dist_mat[:, idx] = new_dists
            current_centers[idx] = new_pos
            current_radius = new_r
            if new_r > best_radius:
                best_radius = new_r
                best_centers = current_centers.copy()
        else:
            delta_E = current_r - new_r
            if delta_E < 1e-15:
                accept = True
            else:
                accept = np.log(rng.uniform()) > -delta_E / T
            if accept:
                dist_mat[idx, :] = new_dists
                dist_mat[:, idx] = new_dists
                current_centers[idx] = new_pos
                current_radius = new_r
    
    return best_centers, best_radius

def _optimize_from_initial(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=6):
    """
    Run L-BFGS-B optimization with increasing beta from initial centers.
    Returns (best_centers, best_radius).
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
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
        
        if time.time() > deadline - 5.0:
            break
    
    return best_centers, best_radius

def _polish(centers, deadline):
    """
    Final high-precision polish using epigraph formulation with auxiliary variable t.
    Maximizes t subject to all constraints being >= t.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    n_constraints = N * 3 + n_pairs  # wall distances + pair distances
    
    def _epigraph_obj(x):
        """Minimize -t where t is bounded by all constraints."""
        centers = x[:N*3].reshape(N, 3)
        t = x[N*3]
        
        # Wall distances: 0.5 - |c_ik| for each coordinate
        wall_dists = 0.5 - np.abs(centers)
        wall_flat = wall_dists.flatten()
        
        # Pair distances / 2
        dists = cdist(centers, centers, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists[iu] / 2.0
        
        # All constraints must be >= t
        # Penalty: sum of max(0, t - constraint)^2
        all_constr = np.concatenate([wall_flat, pair_vals])
        violations = np.maximum(0, t - all_constr)
        penalty = np.sum(violations ** 2)
        
        # Objective: -t + penalty (we want to maximize t while keeping constraints >= t)
        return -t + 1000.0 * penalty
    
    def _epigraph_grad(x):
        """Gradient of epigraph objective."""
        centers = x[:N*3].reshape(N, 3)
        t = x[N*3]
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        wall_flat = wall_dists.flatten()
        
        # Pair distances / 2
        dists = cdist(centers, centers, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        pair_vals = dists[iu] / 2.0
        
        all_constr = np.concatenate([wall_flat, pair_vals])
        violations = np.maximum(0, t - all_constr)
        
        # Gradient w.r.t. t: -1 + 2000 * sum(violations)
        grad_t = -1.0 + 2000.0 * np.sum(violations)
        
        # Gradient w.r.t. centers: from penalty term
        # d/dx_i sum_j max(0, t - c_j(x))
        # For wall: c_j = 0.5 - |x_k|, dc/dx_k = -sign(x_k)
        # For pairs: c_j = |p_i - p_j|/2, dc/dx_i = (x_i - x_j)/|x_i - x_j|
        
        grad_centers = np.zeros((N, 3))
        
        # Wall contributions
        for i in range(N):
            for k in range(3):
                idx = i * 3 + k
                if violations[idx] > 0:
                    # c = 0.5 - |x_k|, dc/dx_k = -sign(x_k)
                    # d penalty/dx_k = 2 * violation * 1000 * (-dc/dx_k) = -2000 * violation * sign(x_k)
                    sign_val = np.sign(centers[i, k])
                    if sign_val == 0:
                        sign_val = 1.0
                    grad_centers[i, k] += -2000.0 * violations[idx] * sign_val
        
        # Pair contributions
        for idx_pair in range(n_pairs):
            if violations[N*3 + idx_pair] > 0:
                i, j = iu[0][idx_pair], iu[1][idx_pair]
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-15:
                    unit = diff / dist
                    # c = dist/2, dc/dx_i = unit/2, dc/dx_j = -unit/2
                    # d penalty/dx_i = 2 * viol * 1000 * (-dc/dx_i) = -2000 * viol * unit/2
                    grad_centers[i] += -1000.0 * violations[N*3 + idx_pair] * unit
                    grad_centers[j] += 1000.0 * violations[N*3 + idx_pair] * unit
        
        grad = np.zeros_like(x)
        grad[:N*3] = grad_centers.flatten()
        grad[N*3] = grad_t
        return grad
    
    x0 = np.concatenate([best_centers.flatten(), [best_radius * 0.999]])
    bounds = [(-0.5, 0.5)] * (N * 3) + [(0.0, 0.5)]
    
    remaining = deadline - time.time()
    if remaining > 10.0:
        try:
            res = minimize(
                _epigraph_obj,
                x0,
                jac=_epigraph_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            new_centers = res.x[:N*3].reshape(N, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    # Final soft-min polish with very high beta
    if time.time() < deadline - 5.0:
        beta = 100000.0
        x0 = best_centers.flatten()
        try:
            res = minimize(
                _soft_min_objective,
                x0,
                args=(beta, iu),
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers
        except Exception:
            pass
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Generate diverse initial configurations
    inits = []
    
    # Grid-based inits (most promising)
    grid_configs = [(5, 5, 4), (4, 5, 5), (5, 4, 5)]
    for nx, ny, nz in grid_configs:
        for noise in [0.003, 0.005, 0.01]:
            c = _build_initial_grid(nx, ny, nz, noise)
            inits.append(c)
    
    # BCC-inspired init
    inits.append(_build_bcc_init(noise_scale=0.005))
    
    # Halton sequence init
    inits.append(_build_halton_init())
    
    # Sort by initial radius (best first)
    init_radii = [_compute_radius(c) for c in inits]
    sorted_indices = np.argsort(init_radii)[::-1]
    inits = [inits[i] for i in sorted_indices]
    init_radii = [init_radii[i] for i in sorted_indices]
    
    # Phase 2: For each init, run annealing then L-BFGS-B
    # Allocate time based on initial quality
    n_inits = len(inits)
    for i, init_centers in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = deadline - 5.0 - elapsed
        
        # Skip if not enough time
        if remaining < 8.0:
            break
        
        # If we already have a good solution, spend less time on worse inits
        if i > 0 and best_radius > 0.110:
            if remaining < 5.0:
                break
        
        r_init = init_radii[i]
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init_centers.copy()
        
        # Run annealing with time allocation
        if i == 0:
            # Best init gets more time
            anneal_deadline = min(deadline - 20.0, start_time + 25.0)
            n_steps = 8000
        elif i == 1:
            anneal_deadline = min(deadline - 15.0, start_time + 18.0)
            n_steps = 5000
        else:
            anneal_deadline = min(deadline - 10.0, start_time + 12.0)
            n_steps = 3000
        
        annealed_centers, anneal_radius = _anneal(
            init_centers, anneal_deadline,
            T_start=0.005, T_end=0.0001, n_steps=n_steps
        )
        
        if anneal_radius > best_radius:
            best_radius = anneal_radius
            best_centers = annealed_centers.copy()
        
        # Run L-BFGS-B from annealed result
        if time.time() > deadline - 8.0:
            break
        
        # Allocate L-BFGS time based on remaining budget
        if i == 0:
            opt_deadline = deadline - 15.0
            n_beta_steps = 10
        elif i == 1:
            opt_deadline = deadline - 12.0
            n_beta_steps = 8
        else:
            opt_deadline = deadline - 8.0
            n_beta_steps = 6
        
        opt_centers, opt_radius = _optimize_from_initial(
            annealed_centers, opt_deadline,
            beta_start=30.0, beta_end=20000.0, n_beta_steps=n_beta_steps
        )
        
        if opt_radius > best_radius:
            best_radius = opt_radius
            best_centers = opt_centers
    
    # Phase 3: Final polish
    if best_centers is not None and time.time() < deadline - 2.0:
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
