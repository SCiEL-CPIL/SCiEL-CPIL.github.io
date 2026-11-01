"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centers):
    """
    Compute the largest common radius for the given centers.
    Centers are in [-0.5, 0.5]^3.
    """
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    
    # Distance to walls
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    
    # Pairwise distances
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_wall_dist, min_pair_dist / 2.0)

def _soft_min_objective_and_grad(centers_flat, beta, iu):
    """
    Objective and gradient for L-BFGS-B using soft-min surrogate.
    """
    centers = centers_flat.reshape(-1, 3)
    n = centers.shape[0]
    
    # 1. Wall distances: 0.5 - |c_i|
    dists_to_walls = 0.5 - np.abs(centers)
    wall_vals = dists_to_walls.flatten()  # shape (3N,)
    
    # 2. Pairwise distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0
    pair_vals_unique = pair_vals[iu]  # shape (N*(N-1)/2,)
    
    all_vals = np.concatenate([wall_vals, pair_vals_unique])
    
    # Calculate LSE: 1/beta * ln(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    shifted = neg_beta_x - max_val
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp == 0:
        return 1e10, np.zeros_like(centers_flat)
    
    obj = (max_val + np.log(sum_exp)) / beta
    
    # Gradient: d/dx_i [1/beta * ln(sum_j exp(-beta * x_j))]
    # = -sum_j exp(-beta*x_j) * (-beta) / (beta * sum_j exp(-beta*x_j))
    # = sum_j exp(-beta*x_j) / sum_j exp(-beta*x_j) * (-1)
    # Wait, let me redo this carefully.
    # f(x) = 1/beta * ln(sum_j exp(-beta * x_j))
    # df/dx_k = 1/beta * (1/sum_j exp(-beta*x_j)) * sum_j d/dx_k exp(-beta*x_j)
    # = 1/beta * (1/sum_j exp(-beta*x_j)) * sum_j exp(-beta*x_j) * (-beta) * delta_{jk}
    # = 1/beta * (-beta) * exp(-beta*x_k) / sum_j exp(-beta*x_j)
    # = -exp(-beta*x_k) / sum_j exp(-beta*x_j)
    # = -exp_shifted[k] / sum_exp
    
    # So the gradient with respect to each constraint value is -exp_shifted/sum_exp
    # But we need gradient with respect to centers_flat.
    
    # Gradient of wall_vals w.r.t. centers:
    # wall_val for center i, coord k: 0.5 - |c_i[k]|
    # d(0.5 - |c_i[k]|)/dc_i[k] = -sign(c_i[k])
    
    # Gradient of pair_vals_unique w.r.t. centers:
    # For pair (i,j), val = |c_i - c_j| / 2
    # d/dc_i = (c_i - c_j) / (2 * |c_i - c_j|) * (1/2) = (c_i - c_j) / (4 * |c_i - c_j|)
    # Wait: pair_val = dist/2, so d(pair_val)/dc_i = (c_i - c_j)/dist * (1/2) = (c_i-c_j)/(2*dist)
    # and d(pair_val)/dc_j = (c_j - c_i)/dist * (1/2) = (c_j-c_i)/(2*dist)
    
    # Build gradient array
    grad = np.zeros_like(centers_flat)
    
    # Wall contributions
    wall_grad = -np.sign(centers)  # shape (N, 3)
    # Weight by -exp_shifted/sum_exp for wall constraints
    wall_weights = -exp_shifted[:3*N] / sum_exp  # shape (3N,)
    # Reshape to (N, 3) and multiply
    grad += (wall_weights.reshape(-1, 3) * wall_grad).flatten()
    
    # Pair contributions
    pair_weights = -exp_shifted[3*N:] / sum_exp  # shape (N*(N-1)/2,)
    
    # For each pair (i,j) with i<j, weight w:
    # grad_i += w * (c_i - c_j) / (2 * dist_ij)
    # grad_j += w * (c_j - c_i) / (2 * dist_ij)
    # Note: (c_i - c_j)/(2*dist) is the unit vector from j to i divided by 2
    
    ii, jj = iu
    for idx in range(len(ii)):
        i, j = ii[idx], jj[idx]
        d = dists[i, j]
        if d < 1e-15:
            continue
        w = pair_weights[idx]
        diff = centers[i] - centers[j]
        unit = diff / d
        grad[i*3:(i+1)*3] += w * unit * 0.5
        grad[j*3:(j+1)*3] += w * (-unit) * 0.5
    
    return obj, grad

def _force_directed_refinement(centers, deadline, max_iters=200):
    """
    Force-directed refinement: push overlapping spheres apart and pull toward center.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        dists = cdist(centers, centers, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < 1e-15:
                    continue
                # If spheres overlap or are close, push apart
                if d < 0.25:  # threshold
                    strength = (0.25 - d) / 0.25
                    diff = centers[i] - centers[j]
                    unit = diff / d
                    forces[i] += strength * unit
                    forces[j] -= strength * unit
        
        # Wall repulsion
        for i in range(N):
            for k in range(3):
                dist_to_wall = 0.5 - abs(centers[i, k])
                if dist_to_wall < 0.05:
                    strength = (0.05 - dist_to_wall) / 0.05
                    forces[i, k] -= strength * np.sign(centers[i, k])
        
        # Small random perturbation to escape local minima
        rng = np.random.default_rng(it)
        forces += rng.normal(0, 0.001, centers.shape)
        
        # Update positions
        step_size = 0.01
        centers += step_size * forces
        centers = np.clip(centers, -0.499, 0.499)
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    return best_centers, best_r

def _init_bcc_like(n=100):
    """
    Build a BCC-inspired initialization.
    """
    # 5x5x4 grid gives 100 points
    nx, ny, nz = 5, 5, 4
    margin = 0.02
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    centers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                centers.append([xs[i], ys[j], zs[k]])
    
    centers = np.array(centers[:n])
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _init_fcc_like(n=100):
    """
    Build an FCC-inspired initialization with body-centered additions.
    """
    # Start with a grid and add body centers
    nx, ny, nz = 4, 4, 4  # 64 corner points
    margin = 0.03
    xs = np.linspace(-0.5 + margin, 0.5 - margin, nx)
    ys = np.linspace(-0.5 + margin, 0.5 - margin, ny)
    zs = np.linspace(-0.5 + margin, 0.5 - margin, nz)
    
    centers = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                centers.append([xs[i], ys[j], zs[k]])
                # Add body center of each cell
                if i < nx-1 and j < ny-1 and k < nz-1:
                    centers.append([(xs[i]+xs[i+1])/2, (ys[j]+ys[j+1])/2, (zs[k]+zs[k+1])/2])
    
    centers = np.array(centers)
    rng = np.random.default_rng(123)
    
    if len(centers) >= n:
        idx = rng.choice(len(centers), n, replace=False)
        centers = centers[idx]
    
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _init_halton(n=100, seed=42):
    """
    Halton sequence initialization.
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
    
    centers = np.zeros((n, 3))
    for i in range(1, n+1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    
    rng = np.random.default_rng(seed)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, -0.499, 0.499)
    
    return centers

def _optimize_soft_min(centers, deadline, beta_start=50.0, beta_end=5000.0, n_beta_steps=5):
    """
    Run L-BFGS-B optimization with gradient and increasing beta.
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
                _soft_min_objective_and_grad,
                x0,
                args=(beta, iu),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            new_centers = res.x.reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            r = _compute_radius(new_centers)
            if r > best_radius:
                best_radius = r
                best_centers = new_centers.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 5.0:
            break
    
    return best_centers, best_radius

def _polish(centers, deadline):
    """
    Final high-precision polish pass.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    beta = 20000.0
    
    try:
        res = minimize(
            _soft_min_objective_and_grad,
            best_centers.flatten(),
            args=(beta, iu),
            jac=True,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
        )
        
        new_centers = res.x.reshape(-1, 3)
        new_centers = np.clip(new_centers, -0.5, 0.5)
        
        r = _compute_radius(new_centers)
        if r > best_radius:
            best_radius = r
            best_centers = new_centers.copy()
    except Exception:
        pass
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try multiple initializations
    inits = [
        ("bcc", _init_bcc_like()),
        ("fcc", _init_fcc_like()),
        ("halton", _init_halton()),
    ]
    
    best_centers = None
    best_r = 0.0
    
    for name, init_centers in inits:
        if time.time() > deadline - 30.0:
            break
        
        # Force-directed refinement first
        c_fd, r_fd = _force_directed_refinement(init_centers, deadline - 40.0, max_iters=100)
        if r_fd > best_r:
            best_r = r_fd
            best_centers = c_fd.copy()
        
        # Then soft-min optimization
        if time.time() > deadline - 20.0:
            break
        c_opt, r_opt = _optimize_soft_min(c_fd, deadline - 15.0)
        if r_opt > best_r:
            best_r = r_opt
            best_centers = c_opt.copy()
    
    # Final polish
    if best_centers is not None and time.time() < deadline - 5.0:
        c_final, r_final = _polish(best_centers, deadline)
        if r_final > best_r:
            best_r = r_final
            best_centers = c_final.copy()
    
    if best_centers is None:
        best_centers = _init_bcc_like()
    
    # Final validation
    final_r = _compute_radius(best_centers)
    if final_r > best_r:
        best_r = final_r
    
    return best_centers

# EVOLVE-BLOCK-END
