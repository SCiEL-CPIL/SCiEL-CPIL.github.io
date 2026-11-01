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

def _soft_min_obj_and_grad(x, beta, iu):
    """
    Objective and gradient for soft-min of constraints.
    x: flattened centers (N*3,)
    Returns (obj, grad)
    """
    centers = x.reshape(-1, 3)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)  # (N, 3)
    wall_vals = wall_dists.flatten()  # (3N,)
    
    # Pair distances / 2
    dists = cdist(centers, centers, metric='euclidean')
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists[iu] / 2.0  # (num_pairs,)
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    n = len(all_vals)
    
    # Soft-min: (1/beta) * log(sum(exp(-beta * x)))
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    exp_vals = np.exp(neg_beta_x - max_val)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0 or not np.isfinite(sum_exp):
        return 1e10, np.zeros_like(x)
    
    obj = (max_val + np.log(sum_exp)) / beta
    
    # Gradient: -sum(exp(-beta*x_i) * x_i) / sum(exp(-beta*x_i))
    # d(obj)/d(all_vals[k]) = -exp_vals[k] / sum_exp
    weights = -exp_vals / sum_exp
    
    # Build gradient for x
    grad = np.zeros_like(x)
    
    # Wall part: all_vals[0:3N] = 0.5 - |centers|
    # d(wall_val)/d(x) for x = centers[i,j]: -sign(centers[i,j])
    wall_weights = weights[:3*N].reshape(N, 3)
    grad -= wall_weights * np.sign(centers)
    
    # Pair part: all_vals[3N:] = dists[i,j]/2 for i<j
    # d(dists[i,j]/2)/d(centers[i]) = (centers[i] - centers[j]) / dists[i,j]
    # d(dists[i,j]/2)/d(centers[j]) = (centers[j] - centers[i]) / dists[i,j]
    pair_weights = weights[3*N:]
    ii, jj = iu
    for k in range(len(pair_weights)):
        i, j = ii[k], jj[k]
        d = dists[i, j]
        if d < 1e-15:
            continue
        w = pair_weights[k]
        diff = centers[i] - centers[j]
        grad[i] += w * diff / d
        grad[j] -= w * diff / d
    
    return obj, grad

def _init_bcc(n=100, noise_scale=0.003):
    """BCC-inspired initialization."""
    nx, ny, nz = 5, 5, 4
    margin = 0.015
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
    
    all_centers = np.array(all_centers)
    rng = np.random.default_rng(42)
    
    if len(all_centers) >= n:
        idx = rng.choice(len(all_centers), n, replace=False)
        centers = all_centers[idx].copy()
    else:
        centers = all_centers
        while len(centers) < n:
            centers = np.vstack([centers, centers[0]])
    
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.498, 0.498)
    return centers

def _init_halton(n=100, seed=42):
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
    
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 3))
    for i in range(1, n+1):
        centers[i-1, 0] = -0.5 + halton(i, 2)
        centers[i-1, 1] = -0.5 + halton(i, 3)
        centers[i-1, 2] = -0.5 + halton(i, 5)
    
    centers += rng.normal(0, 0.003, centers.shape)
    centers = np.clip(centers, -0.498, 0.498)
    return centers

def _init_fcc(n=100, noise_scale=0.003):
    """FCC-inspired initialization."""
    # FCC lattice points in a cube
    a = 1.0 / 4.0  # scale
    points = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                x = -0.5 + i * 0.25
                y = -0.5 + j * 0.25
                z = -0.5 + k * 0.25
                # Corner
                points.append([x, y, z])
                # Face centers
                if i < 4:
                    points.append([x + 0.125, y, z])
                if j < 4:
                    points.append([x, y + 0.125, z])
                if k < 4:
                    points.append([x, y, z + 0.125])
    
    points = np.array(points)
    rng = np.random.default_rng(123)
    
    if len(points) >= n:
        idx = rng.choice(len(points), n, replace=False)
        centers = points[idx].copy()
    else:
        centers = points
        while len(centers) < n:
            centers = np.vstack([centers, centers[0]])
    
    centers += rng.normal(0, noise_scale, centers.shape)
    centers = np.clip(centers, -0.498, 0.498)
    return centers

def _optimize_soft_min(centers, deadline, beta_start=20.0, beta_end=5000.0, n_beta_steps=5):
    """
    Run L-BFGS-B with analytic gradient, increasing beta schedule.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    betas = np.geomspace(beta_start, beta_end, n_beta_steps)
    
    for beta in betas:
        if time.time() > deadline - 2.0:
            break
        
        x0 = best_centers.flatten()
        
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                x0,
                args=(beta, iu),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30}
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

def _force_refine(centers, deadline, n_iterations=200):
    """
    Force-directed local refinement: push apart close pairs, pull toward center if near walls.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    for it in range(n_iterations):
        if time.time() > deadline - 1.0:
            break
        
        centers = best_centers.copy()
        dists = cdist(centers, centers, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        
        # Find close pairs
        min_dist = np.min(dists)
        if min_dist < 1e-10:
            continue
        
        # Target distance based on current radius
        r = _compute_radius(centers)
        target = 2.0 * r * 1.01
        
        # Apply forces
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                if d < target and d > 1e-15:
                    # Push apart
                    f = 0.01 * (target - d) / d
                    diff = centers[i] - centers[j]
                    forces[i] += f * diff
                    forces[j] -= f * diff
        
        # Wall forces: push inward if too close to wall
        for i in range(N):
            for k in range(3):
                if centers[i, k] > 0.5 - r:
                    forces[i, k] -= 0.01 * (centers[i, k] - (0.5 - r))
                elif centers[i, k] < -0.5 + r:
                    forces[i, k] += 0.01 * ((-0.5 + r) - centers[i, k])
        
        # Update
        learning_rate = 0.05 / (1 + it * 0.01)
        centers += learning_rate * forces
        centers = np.clip(centers, -0.499, 0.499)
        
        r_new = _compute_radius(centers)
        if r_new > best_radius:
            best_radius = r_new
            best_centers = centers.copy()
    
    return best_centers, best_radius

def _polish(centers, deadline):
    """
    Final high-precision polish with very high beta.
    """
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    iu = np.triu_indices(N, k=1)
    beta = 10000.0
    
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            best_centers.flatten(),
            args=(beta, iu),
            jac=True,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 20}
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
    
    iu = np.triu_indices(N, k=1)
    
    best_centers = None
    best_r = 0.0
    
    # Try multiple initializations
    inits = [
        _init_bcc(N),
        _init_halton(N),
        _init_fcc(N),
    ]
    
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        
        c_opt, r_opt = _optimize_soft_min(init, deadline)
        if r_opt > best_r:
            best_r = r_opt
            best_centers = c_opt.copy()
    
    # Force-directed refinement
    if time.time() < deadline - 15.0 and best_centers is not None:
        c_fr, r_fr = _force_refine(best_centers, deadline, n_iterations=150)
        if r_fr > best_r:
            best_r = r_fr
            best_centers = c_fr.copy()
    
    # Final polish
    if time.time() < deadline - 5.0 and best_centers is not None:
        c_final, r_final = _polish(best_centers, deadline)
        if r_final > best_r:
            best_r = r_final
            best_centers = c_final.copy()
    
    # Final validation
    if best_centers is None:
        best_centers = _init_bcc(N)
    
    final_r = _compute_radius(best_centers)
    if final_r > best_r:
        best_r = final_r
    
    return best_centers

# EVOLVE-BLOCK-END
