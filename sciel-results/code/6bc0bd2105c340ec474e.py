"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(c):
    """
    Compute the actual score (largest common radius) for a configuration.
    c: (N, 3) array of centers.
    Returns the radius r.
    """
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    face_dists = 0.5 - np.abs(c)
    min_face = np.min(face_dists)
    if min_face < -1e-12:
        return 0.0
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_pair / 2.0, min_face)
    return r


def _soft_min_logsumexp(d, beta):
    """
    Smooth surrogate for min(d) using log-sum-exp.
    """
    d_min = np.min(d)
    terms = np.exp(-beta * (d - d_min))
    s = np.sum(terms)
    return d_min - np.log(s) / beta


def _objective_and_grad(c_flat, beta):
    """
    Vectorized objective and gradient for the soft-min formulation.
    c_flat: (3N,) array
    Returns (objective_value, gradient_array)
    """
    c = c_flat.reshape(N, 3)
    
    # Pairwise distances - vectorized
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1e-10)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0  # radius contribution from pairs
    
    # Face distances
    face_dists = 0.5 - np.abs(c)
    face_flat = face_dists.flatten()
    
    all_dists = np.concatenate([pair_dists, face_flat])
    
    # Soft min
    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    terms = np.exp(-beta * shifted)
    s = np.sum(terms)
    obj_val = d_min - np.log(s) / beta
    
    # Gradient of soft min w.r.t. all_dists
    weights = beta * terms / s
    
    # Vectorized pair gradient computation
    i_idx, j_idx = iu
    pair_weights = weights[:len(i_idx)]
    
    # For each pair (i,j), compute direction vectors
    diff_pairs = c[i_idx] - c[j_idx]  # shape (n_pairs, 3)
    dist_pairs = dists[i_idx, j_idx]  # shape (n_pairs,)
    
    # Weighted direction: w_k * 0.5 * diff_k / dist_k
    # Avoid division by zero
    safe_dist = np.where(dist_pairs > 1e-10, dist_pairs, 1.0)
    weighted_dirs = (pair_weights / (2.0 * safe_dist))[:, np.newaxis] * diff_pairs
    
    # Accumulate gradients using np.add.at
    grad = np.zeros((N, 3))
    np.add.at(grad, i_idx, weighted_dirs)
    np.add.at(grad, j_idx, -weighted_dirs)
    
    # Face contributions
    face_weights = weights[len(pair_dists):]
    face_grad = face_weights.reshape(N, 3) * (-np.sign(c))
    grad += face_grad
    
    # Objective is -soft_min, so gradient is -grad
    return -obj_val, -grad.flatten()


def _objective(c_flat, beta):
    """Objective only (for scipy minimize)."""
    val, _ = _objective_and_grad(c_flat, beta)
    return val


def _objective_grad(c_flat, beta):
    """Objective and gradient (for scipy minimize with jac)."""
    return _objective_and_grad(c_flat, beta)


def _generate_grid_init(m):
    """Generate a grid initialization with m points per axis."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= N:
        return grid[:N].copy()
    else:
        n_pad = N - len(grid)
        rng = np.random.default_rng(123)
        pad = rng.uniform(-0.4, 0.4, size=(n_pad, 3))
        return np.vstack([grid, pad])


def _generate_bcc_init():
    """Generate a BCC lattice initialization."""
    s = 0.35
    x = np.arange(-0.5, 0.5 + s/2, s)
    y = np.arange(-0.5, 0.5 + s/2, s)
    z = np.arange(-0.5, 0.5 + s/2, s)
    
    pts = []
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                pts.append([x[i], y[j], z[k]])
                cx, cy, cz = x[i] + s/2, y[j] + s/2, z[k] + s/2
                if -0.5 <= cx <= 0.5 and -0.5 <= cy <= 0.5 and -0.5 <= cz <= 0.5:
                    pts.append([cx, cy, cz])
    
    pts = np.array(pts)
    mask = np.all(np.abs(pts) <= 0.5, axis=1)
    pts = pts[mask]
    
    if len(pts) > N:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), N, replace=False)
        pts = pts[idx]
    elif len(pts) < N:
        rng = np.random.default_rng(43)
        n_pad = N - len(pts)
        pad = rng.uniform(-0.4, 0.4, size=(n_pad, 3))
        pts = np.vstack([pts, pad])
    
    return pts


def _local_search(c, beta, max_iter=200, tol=1e-10):
    """
    Perform local search using L-BFGS-B with gradient.
    """
    c_flat = c.flatten()
    try:
        res = minimize(
            _objective,
            c_flat,
            args=(beta,),
            method='L-BFGS-B',
            jac=lambda x, b: _objective_grad(x, b)[1],
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol}
        )
        c_opt = res.x.reshape(N, 3)
        c_opt = np.clip(c_opt, -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initializations
    initializations = []
    
    # Grid with different densities
    for m in [3, 4, 5]:
        initializations.append(_generate_grid_init(m))
    
    # BCC
    initializations.append(_generate_bcc_init())
    
    # Random
    initializations.append(rng.uniform(-0.4, 0.4, size=(N, 3)))
    
    # Perturbed grid
    grid4 = _generate_grid_init(4)
    initializations.append(grid4 + rng.normal(0, 0.03, size=(N, 3)))
    initializations.append(grid4 + rng.normal(0, 0.06, size=(N, 3)))
    
    # Best so far
    best_c = initializations[0].copy()
    best_score = _score(best_c)
    
    # Phase 1: Multi-start optimization with increasing beta
    # More beta stages and more iterations per stage enabled by vectorized gradient
    beta_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 10.0:
            break
        
        # Try more initializations
        for init_idx in range(min(len(initializations), 5)):
            if time.time() > deadline - 10.0:
                break
            
            if init_idx == 0:
                c_init = best_c.copy()
            else:
                c_init = initializations[init_idx].copy()
            
            c_init = np.clip(c_init, -0.49, 0.49)
            
            # Local search with more iterations
            c_opt = _local_search(c_init, beta, max_iter=300, tol=1e-10)
            score = _score(c_opt)
            
            if score > best_score:
                best_score = score
                best_c = c_opt.copy()
        
        # Try more perturbed restarts
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            c_pert = best_c + rng.normal(0, 0.02, size=(N, 3))
            c_pert = np.clip(c_pert, -0.49, 0.49)
            c_opt = _local_search(c_pert, beta, max_iter=200, tol=1e-10)
            score = _score(c_opt)
            if score > best_score:
                best_score = score
                best_c = c_opt.copy()
        
        # Early exit if we're doing well
        if best_score > 0.145:
            break
    
    # Phase 2: Final high-precision polish - more rounds
    if time.time() < deadline - 5.0:
        for beta_final in [5000.0, 20000.0, 50000.0]:
            if time.time() > deadline - 5.0:
                break
            c_opt = _local_search(best_c, beta_final, max_iter=500, tol=1e-12)
            score = _score(c_opt)
            if score > best_score:
                best_score = score
                best_c = c_opt.copy()
    
    # Ensure valid output
    best_c = np.clip(best_c, -0.49999, 0.49999)
    return best_c
# EVOLVE-BLOCK-END
