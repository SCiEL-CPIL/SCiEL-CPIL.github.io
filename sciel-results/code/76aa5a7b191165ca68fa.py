import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill

def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Check if inside unit circle
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0 + 1e-9):
        return 0.0
    
    # Calculate min distance to boundary
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    # Calculate min pairwise distance
    if N < 2:
        return r_cont
        
    # Efficient pairwise distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)

def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise and boundary).
    Minimizing this is equivalent to maximizing the soft-min of x_k.
    """
    pts = flat.reshape(N, 2)
    
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff * diff, axis=-1))
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    return obj_val, grad_pts.flatten()

def _force_directed_relax(pts, n_iter=200, dt=0.002, r_target=0.12):
    """Force-directed relaxation to improve packing density."""
    pts = pts.copy()
    for _ in range(n_iter):
        if time.time() > _DEADLINE - 15.0:
            break
        forces = np.zeros_like(pts)
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, 1.0)
        # Repel if closer than 2*r_target
        mask = dists < 2 * r_target
        force_mag = np.where(mask, (2 * r_target - dists) / (2 * r_target), 0.0)
        force_mat = force_mag[:, :, np.newaxis] * diff / dists[:, :, np.newaxis]
        forces += np.sum(force_mat, axis=1)
        # Attraction to center (weak) to keep them in
        forces -= 0.01 * pts
        # Update
        pts += dt * forces
        # Project back to unit disk
        dists_origin = np.linalg.norm(pts, axis=1)
        mask_out = dists_origin > 1.0
        if np.any(mask_out):
            pts[mask_out] /= dists_origin[mask_out, np.newaxis]
    return pts

def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = -1.0
    
    for spacing in [0.24, 0.25, 0.26, 0.27, 0.28]:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        
        mask = radius <= 1.0
        points = points[mask]
        radius = radius[mask]
        
        if len(points) >= N:
            idx = np.argsort(radius, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    
    if best_pts is not None:
        return best_pts
        
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _ring_init():
    """Initialize with concentric rings."""
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Ring 1: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        r = 0.25
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    # Ring 2: 12 points
    for i in range(12):
        angle = i * np.pi / 6 + np.pi / 12
        r = 0.5
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    # Ring 3: 18 points
    for i in range(18):
        angle = i * np.pi / 9
        r = 0.75
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    # Ring 4: remaining points
    remaining = N - len(pts)
    for i in range(remaining):
        angle = i * 2 * np.pi / remaining + np.pi / remaining
        r = 0.9
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    
    return np.array(pts[:N])

def _random_init():
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _perturb(pts, scale=0.02):
    """Add small random perturbation."""
    return pts + scale * np.random.randn(N, 2)

def construct():
    start_time = time.time()
    
    # Initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(4):
        inits.append(_random_init())
        
    best_pts = inits[0]
    best_score = _score(best_pts)
    
    # Multi-stage beta schedule with tighter tolerances
    betas = [10, 50, 200, 1000, 5000, 20000]
    
    for init_pts in inits:
        if time.time() > _DEADLINE - 25.0:
            break
            
        # Force-directed relaxation first
        current_pts = _force_directed_relax(init_pts, n_iter=150, dt=0.003, r_target=0.12)
        cur_score = _score(current_pts)
        if cur_score > best_score:
            best_score = cur_score
            best_pts = current_pts.copy()
        
        for beta in betas:
            if time.time() > _DEADLINE - 15.0:
                break
                
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    current_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
                )
                
                new_pts = res.x.reshape(N, 2)
                dists = np.linalg.norm(new_pts, axis=1)
                mask = dists > 1.0
                if np.any(mask):
                    new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
                    
                new_score = _score(new_pts)
                
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    current_pts = new_pts.copy()
                    
            except Exception:
                pass
        
    # High-precision polish passes with very tight tolerances
    polish_betas = [50000, 200000]
    for beta in polish_betas:
        if time.time() > _DEADLINE - 8.0:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 800, 'ftol': 1e-16, 'gtol': 1e-14, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            dists = np.linalg.norm(new_pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
            
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
            
    # Final ultra-high precision polish
    if time.time() < _DEADLINE - 3.0:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_pts.flatten(),
                args=(500000,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-15, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            dists = np.linalg.norm(new_pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
            
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
            
    return best_pts

# EVOLVE-BLOCK-END
