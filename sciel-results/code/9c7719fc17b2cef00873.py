import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 145.0  # 5s safety margin before 150s kill

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
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    
    # We only need upper triangle for unique pairs
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    # x_k = boundary_dists[i]
    # x_k = pair_dists[j] / 2.0
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    # Gradient
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms
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
    
    grad_flat = grad_pts.flatten()
    return obj_val, grad_flat

def _hex_lattice_init():
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
    """Initialize with concentric rings, mimicking the 1998 packing structure."""
    # 1 center, 6 ring 1, 12 ring 2, ... 
    # Approximate radii for 50 circles
    # r ~ 0.125
    # Ring 0: 1 circle at 0
    # Ring 1: 6 circles at r
    # Ring 2: 12 circles at 2r?
    # Let's try a few configurations
    
    best_pts = None
    best_score = -1.0
    
    for r in [0.12, 0.125, 0.13]:
        pts = np.zeros((N, 2))
        idx = 0
        
        # Center
        if idx < N:
            pts[idx] = [0, 0]
            idx += 1
            
        # Ring 1
        if idx < N:
            n = 6
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            pts[idx:idx+n] = np.column_stack([r*np.cos(angles), r*np.sin(angles)])
            idx += n
            
        # Ring 2
        if idx < N:
            n = 12
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            # Radius approx 2*r
            pts[idx:idx+n] = np.column_stack([2*r*np.cos(angles), 2*r*np.sin(angles)])
            idx += n
            
        # Ring 3
        if idx < N:
            n = 18
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            # Radius approx 3*r
            pts[idx:idx+n] = np.column_stack([3*r*np.cos(angles), 3*r*np.sin(angles)])
            idx += n
            
        # Ring 4 (if needed)
        if idx < N:
            n = N - idx
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            # Radius approx 4*r
            pts[idx:idx+n] = np.column_stack([4*r*np.cos(angles), 4*r*np.sin(angles)])
            idx += n
            
        # Clip to unit circle
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis]
            
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts
            
    if best_pts is not None:
        return best_pts
        
    return _hex_lattice_init()

def _random_init():
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _local_optimize(pts, max_time=10.0):
    """Run L-BFGS with increasing beta."""
    current_pts = pts.copy()
    betas = [50, 200, 1000, 5000]
    
    for beta in betas:
        if time.time() > _DEADLINE - 5.0:
            break
            
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                current_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            new_pts = res.x.reshape(N, 2)
            
            # Project back to unit disk
            dists = np.linalg.norm(new_pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
                
            # Check validity
            if _score(new_pts) > 0:
                current_pts = new_pts
            else:
                break
        except Exception:
            break
            
    return current_pts

def _simulated_annealing(start_pts, time_budget=60.0):
    """
    Simulated Annealing to escape local minima.
    Periodically applies local optimization.
    """
    current_pts = start_pts.copy()
    best_pts = start_pts.copy()
    best_score = _score(best_pts)
    
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.99
    
    start_time = time.time()
    no_improve_count = 0
    polish_interval = 500  # Steps between local optimizations
    
    while time.time() - start_time < time_budget and T > T_min:
        if time.time() > _DEADLINE - 5.0:
            break
            
        # Generate move: perturb 1-3 random points
        n_perturb = np.random.randint(1, 4)
        indices = np.random.choice(N, n_perturb, replace=False)
        perturbations = np.random.normal(0, T, size=(n_perturb, 2))
        
        new_pts = current_pts.copy()
        new_pts[indices] += perturbations
        
        # Keep inside circle
        dists = np.linalg.norm(new_pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
            
        new_score = _score(new_pts)
        
        # Acceptance criterion
        if new_score > current_score or np.random.rand() < np.exp((new_score - current_score) / T):
            current_pts = new_pts
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_pts = current_pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
            
        # Cool down
        T *= cooling_rate
        
        # Periodic local optimization
        if no_improve_count % polish_interval == 0 and time.time() < _DEADLINE - 10.0:
            polished_pts = _local_optimize(current_pts, max_time=5.0)
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts.copy()
            current_pts = polished_pts
            current_score = _score(current_pts)
            no_improve_count = 0
            
    return best_pts

def construct():
    start_time = time.time()
    
    # 1. Generate diverse initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(3):
        inits.append(_random_init())
        
    # 2. Local optimization for each init
    best_pts = inits[0]
    best_score = -1.0
    
    for init_pts in inits:
        if time.time() > _DEADLINE - 30.0:
            break
        opt_pts = _local_optimize(init_pts, max_time=10.0)
        s = _score(opt_pts)
        if s > best_score:
            best_score = s
            best_pts = opt_pts.copy()
            
    # 3. Simulated Annealing on the best candidate
    if time.time() < _DEADLINE - 10.0:
        sa_time_budget = min(60.0, _DEADLINE - time.time() - 10.0)
        if sa_time_budget > 5.0:
            sa_pts = _simulated_annealing(best_pts, time_budget=sa_time_budget)
            sa_score = _score(sa_pts)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts.copy()
                
    # 4. Final Polish
    if time.time() < _DEADLINE - 2.0:
        final_pts = _local_optimize(best_pts, max_time=10.0)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts.copy()
            
    return best_pts

# EVOLVE-BLOCK-END
