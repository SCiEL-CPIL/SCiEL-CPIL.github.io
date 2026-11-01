import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3

def _pairwise_dists_sq(pts):
    """Return matrix of squared pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff * diff, axis=-1)

def _score(pts):
    """Compute (max_dist / min_dist)^2. Returns inf if any points coincide."""
    d2 = _pairwise_dists_sq(pts)
    mask = ~np.eye(N, dtype=bool)
    d2_off = d2[mask]
    if d2_off.size == 0:
        return np.inf
    min_d2 = np.min(d2_off)
    max_d2 = np.max(d2_off)
    if min_d2 <= 1e-12:
        return np.inf
    ratio = np.sqrt(max_d2 / min_d2)
    return ratio * ratio

def _normalize(pts):
    """Center at origin and scale so RMS distance from origin is 1."""
    pts = pts - np.mean(pts, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if rms < 1e-9:
        return pts
    return pts / rms

def _get_24_cell_vertices():
    """Generate vertices of the 24-cell in 4D."""
    verts = []
    # 24 vertices: (±1, ±1, 0, 0) and permutations
    for i in range(4):
        for j in range(4):
            if i == j: continue
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    v = [0, 0, 0, 0]
                    v[i] = s1
                    v[j] = s2
                    verts.append(v)
    # Normalize to unit sphere for consistency
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _smooth_log_ratio_objective(x, tau, mode='direct', source_verts=None):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    mode: 'direct' (x is NxD), 'projection' (x is 3x4 matrix flattened)
    """
    if mode == 'direct':
        pts = x.reshape(N, D)
    else:
        P = x.reshape(3, 4)
        pts = source_verts @ P.T  # (14, 4) @ (4, 3) -> (14, 3)
        
    # Center and normalize to remove invariances
    pts = pts - np.mean(pts, axis=0, keepdims=True)
    sq_dists_mat = _pairwise_dists_sq(pts)
    mean_sq_dist = np.mean(sq_dists_mat[~np.eye(N, dtype=bool)])
    if mean_sq_dist < 1e-12:
        return 1e10
    pts = pts / np.sqrt(mean_sq_dist)
    
    # Recompute distances
    sq_dists_mat = _pairwise_dists_sq(pts)
    mask = ~np.eye(N, dtype=bool)
    sq_dists = sq_dists_mat[mask]
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    log_sq_dists = np.log(sq_dists)
    
    # Soft max of log(squared distances)
    m_max_log = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max_log))
    soft_max_log = m_max_log + np.log(np.sum(exp_max)) / tau
    
    # Soft min of log(squared distances)
    m_min_log = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min_log))
    soft_min_log = m_min_log - np.log(np.sum(exp_min)) / tau
    
    obj = soft_max_log - soft_min_log
    return obj

def _generate_icosahedron():
    """Generate 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ]
    verts = np.array(verts, dtype=float)
    verts = _normalize(verts)
    return verts

def construct():
    deadline = time.time() + 130  # Safety margin
    
    best_pts = None
    best_score = np.inf
    
    rng = np.random.default_rng(42)
    
    # 1. Prepare 24-cell source vertices for projection mode
    verts_24 = _get_24_cell_vertices()
    # Select 14 vertices. We'll pick the first 14, but also try a few random subsets
    # to see if a different subset yields a better projection.
    source_subsets = [
        verts_24[:14],
        verts_24[rng.choice(24, 14, replace=False)]
    ]
    
    # 2. Generate initial candidates for Direct Mode
    direct_inits = []
    
    # Icosahedron + 2 random
    ico = _generate_icosahedron()
    for _ in range(3):
        extra = rng.randn(2, D)
        extra = _normalize(extra)
        pts = np.vstack([ico, extra])
        direct_inits.append(pts.flatten())
        
    # Random spherical
    for _ in range(3):
        pts = rng.randn(N, D)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        direct_inits.append(pts.flatten())
        
    # Fibonacci sphere
    i = np.arange(N)
    phi_fib = np.pi * (3 - np.sqrt(5))
    z = 1 - 2 * i / (N - 1)
    r = np.sqrt(1 - z**2)
    theta = phi_fib * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts_fib = np.column_stack((x, y, z))
    direct_inits.append(pts_fib.flatten())

    # 3. Generate initial candidates for Projection Mode
    proj_inits = []
    for src in source_subsets:
        # Initial P: Identity-like, scaled
        P0 = np.eye(4)[:3, :] * 0.5
        P0 = P0 + rng.normal(scale=0.1, size=(3,4))
        proj_inits.append(P0.flatten())
        
        # Random P
        P_rand = rng.normal(size=(3,4)) * 0.5
        proj_inits.append(P_rand.flatten())

    # Tau schedule for annealing
    tau_schedule = [10.0, 50.0, 200.0, 1000.0]
    
    # Optimization Loop
    all_candidates = []
    
    # Add direct candidates
    for x0 in direct_inits:
        all_candidates.append((x0, 'direct', None))
        
    # Add projection candidates
    for x0 in proj_inits:
        # Assign a source subset. We'll just use the first one for all, 
        # or cycle through. Let's use the first source for all projection inits 
        # for simplicity, as the P matrix is the variable.
        all_candidates.append((x0, 'projection', source_subsets[0]))
        
    # Shuffle candidates
    rng.shuffle(all_candidates)
    
    for x0, mode, src in all_candidates:
        if time.time() > deadline - 15:
            break
            
        try:
            current_x = x0.copy()
            current_score = np.inf
            
            # Run L-BFGS-B with increasing tau
            for tau in tau_schedule:
                if time.time() > deadline - 5:
                    break
                    
                res = minimize(
                    _smooth_log_ratio_objective,
                    current_x,
                    method='L-BFGS-B',
                    args=(tau, mode, src),
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < current_score:
                    current_score = res.fun
                    current_x = res.x
            
            # Evaluate actual score
            if mode == 'direct':
                pts = current_x.reshape(N, D)
            else:
                P = current_x.reshape(3, 4)
                pts = src @ P.T
                
            pts = _normalize(pts)
            score = _score(pts)
            
            if score < best_score:
                best_score = score
                best_pts = pts
                
        except Exception:
            pass

    if best_pts is None:
        # Fallback
        best_pts = np.random.randn(N, D)
        best_pts = _normalize(best_pts)
    
    return best_pts
