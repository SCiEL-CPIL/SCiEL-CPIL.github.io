import numpy as np
from scipy.optimize import minimize
import time

N = 14
D = 3
TIME_LIMIT = 145.0  # Safety margin before 150s limit

def _pairwise_sq_dists(pts):
    """Compute squared pairwise distances."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sum(diff**2, axis=-1)

def _true_score(pts):
    """Calculate the actual score: (max_dist / min_dist)^2"""
    sqd = _pairwise_sq_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    d_sq = sqd[mask]
    mn = np.min(d_sq)
    mx = np.max(d_sq)
    if mn < 1e-20:
        return np.inf
    return mx / mn

def _objective_epigraph(x, tau=0.01):
    """
    Epigraph-style objective: minimize log(max_d^2) - log(min_d^2)
    using smooth approximations. x is flat array of size N*D.
    """
    pts = x.reshape(N, D)
    # Center
    pts = pts - np.mean(pts, axis=0)
    
    # Normalize
    norm_sq = np.sum(pts**2)
    if norm_sq < 1e-10:
        return 1e10
    pts = pts * np.sqrt(N / norm_sq)
    
    sqd = _pairwise_sq_dists(pts)
    mask = ~np.eye(N, dtype=bool)
    d_sq = sqd[mask]
    
    # Penalize too-close points
    min_d_sq = np.min(d_sq)
    if min_d_sq < 1e-10:
        return 1e10
    
    # Work in log space for stability
    log_d_sq = np.log(d_sq)
    
    # Soft max of log_d_sq
    max_log = np.max(log_d_sq)
    soft_max = max_log + tau * np.log(np.sum(np.exp((log_d_sq - max_log) / tau)))
    
    # Soft min of log_d_sq
    min_log = np.min(log_d_sq)
    soft_min = min_log - tau * np.log(np.sum(np.exp((min_log - log_d_sq) / tau)))
    
    return soft_max - soft_min

def _generate_stella_octangula():
    """
    Generate a stella octangula: two interpenetrating regular tetrahedra.
    This gives 8 points. We add 6 more points to reach 14.
    """
    # Regular tetrahedron vertices (on unit sphere)
    # Using the standard construction
    a = 1.0
    tet1 = np.array([
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a]
    ])
    tet2 = np.array([
        [-a, -a, -a],
        [-a, a, a],
        [a, -a, a],
        [a, a, -a]
    ])
    
    # Normalize to unit sphere
    tet1 = tet1 / np.linalg.norm(tet1, axis=1, keepdims=True)
    tet2 = tet2 / np.linalg.norm(tet2, axis=1, keepdims=True)
    
    stella = np.vstack([tet1, tet2])  # 8 points
    
    # Add 6 more points: vertices of an octahedron (slightly perturbed to avoid symmetry)
    # Octahedron vertices
    octa = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    octa = octa / np.linalg.norm(octa, axis=1, keepdims=True)
    
    # Slightly scale octahedron to interleave with stella
    octa_scaled = octa * 0.95
    
    pts = np.vstack([stella, octa_scaled])  # 14 points
    
    return pts

def _generate_icosahedron_plus():
    """Generate icosahedron (12 points) + 2 poles."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Add two points on z-axis, slightly off to break symmetry
    poles = np.array([[0, 0, 0.95], [0, 0, -0.95]])
    
    return np.vstack([verts, poles])

def _generate_fib_sphere(n):
    """Fibonacci sphere distribution of n points."""
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + np.sqrt(5)) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def _generate_initial_guesses():
    """Generate diverse initial configurations."""
    guesses = []
    rng = np.random.default_rng(42)
    
    # 1. Stella octangula based
    stella = _generate_stella_octangula()
    guesses.append(stella)
    
    # Perturbed stella
    for scale in [0.9, 0.95, 1.0, 1.05]:
        pts = stella.copy()
        pts[8:] *= scale  # Scale the octahedron part
        guesses.append(pts)
    
    # 2. Icosahedron + poles
    ico_plus = _generate_icosahedron_plus()
    guesses.append(ico_plus)
    
    # 3. Fibonacci sphere
    guesses.append(_generate_fib_sphere(N))
    
    # 4. Random perturbations of stella
    for _ in range(6):
        pts = stella + rng.normal(0, 0.08, size=stella.shape)
        guesses.append(pts)
    
    # 5. Random perturbations of icosahedron
    for _ in range(6):
        pts = ico_plus + rng.normal(0, 0.08, size=ico_plus.shape)
        guesses.append(pts)
    
    # 6. Pure random
    for _ in range(5):
        pts = rng.normal(0, 1, size=(N, D))
        guesses.append(pts)
    
    # 7. Two rings + poles (different configurations)
    for top_h, bot_h, r in [(0.6, -0.6, 0.75), (0.4, -0.4, 0.9), (0.8, -0.8, 0.55)]:
        n_per_ring = 6
        angles_top = 2 * np.pi * np.arange(n_per_ring) / n_per_ring
        angles_bot = 2 * np.pi * np.arange(n_per_ring) / n_per_ring + np.pi / n_per_ring
        top_pts = np.column_stack([r * np.cos(angles_top), r * np.sin(angles_top), np.full(n_per_ring, top_h)])
        bot_pts = np.column_stack([r * np.cos(angles_bot), r * np.sin(angles_bot), np.full(n_per_ring, bot_h)])
        poles = np.array([[0, 0, 1.1], [0, 0, -1.1]])
        guesses.append(np.vstack([top_pts, bot_pts, poles]))
    
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_score = np.inf
    best_pts = None
    
    guesses = _generate_initial_guesses()
    
    # Multi-stage optimization
    tau_stages = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    # First pass: coarse optimization from all starts
    candidates = []
    for i, guess in enumerate(guesses):
        if time.time() > deadline - 30:
            break
        
        x0 = guess.flatten()
        
        try:
            res = minimize(
                _objective_epigraph,
                x0,
                args=(tau_stages[0],),
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res.fun < 1e5:
                candidates.append(res.x)
        except Exception:
            continue
    
    # Keep top candidates by true score
    if candidates:
        cand_scores = []
        for cx in candidates:
            pts = cx.reshape(N, D)
            pts = pts - np.mean(pts, axis=0)
            norm_sq = np.sum(pts**2)
            if norm_sq > 1e-10:
                pts = pts * np.sqrt(N / norm_sq)
            score = _true_score(pts)
            if score < np.inf:
                cand_scores.append((score, cx))
        
        cand_scores.sort(key=lambda x: x[0])
        # Keep top 10 for fine-tuning
        top_candidates = [c[1] for c in cand_scores[:10]]
    else:
        top_candidates = [g.flatten() for g in guesses[:5]]
    
    # Second pass: fine-tune top candidates with decreasing tau
    for cx in top_candidates:
        if time.time() > deadline - 10:
            break
        
        x_curr = cx.copy()
        
        for tau in tau_stages[1:]:
            if time.time() > deadline - 5:
                break
            
            try:
                res = minimize(
                    _objective_epigraph,
                    x_curr,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 250, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # Only accept if improved
                if res.fun < _objective_epigraph(x_curr, tau) + 1e-8:
                    x_curr = res.x
            except Exception:
                continue
        
        # Evaluate final score
        pts = x_curr.reshape(N, D)
        pts = pts - np.mean(pts, axis=0)
        norm_sq = np.sum(pts**2)
        if norm_sq > 1e-10:
            pts = pts * np.sqrt(N / norm_sq)
        
        score = _true_score(pts)
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
    
    # If we still have time, try more random starts
    if time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        for _ in range(15):
            if time.time() > deadline - 10:
                break
            
            x0 = rng.normal(0, 1, size=(N, D)).flatten()
            
            try:
                # Quick coarse
                res1 = minimize(
                    _objective_epigraph, x0, args=(0.15,),
                    method='L-BFGS-B', options={'maxiter': 80, 'ftol': 1e-9}
                )
                if res1.fun > 1e5:
                    continue
                
                # Medium
                res2 = minimize(
                    _objective_epigraph, res1.x, args=(0.05,),
                    method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-11, 'gtol': 1e-7}
                )
                
                pts = res2.x.reshape(N, D)
                pts = pts - np.mean(pts, axis=0)
                norm_sq = np.sum(pts**2)
                if norm_sq > 1e-10:
                    pts = pts * np.sqrt(N / norm_sq)
                
                score = _true_score(pts)
                if score < best_score:
                    best_score = score
                    best_pts = pts.copy()
                    
            except Exception:
                continue
    
    # Final high-precision polish on the best candidate
    if best_pts is not None and time.time() < deadline - 5:
        x_best = best_pts.flatten()
        try:
            res_final = minimize(
                _objective_epigraph,
                x_best,
                args=(0.003,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res_final.fun < _objective_epigraph(x_best, 0.003) + 1e-10:
                pts_final = res_final.x.reshape(N, D)
                pts_final = pts_final - np.mean(pts_final, axis=0)
                norm_sq = np.sum(pts_final**2)
                if norm_sq > 1e-10:
                    pts_final = pts_final * np.sqrt(N / norm_sq)
                
                score_final = _true_score(pts_final)
                if score_final < best_score:
                    best_score = score_final
                    best_pts = pts_final
        except Exception:
            pass
    
    # If no valid points found, fallback
    if best_pts is None:
        best_pts = _generate_fib_sphere(N)
        
    return best_pts
