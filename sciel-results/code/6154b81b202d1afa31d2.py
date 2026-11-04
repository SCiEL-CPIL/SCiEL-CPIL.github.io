"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START

# Precompute all triangle index triplets for vectorized area computation
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T


def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                          (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas


def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return float(np.min(areas)) / hull_area


def _soft_min_objective(x, beta=100.0):
    """
    Soft-min of triangle areas / hull area, negative for minimization.
    x is a flat array of length 2*N.
    """
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-10:
        return 1e6
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    
    # Soft min using log-sum-exp trick for numerical stability
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    return -soft_min_area / hull_area


def _exact_objective(x):
    """Exact negative score for Nelder-Mead refinement."""
    pts = x.reshape(-1, 2)
    return -_score(pts)


def _generate_initial_candidates():
    """Generate diverse initial candidates for optimization."""
    candidates = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Concentric rings with various splits
    for n_outer in range(6, 11):
        n_inner = N - n_outer
        if n_inner >= 3:
            for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
                angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
                angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
                outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
                inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
                candidates.append(np.vstack([outer, inner]))
    
    # 3. Three-ring configurations
    for n1 in range(7, 10):
        for n2 in range(2, 6):
            n3 = N - n1 - n2
            if n3 >= 1:
                for r2 in [0.5, 0.6, 0.7]:
                    for r3 in [0.2, 0.3, 0.4]:
                        a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
                        a2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / max(n2, 1)
                        a3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + np.pi / max(n3, 1)
                        p1 = np.column_stack([np.cos(a1), np.sin(a1)])
                        p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
                        p3 = np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
                        parts = [p1]
                        if n2 > 0:
                            parts.append(p2)
                        if n3 > 0:
                            parts.append(p3)
                        candidates.append(np.vstack(parts))
    
    # 4. Halton sequence points
    def _halton_sequence(n, base):
        seq = np.zeros(n)
        f = 1.0
        for i in range(1, n + 1):
            f /= base
            seq[i - 1] = 0.0
            j = i
            while j > 0:
                seq[i - 1] += (j % base) * f
                j //= base
        return seq
    
    x = _halton_sequence(N, 2)
    y = _halton_sequence(N, 3)
    pts = np.column_stack([x, y])
    radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    radii[radii < 1e-10] = 1.0
    pts = pts / radii[:, np.newaxis]
    candidates.append(pts)
    
    # 5. Random points on/near circle
    for i in range(50):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.5, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # 6. Perturbed regular polygon
    for i in range(50):
        noise = rng.normal(0, 0.1, (N, 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        base = np.column_stack([np.cos(angles), np.sin(angles)])
        candidates.append(base + noise)
    
    # 7. Random points in unit disk
    for i in range(50):
        radii = np.sqrt(rng.uniform(0, 1, N))
        angles = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    return candidates


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    best_score = 0.0
    best_pts = None
    
    # Generate and evaluate initial candidates
    candidates = _generate_initial_candidates()
    
    scored_candidates = []
    for pts in candidates:
        if time.time() > deadline:
            break
        s = _score(pts)
        scored_candidates.append((s, pts))
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_pts)
    
    # Sort candidates by score
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Take top candidates for refinement
    top_candidates = [pts for _, pts in scored_candidates[:min(15, len(scored_candidates))]]
    
    # Phase 1: L-BFGS-B with progressive beta (Cartesian coordinates)
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1500.0]
    
    for beta in betas:
        if time.time() > deadline - 20:
            break
        
        num_to_refine = min(6, len(top_candidates))
        
        for init_pts in top_candidates[:num_to_refine]:
            if time.time() > deadline - 10:
                break
            
            try:
                res = minimize(
                    _soft_min_objective,
                    init_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 300}
                )
                
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                
                if res_score > best_score:
                    best_score = res_score
                    best_pts = res_pts.copy()
                    # Update top candidates
                    top_candidates = [best_pts.copy()] + [p for p in top_candidates if not np.allclose(p, best_pts)][:5]
                    
            except Exception:
                pass
    
    # Phase 2: Multiple restarts with small perturbations from best
    if time.time() < deadline - 15:
        rng = np.random.default_rng(123)
        for i in range(10):
            if time.time() > deadline - 5:
                break
            
            # Perturb the best solution
            noise = rng.normal(0, 0.02, (N, 2))
            perturbed = best_pts + noise
            
            for beta in [200.0, 500.0]:
                if time.time() > deadline - 3:
                    break
                
                try:
                    res = minimize(
                        _soft_min_objective,
                        perturbed.flatten(),
                        args=(beta,),
                        method='L-BFGS-B',
                        options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 200}
                    )
                    
                    res_pts = res.x.reshape(N, 2)
                    res_score = _score(res_pts)
                    
                    if res_score > best_score:
                        best_score = res_score
                        best_pts = res_pts.copy()
                        
                except Exception:
                    pass
    
    # Phase 3: Final Nelder-Mead polish with exact objective
    if time.time() < deadline - 3:
        try:
            res = minimize(
                _exact_objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    # Phase 4: One more L-BFGS-B pass with very high beta from the best
    if time.time() < deadline - 1:
        try:
            res = minimize(
                _soft_min_objective,
                best_pts.flatten(),
                args=(3000.0,),
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 400}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts.copy()
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
