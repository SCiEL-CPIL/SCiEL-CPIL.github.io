"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_deg(pts):
    """Compute covering radius in degrees by checking hull facets."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 3):
        return 180.0
    if not np.all(np.isfinite(pts)):
        return 180.0
    
    # Check for coincident points
    min_dist = np.min([np.linalg.norm(pts[i] - pts[j]) for i in range(N) for j in range(i+1, N)])
    if min_dist < 1e-12:
        return 180.0
        
    # Check hemisphere condition: if all points in a closed hemisphere, bad
    # We verify that the origin is inside the convex hull.
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0
    
    # Check if origin is inside convex hull
    # A point is in the convex hull if it satisfies all facet inequalities.
    # For scipy ConvexHull, equations are n_i · x + d_i = 0.
    # The interior is where n_i · x + d_i <= 0.
    # For the origin (0,0,0), this means d_i <= 0 for all i.
    # If any d_i > 0 (with some tolerance), the origin is outside.
    if np.any(hull.equations[:, -1] > 1e-10):
        return 180.0

    max_ang = 0.0
    for simplex in hull.simplices:
        v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        # Compute circumcenter of triangle on unit sphere
        # The circumcenter is the point on the unit sphere equidistant from v0, v1, v2.
        # It's in the direction of the cross product of (v1-v0) and (v2-v0) or similar.
        # Actually, for points on the unit sphere, the circumcenter of the spherical triangle 
        # is in the direction of the vector n = v0 × v1 + v1 × v2 + v2 × v0 (normalized)
        n = np.cross(v0, v1) + np.cross(v1, v2) + np.cross(v2, v0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            # Degenerate, use centroid
            c = (v0 + v1 + v2) / 3.0
            c = c / np.linalg.norm(c)
        else:
            c = n / n_norm
        
        # Check if c is inside the spherical triangle
        # A point c is inside the spherical triangle v0,v1,v2 if it's in the same hemisphere as each edge
        # Check: for edge v0-v1, c should be on the same side as v2.
        # Normal to plane through v0,v1,origin: n01 = v0 × v1
        # v2 should satisfy sign(v2·n01), and c should have same sign.
        n01 = np.cross(v0, v1)
        n12 = np.cross(v1, v2)
        n20 = np.cross(v2, v0)
        
        s1 = np.dot(v2, n01)
        s2 = np.dot(v0, n12)
        s3 = np.dot(v1, n20)
        
        t1 = np.dot(c, n01)
        t2 = np.dot(c, n12)
        t3 = np.dot(c, n20)
        
        inside = (s1 * t1 >= -1e-12) and (s2 * t2 >= -1e-12) and (s3 * t3 >= -1e-12)
        
        if inside:
            # Covering radius for this facet is angular distance from c to v0
            cos_ang = np.clip(np.dot(c, v0), -1.0, 1.0)
            ang = np.arccos(cos_ang) * 180.0 / np.pi
        else:
            # Circumcenter is outside, so the farthest point is one of the edge midpoints
            # Check each edge midpoint
            mid01 = (v0 + v1) / np.linalg.norm(v0 + v1)
            mid12 = (v1 + v2) / np.linalg.norm(v1 + v2)
            mid20 = (v2 + v0) / np.linalg.norm(v2 + v0)
            
            ang01 = np.arccos(np.clip(np.dot(mid01, v0), -1.0, 1.0)) * 180.0 / np.pi
            ang12 = np.arccos(np.clip(np.dot(mid12, v1), -1.0, 1.0)) * 180.0 / np.pi
            ang20 = np.arccos(np.clip(np.dot(mid20, v2), -1.0, 1.0)) * 180.0 / np.pi
            
            ang = max(ang01, ang12, ang20)
        
        max_ang = max(max_ang, ang)
    
    return max_ang

def _spectral_energy(x):
    """
    Compute spectral energy: variance of off-diagonal entries of Gram matrix.
    This is a smooth surrogate for uniformity.
    """
    pts = _normalize(x.reshape(N, 3))
    G = pts @ pts.T
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    off_diag = G[mask]
    # Variance
    return np.var(off_diag)

def _spectral_grad(x):
    """
    Compute gradient of spectral energy.
    E = Var(G_off) = E[G_off^2] - (E[G_off])^2
    dE/dx = 2 * (E[G_off^2] - E[G_off]) * d(G_off)/dx - 2 * E[G_off] * d(G_off)/dx
    Actually, let's use a simpler gradient estimation or derive it properly.
    
    Let S = sum_{i!=j} G_ij^2, M = sum_{i!=j} G_ij
    E = S / M_pairs - (M / M_pairs)^2
    dE/dx = (2 * sum_{i!=j} G_ij * dG_ij/dx) / M_pairs - 2 * (M / M_pairs^2) * sum_{i!=j} dG_ij/dx
    
    dG_ij/dx_k = d(p_i^T p_j)/dx_k
    If k is not i or j, dG_ij/dx_k = 0 (since p_i and p_j are fixed relative to x_k in the inner product, 
    wait, x_k is the coordinate of point k. 
    G_ij = p_i^T p_j. 
    d(p_i^T p_j)/dx_k = 0 if k != i and k != j.
    If k == i, d(p_i^T p_j)/dx_i = p_j.
    If k == j, d(p_i^T p_j)/dx_j = p_i.
    
    So dG_ij/dx_k is non-zero only if k is i or j.
    
    Let's compute the gradient vector g of size 3N.
    g[k] = dE/dx_k
    
    For each pair (i, j) with i != j:
    term_S = 2 * G_ij * dG_ij/dx
    term_M = dG_ij/dx
    
    dE/dx = (1/M_pairs) * sum_{i!=j} term_S - 2 * (M/M_pairs^2) * sum_{i!=j} term_M
    
    This is O(N^2) to compute directly.
    """
    pts = _normalize(x.reshape(N, 3))
    G = pts @ pts.T
    mask = ~np.eye(N, dtype=bool)
    off_diag = G[mask]
    M_pairs = N * (N - 1)
    S = np.sum(off_diag ** 2)
    M = np.sum(off_diag)
    
    # Gradient of S = sum_{i!=j} G_ij^2
    # dS/dx_k = 2 * sum_{j!=k} G_kj * p_j + 2 * sum_{i!=k} G_ik * p_i
    # Note G_kj = G_jk
    # sum_{j!=k} G_kj * p_j = (G @ pts)[k] - G_kk * pts[k] = (G @ pts)[k] - pts[k]
    # So dS/dx_k = 2 * 2 * ((G @ pts)[k] - pts[k]) = 4 * ((G @ pts)[k] - pts[k])
    
    Gp = G @ pts
    dS_dx = 4.0 * (Gp - pts)
    
    # Gradient of M = sum_{i!=j} G_ij
    # dM/dx_k = sum_{j!=k} p_j + sum_{i!=k} p_i
    # sum_{j!=k} p_j = sum_all_p - pts[k]
    # So dM/dx_k = 2 * (sum_all_p - pts[k])
    sum_all_p = np.sum(pts, axis=0)
    dM_dx = 2.0 * (sum_all_p[None, :] - pts)
    
    # E = S/M_pairs - (M/M_pairs)^2
    # dE/dx = (1/M_pairs) * dS/dx - 2 * (M/M_pairs^2) * dM/dx
    coeff_S = 1.0 / M_pairs
    coeff_M = -2.0 * M / (M_pairs ** 2)
    
    grad = coeff_S * dS_dx + coeff_M * dM_dx
    
    # Reshape to (3N,)
    return grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pts = None
    best_score = float('inf')
    
    # 1. Generate initial candidates
    candidates = []
    
    # Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    candidates.append(fib_pts.flatten())
    
    # Random points
    np.random.seed(42)
    for seed in range(10):
        rng = np.random.RandomState(seed)
        pts = rng.randn(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
    
    # 2. Optimize each candidate using spectral energy
    for idx, x0 in enumerate(candidates):
        if time.time() > deadline - 10.0:
            break
        
        try:
            res = minimize(
                _spectral_energy,
                x0,
                jac=_spectral_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            opt_pts = _normalize(res.x.reshape(N, 3))
            score = _covering_radius_deg(opt_pts)
            
            if score < best_score:
                best_score = score
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    if best_pts is None:
        # Fallback to first candidate
        best_pts = _normalize(candidates[0].reshape(N, 3))
        best_score = _covering_radius_deg(best_pts)
    
    # 3. Fine-tuning with small random perturbations and spectral re-optimization
    if time.time() < deadline - 20.0:
        current_pts = best_pts.copy()
        current_score = best_score
        step_size = 0.05
        no_improve_count = 0
        max_no_improve = 20
        
        while time.time() < deadline - 10.0 and no_improve_count < max_no_improve:
            rng = np.random.RandomState()
            noise = rng.randn(N, 3) * step_size
            new_pts = _normalize(current_pts + noise)
            x0 = new_pts.flatten()
            
            try:
                res = minimize(
                    _spectral_energy,
                    x0,
                    jac=_spectral_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                opt_pts = _normalize(res.x.reshape(N, 3))
                new_score = _covering_radius_deg(opt_pts)
                
                if new_score < current_score - 1e-6:
                    current_pts = opt_pts
                    current_score = new_score
                    no_improve_count = 0
                    if new_score < best_score:
                        best_score = new_score
                        best_pts = opt_pts.copy()
                    step_size *= 0.9
                else:
                    no_improve_count += 1
                    if no_improve_count % 5 == 0:
                        step_size *= 1.2
            except Exception:
                continue
    
    return best_pts
# EVOLVE-BLOCK-END
