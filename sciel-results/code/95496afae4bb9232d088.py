"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy (structural change: adaptive arc-length reparameterization on a smooth convex boundary):
  1. Primary representation: 14 points placed on a smooth convex boundary defined by
     a radial function r(θ) = 1 + Σ a_k cos(kθ) + b_k sin(kθ).
     The angular positions are NOT fixed; instead, the points are positioned by
     optimizing their cumulative arc-length fractions along the boundary.
     This decouples shape deformation from point distribution, allowing the optimizer
     to cluster points in high-curvature regions and space them out in flatter regions.
  2. The boundary is parameterized by low-frequency Fourier coefficients (smooth, convex).
  3. The point positions are parameterized by 14 arc-length fractions (summing to 1).
  4. Multi-phase optimization:
     - Phase 1: Powell on Fourier coefficients (shape) with fixed equal arc-length fractions.
     - Phase 2: Powell on arc-length fractions (distribution) with fixed shape.
     - Phase 3: Nelder-Mead on combined parameters (shape + distribution).
     - Phase 4: Direct Cartesian refinement with Nelder-Mead.
     - Phase 5: Coordinate descent polish on Cartesian coordinates.
  5. Centrally symmetric initialization (7 antipodal pairs) for even N.
  6. Vectorized triangle-area computation.
  7. Time-guarded to finish well within 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

# Pre-compute all triangle index combinations
tri_indices = np.array(list(itertools.combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    p = points
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


# ---------- Fourier-based smooth convex boundary ----------
# r(θ) = 1 + Σ_{k=1}^{K} (a_k cos(kθ) + b_k sin(kθ))
# For convexity, we need r(θ) > 0 and the curve to be convex.
# Using low frequencies (K=2) ensures smoothness and helps maintain convexity.

K = 2  # Number of Fourier modes

def _fourier_to_radial(fourier_params, theta):
    """
    Convert Fourier parameters to radial function values at angles theta.
    fourier_params: array of shape (2*K,) = [a_1, b_1, a_2, b_2, ...]
    theta: array of angles
    Returns: array of radial values
    """
    r = np.ones_like(theta)
    for k in range(1, K + 1):
        a_k = fourier_params[2*(k-1)]
        b_k = fourier_params[2*(k-1) + 1]
        r += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)
    return r


def _fourier_to_points(fourier_params, arc_fractions):
    """
    Convert Fourier parameters and arc-length fractions to 2D points.
    
    fourier_params: shape (2*K,)
    arc_fractions: shape (N,) - cumulative arc-length fractions (sum to 1)
    
    The points are placed at angles determined by the arc-length fractions
    along the curve defined by r(θ).
    """
    # Fine sampling of the boundary for arc-length computation
    theta_fine = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    r_fine = _fourier_to_radial(fourier_params, theta_fine)
    
    # Ensure r is positive (convexity constraint)
    r_fine = np.maximum(r_fine, 0.1)
    
    x_fine = r_fine * np.cos(theta_fine)
    y_fine = r_fine * np.sin(theta_fine)
    
    # Compute cumulative arc length
    dx = np.diff(x_fine)
    dy = np.diff(y_fine)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    total_s = s[-1]
    
    # Map arc-length fractions to target arc lengths
    # arc_fractions should be cumulative fractions summing to 1
    s_targets = np.cumsum(arc_fractions) * total_s
    s_targets = s_targets % total_s  # Wrap around
    
    # Interpolate to get points at target arc lengths
    x_pts = np.interp(s_targets, s, x_fine)
    y_pts = np.interp(s_targets, s, y_fine)
    
    pts = np.column_stack([x_pts, y_pts])
    
    # Center the points
    pts -= pts.mean(axis=0)
    
    return pts


def _fourier_to_points_equal(fourier_params):
    """
    Convert Fourier parameters to 2D points with equal arc-length spacing.
    """
    arc_fractions = np.ones(N) / N
    return _fourier_to_points(fourier_params, arc_fractions)


def _decode_combined(params):
    """
    Decode combined parameters: [fourier_params (2*K), arc_fractions (N-1)]
    The last arc fraction is determined by the constraint that they sum to 1.
    """
    fourier_params = params[:2*K]
    # Arc fractions: use softmax-like normalization to ensure positivity and sum to 1
    arc_raw = params[2*K:]
    # Use exp to ensure positivity, then normalize
    arc_exp = np.exp(arc_raw - np.max(arc_raw))
    arc_fractions = arc_exp / np.sum(arc_exp)
    
    pts = _fourier_to_points(fourier_params, arc_fractions)
    return pts


def _objective_combined(params):
    pts = _decode_combined(params)
    if np.any(~np.isfinite(pts)):
        return 1e10
    return _neg_score(pts)


def _generate_initial_fourier_configs():
    """Generate diverse initial Fourier coefficient configurations."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Zero (circle)
    inits.append(np.zeros(2*K))
    
    # 2. Small elliptical perturbations
    for rx, ry in [(1.1, 0.9), (0.9, 1.1), (1.2, 0.8), (0.8, 1.2)]:
        params = np.zeros(2*K)
        params[0] = (rx - ry) / 2  # a_1
        params[1] = 0  # b_1
        inits.append(params.copy())
    
    # 3. Random small perturbations
    for seed in range(20):
        rs = np.random.default_rng(seed * 13 + 7)
        params = rs.normal(0, 0.1, size=2*K)
        inits.append(params)
    
    # 4. Larger perturbations
    for seed in range(10):
        rs = np.random.default_rng(seed * 31 + 17)
        params = rs.normal(0, 0.2, size=2*K)
        inits.append(params)
    
    return inits


def _generate_initial_arc_fractions():
    """Generate diverse initial arc-length fraction configurations."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Equal spacing
    equal = np.ones(N-1) / N
    inits.append(equal.copy())
    
    # 2. Perturbed equal spacing
    for seed in range(20):
        rs = np.random.default_rng(seed * 11 + 3)
        raw = np.ones(N-1) + rs.normal(0, 0.1, size=N-1)
        raw = np.maximum(raw, 0.1)
        frac = raw / np.sum(raw)
        inits.append(frac.copy())
    
    # 3. Clustering patterns
    for seed in range(10):
        rs = np.random.default_rng(seed * 23 + 5)
        raw = np.ones(N-1)
        # Add some clustering
        cluster_idx = rs.choice(N-1, size=3, replace=False)
        raw[cluster_idx] *= 0.5
        raw /= np.sum(raw)
        inits.append(raw.copy())
    
    return inits


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_score_val = 0.0
    
    def _update_best(pts):
        nonlocal best_pts, best_score_val
        if pts is None or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score_val:
            best_score_val = s
            best_pts = pts.copy()
    
    # Initial: regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    _update_best(init_pts)
    
    # ---------- Phase 1: Optimize Fourier coefficients with equal arc spacing ----------
    if time.time() < deadline - 100:
        for init_fourier in _generate_initial_fourier_configs()[:10]:
            if time.time() > deadline - 100:
                break
            
            def obj_fourier(fp):
                pts = _fourier_to_points_equal(fp)
                if np.any(~np.isfinite(pts)):
                    return 1e10
                return _neg_score(pts)
            
            try:
                res = minimize(obj_fourier, init_fourier, method='Powell',
                               options={'maxiter': 500, 'ftol': 1e-12, 'xtol': 1e-12})
                pts = _fourier_to_points_equal(res.x)
                _update_best(pts)
            except Exception:
                pass
    
    # ---------- Phase 2: Optimize arc-length fractions with best Fourier shape ----------
    if time.time() < deadline - 60:
        # Use the best Fourier parameters found so far
        # Re-run a quick optimization to find good Fourier params
        best_fourier = np.zeros(2*K)
        best_fourier_score = 0
        for init_fourier in _generate_initial_fourier_configs()[:5]:
            def obj_fourier(fp):
                pts = _fourier_to_points_equal(fp)
                if np.any(~np.isfinite(pts)):
                    return 1e10
                return _neg_score(pts)
            try:
                res = minimize(obj_fourier, init_fourier, method='Powell',
                               options={'maxiter': 200, 'ftol': 1e-10, 'xtol': 1e-10})
                s = -res.fun
                if s > best_fourier_score:
                    best_fourier_score = s
                    best_fourier = res.x
            except Exception:
                pass
        
        # Now optimize arc fractions
        for init_arc in _generate_initial_arc_fractions()[:5]:
            if time.time() > deadline - 60:
                break
            
            def obj_arc(arc_raw):
                # Convert raw to fractions
                arc_exp = np.exp(arc_raw - np.max(arc_raw))
                arc_fractions = arc_exp / np.sum(arc_exp)
                pts = _fourier_to_points(best_fourier, arc_fractions)
                if np.any(~np.isfinite(pts)):
                    return 1e10
                return _neg_score(pts)
            
            try:
                res = minimize(obj_arc, np.zeros(N-1), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
                arc_exp = np.exp(res.x - np.max(res.x))
                arc_fractions = arc_exp / np.sum(arc_exp)
                pts = _fourier_to_points(best_fourier, arc_fractions)
                _update_best(pts)
            except Exception:
                pass
    
    # ---------- Phase 3: Combined optimization (Fourier + arc fractions) ----------
    if time.time() < deadline - 30:
        # Use combined parameterization
        init_params = np.concatenate([best_fourier, np.zeros(N-1)])
        
        try:
            res = minimize(_objective_combined, init_params, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            pts = _decode_combined(res.x)
            _update_best(pts)
        except Exception:
            pass
        
        # Try Powell on combined
        if time.time() < deadline - 15:
            try:
                res = minimize(_objective_combined, init_params, method='Powell',
                               options={'maxiter': 1000, 'ftol': 1e-12, 'xtol': 1e-12})
                pts = _decode_combined(res.x)
                _update_best(pts)
            except Exception:
                pass
    
    # ---------- Phase 4: Direct Cartesian refinement ----------
    if time.time() < deadline - 10 and best_pts is not None:
        try:
            res = minimize(_neg_score, best_pts.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            pts = res.x.reshape(N, 2)
            _update_best(pts)
        except Exception:
            pass
    
    # ---------- Phase 5: Coordinate descent polish ----------
    if time.time() < deadline - 3 and best_pts is not None:
        pts = best_pts.copy()
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            improved = False
            for i in range(N):
                for j in range(2):
                    def obj_1d(val):
                        pts_new = pts.copy()
                        pts_new[i, j] = val
                        return _neg_score(pts_new)
                    
                    try:
                        res_cd = minimize(obj_1d, pts[i, j], method='Brent',
                                          bounds=(-5.0, 5.0), options={'xtol': 1e-12})
                        if res_cd.fun < obj_1d(pts[i, j]):
                            pts[i, j] = res_cd.x
                            improved = True
                    except Exception:
                        pass
            
            _update_best(pts)
            if not improved:
                break
    
    # Final: normalize to unit area
    if best_pts is not None:
        try:
            hull = ConvexHull(best_pts)
            area = hull.volume
            if area > 1e-10:
                scale = np.sqrt(1.0 / area)
                best_pts = best_pts * scale
        except Exception:
            pass
    
    return best_pts
