# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def equidistant_points_polygon20():
    """
    Returns a (20, 2) array of finite vertex coordinates in cyclic order.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s limit

    n = 20
    # Base angles for a regular 20-gon
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    # Fourier series parameters for radial function r(theta)
    # r(theta) = R0 + sum_{k=1}^{K} (A_k * cos(k * m * theta) + B_k * sin(k * m * theta))
    # To maintain strict convexity and avoid self-intersection easily, 
    # we use a small number of harmonics.
    # m is the symmetry order. For n=20, we can use m=1, 2, 4, 5, 10, 20.
    # Let's try a general representation with a few harmonics.
    # To keep the search space small and effective, we'll optimize:
    # R0 (mean radius) is fixed to 1.0 (normalization handles scale, but keeping it near 1 helps).
    # Actually, normalization subtracts centroid and divides by mean distance.
    # So the absolute scale doesn't matter, only the shape.
    # We can fix R0 = 1.0 and optimize the shape coefficients.
    
    # Let's use harmonics 1, 2, 3, 4, 5.
    # Coefficients: c1, s1, c2, s2, c3, s3, c4, s4, c5, s5
    # Total 10 parameters.
    
    def get_points(coeffs):
        """
        coeffs: array of shape (10,)
        [c1, s1, c2, s2, c3, s3, c4, s4, c5, s5]
        """
        r = np.ones(n)
        # Harmonic 1
        r += coeffs[0] * np.cos(1 * base_angles) + coeffs[1] * np.sin(1 * base_angles)
        # Harmonic 2
        r += coeffs[2] * np.cos(2 * base_angles) + coeffs[3] * np.sin(2 * base_angles)
        # Harmonic 3
        r += coeffs[4] * np.cos(3 * base_angles) + coeffs[5] * np.sin(3 * base_angles)
        # Harmonic 4
        r += coeffs[6] * np.cos(4 * base_angles) + coeffs[7] * np.sin(4 * base_angles)
        # Harmonic 5
        r += coeffs[8] * np.cos(5 * base_angles) + coeffs[9] * np.sin(5 * base_angles)
        
        # Ensure r > 0 to avoid singularities or inward points
        r = np.maximum(r, 0.1)
        
        pts = np.column_stack([r * np.cos(base_angles), r * np.sin(base_angles)])
        return pts

    def normalize(pts):
        centroid = pts.mean(axis=0)
        p = pts - centroid
        d = np.linalg.norm(p, axis=1)
        mean_d = d.mean()
        if mean_d <= 0.01:
            return None
        p = p / mean_d
        return p

    def check_strictly_convex(p):
        n_pts = len(p)
        # No repeated points
        for i in range(n_pts):
            if np.linalg.norm(p[i] - p[(i + 1) % n_pts]) < 1e-8:
                return False
        
        # Cross products
        signs = []
        for i in range(n_pts):
            p0 = p[i]
            p1 = p[(i + 1) % n_pts]
            p2 = p[(i + 2) % n_pts]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) < 1e-8:
                return False
            signs.append(1 if cross > 0 else -1)
        
        if signs.count(signs[0]) != n_pts:
            return False
            
        # Total turning
        total_turn = 0.0
        for i in range(n_pts):
            p0 = p[i]
            p1 = p[(i + 1) % n_pts]
            p2 = p[(i + 2) % n_pts]
            v1 = p1 - p0
            v2 = p2 - p1
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_a = np.clip(cos_a, -1, 1)
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
            ang = np.arctan2(sin_a, cos_a)
            total_turn += ang
            
        expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
        if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
            return False
        return True

    def vertex_score(p, v_idx):
        n_pts = len(p)
        v = p[v_idx]
        d = np.array([np.linalg.norm(v - p[j]) for j in range(n_pts) if j != v_idx])
        d_sorted = np.sort(d)
        
        if np.any(d_sorted < 0.05):
            return 0.0
            
        gaps = d_sorted[1:] - d_sorted[:-1]
        k = int(np.argmin(gaps))
        D = (d_sorted[k] + d_sorted[k + 1]) / 2
        if D < 0.1:
            return 0.0
            
        if k == 0:
            selected = d_sorted[0:4]
        elif k == 17:
            selected = d_sorted[15:19]
        else:
            selected = d_sorted[k - 1:k + 3]
            
        if len(selected) < 4:
            return 0.0
            
        raw = np.mean([max(di / D, D / di) for di in selected])
        if raw < 1e-12:
            return 0.0
        return min(1.0, 1.0 / raw)

    def score_polygon(pts):
        """
        Returns the min vertex score. 0 if invalid.
        """
        p = normalize(pts)
        if p is None:
            return 0.0
        if not check_strictly_convex(p):
            return 0.0
            
        min_vs = 1.0
        for i in range(len(p)):
            vs = vertex_score(p, i)
            if vs < min_vs:
                min_vs = vs
                if min_vs < 0.01:
                    break
        return min_vs

    def smooth_score(pts, sharpness=50.0):
        """
        Smooth surrogate for the min vertex score.
        Uses a soft-min to make the objective differentiable.
        """
        p = normalize(pts)
        if p is None:
            return 0.0
        if not check_strictly_convex(p):
            return 0.0
            
        scores = np.array([vertex_score(p, i) for i in range(len(p))])
        # Soft min: -log(sum(exp(-s_i * sharpness))) / sharpness
        # To avoid overflow, shift scores
        s_min = np.min(scores)
        exp_scores = np.exp(-(scores - s_min) * sharpness)
        soft_min = s_min - np.log(np.sum(exp_scores)) / sharpness
        return soft_min

    # Optimization
    # We will use a simple Nelder-Mead implementation or scipy if available.
    # Since scipy is allowed, let's use scipy.optimize.minimize for robustness.
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False

    def objective(coeffs):
        pts = get_points(coeffs)
        # We want to maximize the score, so minimize -score
        return -smooth_score(pts, sharpness=100.0)

    best_score = -1.0
    best_coeffs = np.zeros(10)

    # Initial guesses
    # 1. Regular polygon (all coeffs 0)
    # 2. Slight perturbations
    initial_guesses = [
        np.zeros(10),
        np.array([0.05, 0, 0.05, 0, 0.05, 0, 0.05, 0, 0.05, 0]),
        np.array([0, 0.05, 0, 0.05, 0, 0.05, 0, 0.05, 0, 0.05]),
        np.array([0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1, 0]),
    ]

    if use_scipy:
        for guess in initial_guesses:
            if time.time() > deadline:
                break
            try:
                res = minimize(objective, guess, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.success or res.fun < objective(best_coeffs):
                    # Evaluate actual score to ensure validity
                    pts = get_points(res.x)
                    actual_s = score_polygon(pts)
                    if actual_s > best_score:
                        best_score = actual_s
                        best_coeffs = res.x.copy()
            except Exception:
                pass
    else:
        # Fallback: Simple random search + local refinement
        rng = np.random.default_rng(42)
        best_coeffs = np.zeros(10)
        best_score = score_polygon(get_points(best_coeffs))
        
        for _ in range(2000):
            if time.time() > deadline:
                break
            # Perturb
            c_new = best_coeffs + rng.normal(0, 0.05, 10)
            s_new = score_polygon(get_points(c_new))
            if s_new > best_score:
                best_score = s_new
                best_coeffs = c_new

    # Final check
    if best_score <= 0:
        # Fallback to regular polygon
        best_coeffs = np.zeros(10)

    return get_points(best_coeffs)

# EVOLVE-BLOCK-END
