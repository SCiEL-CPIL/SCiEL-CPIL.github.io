# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _is_strictly_convex(points: np.ndarray) -> bool:
    """Check if the polygon defined by points is strictly convex."""
    n = len(points)
    if n < 3:
        return False
    
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                return False

    cross_products = np.zeros(n)
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross_products[i] = v1[0] * v2[1] - v1[1] * v2[0]
    
    if np.any(np.abs(cross_products) < 1e-12):
        return False
        
    if np.all(cross_products > 0) or np.all(cross_products < 0):
        return True
    else:
        return False


def _score_polygon(polygon: np.ndarray) -> float:
    """Compute the minimum vertex score for a given polygon."""
    n = len(polygon)
    if n != 20:
        return 0.0
        
    if not _is_strictly_convex(polygon):
        return 0.0

    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt((centered ** 2).sum(axis=1))
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    min_vertex_score = 1.0
    for i in range(n):
        diffs = normalized - normalized[i]
        d = np.sqrt((diffs ** 2).sum(axis=1))
        d = np.delete(d, i)
        d = np.sort(d)
        
        if d[0] < 0.05:
            return 0.0
        
        gaps = np.diff(d)
        k = np.argmin(gaps)
        D = (d[k] + d[k + 1]) / 2
        if D < 0.1:
            return 0.0
        
        if k == 0:
            selected = d[0:4]
        elif k == 17:
            selected = d[15:19]
        else:
            selected = d[k-1:k+3]
        
        ratios = np.maximum(selected / D, D / selected)
        raw = ratios.mean()
        vertex_score = min(1.0, 1.0 / raw)
        min_vertex_score = min(min_vertex_score, vertex_score)
    
    return min_vertex_score


def _neg_score_smooth(params: np.ndarray, beta: float = 50.0) -> float:
    """Negative smooth score for minimization with convexity penalty."""
    polygon = params.reshape(20, 2)
    
    if not np.all(np.isfinite(polygon)):
        return 1000.0
        
    n = 20
    cross_products = np.zeros(n)
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        p2 = polygon[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross_products[i] = v1[0] * v2[1] - v1[1] * v2[0]
    
    # Convexity penalty
    neg_cross = np.maximum(0, -cross_products)
    convexity_penalty = np.sum(neg_cross) * 50.0
    small_cross = np.sum(np.maximum(0, 1e-8 - np.abs(cross_products))) * 100.0
    
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.sqrt((centered ** 2).sum(axis=1))
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 1000.0 + convexity_penalty + small_cross
    normalized = centered / mean_dist
    
    vertex_scores = np.ones(n)
    for i in range(n):
        diffs = normalized - normalized[i]
        d = np.sqrt((diffs ** 2).sum(axis=1))
        d = np.delete(d, i)
        d = np.sort(d)
        
        if d[0] < 0.05:
            vertex_scores[i] *= np.exp(-beta * (0.05 - d[0]))
        
        gaps = np.diff(d)
        k = np.argmin(gaps)
        D = (d[k] + d[k + 1]) / 2
        
        if D < 0.1:
            vertex_scores[i] *= np.exp(-beta * (0.1 - D))
        
        if k == 0:
            selected = d[0:4]
        elif k == 17:
            selected = d[15:19]
        else:
            selected = d[k-1:k+3]
        
        a = selected / D
        b = D / selected
        m = np.maximum(a, b)
        soft_max = m + (1.0 / beta) * np.log(np.exp(beta * (a - m)) + np.exp(beta * (b - m)))
        raw = soft_max.mean()
        
        if raw > 1.0:
            vertex_scores[i] = min(1.0, 1.0 / (raw + 1e-8))
        else:
            vertex_scores[i] = 1.0
    
    shifted = -beta * vertex_scores
    max_shifted = np.max(shifted)
    soft_min = -(max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))) / beta
    
    return -soft_min + convexity_penalty + small_cross


def _generate_superellipse(a, b, n_exp):
    """Generate a superellipse with given parameters."""
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = np.zeros(n)
    y = np.zeros(n)
    for i, theta in enumerate(angles):
        cx = np.cos(theta)
        sx = np.sin(theta)
        if n_exp < 1.1:
            n_exp = 1.1
        x[i] = a * np.sign(cx) * (abs(cx) ** (2.0 / n_exp))
        y[i] = b * np.sign(sx) * (abs(sx) ** (2.0 / n_exp))
    return np.column_stack((x, y))


def _harmonic_perturbation(angles, harmonic, phase, amplitude):
    """Generate a polygon with harmonic perturbation on radii."""
    radii = 1.0 + amplitude * np.sin(harmonic * angles + phase)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)


def _coordinate_descent(points, max_iters=200, step_size=0.01, deadline=None):
    """
    Coordinate descent on individual coordinates, maintaining convexity.
    """
    current = points.copy()
    current_obj = _neg_score_smooth(current.flatten())
    n = len(current)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            for coord in range(2):
                for delta in [step_size, -step_size, step_size * 0.5, -step_size * 0.5]:
                    trial = current.copy()
                    trial[i, coord] += delta
                    if not _is_strictly_convex(trial):
                        continue
                    obj = _neg_score_smooth(trial.flatten())
                    if obj < current_obj - 1e-12:
                        current = trial
                        current_obj = obj
                        improved = True
                        break
        if not improved:
            # Try smaller steps
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return current


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n = 20 vertices.
    """
    start_time = time.time()
    deadline = start_time + 570
    
    n = 20
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    regular_polygon = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    best_score = _score_polygon(regular_polygon)
    best_polygon = regular_polygon
    
    # Optimization from regular polygon
    try:
        result = minimize(_neg_score_smooth, regular_polygon.flatten(), method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        if np.all(np.isfinite(result.x)):
            cand = result.x.reshape(20, 2)
            if _is_strictly_convex(cand):
                s = _score_polygon(cand)
                if s > best_score:
                    best_score = s
                    best_polygon = cand
    except Exception:
        pass
    
    # Superellipse optimization - try different exponent values
    for n_exp in [2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        if time.time() > deadline - 60:
            break
        
        init = _generate_superellipse(1.0, 1.0, n_exp)
        if not _is_strictly_convex(init):
            continue
        
        try:
            res = minimize(_neg_score_smooth, init.flatten(), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if np.all(np.isfinite(res.x)):
                cand = res.x.reshape(20, 2)
                if _is_strictly_convex(cand):
                    s = _score_polygon(cand)
                    if s > best_score:
                        best_score = s
                        best_polygon = cand
        except Exception:
            pass
    
    # Multiple restarts with different initial conditions
    np.random.seed(42)
    for trial in range(40):
        if time.time() > deadline - 80:
            break
        
        if trial % 4 == 0:
            noise = np.random.normal(0, 0.03, (20, 2))
            init = regular_polygon + noise
        elif trial % 4 == 1:
            r = 1.0 + np.random.normal(0, 0.05, 20)
            init = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
        elif trial % 4 == 2:
            ang_noise = np.random.normal(0, 0.02, 20)
            ang_noise -= np.mean(ang_noise)
            new_angles = angles + ang_noise
            new_angles = np.sort(new_angles)
            new_angles = (new_angles - new_angles[0]) % (2 * np.pi)
            new_angles = np.sort(new_angles)
            if new_angles[-1] - new_angles[0] < 2 * np.pi - 0.1:
                new_angles[-1] = new_angles[0] + 2 * np.pi - 0.01
            init = np.stack([np.cos(new_angles), np.sin(new_angles)], axis=1)
        else:
            # Harmonic perturbation initialization
            harmonic = np.random.choice([2, 3, 4, 5, 6])
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.02, 0.12)
            init = _harmonic_perturbation(angles, harmonic, phase, amplitude)
        
        if not _is_strictly_convex(init):
            continue
        
        try:
            res = minimize(_neg_score_smooth, init.flatten(), method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            if np.all(np.isfinite(res.x)):
                cand = res.x.reshape(20, 2)
                if _is_strictly_convex(cand):
                    # Coordinate descent refinement
                    if time.time() < deadline - 60:
                        cand_cd = _coordinate_descent(cand, max_iters=100, step_size=0.005,
                                                       deadline=deadline - 40)
                        if _is_strictly_convex(cand_cd):
                            s_cd = _score_polygon(cand_cd)
                            if s_cd > best_score:
                                best_score = s_cd
                                best_polygon = cand_cd
                    s = _score_polygon(cand)
                    if s > best_score:
                        best_score = s
                        best_polygon = cand
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 30:
        try:
            res_polish = minimize(_neg_score_smooth, best_polygon.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if np.all(np.isfinite(res_polish.x)):
                cand = res_polish.x.reshape(20, 2)
                if _is_strictly_convex(cand):
                    s = _score_polygon(cand)
                    if s > best_score:
                        best_score = s
                        best_polygon = cand
        except Exception:
            pass
    
    # Final coordinate descent polish
    if time.time() < deadline - 10:
        try:
            cand_cd = _coordinate_descent(best_polygon, max_iters=150, step_size=0.002,
                                           deadline=deadline - 5)
            if _is_strictly_convex(cand_cd):
                s_cd = _score_polygon(cand_cd)
                if s_cd > best_score:
                    best_score = s_cd
                    best_polygon = cand_cd
        except Exception:
            pass
    
    return best_polygon


# EVOLVE-BLOCK-END
