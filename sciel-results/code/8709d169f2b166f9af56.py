import numpy as np
import time
import sys
from scipy.spatial import ConvexHull

def _triangle_areas(pts):
    """Compute areas of all triangles formed by triples of points."""
    n = len(pts)
    areas = np.zeros((n, n-1, n-2))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0
                areas[i, j-i-1, k-j-1] = area
    return areas

def _score(pts):
    """Compute A_min / A_hull score."""
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(pts)
        a_min = areas.min()
        return a_min / a_hull
    except Exception:
        return 0.0

def _subgradient(pts):
    """Compute subgradient of the minimum triangle area w.r.t. point coordinates."""
    n = len(pts)
    # Find all triangles with minimum area
    min_area = np.inf
    min_indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                area = abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0
                if area < min_area - 1e-12:
                    min_area = area
                    min_indices = [(i, j, k)]
                elif abs(area - min_area) < 1e-12:
                    min_indices.append((i, j, k))
    
    if not min_indices:
        return np.zeros_like(pts)
    
    # Average subgradient over all minimum triangles
    grad = np.zeros_like(pts)
    for (i, j, k) in min_indices:
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        x3, y3 = pts[k]
        
        # Signed area (without /2)
        signed = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)
        sign = np.sign(signed) if signed != 0 else 1.0
        
        # d(area)/d(xi) where area = |signed|/2
        # d(signed)/d(x1) = y2 - y3
        # d(signed)/d(y1) = x3 - x2
        # d(signed)/d(x2) = y3 - y1
        # d(signed)/d(y2) = x1 - x3
        # d(signed)/d(x3) = y1 - y2
        # d(signed)/d(y3) = x2 - x1
        
        grad[i, 0] += sign * (y2 - y3) / 2.0
        grad[i, 1] += sign * (x3 - x2) / 2.0
        grad[j, 0] += sign * (y3 - y1) / 2.0
        grad[j, 1] += sign * (x1 - x3) / 2.0
        grad[k, 0] += sign * (y1 - y2) / 2.0
        grad[k, 1] += sign * (x2 - x1) / 2.0
    
    grad /= len(min_indices)
    return grad

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    deadline = time.time() + 600.0
    
    # Start from regular n-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float64)
    
    best_pts = pts.copy()
    best_score = _score(pts)
    
    # Gradient ascent with adaptive step size
    lr = 0.01
    patience = 0
    max_patience = 2000
    
    for step in range(200000):
        if time.time() > deadline - 30.0:
            break
        if patience >= max_patience:
            # Try different initializations
            break
        
        grad = _subgradient(pts)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            patience += 1
            continue
        
        # Normalize gradient for stable updates
        grad_normed = grad / grad_norm
        
        # Try multiple step sizes
        improved = False
        for trial_lr in [lr, lr*0.5, lr*0.1, lr*2.0]:
            candidate = pts + trial_lr * grad_normed
            cand_score = _score(candidate)
            if cand_score > best_score + 1e-12:
                pts = candidate
                best_pts = pts.copy()
                best_score = cand_score
                lr = trial_lr
                patience = 0
                improved = True
                break
            elif cand_score > best_score - 1e-8:
                # Accept small regressions to escape local minima
                if np.random.random() < 0.1:
                    pts = candidate
                    lr = trial_lr
                    patience = 0
                    improved = True
                    break
        if not improved:
            patience += 1
            lr *= 0.95
            if lr < 1e-6:
                break
    
    # If we exhausted patience, try a few random restarts
    if patience >= max_patience:
        rng = np.random.default_rng(42)
        for restart in range(20):
            if time.time() > deadline - 60.0:
                break
            
            # Random initialization near the regular polygon
            noise_scale = 0.3
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            base = np.column_stack((np.cos(angles), np.sin(angles)))
            pts = base + rng.normal(0, noise_scale, size=(n, 2))
            
            curr_score = _score(pts)
            if curr_score > best_score:
                best_pts = pts.copy()
                best_score = curr_score
            
            # Short gradient ascent from this point
            lr = 0.01
            for step in range(5000):
                if time.time() > deadline - 60.0:
                    break
                grad = _subgradient(pts)
                grad_norm = np.linalg.norm(grad)
                if grad_norm < 1e-10:
                    break
                grad_normed = grad / grad_norm
                candidate = pts + lr * grad_normed
                cand_score = _score(candidate)
                if cand_score > best_score + 1e-12:
                    pts = candidate
                    best_pts = pts.copy()
                    best_score = cand_score
                else:
                    lr *= 0.9
                    if lr < 1e-6:
                        break
    
    return best_pts
