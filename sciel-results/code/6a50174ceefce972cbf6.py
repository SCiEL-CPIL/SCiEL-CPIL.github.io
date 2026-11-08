import numpy as np
import time
from scipy.spatial import ConvexHull
from itertools import combinations
from scipy.optimize import minimize

N = 13

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            if np.std(pts) < 1e-9:
                return 0.0
            hull = ConvexHull(pts)
            area_hull = hull.volume
            if area_hull < 1e-9:
                return 0.0
            
            x = pts[:, 0]
            y = pts[:, 1]
            v1x = x[tri_j] - x[tri_i]
            v1y = y[tri_j] - y[tri_i]
            v2x = x[tri_k] - x[tri_i]
            v2y = y[tri_k] - y[tri_i]
            cross = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * cross
            min_area = np.min(areas)
            return min_area / area_hull
        except Exception:
            return 0.0

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), offset_angle=0.0):
        if n == 1:
            return np.array([center])
        angles = np.linspace(offset_angle, 2 * np.pi + offset_angle, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    # Generate diverse initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. 7 outer + 6 inner
    for r in [0.3, 0.5, 0.7, 0.9]:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, r, offset_angle=np.pi/6)
        initializations.append(np.vstack([outer, inner]))
        
    # 4. 6 outer + 1 center + 6 inner
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(6, 1.0)
        center = np.array([[0.0, 0.0]])
        inner = _regular_polygon(6, r, offset_angle=np.pi/6)
        initializations.append(np.vstack([outer, center, inner]))
        
    # 5. 5 outer + 8 inner
    for r in [0.4, 0.6, 0.8]:
        outer = _regular_polygon(5, 1.0)
        inner = _regular_polygon(8, r, offset_angle=np.pi/8)
        initializations.append(np.vstack([outer, inner]))
        
    # 6. 4 outer + 9 inner
    for r in [0.5, 0.7]:
        outer = _regular_polygon(4, 1.0)
        inner = _regular_polygon(9, r, offset_angle=np.pi/9)
        initializations.append(np.vstack([outer, inner]))
        
    # 7. 8 outer + 5 inner
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(8, 1.0)
        inner = _regular_polygon(5, r, offset_angle=np.pi/5)
        initializations.append(np.vstack([outer, inner]))
        
    # 8. 9 outer + 4 inner
    for r in [0.3, 0.5]:
        outer = _regular_polygon(9, 1.0)
        inner = _regular_polygon(4, r, offset_angle=np.pi/4)
        initializations.append(np.vstack([outer, inner]))

    # Add random initializations
    rng = np.random.RandomState(42)
    for _ in range(10):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0.5, 1.0, N)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        initializations.append(np.column_stack([xs, ys]))

    # Score all initializations
    scores = []
    for init in initializations:
        s = compute_score(init)
        scores.append(s)
        
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    
    best_score = -1
    best_points = None
    
    # Objective function for minimization
    def objective_flat(x_flat):
        pts = x_flat.reshape(N, 2)
        return -compute_score(pts)
        
    # Refine top candidates
    # We pick top 5 distinct configurations
    candidates_to_refine = []
    seen_configs = set()
    for idx in sorted_indices:
        if len(candidates_to_refine) >= 5:
            break
        key = tuple(np.round(initializations[idx], 2).flatten())
        if key not in seen_configs:
            seen_configs.add(key)
            candidates_to_refine.append(initializations[idx])
            
    if not candidates_to_refine:
        candidates_to_refine = [initializations[sorted_indices[0]]]

    for init in candidates_to_refine:
        # Check time budget
        if time.time() > start_time + time_limit - 10:
            break
            
        x0 = init.flatten()
        
        try:
            res = minimize(
                objective_flat, 
                x0, 
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-7,
                    'fatol': 1e-9,
                    'adaptive': True
                }
            )
            refined_pts = res.x.reshape(N, 2)
            score = compute_score(refined_pts)
            
            if score > best_score:
                best_score = score
                best_points = refined_pts.copy()
                
        except Exception:
            continue

    # If no refinement improved, use best initial
    if best_points is None or best_score < max(scores):
        best_idx = sorted_indices[0]
        best_points = initializations[best_idx].copy()
        best_score = scores[best_idx]
        
    # Final safety check
    if best_score < 1e-6:
         best_points = _regular_polygon(13, 1.0)
         best_score = compute_score(best_points)

    return best_points
