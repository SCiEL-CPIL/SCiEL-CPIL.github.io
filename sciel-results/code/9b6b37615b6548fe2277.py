import numpy as np
import time

def _score_polygon(polygon):
    n = len(polygon)
    if not np.all(np.isfinite(polygon)):
        return 0.0
    
    centroid = np.mean(polygon, axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    if mean_dist <= 0.01:
        return 0.0
    normalized = centered / mean_dist
    
    edges = np.roll(normalized, -1, axis=0) - normalized
    edge_lens = np.linalg.norm(edges, axis=1)
    if np.any(edge_lens < 1e-6):
        return 0.0
        
    cross_products = np.cross(edges, np.roll(edges, -1, axis=0))
    if np.any(np.abs(cross_products) < 1e-6):
        return 0.0
    if np.any(cross_products > 0) and np.any(cross_products < 0):
        return 0.0
        
    min_score = 1.0
    for i in range(n):
        v = normalized[i]
        diffs = normalized - v
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf
        sorted_dists = np.sort(dists)
        
        if sorted_dists[0] < 0.05:
            return 0.0
            
        gaps = np.diff(sorted_dists)
        k = np.argmin(gaps)
        
        D = (sorted_dists[k] + sorted_dists[k+1]) / 2.0
        if D < 0.1:
            return 0.0
            
        if k == 0:
            d_selected = sorted_dists[0:4]
        elif k == 17:
            d_selected = sorted_dists[15:19]
        else:
            d_selected = sorted_dists[k-1:k+3]
            
        ratios = np.maximum(d_selected / D, D / d_selected)
        raw = np.mean(ratios)
        
        score = min(1.0, 1.0 / raw)
        if score < min_score:
            min_score = score
            if min_score == 0.0:
                return 0.0
                
    return min_score

def equidistant_points_polygon20() -> np.ndarray:
    n = 20
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    
    best_polygon = None
    best_score = -1.0
    
    # Initial
    init_radii = np.ones(n)
    init_poly = np.stack([init_radii * np.cos(angles), init_radii * np.sin(angles)], axis=1)
    best_score = _score_polygon(init_poly)
    best_polygon = init_poly.copy()
    
    start_time = time.time()
    timeout = 590.0
    
    # Optimize radii
    current_radii = init_radii.copy()
    current_score = best_score
    
    # Use a simple random walk
    step_size = 0.1
    for i in range(200000):
        if time.time() - start_time > timeout:
            break
            
        noise = np.random.normal(0, step_size, n)
        new_radii = current_radii + noise
        new_radii = np.maximum(new_radii, 0.5) # Keep reasonably large
        
        new_polygon = np.stack([new_radii * np.cos(angles), new_radii * np.sin(angles)], axis=1)
        
        new_score = _score_polygon(new_polygon)
        
        if new_score > current_score:
            current_score = new_score
            current_radii = new_radii.copy()
            if current_score > best_score:
                best_score = current_score
                best_polygon = new_polygon.copy()
                if best_score > 0.99:
                    break
        else:
            # Occasionally restart from best to escape local minima
            if i % 1000 == 0:
                current_radii = best_polygon[:, 0] / np.cos(angles) # Approx recovery? No, better to store best radii
                # Actually, let's just keep the best radii found
                pass

    return best_polygon
