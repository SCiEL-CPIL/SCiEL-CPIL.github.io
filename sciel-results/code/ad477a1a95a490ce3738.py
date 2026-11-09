import numpy as np
import time
from scipy.optimize import differential_evolution
import sys

def calc_vertex_score(distances):
    """
    distances: array of 19 distances from vertex i to others.
    Returns score for this vertex.
    """
    if len(distances) < 19:
        return 0.0
    
    d = np.sort(distances)
    
    # Check if any < 0.05
    if np.any(d < 0.05):
        return 0.0
        
    # Find k: first index minimizing d[k+1] - d[k]
    diffs = d[1:] - d[:-1]
    # Note: d has length 19. diffs has length 18.
    # k is index in d, so diffs index j corresponds to gap between d[j] and d[j+1].
    # If diffs[j] is min, k=j.
    # The prompt says "k be the first index minimizing d_{k+1} - d_k".
    # So k is the index of the lower element.
    k = np.argmin(diffs)
    
    D = (d[k] + d[k+1]) / 2.0
    
    if D < 0.1:
        return 0.0
        
    # Select 4 distances
    if k == 0:
        selected = d[0:4]
    elif k == 17: # Index 17 means gap between d[17] and d[18]. 
                  # Prompt: "d_15..d_18 if k=17"
                  # Wait, if k=17, d_k=d[17], d_{k+1}=d[18].
                  # Selected: d[15], d[16], d[17], d[18].
        selected = d[15:19]
    else:
        # d_{k-1}..d_{k+2}
        # Indices: k-1, k, k+1, k+2
        selected = d[k-1:k+3]
        
    # Check length of selected
    if len(selected) != 4:
        # Should not happen if k in [0, 17] and logic is correct
        # k-1 >= 0 => k>=1. k+2 <= 18 => k<=16.
        # If k=0, handled. If k=17, handled.
        # If k in 1..16, k-1>=0, k+2<=18. OK.
        return 0.0

    # raw = mean of max(d/D, D/d)
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    
    score = min(1.0, 1.0 / raw)
    return score

def calc_polygon_score(points):
    """
    points: (20, 2) array.
    Returns minimum vertex score.
    """
    # Normalization
    centroid = np.mean(points, axis=0)
    points_norm = points - centroid
    
    # Mean distance to centroid
    dists_to_c = np.linalg.norm(points_norm, axis=1)
    mean_dist = np.mean(dists_to_c)
    
    if mean_dist <= 0.01:
        return 0.0
        
    points_norm = points_norm / mean_dist
    
    # Convexity Check
    n = len(points_norm)
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(points_norm[i] - points_norm[j]) < 1e-6:
                return 0.0
                
    # Check turns
    # Edge vectors
    edges = np.roll(points_norm, -1, axis=0) - points_norm
    # Cross products (z-component in 2D)
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    
    # All cross products must be same sign and non-zero
    if np.any(np.abs(cross) < 1e-8):
        return 0.0
        
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return 0.0
        
    # Calculate vertex scores
    min_score = 1.0
    for i in range(n):
        # Distances to other vertices
        other_indices = [j for j in range(n) if j != i]
        dists = np.linalg.norm(points_norm[other_indices] - points_norm[i], axis=1)
        s = calc_vertex_score(dists)
        if s < min_score:
            min_score = s
        if min_score == 0.0:
            return 0.0
            
    return min_score

def regular_polygon_points():
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1)

def radii_to_points(radii):
    n = len(radii)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def objective(radii):
    points = radii_to_points(radii)
    score = calc_polygon_score(points)
    return -score # Minimize negative score

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = 630.0 # Safety margin
    
    # Baseline
    best_points = regular_polygon_points()
    best_score = calc_polygon_score(best_points)
    
    # Optimization using Differential Evolution
    n = 20
    
    # Define bounds for radii: [0.5, 1.5]
    bounds = [(0.5, 1.5)] * n
    
    # Run Differential Evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-5,
            callback=lambda x, convergence: None,
            init='latinhypercube'
        )
        
        if result.fun < -best_score:
            r_opt = result.x
            points = radii_to_points(r_opt)
            score = calc_polygon_score(points)
            if score > best_score:
                best_score = score
                best_points = points
    except Exception as e:
        print(f"DE failed: {e}")
        
    # Check time and if we have time left, do local refinement
    if time.time() < deadline - 30.0:
        # Local refinement with Nelder-Mead starting from best found
        try:
            from scipy.optimize import minimize
            # Start from current best radii
            # We need to recover radii from points? No, we stored best_points but not radii.
            # Let's just try a few random restarts of NM on top of DE result if it was better.
            # Or just NM from regular polygon?
            # Let's try NM from the DE result if it exists, else from regular.
            
            # Actually, we didn't store the radii from DE in a variable accessible here if we didn't update best_points logic correctly above?
            # We did: if score > best_score: best_points = points. But we need radii for NM.
            # Let's store best_radii too.
            
            # Let's refactor slightly to keep track of best_radii.
            # But the function signature must return points.
            
            # Let's do a few NM restarts around the best found solution or random ones.
            # Since we don't have best_radii stored in the outer scope easily if we don't refactor,
            # let's just do a few NM runs from random starts or the regular polygon start.
            
            # Actually, let's just do a few NM runs from the regular polygon with small noise.
            for restart in range(10):
                if time.time() > deadline - 10.0:
                    break
                noise_scale = 0.1
                r0 = 1.0 + noise_scale * np.random.randn(n)
                try:
                    res = minimize(
                        objective, 
                        r0, 
                        method='Nelder-Mead', 
                        options={'xatol': 1e-4, 'fatol': 1e-5, 'maxiter': 200, 'maxfev': 500}
                    )
                    points = radii_to_points(res.x)
                    score = calc_polygon_score(points)
                    if score > best_score:
                        best_score = score
                        best_points = points
                except Exception:
                    continue
        except ImportError:
            pass

    return best_points

if __name__ == "__main__":
    # Test
    pts = equidistant_points_polygon20()
    print("Score:", calc_polygon_score(pts))
