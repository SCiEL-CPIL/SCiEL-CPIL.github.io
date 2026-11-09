# EVOLVE-BLOCK-START
import numpy as np

def _vertex_score(distances: np.ndarray) -> float:
    """Compute the score for a single vertex given sorted distances to other 19 vertices."""
    d = np.sort(distances)
    if d[0] < 0.05:
        return 0.0
    gaps = np.diff(d)
    k = int(np.argmin(gaps))
    D = (d[k] + d[k + 1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d[0:4]
    elif k == 17:
        selected = d[15:19]
    else:
        selected = d[k - 1:k + 3]
    ratios = np.maximum(selected / D, D / selected)
    raw = np.mean(ratios)
    return min(1.0, 1.0 / raw)

def _min_vertex_score(points: np.ndarray) -> float:
    """Compute the minimum vertex score over all vertices."""
    n = len(points)
    scores = []
    for i in range(n):
        diffs = points - points[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists = np.delete(dists, i)  # exclude self
        scores.append(_vertex_score(dists))
    return min(scores)

def _evaluate_angles(angles: np.ndarray) -> float:
    """
    Evaluate the negative minimum vertex score for a given set of angles.
    Returns a large positive number (penalty) if the polygon is invalid.
    """
    n = 20
    # Sort angles to ensure cyclic order
    angles = np.sort(angles) % (2.0 * np.pi)
    
    # Check for very close angles (degenerate)
    gaps = np.diff(np.concatenate([angles, [angles[0] + 2.0 * np.pi]]))
    if np.min(gaps) < 0.005:
        return 1e6

    # Generate points on unit circle
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.stack([x, y], axis=1)
    
    # Normalize: subtract centroid, divide by mean distance to centroid
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist <= 0.01:
        return 1e6
        
    points = points / mean_dist
    
    # Check strict convexity
    # For points on a circle sorted by angle, they are always convex.
    # However, after normalization (which is a scaling and translation), 
    # convexity is preserved. 
    # Wait, the points are on a circle centered at origin initially.
    # Subtracting centroid moves the center.
    # Dividing by mean_dist scales.
    # The resulting points are still vertices of a convex polygon inscribed in a circle?
    # No, the centroid is not the center of the circle.
    # The points are on a circle C1. We subtract centroid G. 
    # The new points are on a circle C2 centered at -G.
    # Scaling keeps them on a circle.
    # Points on a circle in angular order are strictly convex.
    # So we don't need to check cross products if we assume they are on a circle.
    # BUT, the task says "Convex 20-gon". It doesn't say "cyclic".
    # My current approach restricts to cyclic polygons (vertices on a circle).
    # The regular polygon is cyclic.
    # Is it possible that a non-cyclic polygon scores better?
    # The current seed uses angles, implying it stays on a circle.
    # If I switch to full coordinates, I am changing the approach family significantly.
    # Let's stick to the "angles on a circle" family but use SA.
    
    score = _min_vertex_score(points)
    return -score

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon using Simulated Annealing on angular parameters.
    """
    n = 20
    base_angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    
    # Initial state: regular polygon with small noise
    np.random.seed(42)
    current_angles = base_angles + np.random.normal(0, 0.05, n)
    current_cost = _evaluate_angles(current_angles)
    
    best_angles = current_angles.copy()
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    T = 0.1
    T_min = 1e-4
    alpha = 0.99
    max_iter = 10000
    
    for i in range(max_iter):
        if T < T_min:
            break
            
        # Propose a move: perturb a few random angles
        num_perturb = np.random.randint(1, 4)
        indices = np.random.choice(n, num_perturb, replace=False)
        new_angles = current_angles.copy()
        new_angles[indices] += np.random.normal(0, T * 0.1, num_perturb)
        
        new_cost = _evaluate_angles(new_angles)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_angles = new_angles
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_angles = current_angles.copy()
                
        T *= alpha
        
    # Final evaluation and normalization
    angles = np.sort(best_angles) % (2.0 * np.pi)
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.stack([x, y], axis=1)
    
    centroid = np.mean(points, axis=0)
    points = points - centroid
    dists_to_centroid = np.linalg.norm(points, axis=1)
    mean_dist = np.mean(dists_to_centroid)
    
    if mean_dist > 0.01:
        points = points / mean_dist
    else:
        # Fallback to regular polygon if something went wrong
        base = np.linspace(0, 2*np.pi, n, endpoint=False)
        points = np.stack([np.cos(base), np.sin(base)], axis=1)
        centroid = np.mean(points, axis=0)
        points = points - centroid
        points = points / np.mean(np.linalg.norm(points, axis=1))
        
    return points

# EVOLVE-BLOCK-END
