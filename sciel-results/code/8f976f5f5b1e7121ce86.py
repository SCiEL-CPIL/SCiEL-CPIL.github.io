import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def _score(points):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if invalid.
    """
    n = len(points)
    if n != 13:
        return 0.0
    
    # Check for NaN/Inf
    if not np.all(np.isfinite(points)):
        return 0.0

    # Calculate min triangle area
    min_area = float('inf')
    
    for i in range(13):
        for j in range(i + 1, 13):
            for k in range(j + 1, 13):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
                if area < min_area:
                    min_area = area

    # Calculate Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area <= 1e-10:
        return 0.0

    return min_area / hull_area

def heilbronn_convex13() -> np.ndarray:
    """
    Uses Simulated Annealing in full Cartesian space to maximize the ratio 
    of the smallest triangle area to the convex hull area.
    
    This approach avoids the local minima traps of local optimizers like Nelder-Mead
    by allowing uphill moves (accepting worse solutions with a probability based on temperature)
    and exploring the full 26D space directly, which can find "clustered" configurations
    that are topologically distinct from regular polygons.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    n_points = 13
    best_score = 0.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Helper to generate random points in a unit square or circle
    def random_points():
        # Mix of uniform square and uniform circle to explore different hull shapes
        if rng.random() > 0.5:
            pts = rng.uniform(-1, 1, (n_points, 2))
        else:
            angles = rng.uniform(0, 2 * np.pi, n_points)
            radii = rng.uniform(0.5, 1.0, n_points)
            pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        return pts

    # Initial best guess: Regular 13-gon
    theta0 = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    init_points = np.column_stack((np.cos(theta0), np.sin(theta0)))
    best_points = init_points.copy()
    best_score = _score(best_points)
    
    # Simulated Annealing Parameters
    T_start = 0.1
    T_end = 0.0001
    alpha = 0.9995 # Cooling rate
    
    # We will run a single long SA chain or multiple shorter ones?
    # Given 600s, a single long chain with periodic restarts or just a long chain is fine.
    # Let's do a long chain with "jumps" if stuck.
    
    current_points = best_points.copy()
    current_score = best_score
    
    T = T_start
    step_size = 0.05
    
    # Number of iterations
    # Each score calculation is ~0.1ms (286 triangles).
    # 580s / 0.0001s = 5,800,000 iterations possible.
    # Let's aim for ~500,000 iterations to be safe with overhead.
    max_iters = 500000
    
    for i in range(max_iters):
        if time.time() > deadline - 10:
            break
            
        # Propose a new point
        # Gaussian perturbation
        noise = rng.normal(0, step_size, (n_points, 2))
        new_points = current_points + noise
        
        # Optional: Soft constraint to keep points roughly in a bounded region
        # to prevent hull area from exploding or shrinking to 0.
        # If points get too far, hull area grows, min area might grow slower.
        # If points cluster too much, min area shrinks.
        # The objective function naturally handles this, but we can add a mild penalty
        # for points going outside [-2, 2] to keep the search focused.
        # Actually, scaling is invariant? No, A_min scales with L^2, A_hull scales with L^2.
        # So the ratio is scale-invariant. We can fix the scale by normalizing.
        # Let's normalize the points to have unit area hull? 
        # No, that's expensive. 
        # Instead, let's just let it float. The ratio is scale invariant, 
        # so the optimizer might drift in scale, but that doesn't affect the score.
        # However, numerical precision might suffer if coordinates get huge.
        # Let's renormalize periodically or just clip.
        # Clipping to [-10, 10] is safe.
        new_points = np.clip(new_points, -10, 10)
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                # Maybe increase step size slightly on improvement? No, standard SA keeps it or cools.
        else:
            # Accept with probability exp((new - old) / T)
            # Since we maximize, delta = new - old is negative.
            # We want to accept if new is close to old.
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_points = new_points
                    current_score = new_score
            
        # Cooling
        T *= alpha
        
        # If T is too low, we might get stuck. 
        # We can implement a simple restart strategy: if no improvement for X iterations,
        # reset to best and increase T slightly? Or just continue.
        # Given the long runtime, a single long run with slow cooling is robust.
        
    # Final check
    if best_points is None or not np.all(np.isfinite(best_points)) or best_points.shape != (13, 2):
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        best_points = np.column_stack((np.cos(theta), np.sin(theta)))
        
    return best_points

# To ensure the function is standalone and ready
if __name__ == "__main__":
    pts = heilbronn_convex13()
    print(_score(pts))
