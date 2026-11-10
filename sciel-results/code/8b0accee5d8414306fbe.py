import numpy as np
from scipy.spatial import ConvexHull

def compute_ratio_fast(points):
    """
    Computes the ratio A_min / A_hull assuming points are in convex position
    and sorted by angle around the origin.
    """
    n = points.shape[0]
    
    # Calculate Convex Hull Area (Area of the polygon P0...Pn-1)
    # Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    x = points[:, 0]
    y = points[:, 1]
    # Shift indices
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    a_hull = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
    
    if a_hull <= 1e-10:
        return -np.inf

    # Calculate all triangle areas
    # n=13, C(13,3) = 286 triangles
    # We can vectorize this
    i, j, k = np.indices((n, n, n))
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]
    
    areas = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    
    a_min = np.min(areas)
    
    if a_min <= 1e-10:
        return -np.inf
        
    return a_min / a_hull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    def get_points(radii):
        return np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Initial state: Regular polygon
    current_radii = np.ones(n)
    current_points = get_points(current_radii)
    current_score = compute_ratio_fast(current_points)
    
    best_radii = current_radii.copy()
    best_score = current_score
    best_points = current_points.copy()
    
    # Simulated Annealing Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    alpha = 0.99  # Cooling rate
    max_iter = 50000  # Number of iterations
    step_size = 0.1
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        # Propose a move: perturb one or more radii
        # Let's perturb a random subset of radii
        num_perturb = rng.integers(1, 4)
        indices = rng.choice(n, size=num_perturb, replace=False)
        
        new_radii = current_radii.copy()
        new_radii[indices] += rng.normal(0, step_size, size=num_perturb)
        
        # Keep radii positive
        new_radii = np.maximum(new_radii, 0.1)
        
        new_points = get_points(new_radii)
        new_score = compute_ratio_fast(new_points)
        
        # Acceptance criteria
        if new_score > current_score:
            current_radii = new_radii
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_radii = current_radii.copy()
                best_points = current_points.copy()
        else:
            # Calculate delta
            delta = current_score - new_score
            if delta < 0:
                # This shouldn't happen if we checked new_score > current_score
                pass
            prob = np.exp(-delta / T)
            if rng.random() < prob:
                current_radii = new_radii
                current_points = new_points
                current_score = new_score
        
        T *= alpha
        if T < T_min:
            break
            
    return best_points
