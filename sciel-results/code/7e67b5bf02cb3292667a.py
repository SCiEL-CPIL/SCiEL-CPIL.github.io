import numpy as np
from scipy.spatial import ConvexHull

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius for a set of points on the unit sphere.
    """
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-9):
        return np.pi
    points = points / norms[:, np.newaxis]
    
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-9:
                continue
            normal = normal / norm_n
            d = np.dot(normal, p1)
            if d <= 1e-9:
                return np.pi
            cos_R = np.clip(np.dot(normal, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_radius:
                max_radius = R
        return max_radius
    except Exception:
        return np.pi

def _thomson_energy(points):
    """
    Calculate the Thomson energy: sum 1/||p_i - p_j||^2
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    mask = np.eye(N, dtype=bool)
    inv_dist_sq = np.zeros_like(dist_sq)
    np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
    return np.sum(inv_dist_sq) / 2.0

def _simulated_annealing(points, max_iters=5000, T0=0.1, T_end=1e-6):
    """
    Run Simulated Annealing to minimize Thomson energy.
    """
    current_points = points.copy()
    current_energy = _thomson_energy(current_points)
    best_points = current_points.copy()
    best_energy = current_energy
    
    for i in range(max_iters):
        T = T0 * (T_end / T0) ** (i / max_iters)
        
        # Generate a perturbation: move a random point by a small random vector
        idx = np.random.randint(N)
        # Random unit vector
        v = np.random.normal(size=3)
        v = v / np.linalg.norm(v)
        
        # Step size decays slightly
        step_size = 0.05 * (T / T0) + 0.001
        
        new_points = current_points.copy()
        new_points[idx] = new_points[idx] + step_size * v
        # Normalize
        new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
        
        new_energy = _thomson_energy(new_points)
        delta_E = new_energy - current_energy
        
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            current_points = new_points
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()
                
    return best_points

def construct():
    # Multi-start Simulated Annealing
    best_points_overall = None
    best_score_overall = np.pi
    
    # Initializations:
    # 1. Fibonacci Spiral
    # 2. Random Uniform
    # 3. Spherical Grid (approx)
    
    initializations = []
    
    # 1. Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    p_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(p_fib)
    
    # 2. Random
    np.random.seed(42)
    for _ in range(5):
        p_rand = np.random.normal(size=(N, 3))
        p_rand /= np.linalg.norm(p_rand, axis=1, keepdims=True)
        initializations.append(p_rand)
        
    # 3. Grid-like (Lat/Long)
    # Distribute points roughly evenly in lat/long
    # This is a crude grid, but helps with symmetry
    lats = np.linspace(-np.pi/2 + np.pi/(2*N), np.pi/2 - np.pi/(2*N), N)
    # Vary longitudes to avoid clustering at poles
    # Simple approach: just use random longitudes for each lat band? 
    # Or a proper spherical grid. Let's stick to random + fib for robustness.
    
    # Run SA for each initialization
    # Limit time/iterations per run to ensure we finish in 120s
    # 5000 iterations * 7 runs * ~100us/iter = ~3.5s. Safe.
    
    for p_init in initializations:
        # Refine with SA
        p_refined = _simulated_annealing(p_init, max_iters=3000)
        score = _covering_radius(p_refined)
        
        if score < best_score_overall:
            best_score_overall = score
            best_points_overall = p_refined
            
    return best_points_overall
