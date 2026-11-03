import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def compute_covering_radius_and_facets(points):
    """
    Computes the covering radius and returns the radii and indices for each facet.
    points: (N, 3) array of unit vectors.
    Returns:
        max_radius: float
        radii: (M,) array of circumradii for each facet
        simplices: (M, 3) array of vertex indices for each facet
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        # Fallback if hull fails
        return np.pi, np.array([]), np.array([])
    
    radii = np.zeros(len(simplices))
    
    for idx, simplex in enumerate(simplices):
        i, j, k = simplex
        a = points[i]
        b = points[j]
        c = points[k]
        
        # Normal vector to the plane of the triangle
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            radii[idx] = np.pi / 2
            continue
            
        u = n / norm_n
        
        # Ensure u points towards the centroid (inside the spherical triangle)
        centroid = (a + b + c) / 3.0
        if np.dot(u, centroid) < 0:
            u = -u
            
        # Circumradius is the angle between u and any vertex
        cos_theta = np.dot(u, a)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii[idx] = np.arccos(cos_theta)
        
    max_radius = np.max(radii) if len(radii) > 0 else np.pi
    return max_radius, radii, simplices

def soft_energy(points, Y, k=20):
    """
    Computes a soft surrogate for the covering radius.
    Minimizing this sum of powers of (1 - max cosine similarity)
    approximates minimizing the max covering radius.
    """
    # Cosine similarities
    cos_sim = Y @ points.T
    # Max similarity for each point in Y
    max_sim = np.max(cos_sim, axis=1)
    # Clip to [0, 1] to handle numerical errors
    max_sim = np.clip(max_sim, 0.0, 1.0)
    # Sum of k-th powers of the "gap" (1 - max_sim).
    # If a point y is far from all P, max_sim is low, gap is high, energy is high.
    return np.sum((1.0 - max_sim) ** k)

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    # Grid for soft energy calculation
    M = 50000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    Y = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial points: Fibonacci sphere
    i_init = np.arange(N) + 0.5
    z_init = 1.0 - 2.0 * i_init / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i_init
    r_init = np.sqrt(1.0 - z_init * z_init)
    init_points = np.column_stack([r_init * np.cos(phi_init), r_init * np.sin(phi_init), z_init])
    
    best_points = init_points.copy()
    best_radius = compute_covering_radius_and_facets(best_points)[0]
    
    # Simulated Annealing Parameters
    T_start = 0.05
    T_end = 0.001
    n_iters = 2000
    k_power = 10 # Exponent for soft energy
    
    current_points = init_points.copy()
    current_energy = soft_energy(current_points, Y, k_power)
    
    # Pre-compute random directions for perturbations
    rng = np.random.default_rng(42)
    
    for it in range(n_iters):
        if time.time() - start_time > time_limit - 15.0:
            break
            
        # Cooling schedule (exponential)
        T = T_start * (T_end / T_start) ** (it / n_iters)
        
        # Perturb all points by a small random vector
        step_size = 0.05 * (T / T_start) + 0.001
        noise = rng.normal(0, step_size, (N, 3))
        new_points = current_points + noise
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        new_points = new_points / norms
        
        # Calculate new energy
        new_energy = soft_energy(new_points, Y, k_power)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_points = new_points
            current_energy = new_energy
            
            # Periodically check true covering radius and update best
            if it % 10 == 0:
                true_radius = compute_covering_radius_and_facets(current_points)[0]
                if true_radius < best_radius:
                    best_radius = true_radius
                    best_points = current_points.copy()
        
    # Final polish with Lloyd's algorithm on the best found configuration
    # Use a smaller grid for speed in final phase
    M_lloyd = 20000
    i_l = np.arange(M_lloyd) + 0.5
    z_l = 1.0 - 2.0 * i_l / M_lloyd
    phi_l = np.pi * (1.0 + np.sqrt(5.0)) * i_l
    r_l = np.sqrt(1.0 - z_l * z_l)
    Y_lloyd = np.column_stack([r_l * np.cos(phi_l), r_l * np.sin(phi_l), z_l])
    
    polished_points = best_points.copy()
    
    # Run a few Lloyd iterations
    for _ in range(20):
        if time.time() - start_time > time_limit - 5.0:
            break
        dot_products = Y_lloyd @ polished_points.T
        nearest_idx = np.argmax(dot_products, axis=1)
        sums = np.zeros((N, 3))
        np.add.at(sums, nearest_idx, Y_lloyd)
        norms = np.linalg.norm(sums, axis=1)
        valid = norms > 1e-10
        polished_points[valid] = sums[valid] / norms[valid, np.newaxis]
        
    final_radius = compute_covering_radius_and_facets(polished_points)[0]
    if final_radius < best_radius:
        best_points = polished_points
        best_radius = final_radius

    return best_points
