"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _triangle_area(p1, p2, p3):
    """Compute area of triangle with vertices p1, p2, p3."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

def _min_triangle_area(points):
    """Compute minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = _triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute area of convex hull of points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if hull fails, use shoelace on sorted points
        # This shouldn't happen for valid convex polygon vertices
        # Sort by angle from centroid
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        order = np.argsort(angles)
        pts = points[order]
        # Shoelace formula
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _score(points):
    """Score = min triangle area / convex hull area."""
    min_tri = _min_triangle_area(points)
    hull_area = _convex_hull_area(points)
    if hull_area <= 0:
        return 0.0
    return min_tri / hull_area

def _fourier_to_points(log_radii, n_points=13):
    """
    Convert log-radii parameters to actual points.
    log_radii: array of shape (n_points,)
    Returns points array of shape (n_points, 2)
    """
    radii = np.exp(log_radii)
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def _points_to_log_radii(points, n_points=13):
    """Convert points to log-radii parameterization."""
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    # Ensure radii are positive
    radii = np.maximum(radii, 1e-10)
    return np.log(radii)

def _objective(log_radii):
    """Objective function to minimize (negative score)."""
    points = _fourier_to_points(log_radii, N)
    return -_score(points)

def _refine_with_slsqp(log_radii_init, max_iter=500):
    """Refine using SLSQP."""
    from scipy.optimize import minimize
    
    def obj(x):
        return _objective(x)
    
    result = minimize(obj, log_radii_init, method='SLSQP', 
                      options={'maxiter': max_iter, 'ftol': 1e-12})
    return result.x, result.fun

def _differential_evolution_search(log_radii_init, population_size=50, max_iter=100):
    """Simple differential evolution implementation."""
    rng = np.random.RandomState(42)
    
    # Initialize population
    pop = [log_radii_init.copy() for _ in range(population_size)]
    for i in range(population_size):
        if i > 0:
            pop[i] = log_radii_init + rng.normal(0, 0.1, N)
    
    best_x = pop[0].copy()
    best_f = _objective(best_x)
    
    for generation in range(max_iter):
        # Evaluate population
        for i in range(population_size):
            f = _objective(pop[i])
            if f < best_f:
                best_f = f
                best_x = pop[i].copy()
        
        # Differential evolution step
        for i in range(population_size):
            # Select three distinct random individuals
            a, b, c = rng.choice(population_size, 3, replace=False)
            F = 0.8  # Scale factor
            CR = 0.9  # Crossover rate
            
            # Create trial vector
            trial = pop[a] + F * (pop[b] - pop[c])
            
            # Crossover
            for j in range(N):
                if rng.random() < CR or j == rng.randint(0, N - 1):
                    trial[j] = pop[a][j] + F * (pop[b][j] - pop[c][j])
                else:
                    trial[j] = pop[i][j]
            
            # Selection
            f_trial = _objective(trial)
            f_current = _objective(pop[i])
            if f_trial < f_current:
                pop[i] = trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
    
    return best_x, best_f

def construct():
    """
    Construct 13 points in a planar convex region to maximize the minimum
    triangle area over all triples, normalized by convex hull area.
    """
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Start with regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_log_radii = np.log(np.ones(N))
    
    best_points = initial_points.copy()
    best_score = _score(best_points)
    
    # Try SLSQP refinement from regular polygon
    try:
        refined_log_radii, refined_fun = _refine_with_slsqp(initial_log_radii, max_iter=200)
        refined_points = _fourier_to_points(refined_log_radii, N)
        refined_score = _score(refined_points)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
    except Exception:
        pass
    
    if time.time() > deadline - 60:
        return best_points
    
    # Try differential evolution with multiple restarts
    rng = np.random.RandomState(42)
    
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Create perturbed initial point
        if restart == 0:
            log_radii_init = initial_log_radii.copy()
        else:
            # Random perturbation
            log_radii_init = initial_log_radii + rng.normal(0, 0.2, N)
        
        try:
            de_log_radii, de_fun = _differential_evolution_search(
                log_radii_init, population_size=30, max_iter=50)
            de_points = _fourier_to_points(de_log_radii, N)
            de_score = _score(de_points)
            
            if de_score > best_score:
                best_score = de_score
                best_points = de_points
        except Exception:
            pass
    
    # Final SLSQP refinement from best point found
    if time.time() < deadline - 10:
        try:
            best_log_radii = _points_to_log_radii(best_points, N)
            final_log_radii, final_fun = _refine_with_slsqp(best_log_radii, max_iter=300)
            final_points = _fourier_to_points(final_log_radii, N)
            final_score = _score(final_points)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_points
        except Exception:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
