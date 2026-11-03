"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def construct():
    """
    Construct 50 equal circles in a unit circle using ring-based packing with optimization.
    Uses a known good structure: 1 center + 4 rings.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Ring structure: 1 + 6 + 12 + 12 + 19 = 50
    ring_sizes = [6, 12, 12, 19]
    n_rings = len(ring_sizes)
    
    # Initial guess for ring radii and rotation angles
    # Based on hexagonal packing intuition
    r0 = np.array([0.3, 0.6, 0.85, 1.0])  # radial distances
    theta0 = np.zeros(n_rings)  # rotation angles
    
    def get_centers(radii, angles):
        """Generate centers given radial distances and rotation angles for each ring."""
        centers = [[0.0, 0.0]]  # Center circle
        
        for i, (r, angle) in enumerate(zip(radii, angles)):
            n = ring_sizes[i]
            # Evenly spaced angles with rotation offset
            base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + angle
            x = r * np.cos(base_angles)
            y = r * np.sin(base_angles)
            centers.extend(zip(x, y))
        
        return np.array(centers)
    
    def objective(params):
        """Negative of the minimum distance (to maximize)."""
        radii = params[:n_rings]
        angles = params[n_rings:]
        
        # Ensure radii are positive and within bounds
        radii = np.clip(radii, 0.05, 1.0)
        
        centers = get_centers(radii, angles)
        
        # Compute all pairwise distances
        n = len(centers)
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_dist:
                    min_dist = d
        
        # Distance to boundary
        for i in range(n):
            dist_to_center = np.linalg.norm(centers[i])
            if dist_to_center > 1.0:
                return -1.0  # Invalid
            boundary_dist = 1.0 - dist_to_center
            if boundary_dist < min_dist:
                min_dist = boundary_dist
        
        return -min_dist  # Negative because we minimize
    
    # Use Nelder-Mead optimization
    x0 = np.concatenate([r0, theta0])
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Multiple restarts with different initial conditions
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Perturb initial guess
        if restart == 0:
            x_init = x0.copy()
        else:
            x_init = x0 + np.random.normal(0, 0.1, size=len(x0))
            x_init[:n_rings] = np.clip(x_init[:n_rings], 0.1, 1.0)
        
        try:
            result = minimize(objective, x_init, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                      'adaptive': True})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Final optimization with tighter tolerance
    try:
        result = minimize(objective, best_x, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10,
                                  'adaptive': True})
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x.copy()
    except Exception:
        pass
    
    # Extract final parameters
    radii = np.clip(best_x[:n_rings], 0.05, 1.0)
    angles = best_x[n_rings:]
    
    centers = get_centers(radii, angles)
    
    # Verify validity
    n = len(centers)
    valid = True
    for i in range(n):
        if np.linalg.norm(centers[i]) > 1.0:
            valid = False
            break
    
    if valid:
        return centers
    
    # Fallback to hexagonal lattice if optimization failed
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98
# EVOLVE-BLOCK-END
