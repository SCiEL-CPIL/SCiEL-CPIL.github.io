import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin before 150s kill
    
    # Structure: 1 center, 6 ring 1, 12 ring 2, 18 ring 3, 13 ring 4
    # This matches the best known packing structure for N=50
    
    def generate_points(params):
        """Generate 50 points from ring parameters."""
        R1, R2, R3, R4, phi2, phi3, phi4 = params
        points = np.zeros((50, 2))
        
        # Ring 0: Center
        points[0] = [0, 0]
        
        # Ring 1: 6 points
        angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i in range(6):
            points[1 + i] = [R1 * np.cos(angles1[i]), R1 * np.sin(angles1[i])]
            
        # Ring 2: 12 points
        angles2 = np.linspace(0, 2*np.pi, 12, endpoint=False) + phi2
        for i in range(12):
            points[7 + i] = [R2 * np.cos(angles2[i]), R2 * np.sin(angles2[i])]
            
        # Ring 3: 18 points
        angles3 = np.linspace(0, 2*np.pi, 18, endpoint=False) + phi3
        for i in range(18):
            points[19 + i] = [R3 * np.cos(angles3[i]), R3 * np.sin(angles3[i])]
            
        # Ring 4: 13 points
        angles4 = np.linspace(0, 2*np.pi, 13, endpoint=False) + phi4
        for i in range(13):
            points[37 + i] = [R4 * np.cos(angles4[i]), R4 * np.sin(angles4[i])]
            
        return points

    def compute_radius(points):
        """Compute the maximum valid radius for given points."""
        # Distance to boundary
        dist_to_center = np.linalg.norm(points, axis=1)
        max_dist = np.max(dist_to_center)
        if max_dist >= 1.0:
            return 0.0
        
        # Minimum distance between any two centers
        n = len(points)
        min_dist = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    min_dist = d
        
        # Valid radius is limited by half the min distance and the container
        r1 = min_dist / 2.0
        r2 = 1.0 - max_dist
        
        return min(r1, r2)

    def objective(params):
        """Negative of the radius (to minimize)."""
        points = generate_points(params)
        r = compute_radius(points)
        return -r

    def run_optimization(initial_params, method='L-BFGS-B', maxiter=1000):
        """Run optimization with given method and initial params."""
        bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99), 
                  (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
        
        if method == 'L-BFGS-B':
            result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12})
        elif method == 'Nelder-Mead':
            result = minimize(objective, initial_params, method='Nelder-Mead', 
                              options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        elif method == 'Powell':
            result = minimize(objective, initial_params, method='Powell',
                              options={'maxiter': maxiter, 'ftol': 1e-12, 'xtol': 1e-12})
        else:
            result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, 
                              options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-12})
        
        return result

    # Initial guesses based on known structure
    # For r_circle ≈ 0.1258, ring radii should be approximately:
    # Ring 1: ~0.25, Ring 2: ~0.50, Ring 3: ~0.75, Ring 4: ~0.95
    
    best_score = 0.0
    best_points = None
    best_params = None
    
    # Try several starting points with different methods
    starts = [
        [0.25, 0.50, 0.75, 0.95, 0, 0, 0],
        [0.24, 0.48, 0.74, 0.94, np.pi/12, np.pi/18, np.pi/13],
        [0.26, 0.52, 0.76, 0.96, np.pi/6, np.pi/9, np.pi/26],
        [0.23, 0.47, 0.73, 0.93, np.pi/4, np.pi/3, np.pi/2],
        [0.255, 0.51, 0.755, 0.955, np.pi/8, np.pi/10, np.pi/15],
        [0.245, 0.495, 0.745, 0.945, np.pi/16, np.pi/20, np.pi/25],
    ]
    
    methods = ['L-BFGS-B', 'Nelder-Mead', 'Powell']
    
    for i, start in enumerate(starts):
        if time.time() > deadline - 30:  # Leave time for final polish
            break
            
        for method in methods:
            if time.time() > deadline - 20:
                break
                
            try:
                maxiter = 500 if method == 'L-BFGS-B' else 2000
                result = run_optimization(start, method=method, maxiter=maxiter)
                
                if result.fun < -best_score:  # result.fun is -radius
                    params = result.x
                    # Clip radii to valid range
                    params = np.array(params)
                    params[:4] = np.clip(params[:4], 0.01, 0.99)
                    params[4:] = np.mod(params[4:], 2*np.pi)
                    
                    points = generate_points(params)
                    score = compute_radius(points)
                    
                    if score > best_score:
                        best_score = score
                        best_points = points
                        best_params = params
            except Exception as e:
                pass
    
    # Final high-precision polish if we have a good starting point
    if best_params is not None and time.time() < deadline - 10:
        try:
            # Very tight tolerance refinement
            result = run_optimization(best_params, method='Nelder-Mead', maxiter=10000)
            if result.fun < -best_score:
                params = result.x
                params = np.array(params)
                params[:4] = np.clip(params[:4], 0.01, 0.99)
                params[4:] = np.mod(params[4:], 2*np.pi)
                
                points = generate_points(params)
                score = compute_radius(points)
                
                if score > best_score:
                    best_score = score
                    best_points = points
                    best_params = params
        except Exception as e:
            pass
    
    # If no optimization succeeded or score is low, fallback to a simple heuristic
    if best_points is None or best_score < 0.1:
        # Fallback: Hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_points = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98
    
    # Final validation
    if best_points is None or len(best_points) != N or not np.all(np.isfinite(best_points)):
        # Emergency fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_points = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98
    
    return best_points
# EVOLVE-BLOCK-END
