import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Ring structure: 1, 6, 12, 18, 13
    # Indices: 0: center, 1: ring 1 (6), 2: ring 2 (12), 3: ring 3 (18), 4: ring 4 (13)
    
    def generate_points(params):
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
        # Compute min distance between all pairs
        n = len(points)
        min_dist = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    min_dist = d
        
        # Compute max radius from center
        max_r = np.max(np.linalg.norm(points, axis=1))
        
        # Valid radius is limited by half the min distance and the container
        r1 = min_dist / 2.0
        r2 = 1.0 - max_r
        
        return min(r1, r2)

    def objective(params):
        points = generate_points(params)
        r = compute_radius(points)
        return -r

    def run_optimization(initial_params, maxiter=1000):
        # Bounds for radii: must be positive and less than 1
        # Angles: 0 to 2*pi
        bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99), (0.01, 0.99), 
                  (0, 2*np.pi), (0, 2*np.pi), (0, 2*np.pi)]
        
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-10})
        return result

    # Initial guesses based on known structure
    # Known optimal r ~ 0.1258
    # This implies distances ~ 0.25
    # Ring radii roughly: 0.25, 0.5, 0.75, 0.95
    
    best_score = 0.0
    best_points = None
    
    # Try several starting points
    starts = [
        [0.25, 0.50, 0.75, 0.95, 0, 0, 0],
        [0.24, 0.48, 0.74, 0.94, np.pi/12, np.pi/18, np.pi/13],
        [0.26, 0.52, 0.76, 0.96, np.pi/6, np.pi/9, np.pi/26],
        [0.23, 0.47, 0.73, 0.93, np.pi/4, np.pi/3, np.pi/2]
    ]
    
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
            
        try:
            result = run_optimization(start, maxiter=500)
            if result.success:
                params = result.x
                points = generate_points(params)
                score = compute_radius(points)
                
                if score > best_score:
                    best_score = score
                    best_points = points
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

    return best_points
# EVOLVE-BLOCK-END
