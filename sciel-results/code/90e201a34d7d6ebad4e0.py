"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

def construct():
    # Vertices of the equilateral triangle
    # (0,0), (1,0), (0.5, sqrt(3)/2)
    H = np.sqrt(3) / 2
    
    # Precompute indices for all triplets
    indices = list(combinations(range(N), 3))
    n_triplets = len(indices)
    
    # Variables: x_0, y_0, x_1, y_1, ..., x_N-1, y_N-1, t
    # Total variables: 2*N + 1
    
    def area_val(p1, p2, p3):
        # p1, p2, p3 are (x, y) tuples or arrays
        return 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))

    def objective(vars):
        # vars: [x0, y0, ..., xN-1, yN-1, t]
        t = vars[-1]
        return -t

    def constraints(vars):
        cons = []
        coords = vars[:-1]
        t = vars[-1]
        
        # 1. Triangle Boundary Constraints
        # y >= 0
        # y <= sqrt(3)*x  =>  sqrt(3)*x - y >= 0
        # y <= sqrt(3)*(1-x) => sqrt(3)*(1-x) - y >= 0
        
        for i in range(N):
            x = coords[2*i]
            y = coords[2*i+1]
            
            # y >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1]})
            
            # sqrt(3)*x - y >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: np.sqrt(3)*v[2*i] - v[2*i+1]})
            
            # sqrt(3)*(1-x) - y >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: np.sqrt(3)*(1-v[2*i]) - v[2*i+1]})

        # 2. Area Constraints: Area(i,j,k) >= t
        for idx in indices:
            i, j, k = idx
            # Extract coords for i, j, k
            # To avoid lambda closure issues in loop, use default args
            cons.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j, k=k: area_val(
                    (v[2*i], v[2*i+1]), 
                    (v[2*j], v[2*j+1]), 
                    (v[2*k], v[2*k+1])
                ) - v[-1]
            })
            
        return cons

    # Bounds for variables
    # x in [0, 1], y in [0, H]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, H))  # y
    bounds.append((0, 0.25))   # t (max possible min area is small, 0.25 is safe upper bound)

    # Multi-start optimization
    best_result = None
    best_score = -1
    
    # Use a fixed seed for reproducibility of the "random" starts, 
    # but vary the start points.
    rng = np.random.RandomState(42)
    
    n_restarts = 30  # Number of random starts
    
    for r in range(n_restarts):
        # Generate random initial points inside the triangle
        # Using barycentric coordinates or rejection sampling
        init_coords = np.zeros(2*N)
        for i in range(N):
            while True:
                u = rng.random()
                v = rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = H * v
                    init_coords[2*i] = x
                    init_coords[2*i+1] = y
                    break
        
        # Initial t guess
        init_t = 0.01
        x0 = np.append(init_coords, init_t)
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success:
                # Extract points
                coords = res.x[:-1]
                points = np.array([[coords[2*i], coords[2*i+1]] for i in range(N)])
                
                # Calculate the actual minimum area to verify and score
                min_area = float('inf')
                for i, j, k in indices:
                    a = area_val(points[i], points[j], points[k])
                    if a < min_area:
                        min_area = a
                
                # Ensure points are strictly inside (add small epsilon if on boundary)
                # The constraints allow boundary, which is fine, but let's check validity.
                valid = True
                for p in points:
                    if p[0] < -1e-9 or p[0] > 1 + 1e-9 or p[1] < -1e-9 or p[1] > H + 1e-9:
                        valid = False
                    if p[1] > np.sqrt(3)*p[0] + 1e-9:
                        valid = False
                    if p[1] > np.sqrt(3)*(1-p[0]) + 1e-9:
                        valid = False
                
                if valid and min_area > best_score:
                    best_score = min_area
                    best_result = points
                    
        except Exception as e:
            continue

    if best_result is None:
        # Fallback to a simple grid or random if optimization fails completely
        # This shouldn't happen, but good for safety
        points = np.zeros((N, 2))
        for i in range(N):
            u, v = rng.random(), rng.random()
            while u + v >= 1:
                u, v = rng.random(), rng.random()
            points[i] = [u + 0.5 * v, H * v]
        return points

    return best_result
# EVOLVE-BLOCK-END
