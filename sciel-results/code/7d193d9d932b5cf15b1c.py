import numpy as np
from scipy.optimize import minimize
import time

N = 11
# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2])
TRIA_AREA = np.sqrt(3)/4

def bary_to_cart(u, v):
    # u, v are barycentric coords for B and C, A is origin
    # P = u*B + v*C
    return u * B + v * C

def calc_min_area(points):
    # points: (N, 2) array
    n = len(points)
    min_a = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                if area < min_a:
                    min_a = area
    return min_a

def objective(params, beta):
    # params: 2*N floats (u_i, v_i)
    u = params[0::2]
    v = params[1::2]
    
    # Penalty for out-of-bounds (u<0, v<0, u+v>1)
    # SLSQP handles bounds, but let's add a soft penalty just in case for numerical stability
    # Actually, SLSQP with bounds and inequality constraints is better.
    
    points = np.array([bary_to_cart(u[i], v[i]) for i in range(N)])
    
    # Calculate areas
    areas = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
                areas.append(area)
    areas = np.array(areas)
    
    # Soft min: -1/beta * log(sum(exp(-beta * area)))
    # To avoid overflow/underflow, use logsumexp trick
    # log(sum(exp(-beta * a))) = -beta * max(a) + log(sum(exp(-beta * (a - max(a)))))
    max_a = np.max(areas)
    if max_a == 0:
        return 0.0
    log_sum = -beta * max_a + np.log(np.sum(np.exp(-beta * (areas - max_a))))
    soft_min = -1.0 / beta * log_sum
    
    return -soft_min # We want to maximize min area, so minimize negative

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial guesses
    best_result = None
    best_score = -np.inf
    
    # Multi-start
    rng = np.random.RandomState(42)
    
    # Try different beta values (annealing)
    betas = [10, 50, 100, 200, 500]
    
    # Generate some random valid starting points
    # u, v >= 0, u+v <= 1
    # Sample from uniform distribution in triangle
    # u = r * cos(theta), v = r * sin(theta) ? No.
    # Simple: u = rng.random(), v = rng.random() * (1-u)
    
    starts = []
    for _ in range(20):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        params = np.zeros(2*N)
        params[0::2] = u
        params[1::2] = v
        starts.append(params)
        
    # Add a symmetric grid start?
    # For N=11, a grid is not trivial.
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        for start in starts:
            if time.time() > deadline:
                break
                
            # Constraints for SLSQP
            # u_i >= 0, v_i >= 0, u_i + v_i <= 1
            # Bounds: u_i in [0, 1], v_i in [0, 1]
            bounds = []
            for i in range(N):
                bounds.append((0.0, 1.0)) # u_i
                bounds.append((0.0, 1.0)) # v_i
                
            constraints = []
            for i in range(N):
                # u_i + v_i <= 1  =>  1 - u_i - v_i >= 0
                def con(i, x):
                    return 1.0 - x[2*i] - x[2*i+1]
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: con(i, x)})
                
            try:
                res = minimize(
                    objective, 
                    start, 
                    args=(beta,),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                
                if res.success:
                    # Calculate true min area (not soft)
                    u = res.x[0::2]
                    v = res.x[1::2]
                    points = np.array([bary_to_cart(u[i], v[i]) for i in range(N)])
                    true_min = calc_min_area(points)
                    
                    if true_min > best_score:
                        best_score = true_min
                        best_result = points.copy()
                        
            except Exception:
                pass
                
    if best_result is None:
        # Fallback to random
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_result = np.array([bary_to_cart(u[i], v[i]) for i in range(N)])
        
    return best_result
