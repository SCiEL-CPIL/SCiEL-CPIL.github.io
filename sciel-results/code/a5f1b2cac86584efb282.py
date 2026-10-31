import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
SQRT3 = np.sqrt(3)
TIME_LIMIT = 115.0 # Safety margin

def params_to_cartesian(params):
    """
    params: array of shape (N, 2) containing (u, w) for each point.
    u in [0, 1], w in [0, 1].
    Barycentric coordinates:
    P = u*A + v*B + (1-u-v)*C
    A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    We use substitution v = (1-u)*w to ensure u>=0, v>=0, u+v<=1.
    x = u + 0.5*v
    y = (sqrt(3)/2)*v
    """
    u = params[:, 0]
    w = params[:, 1]
    v = (1.0 - u) * w
    x = u + 0.5 * v
    y = (SQRT3 / 2.0) * v
    return np.column_stack((x, y))

def compute_min_area_soft(params, beta):
    """
    Compute the negative log-sum-exp of the triangle areas (soft min).
    We want to maximize the min area, so we minimize the negative soft min.
    soft_min(A) = -1/beta * log(sum(exp(-beta * A)))
    minimize -soft_min(A) = 1/beta * log(sum(exp(-beta * A)))
    """
    pts = params_to_cartesian(params)
    n = len(pts)
    areas = []
    # Precompute cross products? 
    # Area = 0.5 * |det(P_j - P_i, P_k - P_i)|
    # To avoid sqrt and abs in the loop if possible, but we need positive areas.
    # Since points are in a triangle, orientation might flip. 
    # We take absolute value.
    
    # Vectorized area calculation for all triplets is memory heavy for N=11?
    # C(11,3) = 165 triplets. Very small.
    
    min_val = np.inf
    # We will compute sum of exp(-beta * area)
    sum_exp = 0.0
    
    for i, j, k in itertools.combinations(range(n), 3):
        x1, y1 = pts[i]
        x2, y2 = pts[j]
        x3, y3 = pts[k]
        area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
        # To prevent underflow/overflow in exp, we can shift, but areas are small.
        # beta will be large.
        sum_exp += np.exp(-beta * area)
        
    return (1.0 / beta) * np.log(sum_exp)

def objective(params, beta):
    p = params.reshape(N, 2)
    return compute_min_area_soft(p, beta)

def construct():
    start_time = time.time()
    best_params = None
    best_score = -np.inf # We minimize the objective, so lower is better.
    
    # Initial guess: Hexagonal-like lattice
    # Let's try a few different initializations
    inits = []
    
    # Init 1: Random
    rng = np.random.default_rng(42)
    for _ in range(5):
        u = rng.uniform(0, 1, N)
        w = rng.uniform(0, 1, N)
        inits.append(np.column_stack((u, w)))
        
    # Init 2: Structured Grid in Barycentric
    # Approximate positions
    # u, v coords
    # Let's just use a few random structured ones
    # A simple grid in u,v space (clipped to u+v<=1)
    for row in range(4):
        for col in range(3):
             # This doesn't easily give 11 points.
             pass
             
    # Let's just use random restarts. L-BFGS-B is robust.
    
    betas = [10, 50, 200, 1000]
    
    for beta in betas:
        for init in inits:
            if time.time() - start_time > TIME_LIMIT:
                break
                
            try:
                res = minimize(
                    objective,
                    init.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(0, 1)] * (2 * N),
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                if res.success:
                    val = res.fun
                    if val < best_score:
                        best_score = val
                        best_params = res.x
            except Exception:
                continue

    if best_params is None:
        # Fallback to seed-like random points
        rng = np.random.default_rng(7)
        points = []
        while len(points) < N:
            u, v = rng.random(), rng.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
        return np.array(points)

    # Convert best params to Cartesian
    best_pts = params_to_cartesian(best_params.reshape(N, 2))
    
    # Verify points are inside
    # The mapping guarantees inside, but let's double check for numerical errors
    # x >= 0, y >= 0, y <= sqrt(3)*x, y <= -sqrt(3)*(x-1)
    
    return best_pts
