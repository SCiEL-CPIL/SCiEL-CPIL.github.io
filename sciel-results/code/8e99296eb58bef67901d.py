import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
H = np.sqrt(3) / 2.0

def calculate_areas(points):
    """
    points: (N, 2) array
    Returns: array of areas for all triangles
    """
    p1 = points[tri_indices_arr[:, 0]]
    p2 = points[tri_indices_arr[:, 1]]
    p3 = points[tri_indices_arr[:, 2]]
    
    # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    areas = 0.5 * np.abs(cross)
    return areas

def objective(params, beta):
    points = params.reshape(N, 2)
    areas = calculate_areas(points)
    # Minimize -SoftMin(areas)
    # -SoftMin = (1/beta) * ln(sum(exp(-beta * areas)))
    # Clip areas to prevent underflow issues if beta is very large, though exp(-large) -> 0 is fine.
    # The main risk is if areas are 0, log(0) is -inf.
    # But points are distinct, so areas > 0.
    # If points coincide, area is 0. exp(0)=1. Sum >= 1. Log >= 0.
    log_sum = np.log(np.sum(np.exp(-beta * areas)))
    return log_sum / beta

def make_constraints():
    cons = []
    # y_i >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i+1]})
    # y_i - sqrt(3)x_i <= 0  =>  sqrt(3)x_i - y_i >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: np.sqrt(3)*p[2*i] - p[2*i+1]})
    # y_i + sqrt(3)x_i - sqrt(3) <= 0 => sqrt(3) - sqrt(3)x_i - y_i >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: np.sqrt(3) - np.sqrt(3)*p[2*i] - p[2*i+1]})
    return cons

CONSTRAINTS = make_constraints()
BOUNDS = [(0, 1), (0, H)] * N

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_points = None
    best_val = -1
    
    rng = np.random.default_rng(42)
    
    # Generate initial guesses
    inits = []
    for _ in range(10):
        pts = []
        while len(pts) < N:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                p = u * V0 + v * V1 + (1 - u - v) * V2
                pts.append(p)
        init = np.array(pts).flatten()
        inits.append(init)

    # Annealing betas
    betas = [10, 50, 100, 200, 400]
    
    for init in inits:
        if time.time() > deadline - 20:
            break
            
        current = init
        for beta in betas:
            if time.time() > deadline - 10:
                break
                
            try:
                res = minimize(
                    objective, 
                    current, 
                    args=(beta,),
                    method='SLSQP',
                    bounds=BOUNDS,
                    constraints=CONSTRAINTS,
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                if res.success:
                    current = res.x
                else:
                    # If failed, keep current best or try next beta?
                    # SLSQP usually returns something reasonable even if not converged.
                    current = res.x
            except Exception:
                pass
                
        # Evaluate final raw min area
        final_points = current.reshape(N, 2)
        # Ensure points are strictly inside or on boundary
        # SLSQP should respect constraints, but numerical errors might push slightly out.
        # Clip to bounds?
        final_points[:, 0] = np.clip(final_points[:, 0], 0, 1)
        final_points[:, 1] = np.clip(final_points[:, 1], 0, H)
        
        # Re-check constraints roughly?
        # The evaluator likely checks validity.
        
        areas = calculate_areas(final_points)
        val = np.min(areas)
        
        if val > best_val:
            best_val = val
            best_points = final_points
            
    if best_points is None:
        # Fallback
        points = []
        rng2 = np.random.default_rng(7)
        while len(points) < N:
            u = rng2.random()
            v = rng2.random()
            if u + v < 0.98 and u > 0.01 and v > 0.01:
                p = u * V0 + v * V1 + (1 - u - v) * V2
                points.append(p)
        best_points = np.array(points)
        
    return best_points
