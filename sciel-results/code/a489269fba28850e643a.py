import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11

def construct():
    start_time = time.time()
    time_limit = 110.0

    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2.0])
    
    # Precompute triplets indices
    triplets = list(itertools.combinations(range(N), 3))
    
    def smooth_objective(points_flat, beta):
        pts = points_flat.reshape(N, 2)
        # Calculate areas for all triplets
        areas = np.zeros(len(triplets))
        for idx, (i, j, k) in enumerate(triplets):
            p1, p2, p3 = pts[i], pts[j], pts[k]
            # Vectorized area calculation
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            areas[idx] = 0.5 * abs(cross)
        
        # Log-sum-exp for numerical stability
        # max_area = np.max(areas)
        # log_sum = max_area + np.log(np.sum(np.exp(-beta * (areas - max_area))))
        # return -1/beta * log_sum
        
        # Simpler: if beta is not too large, direct exp is fine
        # To avoid overflow, shift by max
        max_area = np.max(areas)
        if max_area == 0:
            return 0.0
        log_sum = np.log(np.sum(np.exp(-beta * (areas - max_area)))) + max_area
        return -1.0 / beta * log_sum

    def make_constraints(points_flat):
        pts = points_flat.reshape(N, 2)
        constraints = []
        for i in range(N):
            x, y = pts[i]
            # y >= 0
            constraints.append({'type': 'ineq', 'fun': lambda y=y: y})
            # sqrt(3)*x - y >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x=x, y=y: np.sqrt(3)*x - y})
            # sqrt(3)*(1-x) - y >= 0
            constraints.append({'type': 'ineq', 'fun': lambda x=x, y=y: np.sqrt(3)*(1.0-x) - y})
        return constraints

    best_val = -1.0
    best_points = None
    
    rng = np.random.default_rng(42)
    
    def random_point_in_triangle():
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                return A + u * (B - A) + v * (C - A)
                
    def get_initial_points():
        pts = np.array([random_point_in_triangle() for _ in range(N)])
        return pts.flatten()

    # Homotopy on beta
    betas = [5, 10, 20, 40]
    x_current = get_initial_points()
    
    for beta in betas:
        if time.time() - start_time > time_limit - 20:
            break
            
        # Run SLSQP for this beta
        # Use previous solution as start
        cons = make_constraints(x_current)
        try:
            res = minimize(smooth_objective, x_current, args=(beta,), method='SLSQP', 
                           constraints=cons, options={'maxiter': 1000, 'ftol': 1e-10})
            if res.success:
                x_current = res.x
                # Evaluate true objective
                pts = x_current.reshape(N, 2)
                min_area = float('inf')
                for i, j, k in triplets:
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                    if area < min_area:
                        min_area = area
                if min_area > best_val:
                    best_val = min_area
                    best_points = x_current.reshape(N, 2)
        except Exception:
            pass

    # If we have time, do a final Nelder-Mead on the exact objective
    if time.time() - start_time < time_limit - 10 and best_points is not None:
        def exact_objective(points_flat):
            pts = points_flat.reshape(N, 2)
            min_area = float('inf')
            for i, j, k in triplets:
                p1, p2, p3 = pts[i], pts[j], pts[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
            return -min_area

        # Nelder-Mead doesn't support constraints. 
        # We can use it if we project, but it's risky.
        # Let's stick with SLSQP result if it's good.
        # Or run SLSQP again with higher beta?
        # Let's try one more SLSQP with beta=100
        if time.time() - start_time < time_limit - 5:
            beta = 100
            cons = make_constraints(x_current)
            try:
                res = minimize(smooth_objective, x_current, args=(beta,), method='SLSQP', 
                               constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12})
                if res.success:
                    x_final = res.x
                    pts = x_final.reshape(N, 2)
                    min_area = float('inf')
                    for i, j, k in triplets:
                        p1, p2, p3 = pts[i], pts[j], pts[k]
                        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                        if area < min_area:
                            min_area = area
                    if min_area > best_val:
                        best_val = min_area
                        best_points = pts
            except Exception:
                pass

    if best_points is None:
        centroid = (A + B + C) / 3.0
        best_points = np.tile(centroid, (N, 1))
        
    return best_points
