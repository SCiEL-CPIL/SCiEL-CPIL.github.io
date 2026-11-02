import numpy as np
import time
import scipy.optimize as opt

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety
    
    # Precompute indices for all triples
    triples = [(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)]
    n_triples = len(triples)
    
    # Objective function: Minimize negative score (MinTriArea / HullArea)
    # We use a smooth surrogate for the min to help optimization.
    # But since we want exact min for final score, we track both.
    # Actually, standard Nelder-Mead on the non-smooth min is okay if we use a good initial guess.
    # The regular polygon is a strong local optimum.
    
    def compute_score(theta):
        # Points on unit circle
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Hull Area (assuming points are in angular order for convex hull)
        # Since we start from regular polygon and optimize locally, order is preserved.
        # To be safe, we sort angles to compute hull area.
        sorted_idx = np.argsort(theta)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]
        
        # Shoelace formula
        x_next = np.roll(x_sorted, -1)
        y_next = np.roll(y_sorted, -1)
        area_hull = 0.5 * np.abs(np.sum(x_sorted * y_next - x_next * y_sorted))
        
        if area_hull < 1e-9:
            return 0.0, 0.0  # Invalid
            
        # Triangle areas
        # Vectorized computation of all triangle areas
        # For each triple (i, j, k), area = 0.5 * |det(p_j - p_i, p_k - p_i)|
        # det = (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i)
        #     = x_j*y_k - x_j*y_i - x_i*y_k + x_i*y_i - y_j*x_k + y_j*x_i + y_i*x_k - y_i*x_i
        #     = x_j*y_k - y_j*x_k + x_i*(y_j - y_k) + y_i*(x_k - x_j)
        
        # Let's just loop for clarity and speed is fine for 286 triples
        min_area = float('inf')
        for i, j, k in triples:
            v1x = x[j] - x[i]
            v1y = y[j] - y[i]
            v2x = x[k] - x[i]
            v2y = y[k] - y[i]
            cross = abs(v1x * v2y - v1y * v2x)
            area = 0.5 * cross
            if area < min_area:
                min_area = area
                
        score = min_area / area_hull
        return score, area_hull

    def objective(theta):
        score, _ = compute_score(theta)
        return -score

    # Initial guess: Regular N-gon
    theta0 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # We can also try shifting the regular polygon by small amounts to break symmetry?
    # No, the regular polygon is the global optimum for this kind of problem usually.
    # Let's optimize starting from the regular polygon.
    
    best_score, _ = compute_score(theta0)
    best_theta = theta0.copy()
    
    stop_flag = False
    def callback(xk):
        nonlocal stop_flag
        if time.time() - start_time > time_limit - 5:
            stop_flag = True
            return True  # Stop optimization
        return False

    try:
        # Use Nelder-Mead
        res = opt.minimize(objective, theta0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-10},
                           callback=callback)
        
        if not stop_flag:
            score, _ = compute_score(res.x)
            if score > best_score:
                best_score = score
                best_theta = res.x
        else:
            # If stopped, check the last point?
            # Nelder-Mead doesn't easily give the best point if stopped by callback in all versions.
            # But usually it returns the current simplex vertex.
            # Let's just use the best found so far.
            pass
            
    except Exception as e:
        pass

    # If time remains, we can try a finer polish with L-BFGS-B or similar?
    # L-BFGS-B requires gradients. We can use finite differences.
    # Or just run Nelder-Mead again from the best point with tighter tolerances.
    
    if time.time() - start_time < time_limit - 10:
        try:
            stop_flag2 = False
            def callback2(xk):
                nonlocal stop_flag2
                if time.time() - start_time > time_limit - 5:
                    stop_flag2 = True
                    return True
                return False

            res2 = opt.minimize(objective, best_theta, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-12},
                               callback=callback2)
            if not stop_flag2:
                score2, _ = compute_score(res2.x)
                if score2 > best_score:
                    best_score = score2
                    best_theta = res2.x
        except Exception:
            pass

    # Construct final points
    points = np.column_stack((np.cos(best_theta), np.sin(best_theta)))
    
    return points

# EVOLVE-BLOCK-END
