import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin

    def objective(x):
        # x is a flattened array of 16 floats (8 points, 2 coords each)
        points = x.reshape(8, 2)
        
        # Compute pairwise squared distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        
        # Extract upper triangle indices
        i, j = np.triu_indices(8, k=1)
        d2 = sq_dists[i, j]
        
        # Check for zero distances
        if np.any(d2 < 1e-12):
            return 1e10
            
        # D = max(d) = sqrt(max(d2))
        max_d2 = np.max(d2)
        if max_d2 < 1e-12:
            return 1e10
            
        # Objective: - sum(log(d_ij)) + 28 * log(D)
        sum_log_d = 0.5 * np.sum(np.log(d2))
        log_D = 0.5 * np.log(max_d2)
        
        return -(sum_log_d) + 28.0 * log_D

    def score_points(points):
        """Compute the score Delta/n^n for a set of points."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        i, j = np.triu_indices(8, k=1)
        d2 = sq_dists[i, j]
        
        if np.any(d2 < 1e-12):
            return 0.0
            
        max_d2 = np.max(d2)
        if max_d2 < 1e-12:
            return 0.0
            
        # Scale to diameter 2
        # Current diameter D = sqrt(max_d2)
        # Scale factor s = 2 / D
        # New distances d'_ij = s * d_ij
        # Product of distances Delta = prod(d'_ij) = s^28 * prod(d_ij)
        # prod(d_ij) = exp(sum(log(d_ij))) = exp(0.5 * sum(log(d2)))
        # Score = Delta / 8^8
        
        log_prod_d = 0.5 * np.sum(np.log(d2))
        log_s = np.log(2.0) - 0.5 * np.log(max_d2)
        log_delta = 28.0 * log_s + log_prod_d
        log_score = log_delta - 8.0 * np.log(8.0)
        
        return np.exp(log_score)

    # Initial guesses
    inits = []
    
    # 1. Regular Octagon
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    p_reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(p_reg.flatten())
    
    # 2. Perturbed Regular Octagons
    rng = np.random.default_rng(42)
    for _ in range(15):
        noise = rng.normal(0, 0.2, (8, 2))
        p_pert = p_reg + noise
        inits.append(p_pert.flatten())
        
    # 3. Two Concentric Squares (more varied radii)
    for r1 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]:
        for r2 in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]:
            if abs(r1 - r2) < 0.1: continue
            a1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
            a2 = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
            p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
            p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
            p_two = np.vstack([p1, p2])
            inits.append(p_two.flatten())
            
    # 4. Deformed Octagon (varying angles)
    for amp in [0.1, 0.2, 0.3]:
        for _ in range(3):
            angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            pert = np.sin(angles * 2) * amp + rng.normal(0, 0.05, 8)
            angles_pert = angles + pert
            p_def = np.column_stack([np.cos(angles_pert), np.sin(angles_pert)])
            inits.append(p_def.flatten())

    best_x = inits[0]
    best_score = score_points(best_x.reshape(8, 2))
    best_val = objective(best_x)
    
    # Optimization Loop
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 15.0:
            break
            
        try:
            # Use L-BFGS-B for speed and robustness
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                cur_score = score_points(best_x.reshape(8, 2))
                if cur_score > best_score:
                    best_score = cur_score
                
                # Polish with Powell's method for extra precision
                if time.time() < deadline - 10.0:
                    res2 = minimize(
                        objective, 
                        best_x, 
                        method='Powell', 
                        options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-12}
                    )
                    if res2.fun < best_val:
                        best_val = res2.fun
                        best_x = res2.x
                        cur_score = score_points(best_x.reshape(8, 2))
                        if cur_score > best_score:
                            best_score = cur_score
                            
        except Exception:
            pass

    # Final polish with Nelder-Mead if time allows
    if time.time() < deadline - 5.0:
        try:
            res3 = minimize(
                objective, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-14}
            )
            if res3.fun < best_val:
                best_val = res3.fun
                best_x = res3.x
        except Exception:
            pass

    best_points = best_x.reshape(8, 2)
    return [[float(p[0]), float(p[1])] for p in best_points]
