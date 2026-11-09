import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 8
    num_pairs = n * (n - 1)  # 56 ordered pairs

    def compute_score(points):
        # points is shape (n, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-10)
        
        mask = ~np.eye(n, dtype=bool)
        max_dist = np.max(dists[mask])
        
        if max_dist == 0:
            return -np.inf
            
        log_prod_ordered = np.sum(np.log(dists[mask]))
        
        # Objective to maximize:
        # K = log_prod_ordered - 56 * log(max_dist)
        # We minimize -K
        return log_prod_ordered - 56 * np.log(max_dist)

    def objective_soft(x, beta):
        points = x.reshape(n, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        dists = np.maximum(dists, 1e-10)
        
        mask = ~np.eye(n, dtype=bool)
        
        # Soft-max approximation for diameter
        # log(sum(exp(beta * dists))) / beta approximates max(dists)
        # Using log-sum-exp trick for stability
        d_masked = dists[mask]
        max_d = np.max(d_masked)
        if max_d < 1e-10:
            return 1e10
            
        # log(sum(exp(beta * d))) = max_d * beta + log(sum(exp(beta * (d - max_d))))
        # soft_max = (max_d * beta + log(sum(exp(beta * (d - max_d))))) / beta
        # = max_d + log(sum(exp(beta * (d - max_d)))) / beta
        
        beta_d = beta * (d_masked - max_d)
        # Clip to avoid overflow in exp
        beta_d = np.clip(beta_d, -50, 50)
        
        log_sum_exp = np.log(np.sum(np.exp(beta_d)))
        soft_max = max_d + log_sum_exp / beta
        
        log_prod_ordered = np.sum(np.log(d_masked))
        
        # Score to maximize: log_prod - 56 * soft_max
        # Minimize negative
        score = log_prod_ordered - 56 * soft_max
        if not np.isfinite(score):
            return 1e10
        return -score

    # Initial candidates
    candidates = []
    
    # 1. Regular Octagon
    angles = 2 * np.pi * np.arange(n) / n
    p_oct = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(p_oct.flatten())
    
    # 2. Symmetric Configuration (D2 symmetry)
    def sym_points(params):
        x1, y1, x2, y2 = params
        x1, y1, x2, y2 = abs(x1), abs(y1), abs(x2), abs(y2)
        p1 = np.array([x1, y1])
        p2 = np.array([-x1, y1])
        p3 = np.array([-x1, -y1])
        p4 = np.array([x1, -y1])
        p5 = np.array([x2, y2])
        p6 = np.array([-x2, y2])
        p7 = np.array([-x2, -y2])
        p8 = np.array([x2, -y2])
        return np.vstack([p1, p2, p3, p4, p5, p6, p7, p8])

    rng = np.random.default_rng(42)
    
    # Generate random symmetric candidates
    for _ in range(30):
        x1 = rng.uniform(0.5, 1.5)
        y1 = rng.uniform(0.5, 1.5)
        x2 = rng.uniform(0.5, 1.5)
        y2 = rng.uniform(0.5, 1.5)
        p_sym = sym_points([x1, y1, x2, y2])
        candidates.append(p_sym.flatten())
        
    # 3. Random perturbations of octagon
    for _ in range(30):
        noise = rng.normal(0, 0.1, size=(n, 2))
        p_rand = p_oct + noise
        candidates.append(p_rand.flatten())
        
    # 4. Random points
    for _ in range(30):
        p_rand = rng.uniform(-1, 1, size=(n, 2))
        candidates.append(p_rand.flatten())

    best_score = -np.inf
    best_points = None

    # Shuffle candidates
    idx = np.arange(len(candidates))
    rng.shuffle(idx)
    
    # Phase 1: Coarse optimization with soft-max (low beta)
    betas = [5, 10, 20]
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
            
        for i in idx:
            if time.time() > deadline - 30:
                break
                
            x0 = candidates[i]
            try:
                # Use L-BFGS-B for gradient-based optimization with soft objective
                res = minimize(objective_soft, x0, args=(beta,), method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                
                final_points = res.x.reshape(n, 2)
                final_score = compute_score(final_points)
                
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points
                    
            except Exception:
                pass
        
        # Add best point to candidates for next round if improved
        if best_points is not None:
            candidates.append(best_points.flatten())
            # Re-shuffle or just add to end? Adding to end is fine, we iterate all.
            # Actually, we should probably restart from the best point in the next beta phase
            # But the loop structure above iterates all candidates for each beta.
            # This might be inefficient if we have many candidates.
            # Let's refine: only use the top few candidates for higher betas.
            # But for now, let's stick to this simple structure and rely on the deadline.
            # To be more efficient, we can truncate candidates to top 10 by score after each beta phase.
            # Let's do that.
            scores = []
            for c in candidates:
                p = c.reshape(n, 2)
                s = compute_score(p)
                scores.append(s)
            top_indices = np.argsort(scores)[::-1][:10]
            candidates = [candidates[i] for i in top_indices]

    # Phase 2: Fine-tuning with Nelder-Mead on the true non-smooth objective
    # Starting from the best point found in Phase 1
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        try:
            res = minimize(lambda x: -compute_score(x.reshape(n, 2)), x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            final_points = res.x.reshape(n, 2)
            final_score = compute_score(final_points)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_points
        except Exception:
            pass

    if best_points is None:
        return [[float(x), float(y)] for x, y in p_oct]
        
    return [[float(x), float(y)] for x, y in best_points]
