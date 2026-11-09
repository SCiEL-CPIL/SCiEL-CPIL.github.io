import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin

    # Initial guess: Regular Octagon
    # But we know it's a local minimum (or saddle) for the "max product" problem?
    # Actually, regular polygon is a critical point.
    
    # Let's try a configuration that is known to be better.
    # For n=8, a "flattened" octagon or two distinct rings.
    
    # We will optimize the 16 coordinates.
    # Objective: Maximize sum(log(dist)) - 8 * log(max_dist)
    # To avoid non-smoothness, we use a penalty for max_dist > 2.
    # Or better: Scale the points to diameter 2, then maximize log product.
    # Scaling makes the objective invariant to scale, so we can just maximize log product
    # and then normalize. The score is Delta / 8^8.
    # Delta = product of distances.
    # Score = (Product / D^8) / 8^8 ? No.
    # "Score is Delta/n^n, where Delta is the product ... after scaling the set to diameter 2".
    # So if we have points P, we compute D = max dist.
    # Scaled points P' = P / (D/2).
    # Delta = prod(dist(P'_i, P'_j)).
    # Score = Delta / 8^8.
    
    # Note: dist(P'_i, P'_j) = dist(P_i, P_j) / (D/2) = 2 * dist(P_i, P_j) / D.
    # Delta = prod(2 * d_ij / D) = 2^28 * (prod d_ij) / D^28.
    # Score = (2^28 * prod(d_ij) / D^28) / 8^8.
    # 8^8 = (2^3)^8 = 2^24.
    # Score = 2^4 * prod(d_ij) / D^28 = 16 * prod(d_ij) / D^28.
    
    # We want to maximize this.
    # Equivalent to maximizing log(prod(d_ij)) - 28 * log(D).
    
    def objective(params):
        P = params.reshape(8, 2)
        # Compute all pairwise distances
        # d_ij for i < j
        dists = []
        max_d = 0.0
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(P[i] - P[j])
                dists.append(d)
                if d > max_d:
                    max_d = d
        
        if max_d == 0:
            return 0.0
            
        log_prod = sum(np.log(d) for d in dists if d > 0)
        # Penalty if max_d is very small (degenerate)
        # We want to maximize log_prod - 28 * log(max_d)
        # Minimizing negative of that
        val = log_prod - 28.0 * np.log(max_d)
        return -val

    # Initial guesses
    guesses = []
    
    # 1. Regular Octagon
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    P0 = np.column_stack([np.cos(angles), np.sin(angles)])
    guesses.append(P0.flatten())
    
    # 2. Two squares, r1=1, r2=0.9, offset pi/4
    P1 = np.zeros((8,2))
    for k in range(4):
        a1 = k * np.pi/2
        a2 = k * np.pi/2 + np.pi/4
        P1[2*k] = [1.0 * np.cos(a1), 1.0 * np.sin(a1)]
        P1[2*k+1] = [0.9 * np.cos(a2), 0.9 * np.sin(a2)]
    guesses.append(P1.flatten())
    
    # 3. Two squares, r1=1, r2=0.8, offset pi/4
    P2 = np.zeros((8,2))
    for k in range(4):
        a1 = k * np.pi/2
        a2 = k * np.pi/2 + np.pi/4
        P2[2*k] = [1.0 * np.cos(a1), 1.0 * np.sin(a1)]
        P2[2*k+1] = [0.8 * np.cos(a2), 0.8 * np.sin(a2)]
    guesses.append(P2.flatten())
    
    # 4. Perturbed regular octagon
    P3 = P0 + 0.1 * np.random.normal(0, 1, (8,2))
    guesses.append(P3.flatten())
    
    best_score = -np.inf
    best_points = None
    
    for guess in guesses:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success or res.fun < best_score:
                # Calculate actual score
                P_opt = res.x.reshape(8, 2)
                dists = []
                max_d = 0.0
                for i in range(8):
                    for j in range(i+1, 8):
                        d = np.linalg.norm(P_opt[i] - P_opt[j])
                        dists.append(d)
                        if d > max_d:
                            max_d = d
                if max_d > 1e-6:
                    log_prod = sum(np.log(d) for d in dists)
                    score_val = 16.0 * np.exp(log_prod) / (max_d ** 28)
                    # The score definition in prompt: Delta / n^n
                    # Delta is product of distances after scaling to diameter 2.
                    # My derived formula: Score = 16 * prod(d) / D^28.
                    # Let's verify.
                    # Scaled dist = 2d/D.
                    # Delta = prod(2d/D) = 2^28 prod(d) / D^28.
                    # Score = Delta / 8^8 = 2^28 prod(d) / D^28 / 2^24 = 2^4 prod(d) / D^28 = 16 prod(d) / D^28.
                    # Correct.
                    
                    if score_val > best_score:
                        best_score = score_val
                        best_points = P_opt
        except Exception:
            pass

    # If optimization failed or didn't improve, fall back to regular octagon
    if best_points is None or best_score < 1.0:
        best_points = P0
        best_score = 1.0

    # Normalize best_points to diameter 2
    # Find max dist
    max_d = 0.0
    for i in range(8):
        for j in range(i+1, 8):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d > max_d:
                max_d = d
        
    if max_d > 1e-9:
        scale_factor = 2.0 / max_d
        best_points = best_points * scale_factor
    else:
        # Degenerate, return regular octagon
        best_points = P0

    # Convert to list of lists
    return [[float(x), float(y)] for x, y in best_points]
