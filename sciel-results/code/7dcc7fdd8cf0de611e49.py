import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    n = 8
    _start_time = time.time()
    _DEADLINE = 145.0  # Safety margin
    
    # Regular octagon as initial best
    angles = [2 * math.pi * k / n for k in range(n)]
    reg_pts = [[math.cos(a), math.sin(a)] for a in angles]
    
    def score_points(pts):
        """
        Calculate the score for a set of points.
        pts: list of [x, y]
        Returns score = Delta / n^n after scaling to diameter 2.
        """
        pts = np.array(pts, dtype=float)
        n = len(pts)
        # Pairwise distances
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        diam = d.max()
        if diam <= 1e-15:
            return 0.0
        # Scale to diameter 2
        scaled_d = d / (diam / 2.0)
        # Product over ordered pairs (i != j)
        mask = ~np.eye(n, dtype=bool)
        # Use log to avoid underflow
        # Note: scaled_d[mask] contains distances for all i != j
        # If any distance is 0, log is -inf, score is 0
        if np.any(scaled_d[mask] <= 0):
            return 0.0
        log_prod = np.sum(np.log(scaled_d[mask]))
        # Score = exp(log_prod) / n^n
        log_score = log_prod - n * math.log(n)
        # Avoid overflow/underflow in exp
        if log_score > 700:
            return float('inf')
        if log_score < -700:
            return 0.0
        return math.exp(log_score)
    
    best_score = score_points(reg_pts)
    best_pts = reg_pts
    
    # Annealed softmax diameter surrogate for gradient-based optimization
    # We optimize in 16D: 8 points x 2 coordinates
    # Objective: maximize log(product of scaled distances) - n*log(n)
    # subject to diameter <= 2 (handled via scaling in the score)
    
    def softmax_max(x, beta):
        """Softmax approximation of max for vector x with temperature 1/beta"""
        x = np.asarray(x)
        x_max = x.max()
        # Avoid overflow
        exp_x = np.exp(beta * (x - x_max))
        return x_max + np.log(exp_x.sum()) / beta
    
    def neg_log_score_gradient(params, beta):
        """
        Negative log score using annealed softmax diameter.
        params: 16 floats (8 points x 2 coords)
        """
        pts = params.reshape(8, 2)
        
        # Compute pairwise distances
        diff = pts[:, None, :] - pts[None, :, :]
        d = np.sqrt(np.sum(diff * diff, axis=2))
        
        # Mask diagonal
        mask = ~np.eye(8, dtype=bool)
        d_off = d[mask]
        
        if np.any(d_off <= 1e-15):
            return 1e10
        
        # Softmax approximation of diameter
        diam = softmax_max(d_off, beta)
        
        # Scale distances so diameter = 2
        scaled_d = d_off / (diam / 2.0)
        
        # Log product
        log_prod = np.sum(np.log(scaled_d))
        log_score = log_prod - 8 * math.log(8)
        
        if log_score < -700:
            return 1e10
        return -log_score
    
    def neg_log_score_exact(params):
        """Exact negative log score for Nelder-Mead polishing"""
        pts = params.reshape(8, 2)
        s = score_points(pts)
        if s <= 0:
            return 1e10
        return -math.log(s)
    
    # Initial guess: Regular octagon flattened
    x0 = np.array([p[0] for p in reg_pts] + [p[1] for p in reg_pts])
    
    rng = np.random.RandomState(42)
    
    # Strategy 1: Gradient-based optimization with annealed softmax
    # Start with high beta (sharp max approximation) and anneal down
    betas = [1000, 500, 200, 100, 50, 20, 10, 5]
    
    current_x = x0.copy()
    
    for beta in betas:
        if time.time() > _DEADLINE - 30:
            break
        
        try:
            # Use L-BFGS-B with gradient approximation
            res = minimize(neg_log_score_gradient, current_x, args=(beta,),
                          method='L-BFGS-B',
                          options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-10})
            
            if res.fun < 1e9:
                current_x = res.x
                
                # Evaluate exact score
                cand_score = score_points(current_x.reshape(8, 2))
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = current_x.reshape(8, 2).tolist()
                    
        except Exception:
            pass
    
    # Strategy 2: Nelder-Mead polishing from gradient result
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(neg_log_score_exact, current_x, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            
            if res.fun < 1e9:
                cand_score = score_points(res.x.reshape(8, 2))
                if cand_score > best_score:
                    best_score = cand_score
                    best_pts = res.x.reshape(8, 2).tolist()
                    
        except Exception:
            pass
    
    # Strategy 3: Multiple random restarts with gradient + polish
    for restart in range(10):
        if time.time() > _DEADLINE - 10:
            break
        
        # Random perturbation of regular octagon
        noise_scale = 0.3 * (1 + restart * 0.2)
        x_start = x0 + rng.randn(16) * noise_scale
        
        # Quick gradient optimization
        try:
            res = minimize(neg_log_score_gradient, x_start, args=(50,),
                          method='L-BFGS-B',
                          options={'maxiter': 50, 'gtol': 1e-5})
            
            if res.fun < 1e9:
                # Polish with Nelder-Mead
                res2 = minimize(neg_log_score_exact, res.x, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                
                if res2.fun < 1e9:
                    cand_score = score_points(res2.x.reshape(8, 2))
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = res2.x.reshape(8, 2).tolist()
                        
        except Exception:
            pass
    
    # Final check and return
    # Ensure format is list of lists of floats
    result = [[float(p[0]), float(p[1])] for p in best_pts]
    return result

# EVOLVE-BLOCK-END
