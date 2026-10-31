import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer

    def project_to_polytope(h, S):
        """Project h onto { x : sum(x) = S, 0 <= x_i <= 1 }"""
        if S < 0 or S > N:
            return np.clip(h, 0, 1)
            
        lo = np.min(h) - 1.0
        hi = np.max(h) + 1.0
        
        for _ in range(50):
            mid = (lo + hi) / 2.0
            val = np.sum(np.clip(h - mid, 0.0, 1.0))
            if val > S:
                lo = mid
            else:
                hi = mid
                
        lambda_star = (lo + hi) / 2.0
        return np.clip(h - lambda_star, 0.0, 1.0)

    def compute_overlaps(h):
        overlaps = np.zeros(N)
        for k in range(N):
            shifted = np.roll(h, -k)
            overlaps[k] = N/2 - np.dot(h, shifted)
        return overlaps

    def smooth_max_grad(h, beta):
        overlaps = compute_overlaps(h)
        
        max_o = np.max(overlaps)
        exps = np.exp(beta * (overlaps - max_o))
        sum_exps = np.sum(exps)
        
        weights = exps / sum_exps
        
        val = max_o + np.log(sum_exps) / beta
        
        term_sum = np.zeros(N)
        for k in range(N):
            if weights[k] > 1e-10:
                term_sum += weights[k] * (np.roll(h, -k) + np.roll(h, k))
                
        grad = 1.0 - term_sum
        return val, grad

    def objective_and_grad(x, beta):
        h = project_to_polytope(x, N/2)
        val, grad = smooth_max_grad(h, beta)
        # We need to account for the projection in the gradient, but for simplicity
        # we use the gradient of the smooth max directly and project
        return val, grad

    best_h = None
    best_score = float('inf')
    
    num_restarts = 15
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Diverse initialization
        if r == 0:
            h = np.full(N, 0.5)
        elif r == 1:
            h = 0.5 + 0.3 * np.sin(2 * np.pi * 1 * np.arange(N) / N)
        elif r == 2:
            h = 0.5 + 0.3 * np.cos(2 * np.pi * 2 * np.arange(N) / N)
        else:
            freq = 1 + (r - 3) % (N//2)
            h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N + 0.1 * r)
            h += 0.03 * np.random.randn(N)
        
        h = project_to_polytope(h, N/2)
        
        # Phase 1: Coarse optimization with moderate beta
        beta = 5.0
        alpha = 0.02
        max_steps = 150
        
        for step in range(max_steps):
            if time.time() > deadline - 15:
                break
                
            val, grad = smooth_max_grad(h, beta)
            actual_overlaps = compute_overlaps(h)
            current_max = np.max(actual_overlaps)
            
            if current_max < best_score:
                best_score = current_max
                best_h = h.copy()
                
            h = h - alpha * grad
            h = project_to_polytope(h, N/2)
            
            beta *= 1.03
            alpha *= 0.995
            
            if np.max(np.abs(grad)) < 1e-7:
                break

        # Phase 2: Fine polish with high beta
        if time.time() < deadline - 5:
            beta = 50.0
            alpha = 0.005
            max_steps = 100
            
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                    
                val, grad = smooth_max_grad(h, beta)
                actual_overlaps = compute_overlaps(h)
                current_max = np.max(actual_overlaps)
                
                if current_max < best_score:
                    best_score = current_max
                    best_h = h.copy()
                    
                h = h - alpha * grad
                h = project_to_polytope(h, N/2)
                
                beta *= 1.02
                alpha *= 0.997
                
                if np.max(np.abs(grad)) < 1e-8:
                    break

    if best_h is None:
        return np.full(N, 0.5)
    
    # Final high-precision polish using scipy
    if time.time() < deadline - 3:
        h = best_h.copy()
        
        # Use L-BFGS-B with bounds
        def obj(x):
            hh = project_to_polytope(x, N/2)
            overlaps = compute_overlaps(hh)
            return np.max(overlaps)
        
        # Try a few local optimizations from different starting points
        for trial in range(3):
            if time.time() > deadline - 1:
                    break
                    
            if trial == 0:
                x0 = h.copy()
            else:
                x0 = h + 0.01 * np.random.randn(N)
                x0 = project_to_polytope(x0, N/2)
                
            try:
                res = minimize(obj, x0, method='L-BFGS-B', bounds=[(0, 1)] * N, 
                              options={'maxiter': 200, 'ftol': 1e-12})
                if res.success or res.fun < best_score:
                    h_candidate = project_to_polytope(res.x, N/2)
                    actual_overlaps = compute_overlaps(h_candidate)
                    if np.max(actual_overlaps) < best_score:
                        best_score = np.max(actual_overlaps)
                        best_h = h_candidate.copy()
            except:
                pass

    best_h = project_to_polytope(best_h, N/2)
    
    return best_h
