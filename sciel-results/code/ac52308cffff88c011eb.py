import numpy as np
import time
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    dw = 2.0 / N
    target = N / 2.0
    
    def compute_overlaps_vectorized(h):
        """Vectorized computation of all overlaps using broadcasting."""
        idx = np.arange(N)
        M = h[(idx[:, None] + idx[None, :]) % N]  # M[i,k] = h[(i+k)%N]
        mins = np.minimum(h[:, None], 1.0 - M)
        overlaps = np.sum(mins, axis=0) * dw
        return overlaps
    
    def project_to_feasible(h):
        """Project h to satisfy sum constraint and [0,1] bounds."""
        h = np.clip(h, 0, 1)
        diff = target - h.sum()
        if abs(diff) > 1e-10:
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if mask.sum() > 0:
                h[mask] += diff / mask.sum()
                h = np.clip(h, 0, 1)
        return h
    
    def objective(x):
        """Objective function for optimization: max overlap."""
        h = project_to_feasible(x)
        overlaps = compute_overlaps_vectorized(h)
        return overlaps.max()
    
    def objective_smooth(x, beta=50.0):
        """Smooth soft-max objective for better gradient-based optimization."""
        h = project_to_feasible(x)
        overlaps = compute_overlaps_vectorized(h)
        # Soft max: (1/beta) * log(sum(exp(beta * overlaps)))
        m = overlaps.max()
        return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta
    
    def objective_grad(x):
        """Gradient of the max overlap objective (vectorized)."""
        h = project_to_feasible(x)
        
        # Compute overlaps
        idx = np.arange(N)
        M = h[(idx[:, None] + idx[None, :]) % N]
        mins = np.minimum(h[:, None], 1.0 - M)
        overlaps = np.sum(mins, axis=0) * dw
        
        # Find the shift with maximum overlap
        k_max = np.argmax(overlaps)
        
        # Gradient with respect to h[i] for the max overlap shift
        # overlap_k = dw * sum_i min(h[i], 1 - h[(i+k)%N])
        h_shifted = np.roll(h, -k_max)
        
        # Where h[i] < 1-h_shifted[i], derivative w.r.t. h[i] is 1
        # Where h[i] > 1-h_shifted[i], derivative w.r.t. h[i] is 0
        # At equality, use 0.5
        a = h
        b = 1.0 - h_shifted
        grad_direct = np.where(a < b, 1.0, np.where(a > b, 0.0, 0.5))
        
        # For the "shifted" part: h[i] appears in term j where (j+k)%N = i, j = (i-k)%N
        # min(h[j], 1-h[i]): derivative w.r.t. h[i] is -1 if h[j] > 1-h[i], 0 otherwise
        j_idx = (np.arange(N) - k_max) % N
        a2 = h[j_idx]
        b2 = 1.0 - h
        grad_shifted = -np.where(a2 > b2, 1.0, np.where(a2 < b2, 0.0, 0.5))
        
        grad = (grad_direct + grad_shifted) * dw
        
        # Add projection gradient (approximate)
        h_clipped = np.clip(x, 0, 1)
        diff = target - h_clipped.sum()
        if abs(diff) > 1e-10:
            mask = (h_clipped > 1e-10) & (h_clipped < 1 - 1e-10)
            if mask.sum() > 0:
                grad[mask] += diff / mask.sum()
        
        return grad
    
    # Start with uniform solution
    best_h = np.full(N, 0.5)
    best_score = compute_overlaps_vectorized(best_h).max()
    
    # Try alternating patterns with optimized amplitude
    for A in np.linspace(0, 0.5, 50):
        if time.time() > DEADLINE:
            break
        h = 0.5 + A * (-1) ** np.arange(N)
        h = project_to_feasible(h)
        score = compute_overlaps_vectorized(h).max()
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Try various periodic patterns
    for period in [2, 3, 4, 5, 7, 10]:
        if time.time() > DEADLINE:
            break
        for A in np.linspace(0.1, 0.5, 20):
            if time.time() > DEADLINE:
                break
            base = np.zeros(period)
            base[:period//2] = 1.0
            h = np.tile(base, N // period + 1)[:N]
            h = 0.5 * h + 0.5 * A
            h = project_to_feasible(h)
            score = compute_overlaps_vectorized(h).max()
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Use Nelder-Mead optimization on the best candidate
    if time.time() < DEADLINE - 30:
        x0 = best_h.copy()
        rng = np.random.default_rng(42)
        for trial in range(5):
            if time.time() > DEADLINE - 10:
                break
            x0_trial = x0 + rng.normal(0, 0.05, N)
            x0_trial = np.clip(x0_trial, 0, 1)
            
            result = minimize(objective, x0_trial, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            h_opt = project_to_feasible(result.x)
            score = compute_overlaps_vectorized(h_opt).max()
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
    
    # BFGS with smooth objective for better convergence
    if time.time() < DEADLINE - 15:
        x0 = best_h.copy()
        result = minimize(objective_smooth, x0, method='L-BFGS-B',
                         options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        h_opt = project_to_feasible(result.x)
        score = compute_overlaps_vectorized(h_opt).max()
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Fast coordinate descent: only adjust one element at a time with optimal step
    if time.time() < DEADLINE - 5:
        for iteration in range(500):
            if time.time() > DEADLINE - 3:
                break
            improved = False
            h_trial = best_h.copy()
            for i in range(N):
                # Find optimal value for h[i] given others fixed
                # Sum of others is target - h[i], so we need h[i] such that total = target
                # This is already satisfied, so we just need to find best h[i] in [0,1]
                # that minimizes max overlap
                
                # Try a few candidate values
                candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                              best_h[i] - 0.1, best_h[i] - 0.05, best_h[i] + 0.05, best_h[i] + 0.1]
                candidates = [max(0, min(1, c)) for c in candidates]
                
                best_val = best_h[i]
                best_local_score = best_score
                
                for val in candidates:
                    if abs(val - best_h[i]) < 1e-6:
                        continue
                    h_test = h_trial.copy()
                    h_test[i] = val
                    # Re-project to maintain sum
                    h_test = project_to_feasible(h_test)
                    score = compute_overlaps_vectorized(h_test).max()
                    if score < best_local_score - 1e-10:
                        best_local_score = score
                        best_val = val
                
                if abs(best_val - h_trial[i]) > 1e-6:
                    h_trial[i] = best_val
                    improved = True
            
            # Final projection
            h_trial = project_to_feasible(h_trial)
            score = compute_overlaps_vectorized(h_trial).max()
            if score < best_score - 1e-10:
                best_score = score
                best_h = h_trial.copy()
            if not improved:
                break
    
    # Final high-precision polish
    if time.time() < DEADLINE - 2:
        x0 = best_h.copy()
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
        h_opt = project_to_feasible(result.x)
        score = compute_overlaps_vectorized(h_opt).max()
        if score < best_score:
            best_score = score
            best_h = h_opt.copy()
    
    # Final projection to ensure exact sum
    best_h = project_to_feasible(best_h)
    
    return best_h
