import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import fft, ifft

N = 100

def construct():
    start_time = time.time()
    deadline = 100.0 # seconds
    
    # Initial guess: Random binary sequence with 50 ones
    np.random.seed(42)
    h0 = np.zeros(N)
    indices = np.random.choice(N, N//2, replace=False)
    h0[indices] = 1.0
    
    def autocorrelation(h):
        # Zero pad to 2N to get linear autocorrelation
        h_padded = np.zeros(2*N)
        h_padded[:N] = h
        H = fft(h_padded)
        acf = ifft(H * np.conj(H)).real
        # acf[k] for k=0..N-1 is sum h[i] h[i+k]
        # We want k=1..N-1
        return acf[1:N]
    
    def objective(h):
        # h is flat array
        # Ensure constraints are roughly met during optimization?
        # SLSQP handles constraints, but we can add a penalty or just rely on it.
        # For robustness, let's calculate the raw objective.
        S = autocorrelation(h)
        # Minimize sum of squares to flatten
        return np.sum(S**2)
        
    def constraint_sum(h):
        return np.sum(h) - N/2.0
        
    bounds = [(0, 1) for _ in range(N)]
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    
    best_h = h0.copy()
    best_score = np.inf
    
    # Calculate initial score
    S0 = autocorrelation(best_h)
    best_score = np.max(S0)
    
    # Run optimization
    # SLSQP might be slow for 100 variables?
    # Let's try L-BFGS-B with projection?
    # No, SLSQP is fine for N=100.
    
    try:
        res = minimize(objective, h0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 200, 'ftol': 1e-8})
        
        if res.success or res.fun < objective(h0):
            candidate = res.x
            # Project candidate to ensure strict constraints
            # SLSQP should satisfy them, but let's verify
            # If sum is off, we might need to adjust
            S_cand = autocorrelation(candidate)
            cand_score = np.max(S_cand)
            if cand_score < best_score:
                best_h = candidate
                best_score = cand_score
                
    except Exception as e:
        pass
        
    # Check time
    if time.time() - start_time < deadline:
        # Try a few random restarts if time permits
        for _ in range(5):
            if time.time() - start_time > deadline - 10:
                break
            h0_new = np.zeros(N)
            indices = np.random.choice(N, N//2, replace=False)
            h0_new[indices] = 1.0
            
            try:
                res = minimize(objective, h0_new, method='SLSQP', bounds=bounds, constraints=constraints,
                               options={'maxiter': 100, 'ftol': 1e-8})
                if res.success:
                    candidate = res.x
                    S_cand = autocorrelation(candidate)
                    cand_score = np.max(S_cand)
                    if cand_score < best_score:
                        best_h = candidate
                        best_score = cand_score
            except:
                pass

    # Final check on best_h
    # Ensure sum is exactly N/2
    # SLSQP might have small errors.
    # Adjust best_h to satisfy sum exactly.
    # Simple adjustment: distribute the error.
    err = N/2.0 - np.sum(best_h)
    if abs(err) > 1e-6:
        # Add err/N to all, then clip and re-adjust?
        # This might break bounds.
        # Better: just return best_h, the evaluator might tolerate small errors?
        # "sum is exactly len/2"
        # Let's do a precise projection.
        
        # Projection onto sum=h and 0<=h<=1
        # Use bisection on lambda
        def project(h, target_sum):
            # h is current
            # find lambda such that sum(clamp(h - lambda, 0, 1)) = target_sum
            lo, hi = -10, 10
            for _ in range(50):
                mid = (lo + hi) / 2
                h_proj = np.clip(h - mid, 0, 1)
                s = np.sum(h_proj)
                if s > target_sum:
                    lo = mid
                else:
                    hi = mid
            return np.clip(h - (lo+hi)/2, 0, 1)
            
        best_h = project(best_h, N/2.0)
        
    return best_h

# EVOLVE-BLOCK-END
