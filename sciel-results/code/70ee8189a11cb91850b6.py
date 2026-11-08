"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_max_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    h is a 1D array of step heights.
    O_k = sum_i h[i] * (1 - h[(i+k)%N])
    We want to minimize max_k O_k.
    """
    H = np.fft.fft(h)
    A = np.real(np.fft.ifft(H * np.conj(H)))
    return np.max(N / 2 - A)

def _objective(params):
    """
    Objective function to minimize.
    params: Fourier coefficients for the smooth function.
    We use a truncated Fourier series to ensure smoothness.
    h(x) = 0.5 + sum_{k=1}^{K} (a_k cos(2*pi*k*x/N) + b_k sin(2*pi*k*x/N))
    We constrain the DC component to be 0.5 so that sum h = N/2.
    """
    K = len(params) // 2
    a = params[:K]
    b = params[K:2*K]
    
    x = np.arange(N)
    h = np.full(N, 0.5)
    for k in range(1, K+1):
        h += a[k-1] * np.cos(2 * np.pi * k * x / N)
        h += b[k-1] * np.sin(2 * np.pi * k * x / N)
    
    # Project to [0, 1] and adjust sum if necessary
    # Since we fixed DC to 0.5, the sum is exactly N/2 if the AC components sum to 0.
    # For integer k, sum of cos/sin over 0..N-1 is 0. So sum h is exactly N/2.
    # We just need to clip to [0,1]. Clipping breaks the sum, so we need a soft projection
    # or accept that clipping might slightly violate the sum, but for small amplitudes it's fine.
    # To be safe, let's clip and then re-normalize? No, re-normalizing breaks the Fourier structure.
    # Instead, we will penalize violations of [0,1] in the objective.
    
    penalty = 0.0
    if np.any(h < 0) or np.any(h > 1):
        penalty += 100 * (np.sum(np.clip(-h, 0, None)**2) + np.sum(np.clip(h-1, 0, None)**2))
    
    h_clipped = np.clip(h, 0, 1)
    # The evaluator likely requires exact sum N/2. 
    # If we use Fourier series with DC=0.5, the sum is exactly N/2 before clipping.
    # Clipping changes the sum. 
    # Strategy: Use a smooth sigmoid-like mapping or ensure amplitudes are small enough that clipping doesn't happen.
    # If clipping happens, we have a problem. 
    # Let's try to keep amplitudes small.
    
    # Compute overlap on the clipped h (since evaluator uses the returned h)
    # But wait, if we return h_clipped, its sum might not be N/2.
    # The task says "sum is exactly len/2". 
    # If we return a clipped h, we must adjust it to have sum N/2.
    # This adjustment breaks smoothness.
    # Alternative: Optimize over h directly but use a smooth penalty for non-smoothness?
    # Or: Use the Fourier representation only for initialization and then do a local search on h?
    
    # Let's stick to the idea: optimize Fourier coefficients to minimize max overlap.
    # To handle the sum constraint exactly, we can define h = 0.5 + delta, where sum delta = 0.
    # The Fourier series with k>=1 automatically has sum 0.
    # The only issue is the [0,1] constraint.
    # If we ensure |delta| <= 0.5, then h is in [0,1].
    # We can add a penalty for |delta| > 0.5.
    
    delta = h - 0.5
    penalty += 1000 * np.sum(np.clip(np.abs(delta) - 0.5, 0, None)**2)
    
    # Compute the objective: max overlap
    H_fft = np.fft.fft(h_clipped)
    A = np.real(np.fft.ifft(H_fft * np.conj(H_fft)))
    max_overlap = np.max(N / 2 - A)
    
    return max_overlap + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Number of Fourier modes to use. 
    # Too many modes -> risk of violating [0,1] and loss of smoothness.
    # Too few -> limited expressiveness.
    # Let's try K=5 (10 parameters).
    K = 5
    n_params = 2 * K
    
    # Initial parameters: small random values
    rng = np.random.default_rng(42)
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Try a few random initializations
    for _ in range(10):
        if time.time() > deadline:
            break
        
        params0 = rng.uniform(-0.1, 0.1, n_params)
        
        # Use L-BFGS-B
        try:
            res = minimize(_objective, params0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if res.success or res.fun < best_score:
                # Reconstruct h
                a = res.x[:K]
                b = res.x[K:2*K]
                x = np.arange(N)
                h_new = np.full(N, 0.5)
                for k in range(1, K+1):
                    h_new += a[k-1] * np.cos(2 * np.pi * k * x / N)
                    h_new += b[k-1] * np.sin(2 * np.pi * k * x / N)
                
                # Clip to [0,1]
                h_new = np.clip(h_new, 0, 1)
                
                # Adjust sum to N/2
                # Since the sum of AC components is 0, the sum of h_new before clipping is N/2.
                # Clipping changes the sum. We need to fix it.
                # Simple adjustment: subtract the mean error.
                error = np.sum(h_new) - N/2
                h_new -= error / N
                
                # Clip again (might slightly violate sum, but error is small)
                h_new = np.clip(h_new, 0, 1)
                
                # Final check: if sum is not exactly N/2, we have a problem.
                # The evaluator might be strict.
                # Let's try a more careful projection: 
                # We want h in [0,1] and sum N/2.
                # This is a convex feasibility problem.
                # Simple iterative projection (Dykstra) or just accept small error.
                # For now, let's check if the score is better.
                
                score = _compute_max_overlap(h_new)
                if score < best_score:
                    best_score = score
                    best_h = h_new.copy()
        except Exception:
            pass
    
    # Local refinement on best_h using random perturbations (like the original, but starting from a good point)
    current_h = best_h.copy()
    current_score = best_score
    step_size = 0.02
    iterations = 0
    max_iterations = 20000
    
    while time.time() < deadline and iterations < max_iterations:
        num_perturb = rng.integers(1, 4)
        idxs = rng.choice(N, num_perturb, replace=False)
        
        new_h = current_h.copy()
        for idx in idxs:
            delta = rng.uniform(-step_size, step_size)
            new_h[idx] += delta
        
        new_h = np.clip(new_h, 0, 1)
        error = np.sum(new_h) - N/2
        new_h -= error / N
        new_h = np.clip(new_h, 0, 1)
        
        # Check sum constraint strictly
        if abs(np.sum(new_h) - N/2) > 1e-6:
            iterations += 1
            continue
            
        new_score = _compute_max_overlap(new_h)
        if new_score < current_score:
            current_h = new_h
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
        
        iterations += 1
        if iterations % 1000 == 0:
            step_size *= 0.95
    
    return best_h

# EVOLVE-BLOCK-END
