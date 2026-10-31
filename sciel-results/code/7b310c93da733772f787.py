import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 145.0  # Safety margin for 150s limit

def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        # Use FFT for efficiency
        # Autoconvolution of h with itself
        # h has length N. We need full convolution.
        # Size needs to be at least 2*N - 1. Using power of 2 or 2*N is fine.
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        # The autoconvolution is symmetric. The peak is at lag 0 or near it.
        # We take the max over the whole array to be safe, though lag 0 is usually the max for non-negative h.
        return np.max(conv)

    i = np.arange(N, dtype=float)

    # --- GMM Parameterization ---
    K = 4  # Number of Gaussian components

    def gmm_to_h(params):
        """Convert GMM parameters to discrete heights on N intervals."""
        # params: [w_0, mu_0, sigma_0, w_1, mu_1, sigma_1, ...]
        h = np.zeros(N)
        for k in range(K):
            w_k = max(params[3*k], 0.01)
            mu_k = params[3*k+1]
            sigma_k = max(params[3*k+2], 1.0)
            h += w_k * np.exp(-((i - mu_k)**2) / (2 * sigma_k**2))
        h = np.maximum(h, 1e-10)
        h /= np.sum(h)
        return h

    def gmm_objective(params):
        h = gmm_to_h(params)
        return compute_autoconv_max(h)

    # Bounds for GMM parameters
    # w_k in [0.1, 5.0], mu_k in [0, N-1], sigma_k in [5, N/3]
    bounds = []
    for k in range(K):
        bounds.append((0.1, 5.0))   # weight
        bounds.append((0, N-1))      # mean
        bounds.append((5.0, N/3.0))  # sigma

    # Initial best: uniform
    best_h = np.ones(N) / N
    best_val = compute_autoconv_max(best_h)

    # --- Smooth Peak Objective with Gradient ---
    def smooth_peak_obj_and_grad(h, beta):
        """
        Smooth max autoconvolution with gradient, using temperature beta.
        Objective: m + (1/beta) * log(sum(exp((conv - m)*beta)))
        where m = max(conv)
        """
        h = np.maximum(h, 1e-10)
        h = h / np.sum(h)

        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real

        m = np.max(conv)
        
        # Numerical stability for log-sum-exp
        exp_conv = np.exp((conv - m) * beta)
        weights = exp_conv / np.sum(exp_conv)

        # Gradient calculation
        # d/dh_k (sum_j w_j conv_j) where w_j depends on conv_j
        # conv = ifft(H^2)
        # d conv / d h = 2 * ifft(H * W) where W = fft(weights)
        
        w_full = np.zeros(size)
        w_full[:len(conv)] = weights
        W = np.fft.fft(w_full, n=size)
        
        # grad_h = 2 * ifft(H * W).real
        # But we also need to account for the normalization h/sum(h) if we were doing that inside, 
        # but here we normalize h before calling, so we treat h as fixed sum=1 in the gradient step 
        # and rely on the optimizer to handle the manifold or we project back.
        # Actually, L-BFGS-B works on unconstrained variables. We should probably 
        # optimize on the simplex or use a transformation. 
        # The inspiration used L-BFGS-B directly on h, which is not on the simplex.
        # However, the objective function normalizes h internally. 
        # This means the gradient of the *normalized* objective is what we need.
        
        # Let f(h) = F(h/||h||). 
        # The gradient of F with respect to normalized h is grad_F.
        # The gradient with respect to raw h is (grad_F - h * sum(grad_F * h/||h||^2) ) / ||h|| ?
        # Simpler: Just compute the gradient of the normalized conv.
        
        # Let's stick to the inspiration's approach which seemed to work:
        # It computed grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
        # And used that directly. This implies it's optimizing a function where the normalization 
        # is handled by the objective function's internal normalization, and the gradient 
        # provided is the directional derivative in the direction of increasing h_k 
        # while keeping other h_j fixed, which is slightly inconsistent with the normalization 
        # inside the objective unless the optimizer is robust or the normalization is weak.
        
        # A safer approach for L-BFGS-B is to optimize log(h) or use a simplex projection.
        # But to stay close to the successful inspiration, let's use its gradient formula 
        # and ensure we project back to the simplex after each step or at the end.
        
        grad_h = 2.0 * np.fft.ifft(H * W).real[:N]
        
        val = m + np.log(np.sum(exp_conv)) / beta
        return val, grad_h

    # Stage 1: GMM Optimization to find a good initial shape
    # We will do a few restarts with different GMM inits and polish them.
    
    rng = np.random.RandomState(42)
    
    n_restarts = 5
    
    for r in range(n_restarts):
        if time_left() < 20:
            break
            
        # Generate random GMM parameters
        # Bias towards symmetric or specific structures
        if r == 0:
            # Uniform-like
            params0 = np.zeros(3*K)
            for k in range(K):
                params0[3*k] = 1.0
                params0[3*k+1] = N * (k + 0.5) / K
                params0[3*k+2] = N / (2*K)
        elif r == 1:
            # Bimodal
            params0 = np.zeros(3*K)
            for k in range(K):
                params0[3*k] = 1.0
                if k < K//2:
                    params0[3*k+1] = N * 0.25
                else:
                    params0[3*k+1] = N * 0.75
                params0[3*k+2] = N/10
        else:
            # Random
            params0 = np.zeros(3*K)
            for k in range(K):
                params0[3*k] = rng.uniform(0.5, 2.0)
                params0[3*k+1] = rng.uniform(0, N)
                params0[3*k+2] = rng.uniform(N/20, N/5)
        
        # Clip to bounds
        for k in range(K):
            params0[3*k] = np.clip(params0[3*k], bounds[3*k][0], bounds[3*k][1])
            params0[3*k+1] = np.clip(params0[3*k+1], bounds[3*k+1][0], bounds[3*k+1][1])
            params0[3*k+2] = np.clip(params0[3*k+2], bounds[3*k+2][0], bounds[3*k+2][1])
            
        h_init = gmm_to_h(params0)
        val_init = compute_autoconv_max(h_init)
        
        # If this random start is already better, keep it
        if val_init < best_val:
            best_val = val_init
            best_h = h_init.copy()
            
        # Polish this candidate with Smooth Peak L-BFGS-B
        # We use a sequence of betas (annealing)
        betas = [10.0, 50.0, 200.0, 500.0]
        
        current_h = h_init.copy()
        current_val = val_init
        
        for beta in betas:
            if time_left() < 5:
                break
                
            try:
                # Define the objective and gradient for this beta
                def obj_func(h):
                    return smooth_peak_obj_and_grad(h, beta)[0]
                
                def jac_func(h):
                    return smooth_peak_obj_and_grad(h, beta)[1]
                
                result = minimize(
                    obj_func,
                    current_h,
                    jac=jac_func,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
                )
                
                if result.success or result.fun < current_val:
                    final_h = np.maximum(result.x, 0)
                    if np.sum(final_h) > 0:
                        final_h = final_h / np.sum(final_h)
                        final_val = compute_autoconv_max(final_h)
                        if final_val < current_val:
                            current_val = final_val
                            current_h = final_h.copy()
            except Exception:
                pass

        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()

    # Stage 2: Local Refinement
    # The smooth optimization might leave us slightly off the true optimum due to the 
    # approximation of max. We do a few steps of direct local search.
    
    if time_left() > 10:
        h = best_h.copy()
        current_m = best_val
        local_deadline = time.time() + time_left() - 2.0
        
        np.random.seed(777)
        
        # Random pair moves
        while time.time() < local_deadline:
            i_idx = np.random.randint(N)
            j_idx = np.random.randint(N)
            if i_idx == j_idx:
                continue

            # Try moving a small fraction of mass from j to i
            # We try a few different fractions
            for frac in [0.01, 0.05, 0.1]:
                delta = h[j_idx] * frac
                if delta < 1e-12:
                    continue
                new_h = h.copy()
                new_h[j_idx] -= delta
                new_h[i_idx] += delta
                
                if new_h[j_idx] < 0:
                    continue
                    
                # Normalize
                s = np.sum(new_h)
                if s < 1e-15:
                    continue
                new_h = new_h / s
                
                m_new = compute_autoconv_max(new_h)
                if m_new < current_m:
                    current_m = m_new
                    h = new_h.copy()
                    if current_m < best_val:
                        best_val = current_m
                        best_h = h.copy()
                    break # If one move worked, restart the loop with new h

    # Final check
    if best_h is None or np.sum(best_h) < 1e-15:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / np.sum(best_h)

    return list(best_h)
