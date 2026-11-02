import numpy as np
from scipy.optimize import minimize
import time

def construct():
    N = 100
    n_half = N // 2
    target_sum = N / 2.0
    target_param_sum = N / 4.0  # Since h is symmetric, sum(h) = 2 * sum(params)
    
    start_time = time.time()
    deadline = 150.0
    margin = 5.0
    
    def time_remaining():
        return deadline - (time.time() - start_time)

    def build_h(params):
        """Construct full symmetric h from half params."""
        h = np.zeros(N)
        h[:n_half] = params
        h[n_half:] = params[::-1]
        return h

    def compute_correlations(h):
        """Compute linear autocorrelations A_k for k=1..N-1."""
        n = len(h)
        corrs = np.zeros(n - 1)
        for k in range(1, n):
            # A_k = sum_{i=0}^{n-1-k} h[i] h[i+k]
            corrs[k-1] = np.dot(h[:n-k], h[k:])
        return corrs

    def soft_min_obj_and_grad(params, beta):
        """
        Objective: Minimize 1 - soft_min(corrs)
        This is equivalent to Maximizing soft_min(corrs)
        """
        h = build_h(params)
        corrs = compute_correlations(h)
        
        # Soft min of corrs
        # S = -1/beta * log(sum(exp(-beta * corrs)))
        # To avoid overflow, subtract max (which is min of -corrs, so max of corrs? No.)
        # exp(-beta * corrs). To be stable, let m = max(corrs).
        # exp(-beta * (corrs - m)) = exp(-beta * corrs) * exp(beta * m)
        # log(sum(exp(-beta * corrs))) = beta * m + log(sum(exp(-beta * (corrs - m))))
        # S = -1/beta * (beta * m + log(sum(...))) = -m - 1/beta * log(sum(exp(-beta * (corrs - m))))
        
        m = np.max(corrs)
        exp_terms = np.exp(-beta * (corrs - m))
        log_sum = np.log(np.sum(exp_terms))
        soft_min_val = -m - log_sum / beta
        
        obj = 1.0 - soft_min_val
        
        # Gradient of obj w.r.t params
        # d(obj)/d(params) = - d(soft_min)/d(params)
        
        # Weights for soft min
        # w_k = exp(-beta * corrs_k) / sum(exp(-beta * corrs))
        # w_k = exp(-beta * (corrs_k - m)) / sum(exp(-beta * (corrs_j - m)))
        weights = exp_terms / np.sum(exp_terms)
        
        # Gradient of soft_min w.r.t h
        # d(soft_min)/d(h) = sum_k w_k * d(A_k)/d(h)
        grad_h = np.zeros(N)
        for k in range(1, N):
            w = weights[k-1]
            if w == 0:
                continue
            # d(A_k)/d(h_i) = h_{i+k} (if i < N-k) + h_{i-k} (if i >= k)
            # Vectorized:
            # For i in 0..N-1-k: add h[i+k]
            grad_h[:N-k] += w * h[k:]
            # For i in k..N-1: add h[i-k]
            grad_h[k:] += w * h[:N-k]
            
        # d(obj)/d(h) = - grad_h
        grad_h = -grad_h
        
        # Map grad_h to grad_params
        # h[i] = params[i] for i < N/2
        # h[N-1-i] = params[i]
        # d(params[i])/d(h) is not direct, but:
        # obj = f(h)
        # d(obj)/d(params[i]) = sum_j (d(obj)/d(h_j)) * (d(h_j)/d(params[i]))
        # h_i = params_i => d(h_i)/d(params_i) = 1
        # h_{N-1-i} = params_i => d(h_{N-1-i})/d(params_i) = 1
        # So grad_params[i] = grad_h[i] + grad_h[N-1-i]
        
        grad_params = grad_h[:n_half] + grad_h[N-1:n_half-1:-1]
        
        return obj, grad_params

    def project_params(params):
        """Project params to satisfy sum = target_param_sum and bounds [0,1]."""
        p = params.copy()
        p = np.clip(p, 0.0, 1.0)
        diff = target_param_sum - np.sum(p)
        if abs(diff) > 1e-12:
            # Iterate projection
            for _ in range(10):
                mask = (p > 0.0) & (p < 1.0)
                if np.sum(mask) == 0:
                    break
                p[mask] += diff / np.sum(mask)
                p = np.clip(p, 0.0, 1.0)
                diff = target_param_sum - np.sum(p)
                if abs(diff) < 1e-12:
                    break
        return p

    best_params = np.full(n_half, 0.5)
    best_obj = float('inf')
    
    # Initial evaluation
    h_init = build_h(best_params)
    corrs_init = compute_correlations(h_init)
    # Approximate min for initial score
    best_obj = 1.0 - np.min(corrs_init)

    np.random.seed(42)
    
    # Generate restarts
    restarts = []
    restarts.append(np.full(n_half, 0.5))
    
    # Sinusoids
    x = np.linspace(0, 2*np.pi, n_half)
    restarts.append(0.5 + 0.3*np.sin(x))
    restarts.append(0.5 + 0.3*np.cos(x))
    restarts.append(0.5 + 0.4*np.sin(2*x))
    restarts.append(0.5 + 0.4*np.cos(2*x))
    
    # Linear ramps
    restarts.append(np.linspace(0.2, 0.8, n_half))
    restarts.append(np.linspace(0.8, 0.2, n_half))
    
    # Random
    for _ in range(30):
        restarts.append(np.random.uniform(0, 1, n_half))
        
    # Block structures
    for k in range(1, n_half):
        p = np.full(n_half, 0.5)
        p[:k] = 1.0
        p[k:] = 0.0
        # Adjust to sum
        restarts.append(p)
        
    # Optimization loop
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    for x0 in restarts:
        if time_remaining() < 10.0:
            break
            
        params = project_params(x0)
        
        # Multi-stage beta
        current_params = params
        for beta in betas:
            if time_remaining() < 5.0:
                break
                
            try:
                res = minimize(
                    soft_min_obj_and_grad,
                    current_params,
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(0, 1)] * n_half,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False}
                )
                
                if res.fun < best_obj:
                    # Verify with hard min
                    new_params = project_params(res.x)
                    h_new = build_h(new_params)
                    corrs_new = compute_correlations(h_new)
                    hard_min = np.min(corrs_new)
                    obj_new = 1.0 - hard_min
                    
                    if obj_new < best_obj:
                        best_obj = obj_new
                        best_params = new_params
                        current_params = new_params
            except Exception:
                pass

    # Final polish if time permits
    if time_remaining() > 10.0:
        # Try to optimize the full vector without symmetry constraint?
        # Or just do a high-precision symmetric optimization on the best found.
        try:
            res = minimize(
                soft_min_obj_and_grad,
                best_params,
                args=(1000.0,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * n_half,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'disp': False}
            )
            new_params = project_params(res.x)
            h_new = build_h(new_params)
            corrs_new = compute_correlations(h_new)
            obj_new = 1.0 - np.min(corrs_new)
            
            if obj_new < best_obj:
                best_obj = obj_new
                best_params = new_params
        except Exception:
            pass

    final_h = build_h(best_params)
    
    # Final check and correction
    # Ensure sum is exactly N/2
    diff = target_sum - np.sum(final_h)
    if abs(diff) > 1e-9:
        # Adjust non-bound elements
        mask = (final_h > 1e-6) & (final_h < 1 - 1e-6)
        if np.sum(mask) > 0:
            final_h[mask] += diff / np.sum(mask)
            final_h = np.clip(final_h, 0, 1)
            
    # If still invalid, fallback to constant
    if np.sum(final_h) != target_sum or np.any(final_h < 0) or np.any(final_h > 1):
        final_h = np.full(N, 0.5)
        
    return final_h
