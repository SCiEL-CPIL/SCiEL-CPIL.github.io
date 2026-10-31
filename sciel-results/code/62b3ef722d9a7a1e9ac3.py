import numpy as np
import time
from scipy import optimize
from scipy.fft import fft, ifft

N = 600

def _fft_convolve(h):
    """FFT-based autoconvolution for speed."""
    n = len(h)
    size = 2 * n
    H = fft(h, size)
    conv = np.real(ifft(H * H, size)[:2*n-1])
    return conv

def compute_autoconv_max(h):
    """Compute the normalized maximum of the autoconvolution of step function h."""
    h = np.asarray(h, dtype=float)
    conv = _fft_convolve(h)
    total = sum(h)
    if total < 1e-10:
        return 0.0
    return max(conv) / (total * total)

def _smooth_max_obj_grad(h, beta):
    """
    Compute smooth max autoconvolution value and gradient w.r.t. h.
    Uses log-sum-exp with temperature 1/beta.
    Assumes h is normalized to sum=1 for the objective value,
    but gradient is computed for the unnormalized problem then scaled.
    """
    h = np.maximum(h, 1e-10)
    total = np.sum(h)
    if total < 1e-10:
        return 1e10, np.zeros_like(h)
    
    # Normalize for computation of the shape-dependent part
    h_norm = h / total
    conv = _fft_convolve(h_norm)
    
    m = np.max(conv)
    shifted = (conv - m) * beta
    shifted = np.clip(shifted, -50, 50)
    exp_conv = np.exp(shifted)
    sum_exp = np.sum(exp_conv)
    weights = exp_conv / sum_exp
    
    soft_max_val = m + np.log(sum_exp) / beta
    
    # Gradient of soft_max w.r.t. h_norm
    # d(soft_max)/d(conv_k) = weights[k]
    # d(conv)/d(h_norm) is the autoconvolution operator
    # grad[j] = sum_k weights[k] * (h_norm[k-j] + h_norm[j-k])
    # = 2 * (weights * h_norm)[j] (linear convolution)
    
    # Compute via FFT
    size = 2 * N
    W = fft(weights, size)
    H = fft(h_norm, size)
    grad_norm = np.real(ifft(W * H, size))[:N]
    grad_norm *= 2.0
    
    # Now scale to get gradient w.r.t. h (unnormalized)
    # f(h) = soft_max(h/sum(h))
    # Let S = sum(h). h_norm = h/S.
    # d(f)/d(h_j) = d(f)/d(h_norm_j) * d(h_norm_j)/d(h_j)
    # d(h_norm_j)/d(h_j) = (S - h_j)/S^2
    # d(f)/d(h_norm) = grad_norm (computed above)
    # But wait, the objective in the task is max(conv(h))/S^2.
    # If we use soft_max(conv(h_norm)), it is an approximation to max(conv(h))/S^2?
    # conv(h) = S^2 * conv(h_norm).
    # max(conv(h)) = S^2 * max(conv(h_norm)).
    # So max(conv(h))/S^2 = max(conv(h_norm)).
    # So the objective value is indeed soft_max(conv(h_norm)).
    
    # Gradient chain rule:
    # f(h) = G(h/S) where G(v) = soft_max(conv(v))
    # df/dh_j = sum_k (dG/dv_k) * (dv_k/dh_j)
    # dv_k/dh_j = delta_{kj}/S - v_k/S = (delta_{kj} - h_j/S)/S = (S*delta_{kj} - h_j)/S^2
    # dG/dv_k = grad_norm[k] (from above)
    # df/dh_j = grad_norm[j]/S - (sum_k grad_norm[k] * v_k) / S
    # Note: sum_k grad_norm[k] * v_k = sum_k dG/dv_k * v_k.
    # By Euler's theorem for homogeneous functions?
    # G(v) is not homogeneous.
    # Let's just compute the term.
    
    # Actually, let's use the explicit derivative of f(h) = soft_max(conv(h/S)).
    # Let u = h/S.
    # f = soft_max(conv(u)).
    # df/du_k = grad_norm[k].
    # du_k/dh_j = 1/S * delta_{kj} - u_k/S.
    # df/dh_j = sum_k grad_norm[k] * (delta_{kj}/S - u_k/S)
    #          = grad_norm[j]/S - (1/S) * sum_k grad_norm[k] * u_k.
    
    sum_grad_u = np.dot(grad_norm, h_norm)
    grad_h = (grad_norm - sum_grad_u) / total
    
    return soft_max_val, grad_h

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def time_left():
        return deadline - time.time()

    best_score = np.inf
    best_h = None
    
    # Strategy 1: GMM Initialization + DE
    # Use a Gaussian Mixture Model to parameterize the function
    # h(x) = sum_k w_k * exp(-(x-mu_k)^2 / (2*sigma_k^2))
    
    K = 4 # Number of Gaussians
    
    def gmm_to_h(params):
        """Convert GMM parameters to discrete heights."""
        # params: [w0, mu0, sig0, w1, mu1, sig1, ...]
        # mu and sig are in index space [0, N)
        h = np.zeros(N)
        i = np.arange(N)
        for k in range(K):
            w_k = max(params[3*k], 0.01)
            mu_k = params[3*k+1]
            sig_k = max(params[3*k+2], 1.0)
            h += w_k * np.exp(-((i - mu_k)**2) / (2 * sig_k**2))
        h = np.maximum(h, 0)
        if np.sum(h) > 0:
            h /= np.sum(h)
        else:
            h = np.ones(N) / N
        return h

    def gmm_objective(params):
        h = gmm_to_h(params)
        return compute_autoconv_max(h)

    # Bounds for DE
    bounds = []
    for k in range(K):
        bounds.append((0.1, 5.0))   # weight
        bounds.append((0, N-1))      # mean
        bounds.append((5.0, N/3.0))  # sigma

    # Initial guess: uniform
    best_h = np.ones(N) / N
    best_score = compute_autoconv_max(best_h)
    
    # Run Differential Evolution on GMM parameters
    if time_left() > 60:
        try:
            # Use a reasonable number of iterations
            max_iter_de = int(min(50, time_left() // 1.5))
            if max_iter_de < 10: max_iter_de = 10
            
            de_result = optimize.differential_evolution(
                gmm_objective,
                bounds=bounds,
                seed=42,
                maxiter=max_iter_de,
                popsize=15,
                tol=1e-5,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube',
                updating='deferred'
            )
            if de_result.fun < best_score:
                h_de = gmm_to_h(de_result.x)
                best_score = de_result.fun
                best_h = h_de.copy()
        except Exception:
            pass

    # Strategy 2: Annealed L-BFGS-B Polish
    # Use the smooth max objective with increasing beta (sharpening)
    if time_left() > 30 and best_h is not None:
        betas = [10.0, 50.0, 200.0, 500.0, 1000.0]
        
        current_h = best_h.copy()
        current_val = best_score
        
        for beta in betas:
            if time_left() < 5:
                break
            
            try:
                res = optimize.minimize(
                    lambda h: _smooth_max_obj_grad(h, beta)[0],
                    current_h,
                    jac=lambda h: _smooth_max_obj_grad(h, beta)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 200}
                )
                
                if res.success:
                    final_h = np.maximum(res.x, 0)
                    if np.sum(final_h) > 0:
                        final_h /= np.sum(final_h)
                        final_val = compute_autoconv_max(final_h)
                        if final_val < current_val:
                            current_val = final_val
                            current_h = final_h.copy()
            except Exception:
                pass
        
        if current_val < best_score:
            best_score = current_val
            best_h = current_h.copy()

    # Strategy 3: Multi-restart with random GMM perturbations
    if time_left() > 20:
        n_restarts = int(min(4, time_left() // 5))
        for r in range(n_restarts):
            if time_left() < 8:
                break
            
            # Generate random GMM parameters
            params = np.zeros(3*K)
            for k in range(K):
                params[3*k] = np.random.uniform(0.1, 2.0)
                params[3*k+1] = np.random.uniform(0, N)
                params[3*k+2] = np.random.uniform(10, N/4)
            
            h_init = gmm_to_h(params)
            
            try:
                # Shorter polish
                current_h = h_init
                current_val = compute_autoconv_max(current_h)
                
                for beta in [50.0, 200.0]:
                    if time_left() < 4:
                        break
                    res = optimize.minimize(
                        lambda h: _smooth_max_obj_grad(h, beta)[0],
                        current_h,
                        jac=lambda h: _smooth_max_obj_grad(h, beta)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 100}
                    )
                    if res.success:
                        final_h = np.maximum(res.x, 0)
                        if np.sum(final_h) > 0:
                            final_h /= np.sum(final_h)
                            final_val = compute_autoconv_max(final_h)
                            if final_val < current_val:
                                current_val = final_val
                                current_h = final_h.copy()
                
                if current_val < best_score:
                    best_score = current_val
                    best_h = current_h.copy()
            except Exception:
                pass

    # Strategy 4: Coordinate Descent Fine-tuning
    if time_left() > 10 and best_h is not None:
        current_h = best_h.copy()
        current_val = best_score
        
        # Simple coordinate descent: optimize one bin at a time
        # For each bin j, find h_j that minimizes objective, holding others fixed.
        # This is expensive if done naively, but we can approximate.
        
        # Instead, let's do a few more L-BFGS-B steps with very high beta
        # to sharpen the peak further.
        for _ in range(2):
            if time_left() < 3:
                break
            try:
                res = optimize.minimize(
                    lambda h: _smooth_max_obj_grad(h, 2000.0)[0],
                    current_h,
                    jac=lambda h: _smooth_max_obj_grad(h, 2000.0)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 100}
                )
                if res.success:
                    final_h = np.maximum(res.x, 0)
                    if np.sum(final_h) > 0:
                        final_h /= np.sum(final_h)
                        final_val = compute_autoconv_max(final_h)
                        if final_val < current_val:
                            current_val = final_val
                            current_h = final_h.copy()
            except Exception:
                pass
        
        if current_val < best_score:
            best_score = current_val
            best_h = current_h.copy()

    if best_h is None:
        best_h = np.ones(N)
    
    return list(best_h)
