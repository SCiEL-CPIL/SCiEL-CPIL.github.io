# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _smoothed_area_and_grad(x, eps, n=64, M=20000):
    """
    Compute the smoothed area of the union of triangles and its gradient.
    Uses a log-sum-exp based smooth approximation for min(D, w).
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    
    # L shape: (M, n)
    # x is (n,), a is (n,), y is (M,)
    # L[i, m] = x[i] + a[i] * y[m]
    L = x[None, :] + a[None, :] * y[:, None]
    
    # Sort L at each height
    idx = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, idx, axis=1)
    
    # Gaps D_k = L_sorted[:, k+1] - L_sorted[:, k]
    # D shape: (M, n-1)
    D = L_sorted[:, 1:] - L_sorted[:, :-1]
    
    # Smooth min(D, w)
    # f(D, w) = -eps * log(exp(-D/eps) + exp(-w/eps))
    # To avoid overflow/underflow in exp, we can shift by max(-D, -w)/eps
    # But D and w are positive. -D/eps and -w/eps are negative.
    # exp(negative) is safe.
    
    # We need sum over k of f(D_k, w_m)
    # And gradient terms.
    
    # For numerical stability, let's compute the sigmoid directly for gradient.
    # sig = 1 / (1 + exp((D - w) / eps))
    # This is the derivative of f w.r.t D.
    
    # (D - w) shape: (M, n-1)
    # w shape: (M, 1)
    diff = D - w[:, None]
    
    # Sigmoid: sigma(-diff/eps)
    # To avoid overflow in exp(diff/eps) if diff is large positive:
    # sigma(-z) = 1 / (1 + exp(z))
    # If z > 0, exp(z) can be large.
    # If z < 0, exp(z) is small.
    # We can use np.exp with clipping or a stable sigmoid implementation.
    
    # Stable sigmoid for -z:
    # If z >= 0: 1 / (1 + exp(z))
    # If z < 0: exp(-z) / (1 + exp(-z))
    
    z = diff / eps
    sig = np.zeros_like(z)
    mask_pos = z >= 0
    mask_neg = ~mask_pos
    
    # For z >= 0
    sig[mask_pos] = 1.0 / (1.0 + np.exp(z[mask_pos]))
    
    # For z < 0
    exp_neg_z = np.exp(-z[mask_neg])
    sig[mask_neg] = exp_neg_z / (1.0 + exp_neg_z)
    
    # sig shape: (M, n-1)
    # sig[:, k] is d f(D_k, w) / d D_k
    
    # Now compute the area.
    # f(D, w) = -eps * log(exp(-D/eps) + exp(-w/eps))
    # log(exp(-D/eps) + exp(-w/eps)) = log(exp(max(-D, -w)/eps) * (exp((-D - max)/eps) + exp((-w - max)/eps)))
    # = max(-D, -w)/eps + log(exp((-D - max)/eps) + exp((-w - max)/eps))
    # Let M_exp = max(-D, -w)/eps. Note -D and -w are negative.
    # Actually, simpler:
    # f = -eps * (logaddexp(-D/eps, -w/eps))
    # np.logaddexp is stable.
    
    log_terms = np.logaddexp(-D / eps, -w[:, None] / eps)
    f_vals = -eps * log_terms
    sum_f = np.sum(f_vals, axis=1)
    
    U = w + sum_f
    area = np.mean(U)
    
    # Gradient computation
    # dU/dL_sorted[:, k]
    # g_sorted shape: (M, n)
    g_sorted = np.zeros_like(L_sorted)
    
    # Contribution from D_k (between k and k+1) to L_sorted[:, k] is -sig[:, k]
    # Contribution from D_k (between k and k+1) to L_sorted[:, k+1] is +sig[:, k]
    
    # For k=0 to n-2:
    # g_sorted[:, k] -= sig[:, k]
    # g_sorted[:, k+1] += sig[:, k]
    
    g_sorted[:, :-1] -= sig
    g_sorted[:, 1:] += sig
    
    # Map back to x
    # grad_x[i] = a_i * sum_m g_sorted[m, inv_idx[m, i]] / M
    
    G = np.zeros_like(L)
    np.put_along_axis(G, idx, g_sorted, axis=1)
    grad_x = np.sum(G, axis=0) * a / M
    
    return area, grad_x

def _keich_init(n):
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    x = _keich_init(n)
    
    start_time = time.time()
    deadline = start_time + 590.0 # Safety margin
    
    # Annealing schedule for epsilon
    # Start with larger eps for global convergence, reduce for accuracy
    eps_schedule = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    best_x = x
    best_area = float('inf')
    
    for eps in eps_schedule:
        if time.time() > deadline - 60: # Leave time for final check
            break
            
        def func_and_grad(x_curr):
            return _smoothed_area_and_grad(x_curr, eps)
        
        try:
            res = minimize(
                func_and_grad, 
                x, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < best_area:
                if np.all(np.isfinite(res.x)):
                    # Verify area with a smaller eps or exact? 
                    # For tracking best, we can use the smoothed area as a proxy,
                    # but for the final return, we want the x that minimizes the TRUE area.
                    # The smoothed area is an upper bound? 
                    # min(D, w) <= smooth_min(D, w)?
                    # Let's check: f(D,w) = -eps log(exp(-D/eps)+exp(-w/eps)).
                    # If D < w, f ~ D. If D >> w, f ~ w.
                    # It is a smooth approximation that is >= min(D,w)?
                    # log(exp(a)+exp(b)) >= max(a,b).
                    # -eps log(...) <= -eps max(a,b) = min(D,w).
                    # So smoothed area <= true area? No.
                    # log(exp(a)+exp(b)) is >= max(a,b).
                    # So -log(...) <= -max(a,b).
                    # So f_smooth <= min(D, w).
                    # So we are minimizing a lower bound.
                    # This might lead to over-optimization.
                    # However, as eps -> 0, it converges to the true value.
                    
                    best_x = res.x
                    # We should compute the TRUE area for comparison to ensure we are improving
                    # But true area is nonsmooth.
                    # Let's just trust the annealing.
                    
                    # Update x for next iteration
                    x = res.x
        except Exception as e:
            # If optimization fails, keep current x
            pass
            
    # Final polish with very small eps if time permits
    if time.time() < deadline - 30:
        eps_final = 1e-5
        def func_and_grad_final(x_curr):
            return _smoothed_area_and_grad(x_curr, eps_final)
        
        try:
            res = minimize(
                func_and_grad_final, 
                x, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 100, 'maxfun': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if np.all(np.isfinite(res.x)):
                x = res.x
        except:
            pass

    # Ensure finite
    if not np.all(np.isfinite(x)):
        return _keich_init(n)
        
    return x

# EVOLVE-BLOCK-END
