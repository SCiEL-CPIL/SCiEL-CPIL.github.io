import numpy as np
import scipy.optimize
import time

def kakeya_needle_2d_positions64():
    n = 64
    M = 20000
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1 - y) / n
    
    # Keich Initialization
    k = int(np.log2(n))
    j_idx = np.arange(1, k + 1)
    weights = (1 - j_idx) * (2.0 ** -j_idx) / k
    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        bin_str = bin(i)[2:]
        eps = np.array(list(bin_str), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x[i] = np.sum(eps * weights)
        
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Smooth objective and gradient
    def objective_smooth(x, tau):
        L = x[None, :] + a[None, :] * y[:, None]
        sorted_L = np.sort(L, axis=1)
        gaps = sorted_L[:, 1:] - sorted_L[:, :-1]
        
        # Soft min: min(g, w) approx -tau * log( (exp(-g/tau) + exp(-w/tau)) / 2 )
        # Using logsumexp for stability
        # log( (exp(-g/tau) + exp(-w/tau)) / 2 ) = logsumexp([-g/tau, -w/tau]) - log(2)
        # We need to handle the 2D array.
        # gaps shape (M, n-1), w shape (M,)
        # We want softmin for each element.
        
        # To avoid overflow in exp, we can use the identity:
        # min(a, b) = a + b - max(a, b)
        # max(a, b) = tau * log( (exp(a/tau) + exp(b/tau)) / 2 )
        # So min(a, b) = a + b - tau * (logsumexp([a/tau, b/tau]) - log(2))
        
        a_term = gaps
        b_term = w[:, None]
        
        # logsumexp of [a/tau, b/tau]
        # max_val = np.maximum(a_term/tau, b_term)
        # logsumexp = max_val + np.log(1 + np.exp(-np.abs(a_term/tau - b_term))) - np.log(2)
        # This is numerically stable.
        
        a_scaled = a_term / tau
        b_scaled = b_term / tau
        
        max_scaled = np.maximum(a_scaled, b_scaled)
        # diff = -np.abs(a_scaled - b_scaled)
        # logsumexp_scaled = max_scaled + np.log(1 + np.exp(diff))
        # min_val = a_term + b_term - tau * (logsumexp_scaled - np.log(2))
        
        # Simplified:
        # min_val = a_term + b_term - tau * max_scaled - tau * np.log(1 + np.exp(-np.abs(a_scaled - b_scaled))) + tau * np.log(2)
        
        abs_diff = np.abs(a_scaled - b_scaled)
        # Clip to avoid underflow in exp? exp(-large) -> 0. log(1+0) -> 0.
        # If abs_diff is large, exp(-abs_diff) is 0.
        log_term = np.log1p(np.exp(-abs_diff))
        
        min_val = a_term + b_term - tau * max_scaled - tau * log_term + tau * np.log(2)
        
        union_len = w + np.sum(min_val, axis=1)
        return np.mean(union_len)

    def gradient_smooth(x, tau):
        L = x[None, :] + a[None, :] * y[:, None]
        idx = np.argsort(L, axis=1)
        sorted_L = L[np.arange(M)[:, None], idx]
        gaps = sorted_L[:, 1:] - sorted_L[:, :-1]
        
        # Gradient of softmin(g, w) w.r.t g is:
        # d/da min(a, b) where min is approx by softmin.
        # softmin(a, b) = a + b - tau * (logsumexp([a/tau, b/tau]) - log(2))
        # d/da = 1 - (1/tau) * (exp(a/tau) / (exp(a/tau) + exp(b/tau)))
        # = 1 - 1 / (1 + exp((b-a)/tau))
        # = 1 - sigmoid((b-a)/tau)
        # = sigmoid((a-b)/tau)
        
        a_scaled = gaps / tau
        b_scaled = w[:, None] / tau
        diff = a_scaled - b_scaled
        # sigmoid(diff) = 1 / (1 + exp(-diff))
        # For numerical stability:
        # if diff > 0: exp(-diff) is small, sigmoid ~ 1
        # if diff < 0: exp(diff) is small, sigmoid ~ exp(diff)
        
        # Use stable sigmoid
        sigmoid_vals = np.zeros_like(diff)
        pos_mask = diff >= 0
        sigmoid_vals[pos_mask] = 1.0 / (1.0 + np.exp(-diff[pos_mask]))
        exp_diff = np.exp(diff[~pos_mask])
        sigmoid_vals[~pos_mask] = exp_diff / (1.0 + exp_diff)
        
        # This is d(min_val)/d(gaps)
        
        # Now chain rule: d(gaps)/d(x_i)
        # gaps are differences of sorted L's.
        # g_j = L_sorted[j+1] - L_sorted[j]
        # d(g_j)/d(x_i) = a_i if x_i is the larger one, -a_i if smaller, 0 otherwise.
        
        flat_idx = np.arange(M)[:, None] * (n - 1) + np.arange(n - 1)[None, :]
        # We need to accumulate gradients for each x_i
        # grad_i = sum_m sum_j sigmoid(g_mj, w_m) * d(g_mj)/d(x_i)
        
        # d(g_mj)/d(x_i) is non-zero only if i is the source of sorted_L[m, j] or sorted_L[m, j+1]
        # Let src_j = idx[m, j], src_k = idx[m, j+1]
        # If i == src_k, derivative is a_i
        # If i == src_j, derivative is -a_i
        
        src_j = idx[:, :-1]
        src_k = idx[:, 1:]
        
        # We want to sum sigmoid_vals[m, j] * a[src_k[m, j]] for all m, j
        # and subtract sigmoid_vals[m, j] * a[src_j[m, j]]
        
        # Use bincount for efficiency
        # Weights for src_k are sigmoid_vals * a[src_k]
        # Weights for src_j are -sigmoid_vals * a[src_j]
        
        weights_k = sigmoid_vals * a[src_k]
        weights_j = -sigmoid_vals * a[src_j]
        
        grad = np.bincount(src_k.ravel(), weights=weights_k.ravel(), minlength=n)
        grad -= np.bincount(src_j.ravel(), weights=weights_j.ravel(), minlength=n)
        
        return grad / M

    # Continuation method
    taus = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    best_x = x.copy()
    best_area = objective_smooth(x, 0.001) # Use a small tau for final check? No, just track best
    
    for tau in taus:
        if time.time() > deadline - 30:
            break
            
        try:
            res = scipy.optimize.minimize(
                objective_smooth,
                x,
                args=(tau,),
                jac=lambda x, t: gradient_smooth(x, t),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_area:
                best_area = res.fun
                best_x = res.x.copy()
            x = res.x
        except Exception:
            pass

    # Final polish with very small tau if time permits
    if time.time() < deadline - 10:
        try:
            res = scipy.optimize.minimize(
                objective_smooth,
                best_x,
                args=(1e-5,),
                jac=lambda x, t: gradient_smooth(x, t),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_area:
                best_area = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    return best_x
