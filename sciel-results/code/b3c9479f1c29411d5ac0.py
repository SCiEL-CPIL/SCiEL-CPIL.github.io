import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def score(h):
        h = np.maximum(h, 0)
        l1 = np.sum(h)
        if l1 < 1e-10:
            return 0.0
        l2_sq = np.sum(h**2)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 0.0
        return l2_sq / (l1 * max_conv)

    # Helper to get max multiplicity of differences in a set (for Sidon-like search)
    def get_max_diff_mult(S):
        if len(S) <= 1:
            return 1
        S = np.sort(S)
        diffs = []
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                diffs.append(S[j] - S[i])
        if not diffs:
            return 1
        vals, counts = np.unique(diffs, return_counts=True)
        return np.max(counts)

    # Greedy construction of a Sidon-like set
    def greedy_build(target_size, seed_idx=0):
        S = [seed_idx]
        current_diffs = []
        for _ in range(target_size - 1):
            best_candidate = -1
            best_max_mult = float('inf')
            for cand in range(N):
                if cand in S:
                    continue
                new_diffs = [abs(cand - s) for s in S]
                all_diffs = current_diffs + new_diffs
                if all_diffs:
                    vals, counts = np.unique(all_diffs, return_counts=True)
                    max_mult = np.max(counts)
                else:
                    max_mult = 1
                if max_mult < best_max_mult:
                    best_max_mult = max_mult
                    best_candidate = cand
                    if best_max_mult == 1:
                        break
            if best_candidate != -1:
                S.append(best_candidate)
                for s in S[:-1]:
                    current_diffs.append(abs(S[-1] - s))
            else:
                break
        return S

    # Objective function: minimize negative log ratio
    # We fix sum(h) = 1 to remove scale invariance.
    # Ratio = sum(h^2) / max_conv
    def neg_log_ratio(h):
        h = np.maximum(h, 0)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        h_norm = h / s
        l2_sq = np.sum(h_norm**2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        max_conv = np.max(conv)
        if max_conv < 1e-10:
            return 1e10
        ratio = l2_sq / max_conv
        if ratio <= 0:
            return 1e10
        return -np.log(ratio)

    def jac_neg_log_ratio(h):
        eps = 1e-7
        grad = np.zeros_like(h)
        f0 = neg_log_ratio(h)
        for i in range(len(h)):
            h_old = h[i]
            h[i] += eps
            f1 = neg_log_ratio(h)
            h[i] = h_old
            grad[i] = (f1 - f0) / eps
        return grad

    # Water-filling based spectral optimization
    def spectral_waterfilling(S, max_iters=200, tol=1e-8):
        """
        Iteratively adjust amplitudes to suppress autoconvolution peaks.
        Uses spectral density to guide updates.
        """
        k = len(S)
        if k == 0:
            return np.zeros(N)
        
        # Initialize with uniform amplitudes normalized to sum=1
        x = np.ones(k) / k
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            h_full = np.zeros(N)
            h_full[S] = x
            
            # Compute autoconvolution
            conv = np.convolve(h_full, h_full, mode='full')
            max_conv = np.max(conv)
            
            if max_conv < 1e-10:
                break
                
            # Compute spectral density (power spectrum)
            # FFT of h_full
            H = np.fft.fft(h_full, n=2*N)
            power = np.abs(H)**2
            power = power / np.sum(power)  # Normalize
            
            # Estimate local spectral density at each support point
            # For simplicity, we use the inverse FFT of the power spectrum
            # to get an estimate of the "energy" at each point
            # Actually, a simpler approach: 
            # The gradient of max_conv with respect to h_i is related to (h*h)[i]
            # But since max_conv is non-differentiable, we use a soft-max approximation
            # or simply use the convolution values directly.
            
            # Let's use the convolution values as a proxy for "pressure"
            # We want to reduce amplitudes where they contribute to high conv values.
            
            # Compute contribution of each h_i to the max conv
            # This is complex, so let's use a simpler heuristic:
            # Update x_i proportional to 1 / (1 + conv[i])
            # This is a form of water-filling where high conv values reduce amplitude.
            
            # Get conv values at support points
            # Note: conv is of length 2N-1, indices 0 to 2N-2
            # h_full is of length N, indices 0 to N-1
            # The autoconvolution at index i is sum_j h_j h_{i-j}
            # For i in [0, N-1], conv[i] = sum_{j=0}^i h_j h_{i-j}
            # For i in [N, 2N-2], conv[i] = sum_{j=i-N+1}^{N-1} h_j h_{i-j}
            
            # We want to reduce h_i if it contributes to high conv values.
            # A simple update: x_new_i = x_i / (1 + alpha * conv_at_i)
            # where conv_at_i is an estimate of the peak contribution.
            
            # Let's use the max conv value as a global pressure
            # and distribute the reduction based on the current amplitudes.
            
            # Alternative: use the gradient of a soft-max of conv
            # soft_max = log(sum(exp(conv / T)))
            # grad soft_max = conv / T * exp(conv/T) / sum(exp(conv/T))
            
            T = 0.1  # Temperature
            conv_clipped = np.clip(conv, -100, 100)  # Avoid overflow
            exp_conv = np.exp(conv_clipped / T)
            sum_exp = np.sum(exp_conv)
            if sum_exp < 1e-10:
                break
            weights = exp_conv / sum_exp
            
            # The gradient of the soft-max w.r.t. h_i is sum_j weights_j * (h_{j-i} + h_{j+i})
            # This is complex, so let's use a simpler update.
            
            # Update: reduce amplitudes in proportion to their "pressure"
            # Pressure at i is approximately the sum of weights over j such that j-i or j+i is in support
            # This is getting complicated. Let's use a simpler heuristic.
            
            # Simple heuristic: reduce amplitudes where the local convolution is high.
            # For each support point i, compute the average conv value in a neighborhood.
            
            pressures = np.zeros(k)
            for idx, s in enumerate(S):
                # Conv value at position s (within the first N points)
                if s < len(conv):
                    pressures[idx] = conv[s]
                else:
                    pressures[idx] = 0
            
            # Normalize pressures
            max_p = np.max(pressures)
            if max_p < 1e-10:
                break
            pressures = pressures / max_p
            
            # Update amplitudes
            x_new = x / (1 + 0.5 * pressures)
            x_new = x_new / np.sum(x_new)  # Re-normalize
            
            # Check convergence
            if np.max(np.abs(x_new - x)) < tol:
                x = x_new
                break
            x = x_new
        
        h_final = np.zeros(N)
        h_final[S] = x
        return h_final

    best_h = np.zeros(N)
    best_score = 0.0

    np.random.seed(42)
    
    # 1. Try Greedy Sidon sets of various sizes
    for k in range(2, 15):
        if time.time() > deadline - 20.0:
            break
        S = greedy_build(k)
        if len(S) < k:
            continue
        
        # Use spectral water-filling to find good amplitudes
        h_wf = spectral_waterfilling(S, max_iters=100)
        s_wf = score(h_wf)
        
        if s_wf > best_score:
            best_score = s_wf
            best_h = h_wf.copy()
        
        # Also try SLSQP optimization starting from water-filling result
        x0 = h_wf[S]
        if np.sum(x0) < 1e-10:
            x0 = np.ones(len(S)) / len(S)
        else:
            x0 = x0 / np.sum(x0)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(len(x))}
        ]
        bounds = [(0, 1)] * len(S)
        
        def obj_full(x):
            h_full = np.zeros(N)
            h_full[S] = x
            return neg_log_ratio(h_full)
            
        def jac_full(x):
            h_full = np.zeros(N)
            h_full[S] = x
            grad_full = jac_neg_log_ratio(h_full)
            return grad_full[S]

        try:
            res = minimize(obj_full, x0, method='SLSQP', jac=jac_full, bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-9})
            if res.success:
                h_opt = np.zeros(N)
                h_opt[S] = res.x
                s_opt = score(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt.copy()
        except Exception:
            pass

    # 2. Random search for better supports if time permits
    if best_score < 1.13 and time.time() < deadline - 30.0:
        for _ in range(2000):
            if time.time() > deadline - 10.0:
                break
            k = np.random.randint(3, 10)
            S = np.random.choice(N, k, replace=False)
            
            # Use spectral water-filling
            h_wf = spectral_waterfilling(S, max_iters=50)
            s_wf = score(h_wf)
            
            if s_wf > best_score:
                best_score = s_wf
                best_h = h_wf.copy()
            
            # Try SLSQP
            x0 = h_wf[S]
            if np.sum(x0) < 1e-10:
                x0 = np.ones(k) / k
            else:
                x0 = x0 / np.sum(x0)
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(len(x))}
            ]
            bounds = [(0, 1)] * k
            
            def obj_full(x):
                h_full = np.zeros(N)
                h_full[S] = x
                return neg_log_ratio(h_full)
                
            def jac_full(x):
                h_full = np.zeros(N)
                h_full[S] = x
                grad_full = jac_neg_log_ratio(h_full)
                return grad_full[S]

            try:
                res = minimize(obj_full, x0, method='SLSQP', jac=jac_full, bounds=bounds, constraints=constraints, options={'maxiter': 50, 'ftol': 1e-8})
                if res.success:
                    h_opt = np.zeros(N)
                    h_opt[S] = res.x
                    s_opt = score(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
            except Exception:
                pass

    # 3. Polish the best result found with more iterations
    if best_score > 0 and time.time() < deadline - 5.0:
        support_idx = np.where(best_h > 1e-6)[0]
        if len(support_idx) > 0:
            S = support_idx
            x0 = best_h[S]
            x0 = x0 / np.sum(x0)
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0, 'jac': lambda x: np.ones(len(x))}
            ]
            bounds = [(0, 1)] * len(S)
            
            def obj_full(x):
                h_full = np.zeros(N)
                h_full[S] = x
                return neg_log_ratio(h_full)
                
            def jac_full(x):
                h_full = np.zeros(N)
                h_full[S] = x
                grad_full = jac_neg_log_ratio(h_full)
                return grad_full[S]

            try:
                res = minimize(obj_full, x0, method='SLSQP', jac=jac_full, bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
                if res.success:
                    h_opt = np.zeros(N)
                    h_opt[S] = res.x
                    s_opt = score(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
            except Exception:
                pass

    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    if np.sum(best_h) < 1e-10:
        h = [0.0] * N
        h[N // 2] = 1.0
        return h
        
    return list(best_h)
