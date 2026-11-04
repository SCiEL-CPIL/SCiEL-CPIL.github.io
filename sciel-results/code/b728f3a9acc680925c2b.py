import numpy as np
import time

N = 50

def construct():
    # Objective to maximize: J(f) = ||f||_2^2 / ||f*f||_inf
    # Subject to f >= 0, sum(f) = 1 (we enforce this via normalization in the loop)
    
    def compute_ratio(f):
        # f is assumed to be normalized to sum 1
        norm2_sq = np.sum(f**2)
        conv = np.convolve(f, f, mode='full')
        max_conv = np.max(conv)
        if max_conv == 0:
            return 0, 0
        return norm2_sq / max_conv, max_conv

    def gradient_ascent(x0, steps=2000, lr=0.01):
        # Adam optimizer parameters
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = np.zeros_like(x0)
        v = np.zeros_like(x0)
        best_f = x0.copy()
        best_val = -np.inf
        
        # Initial evaluation
        val, _ = compute_ratio(x0)
        if val > best_val:
            best_val = val
            best_f = x0.copy()
            
        t = 0
        for step in range(steps):
            t += 1
            f = best_f.copy() # Start from best found so far? No, standard GD follows trajectory.
            # Actually, let's just follow the trajectory from x0, but project back to simplex/non-negative.
            # A common trick: optimize in unconstrained space, project to f>=0, sum=1.
            
            # Current state is 'f' which is normalized
            S = np.sum(f**2)
            conv = np.convolve(f, f, mode='full')
            M = np.max(conv)
            if M == 0:
                break
            
            # Gradient of J = S/M
            # dJ/df_i = (M * 2f_i - S * dM/df_i) / M^2
            # dM/df_i = f_{k*-i} + f_i where k* is argmax(conv)
            
            k_star = np.argmax(conv)
            
            # Compute dM/df
            # (f*f)_k = sum_j f_j f_{k-j}
            # d/d f_i (f*f)_{k*} = f_{k*-i} + f_i (if indices in range)
            # Note: conv index k* corresponds to sum index.
            # f has indices 0..N-1.
            # conv has indices 0..2N-2.
            # If k* is the index in conv, the term is sum_{j} f_j f_{k*-j}.
            # The derivative w.r.t f_i is non-zero when j=i (term f_i f_{k*-i}) or k*-j=i => j=k*-i (term f_{k*-i} f_i).
            
            dM_df = np.zeros_like(f)
            # Iterate over valid i
            # f_i contributes to conv[k] for k=i+j.
            # We need d conv[k_star] / d f_i.
            # This is f_{k_star - i} if 0 <= k_star - i < N, plus f_i if i == k_star - j?
            # Let's just use the formula: dM/df_i = f_{k_star-i} + f_i (with bounds checking)
            
            for i in range(N):
                j = k_star - i
                val_grad = 0.0
                if 0 <= j < N:
                    val_grad += f[j]
                # The second term is f_i? 
                # Let's check: sum_j f_j f_{k*-j}.
                # If j=i, term is f_i f_{k*-i}. Deriv is f_{k*-i}.
                # If k*-j=i => j=k*-i, term is f_{k*-i} f_i. Deriv is f_i.
                # So yes, sum is f_{k*-i} + f_i.
                val_grad += f[i]
                dM_df[i] = val_grad
                
            grad_J = (M * 2 * f - S * dM_df) / (M * M)
            
            # Adam update
            m = beta1 * m + (1 - beta1) * grad_J
            v = beta2 * v + (1 - beta2) * (grad_J ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # We want to ASCEND, so we add gradient
            f_new = f + lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Project to f >= 0 and sum = 1
            f_new = np.maximum(f_new, 0)
            s = np.sum(f_new)
            if s == 0:
                f_new = np.ones(N) / N
            else:
                f_new = f_new / s
                
            # Evaluate
            val, _ = compute_ratio(f_new)
            if val > best_val:
                best_val = val
                best_f = f_new.copy()
                
            # If no improvement for a long time, break?
            # For simplicity, fixed steps.
            
        return best_f, best_val

    best_val = -np.inf
    best_f = np.ones(N) / N
    
    start_time = time.time()
    time_limit = 110.0
    
    # Generate diverse initial guesses
    guesses = []
    
    # 1. Uniform
    guesses.append(np.ones(N) / N)
    
    # 2. Box functions
    for k in [2, 3, 5, 10, 20, 25, 30]:
        box = np.zeros(N)
        box[:k] = 1.0 / k
        guesses.append(box)
        # Centered box
        box_c = np.zeros(N)
        start = (N - k) // 2
        box_c[start:start+k] = 1.0 / k
        guesses.append(box_c)
        
    # 3. Gaussian bumps
    for alpha in [0.5, 1.0, 2.0, 5.0, 10.0]:
        i = np.arange(N)
        center = (N - 1) / 2.0
        g = np.exp(-alpha * (i - center)**2)
        g = g / np.sum(g)
        guesses.append(g)
        
    # 4. "Triangle" / Parabolic
    i = np.arange(N)
    center = (N - 1) / 2.0
    tri = np.maximum(0, 1 - np.abs(i - center) / center)
    tri = tri / np.sum(tri)
    guesses.append(tri)
    
    # 5. Random points
    np.random.seed(42)
    for _ in range(20):
        r = np.random.rand(N)
        r = r / np.sum(r)
        guesses.append(r)
        
    # Run gradient ascent for each guess
    # We use a moderate number of steps. 2000 steps * 100 guesses = 200,000 iterations.
    # Each iteration is O(N^2) due to conv. N=50, N^2=2500.
    # 200k * 2500 = 500M ops. Might be too slow in Python.
    # We need to optimize. 
    # 1. Use fewer guesses or fewer steps.
    # 2. Vectorize the gradient computation.
    
    # Let's optimize the inner loop by vectorizing dM_df calculation.
    
    def fast_gradient_ascent(x0, steps=500, lr=0.05):
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = np.zeros_like(x0)
        v = np.zeros_like(x0)
        best_f = x0.copy()
        best_val = -np.inf
        
        val, _ = compute_ratio(x0)
        if val > best_val:
            best_val = val
            best_f = x0.copy()
            
        t = 0
        for step in range(steps):
            t += 1
            f = best_f.copy()
            
            S = np.sum(f**2)
            conv = np.convolve(f, f, mode='full')
            M = np.max(conv)
            if M == 0:
                break
            
            k_star = np.argmax(conv)
            
            # Vectorized dM_df
            # dM_df[i] = f[k_star - i] + f[i]
            # We need to handle bounds for k_star - i.
            # indices i = 0..N-1
            # j = k_star - i
            # valid if 0 <= j < N
            
            i_idx = np.arange(N)
            j_idx = k_star - i_idx
            
            # Mask for valid j
            valid = (j_idx >= 0) & (j_idx < N)
            
            dM_df = np.zeros_like(f)
            dM_df[valid] = f[j_idx[valid]] + f[i_idx[valid]]
            # For invalid j, only f[i] contributes? 
            # If j is out of bounds, f_j is 0. So only f_i term remains?
            # Wait, the formula was f_{k*-i} + f_i.
            # If k*-i is out of bounds, f_{k*-i} = 0.
            # So dM_df[i] = f_i for all i? No.
            # Let's re-verify.
            # Conv term: sum_j f_j f_{k*-j}.
            # Deriv w.r.t f_i:
            # Term where j=i: f_i f_{k*-i}. Deriv: f_{k*-i}.
            # Term where k*-j=i => j=k*-i: f_{k*-i} f_i. Deriv: f_i.
            # So yes, sum is f_{k*-i} + f_i.
            # If k*-i is out of [0, N-1], f_{k*-i} is 0.
            # So dM_df[i] = f_i + (f[k*-i] if valid else 0).
            
            # So we can just add f[i] to all, and add f[j] where valid.
            dM_df[:] = f
            dM_df[valid] += f[j_idx[valid]]
            
            grad_J = (M * 2 * f - S * dM_df) / (M * M)
            
            m = beta1 * m + (1 - beta1) * grad_J
            v = beta2 * v + (1 - beta2) * (grad_J ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            f_new = f + lr * m_hat / (np.sqrt(v_hat) + eps)
            
            f_new = np.maximum(f_new, 0)
            s = np.sum(f_new)
            if s == 0:
                f_new = np.ones(N) / N
            else:
                f_new = f_new / s
                
            val, _ = compute_ratio(f_new)
            if val > best_val:
                best_val = val
                best_f = f_new.copy()
                
        return best_f, best_val

    for x0 in guesses:
        if time.time() - start_time > time_limit:
            break
        try:
            f_res, val_res = fast_gradient_ascent(x0, steps=1000, lr=0.1)
            if val_res > best_val:
                best_val = val_res
                best_f = f_res
        except Exception:
            pass

    # Final check
    if best_val == -np.inf:
        best_f = np.ones(N) / N
        
    # Ensure non-negative and normalized
    best_f = np.maximum(best_f, 0)
    s = np.sum(best_f)
    if s > 0:
        best_f = best_f / s
        
    return list(best_f)
