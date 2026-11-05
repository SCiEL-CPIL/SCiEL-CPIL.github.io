import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    N = 24
    M = 2**15  # FFT size
    sqrt_N = np.sqrt(N)

    # Precompute DFT matrix D where F = D @ c
    # F[k] = sum_{n=0}^{N-1} c_n exp(-2j pi k n / M)
    k = np.arange(M)
    n = np.arange(N)
    D = np.exp(-2j * np.pi * (k[:, None] * n[None, :] / M))

    def get_score(F):
        return np.max(np.abs(F)) / sqrt_N

    def flip_bit_F(F, c, idx):
        return F - 2.0 * c[idx] * D[:, idx]

    def flip_two_bits_F(F, c, idx1, idx2):
        return F - 2.0 * (c[idx1] * D[:, idx1] + c[idx2] * D[:, idx2])

    # Smooth surrogate: soft-max of |F| using log-sum-exp
    def soft_max_score(F, beta=50.0):
        abs_F = np.abs(F)
        max_val = np.max(abs_F)
        shifted = abs_F - max_val
        log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
        return (max_val + log_sum) / sqrt_N

    # High-precision evaluation: find true sup norm via golden section
    def eval_sup_norm(c):
        """Evaluate max |P(z)| over |z|=1 using dense sampling + golden section."""
        # Dense sampling
        n_dense = 10000
        theta = np.linspace(0, 2*np.pi, n_dense, endpoint=False)
        z = np.exp(1j * theta)
        # Evaluate P(z) = sum c_k z^k
        # Use polynomial evaluation via Horner or direct
        vals = np.zeros(n_dense, dtype=complex)
        for kk in range(N):
            vals += c[kk] * (z ** kk)
        abs_vals = np.abs(vals)
        
        # Find top peaks and refine with golden section
        top_k = 20
        idx_sorted = np.argsort(abs_vals)[-top_k:][::-1]
        
        best = 0.0
        for idx in idx_sorted:
            # Golden section refinement around theta[idx]
            t0 = theta[idx]
            # Find local peak by checking neighbors
            # Determine interval
            dt = 2*np.pi / n_dense
            lo = t0 - dt
            hi = t0 + dt
            # Ensure lo, hi are in [0, 2*pi)
            if lo < 0:
                lo += 2*np.pi
            if hi >= 2*np.pi:
                hi -= 2*np.pi
                
            # Golden section search
            gr = (np.sqrt(5) - 1) / 2  # golden ratio ~0.618
            for _ in range(60):
                if time.time() > deadline - 2.0:
                    break
                m1 = lo + gr * (hi - lo)
                m2 = hi - gr * (hi - lo)
                # Evaluate at m1, m2
                z1 = np.exp(1j * m1)
                z2 = np.exp(1j * m2)
                v1 = 0.0
                v2 = 0.0
                for kk in range(N):
                    v1 += c[kk] * (z1 ** kk)
                    v2 += c[kk] * (z2 ** kk)
                a1 = abs(v1)
                a2 = abs(v2)
                if a1 > a2:
                    hi = m2
                else:
                    lo = m1
                
                # Also check endpoints
                z_lo = np.exp(1j * lo)
                z_hi = np.exp(1j * hi)
                v_lo = 0.0
                v_hi = 0.0
                for kk in range(N):
                    v_lo += c[kk] * (z_lo ** kk)
                    v_hi += c[kk] * (z_hi ** kk)
                a_lo = abs(v_lo)
                a_hi = abs(v_hi)
                if a_lo > a1 and a_lo > a2:
                    # Peak might be at lo
                    hi = m2
                elif a_hi > a1 and a_hi > a2:
                    lo = m1
                
                if hi - lo < 1e-12:
                    break
            
            # Evaluate at center
            t_mid = (lo + hi) / 2
            z_mid = np.exp(1j * t_mid)
            v_mid = 0.0
            for kk in range(N):
                v_mid += c[kk] * (z_mid ** kk)
            a_mid = abs(v_mid)
            if a_mid > best:
                best = a_mid
        
        # Also check the dense samples
        if np.max(abs_vals) > best:
            best = np.max(abs_vals)
        
        return best / sqrt_N

    rng = np.random.RandomState(42)

    best_score = float('inf')
    best_c = None
    best_F = None

    # Known good starting points
    golay_24 = np.array([1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1], dtype=float)
    
    initial_candidates = [golay_24.copy()]
    for _ in range(10):
        initial_candidates.append(np.random.choice([-1.0, 1.0], N))
    
    initial_candidates.append(np.ones(N))
    initial_candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))

    # Phase 1: Continuous relaxation with penalty for non-binary values
    if time.time() < deadline - 60.0:
        for _ in range(3):
            if time.time() > deadline - 60.0:
                break
            
            x = rng.uniform(-1.0, 1.0, N)
            
            lr = 0.1
            penalty_lambda = 5.0
            max_iter = 500
            
            for it in range(max_iter):
                if time.time() > deadline - 60.0:
                    break
                
                F = D @ x
                abs_F = np.abs(F)
                max_val = np.max(abs_F)
                
                beta = 50.0
                shifted = abs_F - max_val
                exp_shifted = np.exp(beta * shifted)
                sum_exp = np.sum(exp_shifted)
                weights = exp_shifted / sum_exp
                
                signs = np.sign(F)
                grad_softmax = (D.T @ (weights * signs)).real / sqrt_N
                
                grad_penalty = 4.0 * penalty_lambda * x * (x**2 - 1)
                
                grad = grad_softmax + grad_penalty
                
                x = x - lr * grad
                x = np.clip(x, -1.0, 1.0)
                
                if it % 50 == 0:
                    lr *= 0.95
            
            c_proj = np.sign(x)
            c_proj[c_proj == 0] = 1.0
            
            F_proj = D @ c_proj
            score_proj = get_score(F_proj)
            
            if score_proj < best_score:
                best_score = score_proj
                best_c = c_proj.copy()
                best_F = F_proj.copy()
            
            initial_candidates.append(c_proj.copy())

    # Phase 2: Discrete search from all candidates
    for init_c in initial_candidates:
        if time.time() > deadline:
            break
        
        c = init_c.copy()
        F = D @ c
        current_score = get_score(F)
        
        T = 0.05
        alpha = 0.9995
        T_min = 1e-7
        
        while time.time() < deadline - 10.0 and T > T_min:
            if rng.rand() < 0.7:
                idx = rng.randint(N)
                F_new = flip_bit_F(F, c, idx)
                c_new = c.copy()
                c_new[idx] *= -1
            else:
                idx1, idx2 = rng.choice(N, 2, replace=False)
                F_new = flip_two_bits_F(F, c, idx1, idx2)
                c_new = c.copy()
                c_new[idx1] *= -1
                c_new[idx2] *= -1
            
            new_smooth = soft_max_score(F_new, beta=50.0)
            current_smooth = soft_max_score(F, beta=50.0)
            delta_smooth = new_smooth - current_smooth
            
            if delta_smooth < 0 or rng.rand() < np.exp(-delta_smooth / T):
                F = F_new
                c = c_new
                current_score = get_score(F)
                
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                    best_F = F.copy()
            
            T *= alpha

    # Phase 3: Multi-start 2-flip local search from diverse points
    # Generate additional diverse starting points
    extra_starts = []
    for _ in range(5):
        extra_starts.append(np.random.choice([-1.0, 1.0], N))
    # Also include negation of best
    if best_c is not None:
        extra_starts.append(-best_c.copy())
    
    for init_c in extra_starts:
        if time.time() > deadline - 15.0:
            break
        
        c = init_c.copy()
        F = D @ c
        current_score = get_score(F)
        
        if current_score < best_score:
            best_score = current_score
            best_c = c.copy()
            best_F = F.copy()
        
        # 2-flip local search
        improved = True
        while improved and time.time() < deadline - 15.0:
            improved = False
            best_delta = 0
            best_idx1 = -1
            best_idx2 = -1
            best_F_candidate = None
            for n1 in range(N):
                for n2 in range(n1+1, N):
                    F_candidate = flip_two_bits_F(F, c, n1, n2)
                    new_score = get_score(F_candidate)
                    delta = new_score - current_score
                    if delta < best_delta:
                        best_delta = delta
                        best_idx1 = n1
                        best_idx2 = n2
                        best_F_candidate = F_candidate
            
            if best_idx1 != -1:
                c[best_idx1] *= -1
                c[best_idx2] *= -1
                F = best_F_candidate
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                    best_F = F.copy()
                improved = True

    # Final Local Search on best_c
    if best_c is not None:
        c = best_c.copy()
        F = best_F.copy()
        current_score = best_score
        
        # 1-flip local search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_delta = 0
            best_idx = -1
            best_F_candidate = None
            for idx in range(N):
                F_candidate = flip_bit_F(F, c, idx)
                new_score = get_score(F_candidate)
                delta = new_score - current_score
                if delta < best_delta:
                    best_delta = delta
                    best_idx = idx
                    best_F_candidate = F_candidate
            
            if best_idx != -1:
                c[best_idx] *= -1
                F = best_F_candidate
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                    best_F = F.copy()
                improved = True
        
        # 2-flip local search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_delta = 0
            best_idx1 = -1
            best_idx2 = -1
            best_F_candidate = None
            for n1 in range(N):
                for n2 in range(n1+1, N):
                    F_candidate = flip_two_bits_F(F, c, n1, n2)
                    new_score = get_score(F_candidate)
                    delta = new_score - current_score
                    if delta < best_delta:
                        best_delta = delta
                        best_idx1 = n1
                        best_idx2 = n2
                        best_F_candidate = F_candidate
            
            if best_idx1 != -1:
                c[best_idx1] *= -1
                c[best_idx2] *= -1
                F = best_F_candidate
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                    best_F = F.copy()
                improved = True

    # Final high-precision polish: verify with true sup norm
    if best_c is not None and time.time() < deadline - 3.0:
        true_score = eval_sup_norm(best_c)
        # If true score is significantly different, try a few more 2-flips
        if abs(true_score - best_score) > 1e-6:
            best_score = true_score
            
            # Try 2-flip neighborhood with true evaluation
            c = best_c.copy()
            for _ in range(3):  # A few iterations
                if time.time() > deadline - 3.0:
                    break
                improved = False
                best_delta = 0
                best_idx1 = -1
                best_idx2 = -1
                for n1 in range(N):
                    for n2 in range(n1+1, N):
                        c_test = c.copy()
                        c_test[n1] *= -1
                        c_test[n2] *= -1
                        new_true = eval_sup_norm(c_test)
                        delta = new_true - best_score
                        if delta < best_delta:
                            best_delta = delta
                            best_idx1 = n1
                            best_idx2 = n2
                
                if best_idx1 != -1:
                    c[best_idx1] *= -1
                    c[best_idx2] *= -1
                    best_score += best_delta
                    best_c = c.copy()
                    improved = True
                else:
                    break

    if best_c is None:
        return [1]*24
        
    return [int(x) for x in best_c]
