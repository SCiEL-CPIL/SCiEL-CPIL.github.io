"""Model-and-optimize bootstrap: parameterize step heights and optimize the objective."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    
    Strategy (STRUCTURAL CHANGE):
    1. Use binary sign patterns (±1) with DC offset.
    2. For each pattern, compute circ_corr via FFT and find optimal DC analytically.
    3. Optimize sign patterns via stochastic local search + simulated annealing.
    4. Final polish with continuous coordinate descent.
    """
    start_time = time.time()
    DEADLINE = 140  # 10s safety margin
    
    best_value = np.inf
    best_h = np.ones(N).astype(float)
    
    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid
    
    val_seed, _ = evaluate(best_h)
    best_value = val_seed
    
    # Precompute FFT size
    n_fft = 1 << (2 * N - 1).bit_length()
    
    def circ_corr_fft(h):
        """Compute circular autocorrelation via FFT."""
        H = np.fft.rfft(h, n=n_fft)
        corr = np.fft.irfft(H * np.conj(H), n=n_fft)
        return corr[:N]
    
    def evaluate_binary_pattern(signs):
        """
        For a binary pattern (±1), find optimal DC offset and evaluate.
        h = signs + c
        circ_corr_k(h) = circ_corr_k(signs) + 2c*sum(signs) + N*c^2
        For k=0: N + 2c*S + N*c^2 where S=sum(signs)
        For k>0: corr_k + 2c*S + N*c^2
        max_k = max(N + 2c*S + N*c^2, max_{k>0}(corr_k + 2c*S + N*c^2))
              = max(N, max_{k>0}(corr_k)) + 2c*S + N*c^2
        sum(h) = S + N*c
        value(c) = 2N * (M + 2c*S + N*c^2) / (S + N*c)^2 where M = max(N, max_{k>0} corr_k)
        
        We want to minimize this over c.
        """
        S = np.sum(signs)
        corr = circ_corr_fft(signs.astype(float))
        M = max(N, np.max(corr[1:]))  # max over all k including k=0 (which is N)
        
        # If M <= 0, we can make value negative... but abs() makes it positive
        # Actually M >= N > 0 always since corr[0] = N
        
        # Minimize f(c) = 2N*(M + 2c*S + N*c^2) / (S + N*c)^2
        # Try a range of c values
        # Optimal c is where derivative = 0
        # f'(c) = 2N * [(2S + 2Nc)(S+Nc)^2 - (M+2cS+Nc^2)*2N(S+Nc)] / (S+Nc)^4
        #       = 2N * (S+Nc) * [(2S+2Nc)(S+Nc) - 2N(M+2cS+Nc^2)] / (S+Nc)^4
        #       = 2N * [(2S+2Nc)(S+Nc) - 2N(M+2cS+Nc^2)] / (S+Nc)^3
        # Numerator: (2S+2Nc)(S+Nc) - 2N(M+2cS+Nc^2)
        #   = 2S^2 + 2SNc + 2NSc + 2N^2c^2 - 2NM - 4NcS - 2N^2c^2
        #   = 2S^2 + 4SNc - 4NcS - 2NM
        #   = 2S^2 - 2NM
        #   = 2(S^2 - NM)
        # So f'(c) = 2N * 2(S^2 - NM) / (S+Nc)^3 = 4N(S^2 - NM)/(S+Nc)^3
        
        # If S^2 > NM: f' > 0 for c > -S/N, so minimum at c -> -S/N (but sum->0)
        # If S^2 < NM: f' < 0 for c > -S/N, so f decreases as c increases
        # If S^2 = NM: f is constant
        
        # Actually wait, this means f is monotone! Let me recheck...
        # The key is that for binary patterns, the optimal c depends on the sign of (S^2 - NM)
        
        # Let's just try several c values
        best_c_val = np.inf
        best_c = 0.0
        
        # If S^2 > NM, optimal c is near -S/N (small sum)
        # If S^2 < NM, optimal c is large (large sum)
        
        # Try a range
        if abs(S) > 1e-10:
            c_candidates = np.linspace(-S/N - 5, -S/N + 50, 500)
            # Also try large c
            c_candidates = np.concatenate([c_candidates, np.linspace(50, 200, 100)])
        else:
            c_candidates = np.linspace(-5, 200, 500)
        
        for c in c_candidates:
            s_h = S + N * c
            if abs(s_h) < 1e-10:
                continue
            max_corr = M + 2*c*S + N*c*c
            val = abs(2.0 * N * max_corr / (s_h * s_h))
            if val < best_c_val:
                best_c_val = val
                best_c = c
        
        # Also try negative c range if S < 0
        if S < 0:
            c_candidates2 = np.linspace(-200, -S/N + 5, 200)
            for c in c_candidates2:
                s_h = S + N * c
                if abs(s_h) < 1e-10:
                    continue
                max_corr = M + 2*c*S + N*c*c
                val = abs(2.0 * N * max_corr / (s_h * s_h))
                if val < best_c_val:
                    best_c_val = val
                    best_c = c
        
        h = signs.astype(float) + best_c
        return best_c_val, best_c, h
    
    # Phase 1: Generate and optimize binary patterns
    rng = np.random.RandomState(123)
    
    # Start with random patterns
    n_patterns = 200
    best_pattern = None
    best_pattern_val = np.inf
    
    for p in range(n_patterns):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Generate random pattern
        signs = rng.choice([-1, 1], size=N)
        
        val, c, h = evaluate_binary_pattern(signs)
        if val < best_pattern_val:
            best_pattern_val = val
            best_pattern = signs.copy()
            best_c = c
            best_h = h.copy()
            best_value = val
        
        if best_value < 1.1:
            break
    
    # Phase 2: Local search on sign patterns
    if best_pattern is not None and time.time() - start_time < DEADLINE - 15:
        signs = best_pattern.copy()
        current_val = best_pattern_val
        T = 0.01  # Temperature for simulated annealing
        
        for it in range(2000):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            # Flip a random sign
            idx = rng.randint(N)
            signs[idx] *= -1
            
            new_val, new_c, new_h = evaluate_binary_pattern(signs)
            
            # Accept if better, or with probability exp(-delta/T)
            if new_val < current_val:
                current_val = new_val
                if new_val < best_value:
                    best_value = new_val
                    best_h = new_h.copy()
            else:
                delta = new_val - current_val
                if delta < 0.5:  # Don't accept too bad
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_val = new_val
                    else:
                        signs[idx] *= -1  # Revert
                else:
                    signs[idx] *= -1  # Revert
            
            # Cool down
            T *= 0.999
            T = max(T, 0.001)
    
    # Phase 3: Multi-flip local search
    if time.time() - start_time < DEADLINE - 10:
        signs = best_pattern.copy()
        current_val = best_pattern_val
        
        for it in range(500):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Flip 2-3 random signs
            n_flips = rng.randint(2, 4)
            indices = rng.choice(N, size=n_flips, replace=False)
            signs[indices] *= -1
            
            new_val, new_c, new_h = evaluate_binary_pattern(signs)
            
            if new_val < current_val:
                current_val = new_val
                if new_val < best_value:
                    best_value = new_val
                    best_h = new_h.copy()
            else:
                signs[indices] *= -1  # Revert
    
    # Phase 4: Continuous coordinate descent polish
    if time.time() - start_time < DEADLINE - 5:
        h_curr = best_h.copy()
        
        def lse_surrogate_and_grad(h, beta):
            s = np.sum(h)
            if abs(s) < 1e-10:
                return 1e10, np.zeros_like(h)
            
            conv = np.convolve(h, h)
            max_conv = np.max(conv)
            shifted = beta * (conv - max_conv)
            exp_shifted = np.exp(shifted)
            sum_exp = np.sum(exp_shifted)
            
            if sum_exp < 1e-300:
                soft_max = max_conv
                weights = np.zeros_like(conv)
                weights[np.argmax(conv)] = 1.0
            else:
                soft_max = max_conv + np.log(sum_exp) / beta
                weights = exp_shifted / sum_exp
            
            n_fft_local = 1 << (2 * N - 1).bit_length()
            W = np.fft.rfft(weights, n=n_fft_local)
            H = np.fft.rfft(h, n=n_fft_local)
            corr = np.fft.irfft(W * np.conj(H), n=n_fft_local)
            grad_soft_max = 2.0 * corr[:N]
            
            f = 2.0 * N * soft_max / (s * s)
            grad_f = (2.0 * N / (s * s)) * grad_soft_max - (4.0 * N * soft_max / (s ** 3))
            
            return f, grad_f
        
        for beta in [100, 300, 1000]:
            if time.time() - start_time > DEADLINE - 5:
                break
            
            lr = 0.005
            momentum = 0.9
            v = np.zeros_like(h_curr)
            
            for it in range(300):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                f, grad = lse_surrogate_and_grad(h_curr, beta)
                
                if not np.isfinite(f):
                    break
                
                v = momentum * v + (1 - momentum) * grad
                h_new = h_curr - lr * v
                
                val_new, valid_new = evaluate(h_new)
                
                if valid_new and np.isfinite(val_new) and val_new < best_value:
                    best_value = val_new
                    best_h = h_new.copy()
                    h_curr = h_new
                else:
                    lr *= 0.7
                    if lr < 1e-9:
                        break
    
    return best_h
# EVOLVE-BLOCK-END
