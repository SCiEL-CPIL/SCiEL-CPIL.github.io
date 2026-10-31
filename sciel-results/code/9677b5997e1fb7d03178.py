import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin for 150s kill

    best_h = np.zeros(N, dtype=np.float64)
    best_score = -np.inf

    def update_best(h, score):
        nonlocal best_h, best_score
        if score > best_score:
            best_score = score
            best_h = h.copy()

    def time_left():
        return deadline - time.time()

    # --- Phase 1: Generate Structured Candidates ---
    
    # 1. Single spike
    h_spike = np.zeros(N)
    h_spike[N//2] = 1.0
    update_best(h_spike, compute_ratio(h_spike))

    # 2. Two-point distributions
    for i in range(N):
        for j in range(i+1, N):
            d = j - i
            if d <= 3:
                a_vals = np.linspace(0.01, 0.99, 100)
            else:
                a_vals = np.linspace(0.01, 0.99, 20)
            
            for a in a_vals:
                h = np.zeros(N)
                h[i] = a
                h[j] = 1.0 - a
                sc = compute_ratio(h)
                update_best(h, sc)
            if time_left() < 30.0:
                return list(best_h)

    # 3. NEW: Spectral basis candidates (DCT/Fourier modes)
    # These create smoother distributions with potentially flatter autoconvolution
    x = np.arange(N)
    
    # DCT-like basis functions
    for k in range(1, 15):
        # Cosine modes
        h_cos = np.cos(2 * np.pi * k * x / N)
        h_cos = np.maximum(h_cos, 0.0)
        s = np.sum(h_cos)
        if s > 1e-10:
            h_cos = h_cos / s
            sc = compute_ratio(h_cos)
            update_best(h_cos, sc)
        
        # Sine modes (shifted to be non-negative)
        h_sin = np.sin(2 * np.pi * k * x / N)
        h_sin = np.maximum(h_sin, 0.0)
        s = np.sum(h_sin)
        if s > 1e-10:
            h_sin = h_sin / s
            sc = compute_ratio(h_sin)
            update_best(h_sin, sc)
    
    # Mixed spectral candidates: sum of a few modes
    for k1 in range(1, 8):
        for k2 in range(k1+1, 12):
            h_mix = np.cos(2 * np.pi * k1 * x / N) + np.cos(2 * np.pi * k2 * x / N)
            h_mix = np.maximum(h_mix, 0.0)
            s = np.sum(h_mix)
            if s > 1e-10:
                h_mix = h_mix / s
                sc = compute_ratio(h_mix)
                update_best(h_mix, sc)
    
    if time_left() < 30.0:
        return list(best_h)

    # 4. Sidon-like sets (Sparse, distinct sums)
    def find_sidon_set_greedy(n, start_idx=0, step=1):
        A = []
        used_sums = set()
        for x_val in range(start_idx, N, step):
            valid = True
            new_sums = []
            for a in A:
                s = a + x_val
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            if 2 * x_val in used_sums:
                valid = False
            if valid:
                A.append(x_val)
                used_sums.update(new_sums)
                used_sums.add(2 * x_val)
        return A

    for step in [1, 2, 3]:
        for start in range(0, 5, step):
            A = find_sidon_set_greedy(N, start_idx=start, step=step)
            if len(A) >= 2:
                h = np.zeros(N)
                for idx in A:
                    h[idx] = 1.0
                sc = compute_ratio(h)
                update_best(h, sc)
            if time_left() < 30.0:
                return list(best_h)

    # 5. Random Sparse Sets
    rng = np.random.default_rng(42)
    for _ in range(50):
        k = rng.integers(2, 10)
        idx = rng.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        sc = compute_ratio(h)
        update_best(h, sc)
        if time_left() < 30.0:
            return list(best_h)

    # 6. Gaussian-like
    x_center = np.arange(N) - N // 2
    for sigma in np.linspace(0.5, 10.0, 50):
        h = np.exp(-0.5 * (x_center/sigma)**2)
        sc = compute_ratio(h)
        update_best(h, sc)
        if time_left() < 30.0:
            return list(best_h)

    # 7. NEW: Smooth max surrogate optimization
    # Use soft-max to create a differentiable objective for better gradient-based search
    
    def soft_max(x, beta=100.0):
        """Approximate max using log-sum-exp"""
        x_max = np.max(x)
        return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta
    
    def smooth_ratio(h, beta=100.0):
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(h, 0.0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        h_norm = h / l1
        h2_sq = np.sum(h_norm**2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        conv_soft_max = soft_max(conv, beta)
        if conv_soft_max < 1e-15:
            return 0.0
        return h2_sq / conv_soft_max
    
    # Use smooth objective for local optimization
    def local_optimize_smooth(h_init, beta=100.0):
        h = h_init.copy()
        h = np.maximum(h, 0.0)
        s = np.sum(h)
        if s < 1e-15: return h, 0.0
        h = h / s
        
        current_score = smooth_ratio(h, beta)
        best_local_h = h.copy()
        best_local_score = current_score
        
        # Coordinate ascent with line search on smooth objective
        for _ in range(300):
            if time_left() < 15.0:
                break
            
            improved = False
            # Try perturbing a random subset of coordinates
            num_perturb = rng.integers(1, 4)
            indices = rng.choice(N, size=num_perturb, replace=False)
            
            h_new = h.copy()
            for idx in indices:
                h_new[idx] += rng.uniform(-0.1, 0.1)
            
            h_new = np.maximum(h_new, 0.0)
            s_new = np.sum(h_new)
            if s_new < 1e-15:
                continue
            h_new = h_new / s_new
            
            sc_new = smooth_ratio(h_new, beta)
            if sc_new > current_score:
                h = h_new
                current_score = sc_new
                improved = True
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_h = h.copy()
            
            if not improved:
                # Try smaller steps
                small_improved = False
                for _ in range(5):
                    h_new = h.copy()
                    idx = rng.choice(N)
                    h_new[idx] += rng.uniform(-0.01, 0.01)
                    h_new = np.maximum(h_new, 0.0)
                    s_new = np.sum(h_new)
                    if s_new < 1e-15: continue
                    h_new = h_new / s_new
                    sc_new = smooth_ratio(h_new, beta)
                    if sc_new > current_score:
                        h = h_new
                        current_score = sc_new
                        small_improved = True
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_h = h.copy()
                if not small_improved:
                    break
                    
        # Evaluate final result with exact objective
        final_score = compute_ratio(best_local_h)
        return best_local_h, final_score

    # Refine the global best using smooth optimization
    h_opt, sc_opt = local_optimize_smooth(best_h)
    update_best(h_opt, sc_opt)

    # Refine spectral candidates
    x = np.arange(N)
    for k in range(1, 10):
        h_cos = np.cos(2 * np.pi * k * x / N)
        h_cos = np.maximum(h_cos, 0.0)
        s = np.sum(h_cos)
        if s > 1e-10:
            h_cos = h_cos / s
            h_opt, sc_opt = local_optimize_smooth(h_cos)
            update_best(h_opt, sc_opt)
    
    if time_left() < 15.0:
        return list(best_h)

    # Refine some other good candidates from random starts
    for _ in range(15):
        if time_left() < 15.0:
            break
        # Random start: sparse
        k = rng.integers(2, 8)
        idx = rng.choice(N, size=k, replace=False)
        h_rand = np.zeros(N)
        h_rand[idx] = 1.0
        h_opt, sc_opt = local_optimize_smooth(h_rand)
        update_best(h_opt, sc_opt)

    # Refine from the specific 2-point [2/3, 1/3] which was analytically good
    h_2pt = np.zeros(N)
    h_2pt[N//2] = 2.0/3.0
    h_2pt[N//2 + 1] = 1.0/3.0
    h_opt, sc_opt = local_optimize_smooth(h_2pt)
    update_best(h_opt, sc_opt)

    return list(best_h)
