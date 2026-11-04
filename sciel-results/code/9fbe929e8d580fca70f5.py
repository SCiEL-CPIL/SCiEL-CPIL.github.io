"""
Combined approach:
1. Fast FFT-based correlation computation.
2. Multi-start L-BFGS-B with smooth epigraph formulation.
3. Diverse structured initializations.
4. Efficient local polish with gradient-informed steps.
"""
import numpy as np
import time
import scipy.optimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (150s limit)
    
    target_sum = N / 2.0
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Precompute FFT sizes and indices for fast correlation
    fft_size = 2 * N - 1
    
    def get_correlations_fast(h):
        """Compute linear correlations of h and 1-h for all non-zero shifts using FFT."""
        g = 1.0 - h
        H = np.fft.fft(h, n=fft_size)
        G = np.fft.fft(g, n=fft_size)
        c = np.real(np.fft.ifft(H * np.conj(G)))
        c_pos = c[1:N]
        c_neg = c[fft_size-N+1:fft_size]
        return np.concatenate([c_neg, c_pos])

    def true_score(h):
        """Calculate the raw score (max overlap)."""
        c_all = get_correlations_fast(h)
        return np.max(c_all)

    def project_to_feasible(h):
        """Project h to satisfy sum(h) = N/2 and 0 <= h <= 1."""
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        
        if abs(diff) > 1e-12:
            for _ in range(100):
                if abs(diff) < 1e-15:
                    break
                h += diff / N
                h = np.clip(h, 0.0, 1.0)
                current_sum = np.sum(h)
                diff = target_sum - current_sum
        return h

    def make_objective_epigraph(beta, lambda_sum):
        """
        Epigraph formulation: minimize t + penalty
        subject to c_i <= t for all i (softened with smooth log-sum-exp).
        Variables: [h_0, ..., h_{N-1}, t]
        """
        def obj(x):
            h = x[:N]
            t = x[N]
            
            c_all = get_correlations_fast(h)
            
            # Smooth penalty for c_i > t using always-smooth log-sum-exp
            excess = c_all - t
            # Shift for numerical stability
            max_excess = np.max(excess)
            shifted = beta * (excess - max_excess)
            # Avoid overflow: clip shifted values
            shifted = np.clip(shifted, -50, 50)
            lse = max_excess + np.log(np.sum(np.exp(shifted))) / beta
            
            sum_h = np.sum(h)
            sum_pen = lambda_sum * (sum_h - target_sum)**2
            
            return t + lse + sum_pen
        return obj

    def make_objective_lse(beta, lambda_sum):
        """LSE formulation for final polish."""
        def obj(h):
            c_all = get_correlations_fast(h)
            max_c = np.max(c_all)
            if beta > 0:
                shifted = beta * (c_all - max_c)
                shifted = np.clip(shifted, -50, 50)
                lse = max_c + np.log(np.sum(np.exp(shifted))) / beta
            else:
                lse = max_c
            
            sum_h = np.sum(h)
            sum_pen = lambda_sum * (sum_h - target_sum)**2
            
            return lse + sum_pen
        return obj

    def local_polish(h, max_iters=300, tol=1e-9):
        """
        Efficient local polish: use gradient information from LSE to guide
        small steps, with occasional random restarts of the local search.
        """
        h = h.copy()
        best_score = true_score(h)
        c_all = get_correlations_fast(h)
        max_c = best_score
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            
            # Find critical shifts (those near the max)
            threshold = max_c - 0.01
            critical_mask = c_all >= threshold
            n_critical = np.sum(critical_mask)
            
            if n_critical == 0:
                break
            
            # Compute gradient of the LSE objective w.r.t. h
            beta_local = 100.0
            excess = c_all - max_c
            max_excess = np.max(excess)
            shifted = beta_local * (excess - max_excess)
            shifted = np.clip(shifted, -50, 50)
            weights = np.exp(shifted) / np.sum(np.exp(shifted))
            
            # Gradient of c_k w.r.t. h_i for each shift k
            # c_k = sum_i h[i] * (1 - h[(i+k) % N])
            # dc_k/dh_i = (1 - h[(i+k)%N]) - h[i] * delta_{i, (j-k)%N} for all j... 
            # Actually: c_k = sum_j h[j] * (1 - h[(j+k)%N])
            # dc_k/dh_i = (1 - h[(i+k)%N]) + sum_j h[j] * (-delta_{(j+k)%N, i})
            #           = (1 - h[(i+k)%N]) - h[(i-k)%N]
            
            grad = np.zeros(N)
            for k in range(N):
                if critical_mask[k]:
                    w = weights[k]
                    if w < 1e-10:
                        continue
                    # dc_k/dh_i = (1 - h[(i+k)%N]) - h[(i-k)%N]
                    i_idx = np.arange(N)
                    grad += w * ((1 - h[(i_idx + k) % N]) - h[(i_idx - k) % N])
            
            # Project gradient to maintain sum constraint
            grad -= np.mean(grad)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            grad /= grad_norm
            
            # Try a few step sizes in the negative gradient direction
            improved = False
            for step_size in [0.02, 0.01, 0.005, 0.002, 0.001]:
                h_trial = h - step_size * grad
                h_trial = project_to_feasible(h_trial)
                
                score_trial = true_score(h_trial)
                if score_trial < best_score - 1e-12:
                    h = h_trial
                    best_score = score_trial
                    c_all = get_correlations_fast(h)
                    max_c = best_score
                    improved = True
                    break
            
            # If no improvement from gradient step, try random perturbations
            if not improved:
                # Try small random perturbations focused on critical regions
                for _ in range(5):
                    h_trial = h + np.random.normal(0, 0.005, N)
                    h_trial = project_to_feasible(h_trial)
                    score_trial = true_score(h_trial)
                    if score_trial < best_score - 1e-12:
                        h = h_trial
                        best_score = score_trial
                        c_all = get_correlations_fast(h)
                        max_c = best_score
                        improved = True
                        break
            
            if not improved:
                # Try coordinate descent on a few random coordinates
                for _ in range(3):
                    i = np.random.randint(N)
                    for delta in [0.01, -0.01, 0.005, -0.005]:
                        h_trial = h.copy()
                        h_trial[i] += delta
                        if h_trial[i] < 0 or h_trial[i] > 1:
                            continue
                        diff = target_sum - np.sum(h_trial)
                        if abs(diff) > 1e-10:
                            others = np.ones(N)
                            others[i] = 0
                            h_trial[others.astype(bool)] += diff / (N - 1)
                            h_trial = np.clip(h_trial, 0.0, 1.0)
                            diff2 = target_sum - np.sum(h_trial)
                            h_trial += diff2 / N
                            h_trial = np.clip(h_trial, 0.0, 1.0)
                        
                        score_trial = true_score(h_trial)
                        if score_trial < best_score - 1e-12:
                            h = h_trial
                            best_score = score_trial
                            c_all = get_correlations_fast(h)
                            max_c = best_score
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return h, best_score

    # Initial candidates
    rng = np.random.default_rng(42)
    candidates = []
    
    # Uniform
    candidates.append(np.full(N, 0.5))
    
    # Random
    for _ in range(8):
        h_rand = rng.uniform(0.1, 0.9, N)
        candidates.append(h_rand)
        
    # Sine waves with various frequencies
    t_arr = np.arange(N)
    for freq in [1, 2, 3, 4, 5, 7, 10, 15]:
        candidates.append(0.5 + 0.4 * np.sin(2 * np.pi * freq * t_arr / N))
        candidates.append(0.5 + 0.4 * np.cos(2 * np.pi * freq * t_arr / N))
    
    # Thue-Morse sequence
    def thue_morse_seq(n):
        tm = np.zeros(n, dtype=int)
        for i in range(1, n):
            tm[i] = tm[i >> 1] ^ (i & 1)
        return tm
    
    tm = thue_morse_seq(N)
    candidates.append(0.5 + 0.45 * (2 * tm - 1))
    
    # Random binary
    h_bin = rng.integers(0, 2, N).astype(float)
    candidates.append(0.5 + 0.45 * (2 * h_bin - 1))
    
    # Van der Corput low-discrepancy sequence
    def van_der_corput(n, base=2):
        vdc = np.zeros(n)
        for i in range(1, n):
            f = 1.0
            j = i
            while j > 0:
                f /= base
                vdc[i] += f * (j % base)
                j //= base
        return vdc
    
    h_vdc = van_der_corput(N)
    h_vdc = (h_vdc - np.min(h_vdc)) / (np.max(h_vdc) - np.min(h_vdc) + 1e-9)
    candidates.append(h_vdc)
    
    # Halving pattern
    h_half = np.zeros(N)
    h_half[::2] = 0.9
    h_half[1::2] = 0.1
    candidates.append(h_half)

    best_h = None
    best_score = float('inf')
    
    # Phase 1: Evaluate candidates
    for h_cand in candidates:
        h_proj = project_to_feasible(h_cand)
        score = true_score(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj.copy()

    # Phase 2: Multi-start optimization using epigraph formulation
    starts = [best_h.copy()]
    
    # Add more diverse starts
    for _ in range(12):
        if rng.random() < 0.5:
            h_start = rng.uniform(0.15, 0.85, N)
        else:
            freq = rng.integers(1, 25)
            phase = rng.uniform(0, 2 * np.pi)
            h_start = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
            h_start += rng.normal(0, 0.05, N)
        starts.append(project_to_feasible(h_start))
    
    for start_idx, h_start in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        # Use epigraph formulation with varying beta
        if start_idx == 0:
            beta = 50.0
        elif start_idx < 5:
            beta = 30.0
        else:
            beta = 20.0
        
        # Initialize t to current max overlap
        t_init = true_score(h_start)
        x0 = np.concatenate([h_start, [t_init]])
        bounds_ext = bounds + [(0.0, 2.0)]  # t in [0, 2]
        
        obj_func = make_objective_epigraph(beta, 500.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds_ext, 
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            h_opt = project_to_feasible(res.x[:N])
            score_opt = true_score(h_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # Phase 3: Local polish with gradient-informed steps
    if time.time() < deadline - 15.0:
        h_polished, score_polished = local_polish(best_h, max_iters=200)
        if score_polished < best_score:
            best_score = score_polished
            best_h = h_polished.copy()

    # Phase 4: Final high-precision polish with LSE (very high beta)
    if time.time() < deadline - 8.0:
        obj_func = make_objective_lse(500.0, 5000.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                best_h, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
            )
            h_opt = project_to_feasible(res.x)
            score_opt = true_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Phase 5: Another local polish from slightly perturbed best
    if time.time() < deadline - 4.0:
        h_perturbed = best_h + rng.normal(0, 0.005, N)
        h_perturbed = project_to_feasible(h_perturbed)
        h_final, score_final = local_polish(h_perturbed, max_iters=150)
        if score_final < best_score:
            best_score = score_final
            best_h = h_final.copy()

    # Phase 6: Final tiny LSE polish if time remains
    if time.time() < deadline - 2.0:
        obj_func = make_objective_lse(1000.0, 10000.0)
        try:
            res = scipy.optimize.minimize(
                obj_func, 
                best_h, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 50}
            )
            h_opt = project_to_feasible(res.x)
            score_opt = true_score(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    return best_h
