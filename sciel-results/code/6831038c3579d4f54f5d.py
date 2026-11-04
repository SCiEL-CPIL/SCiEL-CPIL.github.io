import numpy as np
import scipy.optimize
import time

N = 100

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s buffer for safety
    
    def get_correlations_circular(h):
        """
        Compute circular correlations of h with 1-h for all non-zero shifts.
        C(k) = sum_i h[i] * (1 - h[(i+k) % N]) for k = 1, ..., N-1
        Uses FFT for efficiency.
        """
        g = 1.0 - h
        # Circular cross-correlation via FFT
        n_fft = 2 * N
        F_h = np.fft.fft(h, n=n_fft)
        F_g = np.fft.fft(g, n=n_fft)
        # Cross-correlation: sum_i h[i] * g[(i+k) % N]
        # This is the circular cross-correlation of h and g
        corr = np.real(np.fft.ifft(F_h * np.conj(F_g)))[:N]
        # corr[k] = sum_i h[i] * g[(i+k) % N] for k=0..N-1
        # We want non-zero shifts: k=1..N-1
        return corr[1:]

    def project_to_feasible(h):
        """
        Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2.
        """
        h = np.clip(h, 0.0, 1.0)
        for _ in range(50):
            current_sum = np.sum(h)
            diff = N / 2.0 - current_sum
            if abs(diff) < 1e-14:
                break
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
        # Final exact correction
        residual = N / 2.0 - np.sum(h)
        if abs(residual) > 1e-15:
            interior = (h > 1e-10) & (h < 1.0 - 1e-10)
            if np.sum(interior) > 0:
                h[interior] += residual / np.sum(interior)
                h = np.clip(h, 0.0, 1.0)
        return h

    def lse_objective(h, beta):
        """
        LogSumExp of correlations for smooth approximation of max.
        """
        c_all = get_correlations_circular(h)
        max_c = np.max(c_all)
        if max_c > 500:
            max_c = 500.0
        lse = max_c + np.log(np.sum(np.exp(beta * (c_all - max_c)))) / beta
        return lse

    def optimize_instance(h0, beta, max_iter, ftol=1e-10, gtol=1e-10):
        """
        Optimize using L-BFGS-B with projection-based constraint handling.
        """
        bounds = [(0, 1) for _ in range(N)]
        
        def obj(h):
            h_proj = project_to_feasible(h)
            return lse_objective(h_proj, beta)

        try:
            res = scipy.optimize.minimize(
                obj, 
                h0, 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
            )
            h_opt = res.x
        except Exception:
            h_opt = h0
        
        h_opt = project_to_feasible(h_opt)
        return h_opt

    def local_search_polish(h, max_iters=2000):
        """
        Local search: try small perturbations to improve max overlap.
        """
        best = h.copy()
        best_score = np.max(get_correlations_circular(best))
        rng = np.random.default_rng(123)
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Pick a random pair and try to adjust
            i = rng.integers(N)
            j = rng.integers(N)
            if i == j:
                continue
            
            # Try small transfer from one to another
            delta = rng.uniform(-0.05, 0.05)
            new_h = best.copy()
            new_h[i] += delta
            new_h[j] -= delta
            new_h = project_to_feasible(new_h)
            
            new_score = np.max(get_correlations_circular(new_h))
            if new_score < best_score - 1e-12:
                best = new_h
                best_score = new_score
        
        return best, best_score

    def score_h(h):
        """Calculate the true max overlap score."""
        c_all = get_correlations_circular(h)
        return np.max(c_all)

    best_h = np.full(N, 0.5)
    best_score = score_h(best_h)  # Should be N/4 = 25.0

    rng = np.random.default_rng(42)
    
    # Multi-restart strategy with varying beta and initializations
    restarts = []
    
    # 1. Uniform start
    restarts.append((np.full(N, 0.5), 5.0, 800))
    
    # 2-6. Random perturbations with increasing beta
    for i in range(5):
        h_rand = rng.uniform(0.1, 0.9, size=N)
        h_rand = project_to_feasible(h_rand)
        beta = 5.0 + i * 5.0
        restarts.append((h_rand, beta, 800))
        
    # 7-12. Bimodal initializations with higher beta
    for i in range(6):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        h_rand = np.where(h_rand > 0.5, 0.8, 0.2)
        h_rand = project_to_feasible(h_rand)
        beta = 20.0 + i * 5.0
        restarts.append((h_rand, beta, 600))
    
    # 13-18. Smooth sinusoidal initializations
    for freq in [1, 2, 3, 5, 7, 10]:
        x = np.linspace(0, 2*np.pi*freq, N, endpoint=False)
        h_sine = 0.5 + 0.3*np.sin(x)
        h_sine = project_to_feasible(h_sine)
        beta = 10.0 + freq
        restarts.append((h_sine, beta, 600))

    # Run all restarts
    for idx, (h0, beta, max_iter) in enumerate(restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        h_opt = optimize_instance(h0, beta, max_iter)
        current_score = score_h(h_opt)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h_opt.copy()
            
    # High-precision polish pass on the best solution
    if time.time() - start_time < time_limit - 30.0:
        # Use very high beta and tight tolerances
        polish_beta = 100.0
        h_polished = optimize_instance(best_h, polish_beta, 2000, ftol=1e-12, gtol=1e-12)
        polished_score = score_h(h_polished)
        
        if polished_score < best_score:
            best_score = polished_score
            best_h = h_polished.copy()
    
    # Local search polish
    if time.time() - start_time < time_limit - 10.0:
        h_ls, ls_score = local_search_polish(best_h, max_iters=3000)
        if ls_score < best_score:
            best_score = ls_score
            best_h = h_ls.copy()
    
    # Final constraint verification
    best_h = project_to_feasible(best_h)
    
    return best_h
