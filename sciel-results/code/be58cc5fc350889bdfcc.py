import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    def get_correlations_linear(h):
        """
        Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
        Returns array of C_k for k != 0.
        """
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # c[N-1] is zero-shift. We exclude it.
        # c[0:N-1] are negative shifts, c[N:2N-1] are positive shifts.
        return np.concatenate([c[:N-1], c[N:]])

    def get_correlations_circular(h):
        """
        Compute circular cross-correlation of h with (1-h) for all non-zero shifts.
        Returns array of C_k for k = 1, ..., N-1.
        """
        g = 1.0 - h
        # FFT-based circular cross-correlation
        H = np.fft.fft(h)
        G = np.fft.fft(g)
        C = np.fft.ifft(H * np.conj(G)).real
        # C[0] is zero-shift, C[1:N] are positive shifts, C[N-1::-1] are negative
        # Actually for cross-correlation: C[k] = sum_i h[i] * g[(i-k) % N]
        # We want all non-zero shifts
        return np.concatenate([C[1:], C[:0]])  # Just positive shifts 1 to N-1

    def get_all_circular_overlaps(h):
        """
        Get all circular cross-correlation values for non-zero shifts.
        """
        g = 1.0 - h
        H = np.fft.fft(h)
        G = np.fft.fft(g)
        C = np.fft.ifft(H * np.conj(G)).real
        # C[0] is zero-shift, exclude it
        return C[1:]

    def max_overlap(h):
        """Return the maximum overlap value."""
        c = get_correlations_linear(h)
        if len(c) == 0:
            return 0.0
        return np.max(c)

    def spectral_factorization_init(target_flatness=1.0, noise_level=0.01):
        """
        Initialize h using spectral factorization.
        
        We want the cross-power spectrum |H(f)|^2 * |G(f)|^2 to be flat across
        non-zero frequencies. Since G = 1 - h, we have G_hat = 1 - H_hat (in frequency domain,
        but this isn't quite right since 1 is a constant).
        
        Actually, let's think differently. We want:
        C_k = sum_i h[i] * (1-h)[(i-k) mod N] to be constant for k != 0.
        
        In frequency domain, the circular cross-correlation is:
        C_k = ifft( H * conj(G) )
        
        We want C_k = constant for k != 0, which means the DFT of C should have
        only DC component (and we don't care about C_0).
        
        Let S(f) = H(f) * conj(G(f)) = H(f) * conj(1 - H(f)) for the circular case.
        Wait, G = 1 - h, so G_hat = 1_hat - H_hat. But 1_hat is [N, 0, 0, ...] in frequency.
        So G_hat[0] = N - H_hat[0], G_hat[k] = -H_hat[k] for k != 0.
        
        This is getting complex. Let's use a simpler approach:
        Start with a random h, compute the cross-power spectrum, and try to flatten it
        by adjusting h in the frequency domain.
        """
        rng = np.random.default_rng(42)
        h = 0.5 + noise_level * rng.normal(0, 1, N)
        h = np.clip(h, 0, 1)
        
        # Iteratively adjust to flatten the cross-power spectrum
        for iteration in range(50):
            g = 1.0 - h
            H = np.fft.fft(h)
            G = np.fft.fft(g)
            S = H * np.conj(G)
            
            # Target: S should be flat (constant) for all frequencies
            # But we only care about the resulting C_k being constant for k != 0
            # Let's compute C and try to minimize variance
            C = np.fft.ifft(S).real
            C_nonzero = np.concatenate([C[1:], C[:0]])  # Just positive shifts
            
            # Compute gradient of variance with respect to h
            # This is complex, so let's use a simpler heuristic:
            # If C_k is too high, reduce h at positions that contribute to that shift
            # If C_k is too low, increase h at those positions
            
            mean_C = np.mean(C_nonzero)
            var_C = np.var(C_nonzero)
            
            if var_C < 1e-10:
                break
            
            # Adjust h to reduce variance
            # For each shift k, if C_k > mean, we want to reduce correlation at that shift
            # This means reducing h[i] * g[i-k] for large values
            adjustment = np.zeros(N)
            for k in range(1, N):
                diff = C[k] - mean_C
                if diff > 0:
                    # Reduce correlation: decrease h where g is large, or vice versa
                    for i in range(N):
                        idx_g = (i - k) % N
                        adjustment[i] -= diff * h[i] * g[idx_g] / N
                else:
                    # Increase correlation
                    for i in range(N):
                        idx_g = (i - k) % N
                        adjustment[i] += (-diff) * h[i] * g[idx_g] / N
            
            h = h + 0.01 * adjustment
            h = np.clip(h, 0, 1)
        
        return h

    def project(h):
        """
        Project h onto feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
        Uses bisection on a Lagrange multiplier for exact projection.
        """
        h = np.clip(h, 0.0, 1.0).copy()
        current = np.sum(h)
        
        if abs(current - TARGET_SUM) < 1e-12:
            return h
        
        # Bisection on lambda: find lambda such that sum(clip(h - lambda, 0, 1)) = TARGET_SUM
        # When lambda < 0, we're adding; when lambda > 0, we're subtracting
        lo, hi = -max(h) - 1, 1 - min(h) + 1
        
        def sum_at_lambda(lam):
            return np.sum(np.clip(h - lam, 0, 1))
        
        for _ in range(60):
            mid = (lo + hi) / 2
            s = sum_at_lambda(mid)
            if abs(s - TARGET_SUM) < 1e-12:
                break
            if s > TARGET_SUM:
                lo = mid
            else:
                hi = mid
        
        lam = (lo + hi) / 2
        h = np.clip(h - lam, 0, 1)
        
        # Final adjustment for any residual error
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) > 1e-14:
            # Distribute residual to elements not at bounds
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
        
        return h

    def spectral_objective(h, beta=50):
        """
        Objective based on variance of circular cross-correlation.
        We want C_k to be constant for k != 0, so minimize variance.
        """
        g = 1.0 - h
        H = np.fft.fft(h)
        G = np.fft.fft(g)
        S = H * np.conj(G)
        C = np.fft.ifft(S).real
        C_nonzero = np.concatenate([C[1:], C[N//2+1:]])  # Positive and negative shifts
        
        if len(C_nonzero) == 0:
            return 0.0
        
        mean_C = np.mean(C_nonzero)
        var_C = np.var(C_nonzero)
        
        # Also add a term to keep the mean overlap low
        # The actual score is max(C_k), so we want both low mean and low variance
        return beta * var_C + mean_C

    def objective_lse(x, beta, lam_sum):
        """
        Objective function using LogSumExp approximation of the max overlap.
        """
        h = np.clip(x, 0.0, 1.0)
        overlaps = get_correlations_linear(h)
        
        if len(overlaps) == 0:
            max_ov = 0.0
            lse = 0.0
        else:
            max_ov = np.max(overlaps)
            # Shift for numerical stability
            shifted = overlaps - max_ov
            # Avoid overflow in exp
            # If beta is large, shifted * beta might be large negative, which is fine.
            # If shifted is 0, exp(0)=1.
            # We sum exp(beta * shifted).
            # To avoid underflow/overflow issues, we can just use the formula directly.
            try:
                from scipy.special import logsumexp
                lse = max_ov + logsumexp(beta * shifted) / beta
            except ImportError:
                exp_vals = np.exp(beta * shifted)
                lse = max_ov + np.log(np.sum(exp_vals)) / beta
        
        # Penalty for sum constraint
        sum_err = np.sum(h) - TARGET_SUM
        sum_penalty = lam_sum * (sum_err ** 2)
        
        return lse + sum_penalty

    def objective_spectral(x, beta=50, lam_sum=10):
        """
        Objective combining spectral flatness and max overlap.
        """
        h = np.clip(x, 0.0, 1.0)
        
        # Spectral component: variance of circular cross-correlation
        g = 1.0 - h
        H = np.fft.fft(h)
        G = np.fft.fft(g)
        S = H * np.conj(G)
        C = np.fft.ifft(S).real
        C_nonzero = np.concatenate([C[1:], C[N//2+1:]])
        
        if len(C_nonzero) == 0:
            var_term = 0.0
            mean_term = 0.0
        else:
            var_term = np.var(C_nonzero)
            mean_term = np.mean(C_nonzero)
        
        # Max overlap component
        overlaps = get_correlations_linear(h)
        if len(overlaps) == 0:
            max_ov = 0.0
        else:
            max_ov = np.max(overlaps)
        
        # Penalty for sum constraint
        sum_err = np.sum(h) - TARGET_SUM
        sum_penalty = lam_sum * (sum_err ** 2)
        
        return beta * var_term + mean_term + max_ov + sum_penalty

    def run_optimization(x0, beta, lam_sum, maxiter=200, use_spectral=False):
        """
        Run L-BFGS-B optimization from x0 with given parameters.
        """
        bounds = [(0.0, 1.0) for _ in range(N)]
        
        if use_spectral:
            def obj_func(x):
                return objective_spectral(x, beta=beta, lam_sum=lam_sum)
        else:
            def obj_func(x):
                return objective_lse(x, beta, lam_sum)
        
        try:
            result = minimize(
                obj_func, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return result.x
        except Exception:
            return x0

    def local_search_coordinate(h, max_iters=500):
        """
        Perform coordinate descent local search.
        Tries adjusting individual coordinates and re-projecting.
        """
        current_h = h.copy()
        current_score = max_overlap(current_h)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            improved = False
            # Try adjusting each coordinate slightly
            for i in range(N):
                # Try a few small steps
                for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    new_h = current_h.copy()
                    new_h[i] += delta
                    new_h = project(new_h)
                    
                    new_score = max_overlap(new_h)
                    if new_score < current_score - 1e-12:
                        current_h = new_h
                        current_score = new_score
                        improved = True
                        break
                        
            if not improved:
                break
                
        return current_h, current_score

    # --- Initialization ---
    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.full(N, 0.5))
    
    # 2. Alternating binary-like (smoothed)
    h_alt = np.zeros(N)
    h_alt[::2] = 0.6
    h_alt[1::2] = 0.4
    candidates.append(h_alt)
    
    # 3. Stronger alternating patterns
    h_alt2 = np.zeros(N)
    h_alt2[::2] = 0.8
    h_alt2[1::2] = 0.2
    candidates.append(h_alt2)
    
    h_alt3 = np.zeros(N)
    h_alt3[::2] = 0.9
    h_alt3[1::2] = 0.1
    candidates.append(h_alt3)
    
    # 4. Block patterns
    for block_size in [2, 3, 4, 5]:
        h_block = np.zeros(N)
        for i in range(0, N, block_size * 2):
            h_block[i:i+block_size] = 0.7
            h_block[i+block_size:i+2*block_size] = 0.3
        candidates.append(h_block)
        
    # 5. Smooth "blob" patterns (to encourage positive autocorrelation)
    t = np.arange(N)
    for center in [0, N//4, N//2, 3*N//4]:
        for width in [5, 10, 20]:
            h_blob = 0.5 + 0.3 * np.exp(-((t - center)**2) / (2 * width**2))
            # Shift to mean 0.5? The projection will handle the sum.
            candidates.append(h_blob)

    # 6. Sinusoidal patterns
    for freq in range(1, 10):
        for amp in [0.1, 0.2, 0.3, 0.4]:
            for phase in [0, np.pi/2]:
                trial = 0.5 + amp * np.sin(2*np.pi*freq*t/N + phase)
                candidates.append(trial.copy())
                
    # 7. Random
    for _ in range(20):
        h_rand = rng.uniform(0.0, 1.0, size=N)
        candidates.append(h_rand)
    
    # 8. Spectral factorization initializations
    for noise_level in [0.005, 0.01, 0.02, 0.05]:
        h_spec = spectral_factorization_init(target_flatness=1.0, noise_level=noise_level)
        candidates.append(h_spec)
    
    # 9. Frequency-domain optimized initializations
    # Start with flat power spectrum in frequency domain
    for freq_pattern in ['flat', 'lowpass', 'bandpass']:
        H_target = np.zeros(N, dtype=complex)
        if freq_pattern == 'flat':
            H_target[1:] = 1.0  # Flat across non-DC frequencies
        elif freq_pattern == 'lowpass':
            H_target[1:5] = 1.0
        elif freq_pattern == 'bandpass':
            H_target[2:8] = 1.0
        
        # Inverse FFT to get time-domain signal
        h_init = np.real(np.fft.ifft(H_target))
        # Normalize to have mean 0.5 and values in [0,1]
        h_init = h_init - np.mean(h_init)
        if np.std(h_init) > 0:
            h_init = 0.5 + 0.3 * h_init / np.std(h_init)
        h_init = np.clip(h_init, 0, 1)
        candidates.append(h_init)
        
    # Evaluate and select best initial candidate
    best_h = None
    best_score = np.inf
    
    for h_cand in candidates:
        h_proj = project(h_cand)
        score = max_overlap(h_proj)
        if score < best_score:
            best_score = score
            best_h = h_proj
            
    # --- Optimization Phases ---
    # We increase beta to sharpen the max approximation.
    # We also increase the sum penalty lambda to enforce constraint more strictly as we converge.
    
    beta_schedule = [5, 10, 20, 50, 100, 200]
    
    current_h = best_h.copy()
    
    # Phase 1: Spectral optimization to flatten the cross-power spectrum
    if time.time() < deadline - 30.0:
        for beta_spec in [10, 20, 50]:
            if time.time() > deadline - 30.0:
                break
            lam_spec = 10.0
            new_h = run_optimization(current_h, beta_spec, lam_spec, maxiter=200, use_spectral=True)
            new_h_proj = project(new_h)
            score_new = max_overlap(new_h_proj)
            
            if score_new < best_score:
                best_score = score_new
                best_h = new_h_proj
                current_h = new_h_proj
    
    # Phase 2: Standard LSE optimization
    for beta in beta_schedule:
        if time.time() > deadline - 15.0:
            break
            
        # Increase lambda as beta increases to keep sum constraint tight
        lam_sum = 10.0 * (beta / 5.0)
        
        # Run optimization from current best
        new_h = run_optimization(current_h, beta, lam_sum, maxiter=300, use_spectral=False)
        
        # Project to ensure strict feasibility for evaluation
        new_h_proj = project(new_h)
        score_new = max_overlap(new_h_proj)
        
        if score_new < best_score:
            best_score = score_new
            best_h = new_h_proj
            current_h = new_h_proj
            
        # Try a few random restarts at lower beta to escape local minima
        if beta <= 20:
            for _ in range(3):
                if time.time() > deadline - 15.0:
                    break
                x0 = rng.uniform(0.2, 0.8, size=N)
                x0 = project(x0)
                new_h = run_optimization(x0, beta, lam_sum, maxiter=100, use_spectral=False)
                new_h_proj = project(new_h)
                score_new = max_overlap(new_h_proj)
                if score_new < best_score:
                    best_score = score_new
                    best_h = new_h_proj
                    current_h = new_h_proj

    # --- Local Search Phase ---
    # Use coordinate descent to polish the gradient-optimized solution
    if time.time() < deadline - 10.0:
        ls_h, ls_score = local_search_coordinate(best_h, max_iters=1000)
        if ls_score < best_score:
            best_score = ls_score
            best_h = ls_h

    # --- Final Polish ---
    # Use very high beta and high lambda for final precision
    if time.time() < deadline - 5.0:
        beta_final = 500
        lam_final = 1000.0
        new_h = run_optimization(best_h, beta_final, lam_final, maxiter=500, use_spectral=False)
        new_h_proj = project(new_h)
        score_new = max_overlap(new_h_proj)
        
        if score_new < best_score:
            best_score = score_new
            best_h = new_h_proj
        
        # Also try spectral polish
        if time.time() < deadline - 2.0:
            beta_spec_final = 100
            lam_spec_final = 100.0
            new_h_spec = run_optimization(best_h, beta_spec_final, lam_spec_final, maxiter=300, use_spectral=True)
            new_h_spec_proj = project(new_h_spec)
            score_spec = max_overlap(new_h_spec_proj)
            
            if score_spec < best_score:
                best_score = score_spec
                best_h = new_h_spec_proj

    # Final Validation
    best_h = project(best_h)
    
    # Ensure strict constraints
    assert np.all(best_h >= 0) and np.all(best_h <= 1), "h out of bounds"
    
    # Re-project one more time with tighter tolerance if needed
    diff = TARGET_SUM - np.sum(best_h)
    if abs(diff) > 1e-9:
        mask = (best_h > 1e-6) & (best_h < 1 - 1e-6)
        if np.any(mask):
            adj = diff / np.sum(mask)
            best_h[mask] += adj
            best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
