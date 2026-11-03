from scipy.optimize import minimize
import numpy as np
import time

N = 600
TIME_LIMIT = 135.0 # Safety margin before 150s kill

def fft_autocorr(f):
    """Compute normalized autocorrelation using FFT."""
    n = len(f)
    # Pad to next power of 2 for efficiency, or use n*2 for full linear autocorr
    # For circular autocorrelation (which is what fftconvolve with 'full' or similar might imply if not careful),
    # but the task implies standard autocorrelation. 
    # Let's compute linear autocorrelation.
    # f is length N.
    # We want R[k] = sum f[i] f[i+k].
    # Using FFT: IFFT(|FFT(f)|^2) gives circular autocorrelation.
    # To get linear, we need to zero-pad to length 2N.
    f_pad = np.zeros(2 * n)
    f_pad[:n] = f
    F = np.fft.rfft(f_pad)
    acf = np.fft.irfft(F * np.conj(F))
    # The first N values are the positive lags (including 0).
    # acf[0] is sum(f^2) ? No, acf[0] = sum(f[i]^2) if we used energy? 
    # Wait, autocorrelation definition: R[k] = sum f[i] f[i+k].
    # If we use f_pad, acf[0] = sum(f[i]^2).
    # acf[k] = sum f[i] f[i+k] for k=0..N-1.
    # Note: The "normalized maximum" usually implies dividing by max possible or sum?
    # Task says: "minimize the normalized maximum of its autoconvolution".
    # "autoconvolution" usually means (f*f)[k] = sum f[i] f[k-i].
    # For real signals, autoconvolution is related to autocorrelation.
    # Let's assume the score is max(autoconv) / (sum(f))^2 or similar.
    # The seed code used: score = smooth_max_val / (s * s) where s = sum(f).
    # So we need autoconvolution.
    # Autoconvolution C[k] = sum_i f[i] f[k-i].
    # This is equivalent to correlation if we reverse one, but for symmetric distributions it's similar.
    # Let's compute C = IFFT(FFT(f) * FFT(f)) for circular? No, linear.
    # Linear autoconv:
    F_full = np.fft.fft(f, n=2*n)
    C = np.fft.ifft(F_full * F_full).real
    return C[:n]

def compute_score(f):
    """Compute the raw score: normalized maximum of autoconvolution."""
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    conv = fft_autocorr(f)
    return np.max(conv) / (s * s)

def initial_guess_parabolic():
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = 1.0 - (4.0 * x) ** 2
    f[f < 0] = 0.0
    return f

def initial_guess_cosine_wave():
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = 0.5 + 0.5 * np.cos(4.0 * np.pi * x)
    return f

def parametric_to_step(params, n_c):
    """
    params: [c1, c2, ..., cn, w1, w2, ..., wn, h1, h2, ..., hn]
    """
    centers = params[:n_c]
    widths = params[n_c:2*n_c]
    heights = params[2*n_c:3*n_c]
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    f = np.zeros(N)
    for c, w, h in zip(centers, widths, heights):
        f += h * np.exp(-0.5 * ((x - c) / w) ** 2)
    return f

def objective(params, n_c, beta=10.0):
    """
    Objective for parametric optimization.
    Minimizes smooth max of autoconv + penalty for negative values.
    """
    f = parametric_to_step(params, n_c)
    
    # Penalty for negative values
    neg_part = np.maximum(-f, 0.0)
    penalty = 10.0 * np.sum(neg_part**2)
    
    s = np.sum(f)
    if s < 1e-15:
        return 1e10 + penalty
        
    conv = fft_autocorr(f)
    max_c = np.max(conv)
    if max_c < 1e-15:
        return penalty
    
    # Smooth max
    shifted = conv - max_c
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    smooth_max_val = max_c + log_sum
    
    score = smooth_max_val / (s * s)
    return score + penalty

def chebyshev_to_step(coeffs, n=600):
    """
    Convert Chebyshev coefficients to step function values on [-1/4, 1/4].
    coeffs: array of length K+1, coeffs[k] is coefficient for T_k.
    The domain is mapped to [-1, 1] for Chebyshev polynomials.
    """
    # x in [-1/4, 1/4]
    x = np.linspace(-0.25, 0.25, n, endpoint=False)
    # Map to [-1, 1] for Chebyshev
    t = 4.0 * x  # x in [-0.25, 0.25] -> t in [-1, 1]
    
    # Compute Chebyshev polynomials T_k(t)
    # Using recurrence: T_0 = 1, T_1 = t, T_k = 2t T_{k-1} - T_{k-2}
    K = len(coeffs) - 1
    f = np.zeros(n)
    if K >= 0:
        f += coeffs[0] * np.ones(n)
    if K >= 1:
        f += coeffs[1] * t
    if K >= 2:
        T_prev2 = np.ones(n)
        T_prev1 = t
        for k in range(2, K + 1):
            T_curr = 2.0 * t * T_prev1 - T_prev2
            f += coeffs[k] * T_curr
            T_prev2 = T_prev1
            T_prev1 = T_curr
            
    return f

def chebyshev_objective(coeffs, n=600, beta=50.0, penalty_weight=1.0):
    """
    Objective function for Chebyshev coefficient optimization.
    Minimizes smooth max of autoconvolution + penalty for negative values.
    """
    f = chebyshev_to_step(coeffs, n)
    
    # Penalty for negative values to encourage non-negativity
    neg_part = np.maximum(-f, 0.0)
    penalty = penalty_weight * np.sum(neg_part**2)
    
    # Compute autoconvolution
    s = np.sum(f)
    if s < 1e-15:
        return 1e10 + penalty
    
    conv = fft_autocorr(f)
    
    # Smooth max
    max_c = np.max(conv)
    if max_c < 1e-15:
        return penalty
    
    shifted = conv - max_c
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    smooth_max_val = max_c + log_sum
    
    # Normalized score
    score = smooth_max_val / (s * s)
    
    return score + penalty

def chebyshev_objective(coeffs, n=600, beta=50.0, penalty_weight=1.0):
    """
    Objective function for Chebyshev coefficient optimization.
    """
    f = chebyshev_to_step(coeffs, n)
    
    # Penalty for negative values
    neg_part = np.maximum(-f, 0.0)
    penalty = penalty_weight * np.sum(neg_part**2)
    
    s = np.sum(f)
    if s < 1e-15:
        return 1e10 + penalty
    
    conv = fft_autocorr(f)
    
    # Smooth max
    max_c = np.max(conv)
    if max_c < 1e-15:
        return penalty
    
    shifted = conv - max_c
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    smooth_max_val = max_c + log_sum
    
    score = smooth_max_val / (s * s)
    
    return score + penalty

def optimize_chebyshev(start_time, max_time, n_coeffs=15, rng=None):
    """
    Optimize Chebyshev coefficients to find a good initial guess.
    """
    if rng is None:
        rng = np.random.RandomState(123)
    
    best_f = None
    best_score = np.inf
    
    # Try a few different initializations and orders
    orders_to_try = [n_coeffs, n_coeffs + 5] if n_coeffs + 5 <= 30 else [n_coeffs]
    
    for K in orders_to_try:
        n_c = K + 1
        # Initial guess: mostly flat with small perturbations
        x0 = np.zeros(n_c)
        x0[0] = 1.0
        for i in range(1, n_c):
            x0[i] = rng.normal(0, 0.1)
            
        try:
            res = minimize(
                chebyshev_objective, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'maxfun': 500}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_f = chebyshev_to_step(res.x, N)
                
        except Exception:
            pass
        
        # Check time against current time, not start_time
        if time.time() > max_time:
            break
            
    if best_f is not None:
        best_f = np.maximum(best_f, 0.0)
        s = np.sum(best_f)
        if s > 1e-12:
            best_f = best_f * (N / s)
        return best_f
        
    return None

def peak_shaving_sa(f, start_time, deadline, rng, T_init=0.06, cooling=0.9975):
    """
    Simulated annealing to shave peaks in autoconvolution.
    """
    current_f = f.copy()
    current_score = compute_score(current_f)
    best_f = current_f.copy()
    best_score = current_score
    T = T_init
    
    n = len(current_f)
    while time.time() < deadline:
        # Pick a random bin to reduce (peak shaving)
        # Or pick a bin with high contribution to peak?
        # Simple random move: reduce one bin, increase another (mass conservation)
        i = rng.randint(n)
        j = rng.randint(n)
        if i == j:
            continue
            
        # Try to move mass from i to j
        # We want to reduce peaks, so maybe reduce high value bins?
        # Let's just do random exchange with small amount
        amount = rng.uniform(0.01, 0.05) * np.mean(current_f)
        
        if current_f[i] < amount:
            continue
            
        new_f = current_f.copy()
        new_f[i] -= amount
        new_f[j] += amount
        
        new_score = compute_score(new_f)
        
        if new_score < current_score or rng.rand() < np.exp(-(new_score - current_score) / T):
            current_f = new_f
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_f = current_f.copy()
        
        T *= cooling
        if T < 1e-6:
            T = 1e-6
            
    return best_f, best_score

def spectral_refinement(f, start_time, deadline):
    """
    Refine by flattening the power spectrum (which relates to autoconv).
    Uses local random steps.
    """
    current_f = f.copy()
    best_f = current_f.copy()
    best_score = compute_score(current_f)
    local_rng = np.random.RandomState(42)
    
    while time.time() < deadline:
        # Simple local search step
        i = local_rng.randint(len(current_f))
        j = local_rng.randint(len(current_f))
        if i == j: continue
        
        amount = 0.01 * np.mean(current_f)
        if current_f[i] < amount: continue
        
        new_f = current_f.copy()
        new_f[i] -= amount
        new_f[j] += amount
        
        new_score = compute_score(new_f)
        if new_score < best_score:
            best_score = new_score
            best_f = new_f.copy()
            current_f = new_f
            
    return best_f, best_score

def polish_pass(f, start_time, deadline):
    """
    Final polish: coordinate descent or small SA.
    """
    current_f = f.copy()
    best_f = current_f.copy()
    best_score = compute_score(current_f)
    
    # Coordinate descent: try to adjust each bin slightly
    n = len(current_f)
    step = 0.05 * np.mean(current_f)
    
    while time.time() < deadline:
        improved = False
        for i in range(n):
            # Try decreasing
            if current_f[i] > step:
                new_f = current_f.copy()
                new_f[i] -= step
                # Distribute to neighbors? Or just keep sum same?
                # If we just decrease, sum changes. 
                # The score is normalized by (sum f)^2.
                # Decreasing f[i] decreases sum, which increases the denominator (good) 
                # but also decreases numerator (autoconv).
                # Let's check both directions.
                new_score = compute_score(new_f)
                if new_score < best_score:
                    best_score = new_score
                    best_f = new_f.copy()
                    current_f = new_f
                    improved = True
                    break # Restart loop for efficiency? Or continue?
            
            # Try increasing
            new_f = current_f.copy()
            new_f[i] += step
            new_score = compute_score(new_f)
            if new_score < best_score:
                best_score = new_score
                best_f = new_f.copy()
                current_f = new_f
                improved = True
                break
        
        if not improved:
            # If no single step improves, try smaller step or break
            step *= 0.9
            if step < 1e-5:
                break
                
    return best_f, best_score

def construct():
    start_time = time.time()
    rng = np.random.RandomState(7)

    # ---- Phase 1: Generate diverse initializations ----
    best_f = np.ones(N)
    best_score = compute_score(best_f)

    # Parabolic
    f_par = initial_guess_parabolic()
    sc = compute_score(f_par)
    if sc < best_score:
        best_f, best_score = f_par, sc

    # Cosine wave
    f_cos = initial_guess_cosine_wave()
    sc = compute_score(f_cos)
    if sc < best_score:
        best_f, best_score = f_cos, sc

    # Box functions
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
    for w in [0.08, 0.12, 0.16, 0.2, 0.25, 0.3]:
        f_box = np.zeros(N)
        f_box[np.abs(x) <= w / 2.0] = 1.0
        sc = compute_score(f_box)
        if sc < best_score:
            best_f, best_score = f_box, sc

    # Parametric cosine bumps with Nelder-Mead
    if time.time() - start_time < TIME_LIMIT - 60:
        for n_c in [3, 4]:
            if time.time() - start_time > TIME_LIMIT - 50:
                break
            best_params = None
            best_obj = np.inf
            for beta in [10.0, 25.0]:
                if time.time() - start_time > TIME_LIMIT - 40:
                    break
                for restart in range(3):
                    if time.time() - start_time > TIME_LIMIT - 45:
                        break
                    if restart == 0:
                        centers = np.linspace(-0.15, 0.15, n_c)
                        widths = np.full(n_c, 0.15)
                        heights = np.full(n_c, 1.0)
                    else:
                        centers = rng.uniform(-0.2, 0.2, size=n_c)
                        widths = rng.uniform(0.04, 0.28, size=n_c)
                        heights = rng.uniform(0.3, 2.5, size=n_c)
                    x0 = np.concatenate([centers, widths, heights])
                    try:
                        result = minimize(
                            objective, x0, args=(n_c, beta),
                            method='Nelder-Mead',
                            options={'maxiter': 600, 'xatol': 1e-6,
                                     'fatol': 1e-8, 'adaptive': True}
                        )
                        if result.fun < best_obj:
                            best_obj = result.fun
                            best_params = result.x
                    except Exception:
                        pass
            if best_params is not None:
                f = parametric_to_step(best_params, n_c)
                actual = compute_score(f)
                if actual < best_score:
                    best_f, best_score = f, actual

    # ---- Phase 1.5: Chebyshev Low-Rank Optimization ----
    # This reduces dimensionality and might find a better global basin
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 30.0:
        cheb_deadline = time.time() + min(remaining * 0.2, 20.0)
        f_cheb = optimize_chebyshev(start_time, cheb_deadline, n_coeffs=15, rng=rng)
        if f_cheb is not None:
            sc_cheb = compute_score(f_cheb)
            if sc_cheb < best_score:
                best_f, best_score = f_cheb, sc_cheb

    # ---- Phase 2: Simulated-annealing peak-shaving (the key improvement) ----
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 20.0:
        sa_time = min(remaining * 0.45, 55.0)
        sa_deadline = time.time() + sa_time
        sa_f, sa_score = peak_shaving_sa(
            best_f, start_time, sa_deadline, rng=rng,
            T_init=0.06, cooling=0.9975
        )
        if sa_score < best_score:
            best_f, best_score = sa_f, sa_score

    # ---- Phase 3: Spectral refinement ----
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 15.0:
        spec_time = min(remaining * 0.25, 30.0)
        spec_deadline = time.time() + spec_time
        spec_f, spec_score = spectral_refinement(
            best_f, start_time, spec_deadline
        )
        if spec_score < best_score:
            best_f, best_score = spec_f, spec_score

    # ---- Phase 4: Multi-scale polish ----
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 8.0:
        polish_deadline = time.time() + min(remaining * 0.30, 35.0)
        pol_f, pol_score = polish_pass(
            best_f, start_time, polish_deadline
        )
        if pol_score < best_score:
            best_f, best_score = pol_f, pol_score

    # ---- Phase 5: Final SA polish if time remains ----
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 12.0:
        final_deadline = time.time() + min(remaining * 0.40, 25.0)
        fin_f, fin_score = peak_shaving_sa(
            best_f, start_time, final_deadline, rng=rng,
            T_init=0.02, cooling=0.9985
        )
        if fin_score < best_score:
            best_f, best_score = fin_f, fin_score

    # Ensure nonnegative
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
