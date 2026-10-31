import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50
DEADLINE_MARGIN = 10.0  # Return 10s before the 150s kill
SOFTMAX_TEMPERATURE = 0.01 # Temperature for smooth max approximation

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    The ratio is homogeneous of degree 0, so scaling h does not change the value.
    We assume h is non-negative.
    """
    h = np.asarray(h, dtype=np.float64)
    # Enforce non-negativity for safety, though optimizers should maintain it
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    
    # Normalize to sum=1 to improve numerical stability and simplify ratio
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    
    # Full convolution
    # Using np.convolve for exactness. N=50 is small.
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    
    if conv_inf < 1e-15:
        return 0.0
        
    return h2_sq / conv_inf

def spectral_to_time(spectral_mags):
    """
    Convert non-negative spectral magnitudes to time-domain heights via IDFT.
    We use the real part of the IDFT and clip to ensure non-negativity.
    To maintain a valid signal, we assume the spectrum is Hermitian.
    """
    # spectral_mags is the magnitude of the FFT coefficients.
    # For a real signal, H[k] = H[N-k].
    # We construct H such that |H[k]| = spectral_mags[k].
    # Phase is set to 0 for simplicity (real, even spectrum).
    H = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        H[k] = spectral_mags[k]
    
    h = np.fft.ifft(H).real
    return h

def smooth_max(x, temp=1.0):
    """
    Approximate max(x) using log-sum-exp for smoothness.
    max(x) ~ temp * log(sum(exp(x/temp)))
    """
    x_max = np.max(x)
    # Numerical stability: subtract max
    return x_max + temp * np.log(np.sum(np.exp((x - x_max) / temp)))

def spectral_objective(log_mags):
    """
    Objective function for spectral optimization.
    We optimize the log-magnitudes of the spectrum to ensure positivity.
    Returns negative ratio (to be minimized).
    """
    mags = np.exp(log_mags)
    if not np.all(np.isfinite(mags)):
        return 1e10
    
    # Convert to time domain
    h = spectral_to_time(mags)
    h = np.maximum(h, 0.0)
    
    if np.sum(h) < 1e-15:
        return 1e10
        
    return -compute_ratio(h)

def neg_ratio_smooth(params, temp=SOFTMAX_TEMPERATURE):
    """
    Objective function for L-BFGS-B using a smooth surrogate for the max.
    We parameterize h using log-space to ensure positivity: h = exp(params).
    """
    h = np.exp(params)
    # Check for overflow
    if not np.all(np.isfinite(h)):
        return 1e10
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    
    conv = np.convolve(h_norm, h_norm, mode='full')
    
    # Use smooth max for the denominator to provide gradients
    conv_soft_max = smooth_max(conv, temp)
    
    if conv_soft_max < 1e-15:
        return 1e10
        
    return -(h2_sq / conv_soft_max)

def spectral_projected_gradient_descent(init_log_mags, max_iters=1000, lr=0.01, deadline=None):
    """
    Perform projected gradient descent on the spectral log-magnitudes.
    """
    log_mags = init_log_mags.copy()
    best_val = spectral_objective(log_mags)
    best_log_mags = log_mags.copy()
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        val = spectral_objective(log_mags)
        if val < best_val:
            best_val = val
            best_log_mags = log_mags.copy()
            
        # Numerical gradient
        grad = np.zeros_like(log_mags)
        for j in range(N):
            log_mags_plus = log_mags.copy()
            log_mags_plus[j] += 1e-5
            val_plus = spectral_objective(log_mags_plus)
            grad[j] = (val_plus - val) / 1e-5
            
        log_mags -= lr * grad
        
        # Project: keep values in a reasonable range to avoid overflow/underflow
        log_mags = np.clip(log_mags, -10, 10)
        
        # Adaptive learning rate
        if i % 10 == 0:
            lr = max(1e-4, lr * 0.95)
            
    return best_log_mags

def neg_ratio(params):
    """
    Objective function for L-BFGS-B.
    We parameterize h using log-space to ensure positivity: h = exp(params).
    Since the ratio is scale-invariant, we can just compute the ratio of exp(params).
    However, L-BFGS-B works better if the objective is somewhat well-scaled.
    Let's normalize inside the objective to keep values bounded.
    """
    h = np.exp(params)
    # Check for overflow
    if not np.all(np.isfinite(h)):
        return 1e10
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    h_norm = h / l1
    return -compute_ratio(h_norm)

def neg_ratio_normalized(x):
    """
    Objective for SLSQP on normalized vector x (sum=1, x>=0).
    """
    h = np.maximum(x, 0)
    s = np.sum(h)
    if s < 1e-15:
        return 1e10
    h_norm = h / s
    return -compute_ratio(h_norm)

def time_to_spectral(h):
    """
    Convert time-domain heights to spectral magnitudes.
    """
    H = np.fft.fft(h)
    mags = np.abs(H)
    # Ensure non-negative
    mags = np.maximum(mags, 1e-10)
    return np.log(mags)

def epigraph_objective(params, temp=0.01):
    """
    Epigraph formulation: maximize h2_sq / t subject to smooth_max(conv) <= t,
    sum(h)=1, h>=0. We use log-space for h.
    params = [log(h_0), ..., log(h_{N-1}), log(t)]
    We minimize the negative of the ratio plus a penalty for constraint violation.
    """
    log_h = params[:N]
    log_t = params[N]
    
    h = np.exp(log_h)
    if not np.all(np.isfinite(h)):
        return 1e10
    
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1e10
    
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    
    conv = np.convolve(h_norm, h_norm, mode='full')
    
    # Smooth max of conv
    conv_soft_max = smooth_max(conv, temp)
    
    t = np.exp(log_t)
    if t < 1e-10 or not np.isfinite(t):
        return 1e10
    
    # Penalty for violating constraint conv_soft_max <= t
    # Use a smooth penalty: exp((conv_soft_max - t)/temp) - 1
    penalty = np.exp((conv_soft_max - t) / temp) - 1.0
    
    # Objective: minimize -h2_sq/t + penalty
    # This is equivalent to maximizing h2_sq/t subject to conv_max <= t
    if h2_sq < 1e-15:
        return 1e10
    
    return -(h2_sq / t) + penalty

def spectral_design_optimize(init_h, deadline, max_iters=500):
    """
    Optimize using epigraph formulation in log-space with temperature annealing.
    Uses log(t) as the last parameter for better conditioning.
    """
    h_init = np.maximum(init_h, 1e-10)
    log_h_init = np.log(h_init)
    
    # Initial t: current conv peak
    h_norm_init = h_init / np.sum(h_init)
    conv_init = np.convolve(h_norm_init, h_norm_init, mode='full')
    t_init = np.max(conv_init) * 1.05  # Slightly above current peak
    
    # Use log(t) for better conditioning
    log_t_init = np.log(t_init)
    
    params_init = np.concatenate([log_h_init, [log_t_init]])
    
    # Multi-stage optimization with decreasing temperature
    temps = [0.05, 0.02, 0.01, 0.005, 0.002]
    current_params = params_init.copy()
    
    for temp in temps:
        if time.time() > deadline:
            break
        
        def objective(params):
            return epigraph_objective(params, temp=temp)
        
        # Use L-BFGS-B with bounds
        # log_h in [-15, 5], log_t in [-10, 1]
        bounds = [(-15, 5)] * N + [(-10, 1.0)]
        
        try:
            res = minimize(objective, current_params, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': max(20, max_iters // len(temps)), 'ftol': 1e-12, 'gtol': 1e-10})
            
            if time.time() > deadline:
                break
                
            current_params = res.x
        except Exception:
            pass
    
    if time.time() > deadline:
        return None
        
    log_h_opt = current_params[:N]
    
    h_opt = np.exp(log_h_opt)
    h_opt = np.maximum(h_opt, 0.0)
    if np.sum(h_opt) > 1e-15:
        h_opt = h_opt / np.sum(h_opt)
    
    return h_opt

def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_MARGIN
    
    def elapsed():
        return time.time() - t0
    
    # Initial best: single spike
    best_h = np.zeros(N, dtype=np.float64)
    best_h[N // 2] = 1.0
    best_score = compute_ratio(best_h)
    
    candidates = []
    
    # Helper to add candidate
    def add_candidate(h):
        nonlocal best_h, best_score
        h = np.maximum(h, 0.0)
        if np.sum(h) < 1e-15:
            return
        sc = compute_ratio(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
        candidates.append((sc, h.copy()))
        
    # --- Phase 1: Generate Structured Candidates ---
    
    # 1. Single spike (already set)
    add_candidate(best_h)
    
    # 2. Two spikes
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 0.5
            h[j] = 0.5
            add_candidate(h)
            if elapsed() > deadline - 5:
                return list(best_h)
                
    # 3. Three spikes
    # Sample a subset of triplets to save time
    # Focus on "spread out" triplets
    for i in range(0, N, 2):
        for j in range(i+2, N, 2):
            for k in range(j+2, N, 2):
                h = np.zeros(N)
                h[i] = 1.0/3
                h[j] = 1.0/3
                h[k] = 1.0/3
                add_candidate(h)
                if elapsed() > deadline - 5:
                    return list(best_h)

    # 4. Sidon-like sets
    # A Sidon set ensures all pair sums are distinct, minimizing collisions in convolution.
    # This tends to lower ||h*h||_inf relative to ||h||_2^2.
    
    def find_sidon_set_greedy(n, start_idx=0, step=1):
        """
        Greedy construction of a Sidon set.
        """
        A = []
        used_sums = set()
        # We want to pick indices from 0 to N-1
        for x in range(start_idx, N, step):
            valid = True
            new_sums = []
            for a in A:
                s = a + x
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            if 2 * x in used_sums:
                valid = False
            if valid:
                A.append(x)
                used_sums.update(new_sums)
                used_sums.add(2 * x)
        return A

    # Try different steps and starts to find diverse Sidon sets
    for step in [1, 2, 3]:
        for start in range(0, 5, step):
            A = find_sidon_set_greedy(N, start_idx=start, step=step)
            if len(A) >= 2:
                h = np.zeros(N)
                for idx in A:
                    h[idx] = 1.0
                add_candidate(h)
                if elapsed() > deadline - 5:
                    return list(best_h)

    # 5. Random Sparse Sets
    rng = np.random.default_rng(42)
    # Increase number of random restarts and vary sparsity
    for _ in range(120):
        k = rng.integers(2, 25) # Wider range of sparsity
        idx = rng.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        add_candidate(h)
        if elapsed() > deadline - 5:
            return list(best_h)

    # 6. Uniform
    add_candidate(np.ones(N))
    
    # 7. Gaussian
    x = np.arange(N)
    for sigma in [2, 5, 10, 20]:
        h = np.exp(-0.5 * ((x - N/2) / sigma)**2)
        add_candidate(h)

    # 8. Structured Sparse Patterns (e.g., arithmetic progressions)
    for start in range(0, N, 3):
        for step in range(2, 15):
            indices = list(range(start, N, step))
            if len(indices) >= 2:
                h = np.zeros(N)
                h[indices] = 1.0
                add_candidate(h)
                if elapsed() > deadline - 5:
                    return list(best_h)

    # 9. Quadratic residue-like patterns
    for base in [2, 3, 5, 7]:
        h = np.zeros(N)
        for i in range(N):
            if (i * i) % base == 0:
                h[i] = 1.0
        if np.sum(h) >= 2:
            add_candidate(h)
        if elapsed() > deadline - 5:
            return list(best_h)

    # 10. Fibonacci-like spacing
    fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34]
    h = np.zeros(N)
    pos = 0
    for f in fibs:
        if pos < N:
            h[pos] = 1.0
            pos += f
    if np.sum(h) >= 2:
        add_candidate(h)
    if elapsed() > deadline - 5:
        return list(best_h)

    # 11. Spectral flatness candidates: optimize power spectrum to be flat
    # This distributes autoconvolution energy across more lags
    for trial in range(20):
        if elapsed() > deadline - 5:
            break
        # Random phase spectrum
        rng_trial = np.random.default_rng(42 + trial)
        phases = rng_trial.uniform(0, 2*np.pi, N)
        # Ensure Hermitian symmetry
        for k in range(1, N):
            phases[N-k] = -phases[k]
        H = np.exp(1j * phases)
        h = np.fft.ifft(H).real
        h = np.maximum(h, 0.0)
        if np.sum(h) > 1e-15:
            add_candidate(h)

    # Sort candidates by score, keep top few for refinement
    candidates.sort(key=lambda x: -x[0])
    
    # --- Phase 2: Refinement ---
    
    # We refine the top candidates.
    # We will use SLSQP on the normalized vector to handle the non-smooth max better.
    
    top_candidates = candidates[:15]
    
    for sc, h_init in top_candidates:
        if elapsed() > deadline - 2:
            break
            
        # Normalize h_init
        h_norm_init = h_init / np.sum(h_init)
        
        # Constraints for SLSQP: sum(h) = 1, h >= 0
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        bounds = [(0, 1)] * N
        
        try:
            # Use SLSQP with finite difference gradient
            # This is robust to the kink in the max function
            res = minimize(neg_ratio_normalized, h_norm_init, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 300, 'ftol': 1e-13})
            
            if elapsed() > deadline - 1:
                break
                
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
                
        except Exception:
            pass

    # --- Phase 2.5: Spectral Design Refinement (NEW) ---
    # Use epigraph formulation to directly minimize autoconvolution peak
    if elapsed() < deadline - 20:
        top_for_spectral_design = candidates[:15]
        for sc, h_init in top_for_spectral_design:
            if elapsed() > deadline - 5:
                break
                
            h_norm_init = h_init / np.sum(h_init)
            
            try:
                h_opt = spectral_design_optimize(h_norm_init, deadline - 5, max_iters=600)
                if h_opt is not None:
                    sc_opt = compute_ratio(h_opt)
                    if sc_opt > best_score:
                        best_score = sc_opt
                        best_h = h_opt.copy()
            except Exception:
                pass
        
        # After spectral design, do a final SLSQP polish on the best result
        if elapsed() < deadline - 3:
            h_norm_best = best_h / np.sum(best_h)
            try:
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
                bounds = [(0, 1)] * N
                res = minimize(neg_ratio_normalized, h_norm_best, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 400, 'ftol': 1e-14})
                h_opt = res.x
                h_opt = np.maximum(h_opt, 0.0)
                if np.sum(h_opt) > 1e-15:
                    h_opt = h_opt / np.sum(h_opt)
                sc_opt = compute_ratio(h_opt)
                if sc_opt > best_score:
                    best_score = sc_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # --- Phase 2.6: Spectral Refinement (Original) ---
    # For the top candidates, try spectral optimization
    if elapsed() < deadline - 10:
        top_for_spectral = candidates[:8]
        for sc, h_init in top_for_spectral:
            if elapsed() > deadline - 5:
                break
                
            h_norm_init = h_init / np.sum(h_init)
            init_log_mags = time_to_spectral(h_norm_init)
            
            try:
                optimized_log_mags = spectral_projected_gradient_descent(
                    init_log_mags, max_iters=300, lr=0.01, deadline=deadline - 5
                )
                h_opt = spectral_to_time(np.exp(optimized_log_mags))
                h_opt = np.maximum(h_opt, 0.0)
                if np.sum(h_opt) > 1e-15:
                    h_opt = h_opt / np.sum(h_opt)
                
                sc_opt = compute_ratio(h_opt)
                if sc_opt > best_score:
                    best_score = sc_opt
                    best_h = h_opt.copy()
            except Exception:
                pass

    # --- Phase 3: Final Polish ---
    # If we have time, do a high-precision pass on the best solution found.
    # Use SLSQP with tight tolerance for final refinement.
    if elapsed() < deadline - 2:
        h_norm_best = best_h / np.sum(best_h)
        try:
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
            bounds = [(0, 1)] * N
            res = minimize(neg_ratio_normalized, h_norm_best, method='SLSQP',
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 600, 'ftol': 1e-15, 'iprint': 0})
            h_opt = res.x
            h_opt = np.maximum(h_opt, 0.0)
            if np.sum(h_opt) > 1e-15:
                h_opt = h_opt / np.sum(h_opt)
            sc_opt = compute_ratio(h_opt)
            if sc_opt > best_score:
                best_score = sc_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
