"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _overlap_value(h):
    """Compute the maximum overlap of h with shifted (1-h).
    
    The overlap at shift k is sum_i h[i] * (1-h[(i+k) % N]).
    We want to minimize the maximum over all shifts.
    """
    N = len(h)
    one_minus_h = 1.0 - h
    # Compute all circular cross-correlations
    # overlap[k] = sum_i h[i] * (1-h[(i+k) % N])
    # Use FFT for efficiency
    h_ext = np.zeros(2 * N)
    h_ext[:N] = h
    omh_ext = np.zeros(2 * N)
    omh_ext[:N] = one_minus_h
    
    # Cross-correlation via FFT
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    O = np.fft.rfft(one_minus_h, n=n_fft)
    # correlation: sum_i h[i] * omh[(i+k) % N]
    # This is the circular cross-correlation
    corr = np.fft.irfft(H * np.conj(O), n=n_fft)
    # corr[k] = sum_i h[(i-k) % N] * omh[i] ... need to be careful
    # Actually: ifft(fft(a) * conj(fft(b))) gives circular correlation
    # corr[k] = sum_i a[(i-k) mod N] * b[i]
    # We want: overlap[k] = sum_i h[i] * omh[(i+k) mod N]
    # Let's just compute directly for N=100 (fast enough)
    
    overlaps = np.zeros(N)
    for k in range(N):
        shifted_omh = np.roll(one_minus_h, -k)
        overlaps[k] = np.dot(h, shifted_omh)
    
    return np.max(overslaps) if 'overslaps' in dir() else np.max(overlaps)


def _overlap_value_fast(h):
    """Compute max overlap using FFT."""
    N = len(h)
    one_minus_h = 1.0 - h
    
    # overlap[k] = sum_i h[i] * omh[(i+k) % N]
    # = sum_i h[i] * omh_shifted[k][i]
    # Using FFT: let A = fft(h), B = fft(omh)
    # We want the circular cross-correlation
    # corr[k] = sum_i h[(i-k)%N] * omh[i]
    # We want sum_i h[i] * omh[(i+k)%N]
    # Let j = i+k, then i = j-k: sum_j h[j-k] * omh[j] = corr[k]
    # So overlap[k] = corr[k] where corr is standard circular cross-correlation
    
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    O = np.fft.rfft(one_minus_h, n=n_fft)
    # Circular cross-correlation: ifft(H * conj(O))
    corr = np.fft.irfft(H * np.conj(O), n=n_fft)
    # corr[k] for k in 0..N-1 gives sum_i h[(i-k)%N] * omh[i]
    # But we want sum_i h[i] * omh[(i+k)%N]
    # Note: sum_i h[i] * omh[(i+k)%N] = sum_j h[j-k] * omh[j] (substituting j=i+k)
    # So overlap[k] = corr[k]
    
    return np.max(corr[:N])


def _objective(params):
    """Objective: maximize the log-sum-exp surrogate of the overlaps."""
    h = params
    # Enforce bounds softly
    N = len(h)
    one_minus_h = 1.0 - h
    
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    O = np.fft.rfft(one_minus_h, n=n_fft)
    corr = np.fft.irfft(H * np.conj(O), n=n_fft)
    overlaps = corr[:N]
    
    # Log-sum-exp surrogate for max
    alpha = 50.0
    m = np.max(overlaps)
    lse = m + np.log(np.sum(np.exp(alpha * (overlaps - m)))) / alpha
    
    # Penalty for constraint violations
    # h in [0,1], sum(h) = N/2
    penalty = 0.0
    penalty += 100.0 * np.sum(np.maximum(0, h - 1.0)) ** 2
    penalty += 100.0 * np.sum(np.maximum(0, -h)) ** 2
    penalty += 100.0 * (np.sum(h) - N / 2.0) ** 2
    
    return lse + penalty


def _objective_logsumexp(params, alpha=50.0):
    """Log-sum-exp surrogate objective."""
    h = params
    N = len(h)
    one_minus_h = 1.0 - h
    
    n_fft = 2 * N
    H = np.fft.rfft(h, n=n_fft)
    O = np.fft.rfft(one_minus_h, n=n_fft)
    corr = np.fft.irfft(H * np.conj(O), n=n_fft)
    overlaps = corr[:N]
    
    m = np.max(overlaps)
    lse = m + np.log(np.sum(np.exp(alpha * (overlaps - m)))) / alpha
    
    # Soft penalties
    penalty = 50.0 * (np.sum(h) - N / 2.0) ** 2
    penalty += 10.0 * np.sum(np.maximum(0, h - 1.0)) ** 2
    penalty += 10.0 * np.sum(np.maximum(0, -h)) ** 2
    
    return lse + penalty


def _project_to_feasible(h):
    """Project h onto {h in [0,1], sum(h) = N/2} approximately."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to make sum = N/2
    diff = N / 2.0 - np.sum(h)
    if abs(diff) < 1e-10:
        return h
    # Distribute adjustment
    h = h + diff / N
    return np.clip(h, 0.0, 1.0)


def construct():
    """Construct step heights minimizing max overlap of h with shifted 1-h."""
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, use 120s budget
    
    best_h = np.full(N, 0.5)
    best_score = _overlap_value_fast(best_h)
    
    # Strategy 1: Use differential evolution for global search
    # Then polish with SLSQP
    
    def _de_objective(x):
        h = np.clip(x, 0, 1)
        # Enforce sum constraint via penalty
        penalty = 100.0 * (np.sum(h) - N / 2.0) ** 2
        val = _overlap_value_fast(h) + penalty
        return val
    
    # Use a smaller subset of variables for DE to be faster
    # Actually N=100 is manageable for DE with limited iterations
    
    try:
        # Differential evolution with bounded variables
        bounds = [(0.0, 1.0)] * N
        
        def _de_obj(x):
            h = np.clip(x, 0, 1)
            penalty = 50.0 * (np.sum(h) - N / 2.0) ** 2
            return _overlap_value_fast(h) + penalty
        
        # Use moderate settings
        de_result = differential_evolution(
            _de_obj, bounds,
            maxiter=30, popsize=15,
            tol=1e-4, mutation=(0.3, 1.0),
            recombination=0.7, seed=42,
            polish=False, init='latinhypercube'
        )
        
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            h_de = np.clip(de_result.x, 0, 1)
            # Project to feasible
            h_de = _project_to_feasible(h_de)
            score_de = _overlap_value_fast(h_de)
            if score_de < best_score:
                best_score = score_de
                best_h = h_de.copy()
    except Exception:
        pass
    
    # Strategy 2: Polish with SLSQP using log-sum-exp surrogate
    try:
        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            # Use log-sum-exp with increasing alpha
            for alpha in [20.0, 50.0, 100.0, 200.0]:
                elapsed = time.time() - start_time
                if elapsed > deadline - 10:
                    break
                
                def _lsq_obj(x):
                    h = np.clip(x, 0, 1)
                    N_local = len(h)
                    one_minus_h = 1.0 - h
                    n_fft = 2 * N_local
                    H = np.fft.rfft(h, n=n_fft)
                    O = np.fft.rfft(one_minus_h, n=n_fft)
                    corr = np.fft.irfft(H * np.conj(O), n=n_fft)
                    overlaps = corr[:N_local]
                    m = np.max(overlaps)
                    lse = m + np.log(np.sum(np.exp(alpha * (overlaps - m)))) / alpha
                    return lse
                
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0},
                ]
                bounds = [(0.0, 1.0)] * N
                
                result = minimize(
                    _lsq_obj, best_h, method='SLSQP',
                    bounds=bounds, constraints=constraints,
                    options={'maxiter': 100, 'ftol': 1e-10}
                )
                
                if result.success:
                    h_new = np.clip(result.x, 0, 1)
                    # Re-project
                    diff = N / 2.0 - np.sum(h_new)
                    h_new = h_new + diff / N
                    h_new = np.clip(h_new, 0, 1)
                    
                    score_new = _overlap_value_fast(h_new)
                    if score_new < best_score:
                        best_score = score_new
                        best_h = h_new.copy()
    except Exception:
        pass
    
    # Strategy 3: Try structured patterns
    try:
        elapsed = time.time() - start_time
        if elapsed < deadline - 10:
            # Try alternating patterns
            for period in [2, 3, 4, 5, 7, 10, 13, 17, 20, 25]:
                for phase in range(period):
                    h_pat = np.zeros(N)
                    for i in range(N):
                        h_pat[i] = 1.0 if (i + phase) % period < period // 2 else 0.0
                    # Scale to sum = N/2
                    s = np.sum(h_pat)
                    if s > 0:
                        h_pat = h_pat * (N / 2.0) / s
                        h_pat = np.clip(h_pat, 0, 1)
                        score_pat = _overlap_value_fast(h_pat)
                        if score_pat < best_score:
                            best_score = score_pat
                            best_h = h_pat.copy()
    except Exception:
        pass
    
    # Final projection and verification
    best_h = np.clip(best_h, 0, 1)
    diff = N / 2.0 - np.sum(best_h)
    best_h = best_h + diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
