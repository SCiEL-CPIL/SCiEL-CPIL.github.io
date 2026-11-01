"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # seconds, safety margin before 150s kill


def _autocorr_sidelobes(x):
    """Compute autocorrelation sidelobes (exclude lag 0).
    x: array of length N with values in {-1, 1}.
    Returns array of length N-1 with sidelobe values at lags 1..N-1.
    """
    # Use FFT-based autocorrelation
    n = len(x)
    size = 1
    while size < 2 * n:
        size <<= 1
    X = np.fft.rfft(x, size)
    acf = np.fft.irfft(X * np.conj(X), size)[:n]
    return acf[1:]


def _max_abs_sidelobe(x):
    """Return max absolute value of autocorrelation sidelobes."""
    sl = _autocorr_sidelobes(x)
    return np.max(np.abs(sl))


def _soft_max_abs_sidelobe(x, beta=10.0):
    """Smooth surrogate: soft-max of |sidelobes| via log-sum-exp."""
    sl = _autocorr_sidelobes(x)
    a = np.abs(sl)
    # log-sum-exp for numerical stability
    m = np.max(a)
    return m + np.log(np.sum(np.exp(a - m))) / len(a)


def _flip_eval(x, i, sl_cache):
    """Evaluate effect of flipping bit i on max sidelobe.
    sl_cache: precomputed autocorrelation sidelobes for current x.
    Returns new max sidelobe after flip.
    """
    # Flipping x[i] changes autocorrelation:
    # acf_new[k] = acf[k] - 2*x[i]*sum_{j} x[j]*x[j+k] (for appropriate ranges)
    # Actually, flipping x[i] to -x[i]:
    # acf[k] = sum_j x[j]*x[j+k]
    # new_acf[k] = acf[k] - 2*x[i]*x[i+k] (if i+k in range) - 2*x[i]*x[i-k] (if i-k in range, k>0)
    # More precisely:
    # acf[k] = sum_{j=0}^{N-1-k} x[j]*x[j+k]
    # After flipping x[i]:
    #   If i <= N-1-k: term x[i]*x[i+k] becomes -x[i]*x[i+k], change = -2*x[i]*x[i+k]
    #   If i+k >= N: no change from that term
    #   Also for the "other direction": acf[k] also includes contributions where j+k = i, i.e., j = i-k
    #   Wait, acf[k] = sum_j x[j]*x[j+k] only goes one way. But autocorrelation is symmetric.
    #   Actually for real signals, acf[k] = sum_j x[j]*x[j+k] for k>=0.
    #   Flipping x[i] affects:
    #     - terms where j = i (if i+k < N): change -2*x[i]*x[i+k]
    #     - terms where j+k = i, i.e., j = i-k (if i-k >= 0): change -2*x[i-k]*x[i]
    #   So total change to acf[k] = -2*x[i]*(x[i+k] if i+k<N else 0) - 2*x[i]*(x[i-k] if i-k>=0 else 0)
    
    n = len(x)
    xi = x[i]
    new_sl = np.empty_like(sl_cache)
    
    for k in range(1, n):
        delta = 0.0
        if i + k < n:
            delta -= 2.0 * xi * x[i + k]
        if i - k >= 0:
            delta -= 2.0 * xi * x[i - k]
        new_sl[k - 1] = sl_cache[k - 1] + delta
    
    return np.max(np.abs(new_sl))


def _local_search(x, max_iters=5000, time_limit=5.0):
    """Greedy local search: flip single bits to reduce max sidelobe."""
    n = len(x)
    sl = _autocorr_sidelobes(x)
    best_val = np.max(np.abs(sl))
    improved = True
    iter_count = 0
    start_time = time.time()
    
    while improved and iter_count < max_iters:
        if time.time() - start_time > time_limit:
            break
        improved = False
        best_flip = -1
        best_new_val = best_val
        
        for i in range(n):
            new_val = _flip_eval(x, i, sl)
            if new_val < best_new_val - 1e-12:
                best_new_val = new_val
                best_flip = i
            iter_count += 1
            if time.time() - start_time > time_limit:
                break
        
        if best_flip >= 0:
            x[best_flip] = -x[best_flip]
            sl = _autocorr_sidelobes(x)
            best_val = best_new_val
            improved = True
    
    return x, best_val


def _multi_start_local_search(n_starts=20, time_limit=5.0):
    """Run local search from multiple random starts."""
    n = N
    best_x = None
    best_val = np.inf
    
    for s in range(n_starts):
        if time.time() > _DEADLINE - 5:
            break
        # Random signed step function
        x = np.random.choice([-1.0, 1.0], size=n)
        x, val = _local_search(x, max_iters=2000, time_limit=time_limit / max(1, n_starts))
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    return best_x, best_val


def _anneal(x, time_limit=5.0):
    """Simple simulated annealing to escape local minima."""
    n = len(x)
    sl = _autocorr_sidelobes(x)
    best_val = np.max(np.abs(sl))
    best_x = x.copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    start_time = time.time()
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        i = np.random.randint(n)
        x[i] = -x[i]
        new_sl = _autocorr_sidelobes(x)
        new_val = np.max(np.abs(new_sl))
        
        delta = new_val - best_val
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-8)):
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
            sl = new_sl
        else:
            x[i] = -x[i]  # revert
        
        T *= cooling
    
    return best_x, best_val


def _refine(x, time_limit=10.0):
    """Final refinement: alternating local search and annealing."""
    best_x = x.copy()
    best_val = _max_abs_sidelobe(x)
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Local search
        x, val = _local_search(x, max_iters=1000, time_limit=2.0)
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        if time.time() - start_time > time_limit:
            break
        
        # Annealing to escape
        x, val = _anneal(x, time_limit=2.0)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    return best_x, best_val


def construct():
    """Construct signed step function minimizing max autocorrelation sidelobe."""
    np.random.seed(42)
    
    # Phase 1: Multi-start local search
    x, val = _multi_start_local_search(n_starts=15, time_limit=20.0)
    
    # Phase 2: Refinement with alternating LS and SA
    x, val = _refine(x, time_limit=30.0)
    
    # Phase 3: Final local search polish
    x, val = _local_search(x, max_iters=5000, time_limit=10.0)
    
    # Return as integer sequence
    return np.round(x).astype(int)
# EVOLVE-BLOCK-END
