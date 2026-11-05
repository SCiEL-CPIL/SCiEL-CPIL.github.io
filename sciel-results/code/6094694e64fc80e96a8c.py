"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_bound(a):
    """
    Compute the upper bound on C3 for a signed step function a of length N.
    The bound is max over lags k>=1 of |sum_i a[i]*a[i+k]|.
    We use FFT-based autocorrelation.
    """
    n = len(a)
    # Zero-pad to next power of 2 for FFT efficiency (or just 2*n)
    f = np.fft.fft(a, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # Exclude lag 0
    if n <= 1:
        return 0.0
    return float(np.max(np.abs(acf[1:])))

def _random_init(seed=None):
    """Generate a random signed step function."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng.choice([-1.0, 1.0], size=N)

def _logistic_map_sequence(r, x0, n):
    """
    Generate a sequence of signs based on logistic map.
    x_{k+1} = r * x_k * (1 - x_k)
    sign is +1 if x_k > 0.5 else -1
    """
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = r * x[i] * (1.0 - x[i])
    return np.where(x > 0.5, 1.0, -1.0)

def _chaos_init(r, x0):
    """
    Initialize using logistic map in chaotic regime.
    r in (3.57, 4.0) is chaotic.
    """
    return _logistic_map_sequence(r, x0, N)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    best_a = None
    best_bound = float('inf')
    
    # Phase 1: Try several chaotic initializations
    # Logistic map parameters in chaotic regime
    r_values = [3.9, 3.99, 3.7, 3.57, 3.8, 3.95, 3.6, 3.85]
    x0_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.123, 0.456, 0.789]
    
    candidates = []
    for r in r_values:
        for x0 in x0_values:
            if time.time() > deadline - 30:
                break
            a = _chaos_init(r, x0)
            bound = _autoconv_bound(a)
            candidates.append((bound, a))
    
    # Also add a few random initializations
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        a = _random_init()
        bound = _autoconv_bound(a)
        candidates.append((bound, a))
    
    # Sort by bound and take the best few
    candidates.sort(key=lambda x: x[0])
    
    # Take top 5 candidates for local search
    top_k = min(5, len(candidates))
    
    # Phase 2: Local search (greedy + simulated annealing) on each top candidate
    for i in range(top_k):
        if time.time() > deadline - 10:
            break
        
        a = candidates[i][1].copy()
        current_bound = candidates[i][0]
        
        # Greedy local search
        no_improve_count = 0
        max_greedy_iters = 5000
        for it in range(max_greedy_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Try flipping random positions
            for _ in range(50):
                if time.time() > deadline - 10:
                    break
                idx = np.random.randint(N)
                old_val = a[idx]
                a[idx] = -a[idx]
                new_bound = _autoconv_bound(a)
                if new_bound < current_bound:
                    current_bound = new_bound
                    if current_bound < best_bound:
                        best_bound = current_bound
                        best_a = a.copy()
                    improved = True
                else:
                    a[idx] = old_val
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 50:
                    break
            else:
                no_improve_count = 0
        
        # Simulated annealing
        if time.time() > deadline - 10:
            continue
        
        T = current_bound * 0.1
        T_min = current_bound * 0.001
        cooling = 0.9995
        max_sa_iters = 100000
        
        while time.time() < deadline - 5 and T > T_min:
            idx = np.random.randint(N)
            old_val = a[idx]
            a[idx] = -a[idx]
            new_bound = _autoconv_bound(a)
            delta = new_bound - current_bound
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                current_bound = new_bound
                if current_bound < best_bound:
                    best_bound = current_bound
                    best_a = a.copy()
            else:
                a[idx] = old_val
            T *= cooling
        
        if best_a is None:
            best_a = a.copy()
            best_bound = current_bound
    
    # If no best found, return a random one
    if best_a is None:
        best_a = _random_init()
        best_bound = _autoconv_bound(best_a)
    
    return best_a
# EVOLVE-BLOCK-END
