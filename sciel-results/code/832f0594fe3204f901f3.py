"""Discrete logistic map signed step function with parameter search and local refinement."""
import numpy as np
import time

N = 400
DEADLINE = 145.0  # safety margin before 150s kill


def _compute_c3_bound(signs):
    """Compute the autoconvolution upper bound C3 for a signed step function.
    
    The bound is max over k of |sum_{i} signs[i] * signs[i+k]| (autocorrelation),
    normalized appropriately. Lower is better.
    """
    signs = np.asarray(signs)
    n = len(signs)
    # Compute autocorrelation via FFT for speed
    # autocorr[k] = sum_i signs[i] * signs[i+k] for k in 0..n-1
    # Use FFT: autocorr = ifft(fft(signs) * conj(fft(signs)))
    f = np.fft.fft(signs, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f)).real[:n]
    # The upper bound C3 is related to max |acf[k]| for k != 0
    # For the third autocorrelation inequality, C3 bound = max_{k!=0} |acf[k]| / acf[0]
    # Actually, let's use the standard: C3 = max_{k!=0} |acf[k]|
    # Normalize by acf[0] which is sum of squares = n (since signs are +/-1)
    if acf[0] > 1e-12:
        bound = np.max(np.abs(acf[1:])) / acf[0]
    else:
        bound = 1.0
    return bound


def _discrete_logistic(mu, x0, n):
    """Generate sequence using discrete logistic map x_{n+1} = floor(mu * x_n * (1 - x_n)).
    
    mu is in [0, 4], x0 in [0, 1]. The output values are integers in [0, floor(mu/4)].
    We map to signs: +1 if value > mid, -1 otherwise.
    """
    seq = np.zeros(n, dtype=np.float64)
    x = x0
    for i in range(n):
        seq[i] = x
        x = np.floor(mu * x * (1.0 - x))
        if x < 0:
            x = 0.0
        if x > 1:
            x = 1.0
    return seq


def _seq_to_signs(seq, threshold=None):
    """Convert a sequence to +/-1 signs based on threshold."""
    if threshold is None:
        threshold = np.median(seq)
    signs = np.where(seq >= threshold, 1.0, -1.0)
    # Ensure balanced signs if possible
    pos = np.sum(signs > 0)
    neg = np.sum(signs < 0)
    if pos > N // 2:
        # Flip some +1 to -1
        idx = np.where(signs > 0)[0]
        np.random.shuffle(idx)
        for i in range(pos - N // 2):
            signs[idx[i]] = -1.0
    elif neg > N // 2:
        idx = np.where(signs < 0)[0]
        np.random.shuffle(idx)
        for i in range(neg - N // 2):
            signs[idx[i]] = 1.0
    return signs


def _local_search(signs, budget):
    """Local search: try single flips and pairwise swaps to reduce C3 bound."""
    current = signs.copy()
    current_bound = _compute_c3_bound(current)
    best = current.copy()
    best_bound = current_bound
    
    n = len(current)
    attempts = 0
    max_attempts = 20000
    
    while attempts < max_attempts:
        if time.time() > budget:
            break
        
        # Try random single flips
        i = np.random.randint(n)
        trial = current.copy()
        trial[i] = -trial[i]
        b = _compute_c3_bound(trial)
        if b < current_bound - 1e-12:
            current = trial
            current_bound = b
            if b < best_bound - 1e-12:
                best = current.copy()
                best_bound = b
        attempts += 1
        
        # Also try pairwise swaps occasionally
        if attempts % 3 == 0:
            i, j = np.random.choice(n, 2, replace=False)
            trial = current.copy()
            trial[i], trial[j] = -trial[i], -trial[j]
            b = _compute_c3_bound(trial)
            if b < current_bound - 1e-12:
                current = trial
                current_bound = b
                if b < best_bound - 1e-12:
                    best = current.copy()
                    best_bound = b
            attempts += 1
    
    return best, best_bound


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_signs = None
    best_bound = float('inf')
    
    # Phase 1: Search over logistic map parameters
    # mu in [0, 4], x0 in [0, 1]
    # Use a grid + random sampling
    np.random.seed(42)
    
    # Grid search over mu
    mu_values = np.linspace(0.5, 4.0, 80)
    x0_values = np.linspace(0.05, 0.95, 20)
    
    # First pass: coarse grid
    for mu in mu_values:
        if time.time() > deadline - 60:
            break
        for x0 in x0_values:
            if time.time() > deadline - 60:
                break
            seq = _discrete_logistic(mu, x0, N)
            signs = _seq_to_signs(seq)
            b = _compute_c3_bound(signs)
            if b < best_bound:
                best_bound = b
                best_signs = signs.copy()
    
    # Phase 2: Local search around best found
    if best_signs is not None:
        remaining = deadline - time.time()
        if remaining > 10:
            best_signs, best_bound = _local_search(best_signs, deadline - 5)
    
    # Phase 3: Random restarts with local search
    if time.time() < deadline - 30:
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            mu = np.random.uniform(0.5, 4.0)
            x0 = np.random.uniform(0.05, 0.95)
            seq = _discrete_logistic(mu, x0, N)
            signs = _seq_to_signs(seq)
            b = _compute_c3_bound(signs)
            
            if b < best_bound:
                best_bound = b
                best_signs = signs.copy()
            
            # Local search on this candidate
            if time.time() < deadline - 15:
                signs, b = _local_search(signs, deadline - 5)
                if b < best_bound:
                    best_bound = b
                    best_signs = signs
    
    # Phase 4: Final local search on best
    if time.time() < deadline - 5:
        if best_signs is None:
            best_signs = np.ones(N)
        best_signs, best_bound = _local_search(best_signs, deadline - 1)
    
    if best_signs is None:
        best_signs = np.ones(N)
    
    return best_signs


# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    signs = construct()
    t1 = time.time()
    print(f"Time: {t1 - t0:.2f}s")
    print(f"C3 bound: {_compute_c3_bound(signs):.6f}")
    print(f"Score: {1.0 / _compute_c3_bound(signs):.6f}")
