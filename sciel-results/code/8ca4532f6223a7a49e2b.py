import numpy as np
import time

N = 400

def _max_autocorr_sidelobe(h):
    """Compute max autocorrelation at non-zero lags for a ±1 sequence."""
    # Use FFT to compute linear autocorrelation
    n_fft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(h, n=n_fft)
    power = F * np.conj(F)
    acf = np.fft.irfft(power, n=n_fft)[:N]
    # acf[0] is the energy, acf[1:] are the sidelobes
    return np.max(np.abs(acf[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Strategy 1: Random initialization
    rng = np.random.RandomState(42)
    h = rng.choice([1, -1], N)
    best_h = h.copy()
    best_score = _max_autocorr_sidelobe(best_h)
    
    # Strategy 2: Local search / greedy improvement
    def local_search(h, max_iter=100000):
        h = h.copy()
        score = _max_autocorr_sidelobe(h)
        improved = True
        iters = 0
        while improved and iters < max_iter:
            improved = False
            iters += 1
            # Try flipping a random position
            idx = rng.randint(N)
            h_new = h.copy()
            h_new[idx] = -h_new[idx]
            new_score = _max_autocorr_sidelobe(h_new)
            if new_score < score:
                h = h_new
                score = new_score
                improved = True
        return h, score
    
    # Run local search from random start
    h_ls, score_ls = local_search(best_h, max_iter=50000)
    if score_ls < best_score:
        best_h = h_ls
        best_score = score_ls
    
    # Check time
    if time.time() > deadline - 20.0:
        return best_h
    
    # Strategy 3: Simulated annealing
    def simulated_annealing(h, max_iter=200000):
        h = h.copy()
        score = _max_autocorr_sidelobe(h)
        T = 1.0
        T_min = 0.01
        cooling = 0.9999
        for _ in range(max_iter):
            if time.time() > deadline - 10.0:
                break
            idx = rng.randint(N)
            h_new = h.copy()
            h_new[idx] = -h_new[idx]
            new_score = _max_autocorr_sidelobe(h_new)
            delta = new_score - score
            if delta < 0 or rng.random() < np.exp(-delta / T):
                h = h_new
                score = new_score
            T = max(T * cooling, T_min)
        return h, score
    
    h_sa, score_sa = simulated_annealing(best_h, max_iter=100000)
    if score_sa < best_score:
        best_h = h_sa
        best_score = score_sa
    
    # Final check
    if time.time() > deadline - 5.0:
        return best_h
    
    # Strategy 4: More local search rounds
    h_ls2, score_ls2 = local_search(best_h, max_iter=50000)
    if score_ls2 < best_score:
        best_h = h_ls2
        best_score = score_ls2
    
    return best_h
