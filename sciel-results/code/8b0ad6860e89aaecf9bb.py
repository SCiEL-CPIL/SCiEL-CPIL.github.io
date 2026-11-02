"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _autocorr(x):
    """Compute autocorrelation of x, returning R[0..N-1] where R[k] = sum x[i]*x[i+k]."""
    n = len(x)
    R = np.zeros(n)
    for k in range(n):
        R[k] = np.dot(x[:n-k], x[k:])
    return R

def _peak_sidelobe(x):
    """Return max absolute value of autocorrelation for lags 1..N-1."""
    R = _autocorr(x)
    return np.max(np.abs(R[1:]))

def _l2_sidelobe(x):
    """Return L2 norm of autocorrelation for lags 1..N-1."""
    R = _autocorr(x)
    return np.sqrt(np.sum(R[1:]**2))

def _construct_continuous_init():
    """Use gradient descent on continuous relaxation to get good initial ±1 seed."""
    np.random.seed(42)
    x = np.random.uniform(-1, 1, N)
    
    lr = 0.1
    for step in range(2000):
        # Compute autocorrelation
        R = _autocorr(x)
        # Gradient of sum(R[k]^2) for k=1..N-1 w.r.t. x
        # d/dx[i] of R[k]^2 = 2*R[k] * dR[k]/dx[i]
        # R[k] = sum_{j} x[j]*x[j+k]
        # dR[k]/dx[i] = x[i+k] (if i+k < N) + x[i-k] (if i-k >= 0)
        grad = np.zeros(N)
        for k in range(1, N):
            Rk = R[k]
            # For each i, contribution to grad[i] from lag k
            # dR[k]/dx[i] = x[i+k] if i+k < N, plus x[i-k] if i-k >= 0
            # So grad[i] += 2*Rk * (x[i+k] if i+k<N else 0 + x[i-k] if i-k>=0 else 0)
            # This is expensive, let's use FFT-based approach or just do it directly
            pass
        
        # Simpler: use numerical gradient or just do direct computation
        # Actually, let's use a simpler approach: the gradient of sum_k R[k]^2 
        # can be computed via the fact that this is related to the power spectrum
        
        # Use FFT: |FFT(x)|^2 = sum_k R[k] exp(-2pi i k n/N)
        # The L2 norm of sidelobes relates to total energy minus DC
        X = np.fft.fft(x)
        power = np.abs(X)**2
        # Total sidelobe energy = sum(power) - power[0] (DC term)
        # Gradient of sum_k R[k]^2 for k>=1 w.r.t. x
        # This equals 2 * ifft(fft(x) * conj(fft(x)) with DC zeroed) ... approximately
        
        # Let's just do direct gradient computation for simplicity
        grad = np.zeros(N)
        for k in range(1, N):
            Rk = R[k]
            if Rk == 0:
                continue
            # dR[k]/dx[i] = x[(i+k) % N] for circular, but we have linear
            # For linear autocorrelation:
            # R[k] = sum_{i=0}^{N-1-k} x[i]*x[i+k]
            # dR[k]/dx[i] = x[i+k] (for i=0..N-1-k) + x[i-k] (for i=k..N-1)
            # So grad[i] += 2*Rk * (x[i+k] if i+k < N else 0) + 2*Rk * (x[i-k] if i-k >= 0 else 0)
            # This is O(N^2) total, which for N=400 is 160k ops per step, 2000 steps = 320M, might be slow
            # Let's vectorize:
            upper = x[k:]  # x[i+k] for i=0..N-1-k
            lower = x[:N-k]  # x[i-k] for i=k..N-1
            grad[:N-k] += 2 * Rk * upper
            grad[k:] += 2 * Rk * lower
        
        x -= lr * grad
        # Clip to keep bounded
        x = np.clip(x, -2, 2)
    
    # Threshold to ±1
    x_bin = np.sign(x)
    x_bin[x_bin == 0] = 1
    return x_bin

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Phase 1: Get good initial seed via continuous relaxation
    x = _construct_continuous_init()
    best_x = x.copy()
    best_score = _peak_sidelobe(x)
    
    # Phase 2: Local search - flip individual bits
    np.random.seed(123)
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Try random flips
        for _ in range(5000):
            if (time.time() - start_time) > deadline:
                break
            i = np.random.randint(N)
            x_trial = x.copy()
            x_trial[i] *= -1
            score = _peak_sidelobe(x_trial)
            if score < best_score:
                best_score = score
                best_x = x_trial.copy()
                x = x_trial.copy()
                improved = True
    
    # Phase 3: Try segment flips
    if (time.time() - start_time) < deadline:
        for seg_len in [2, 3, 4]:
            if (time.time() - start_time) > deadline:
                break
            for _ in range(3000):
                if (time.time() - start_time) > deadline:
                    break
                i = np.random.randint(N - seg_len + 1)
                x_trial = x.copy()
                x_trial[i:i+seg_len] *= -1
                score = _peak_sidelobe(x_trial)
                if score < best_score:
                    best_score = score
                    best_x = x_trial.copy()
                    x = x_trial.copy()
    
    return best_x
# EVOLVE-BLOCK-END
