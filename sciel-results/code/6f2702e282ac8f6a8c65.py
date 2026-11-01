"""Spectral phase retrieval + simulated annealing for autocorrelation minimization."""
import numpy as np
import time

N = 400

def _max_sidelobe(x):
    """Compute the maximum absolute autocorrelation sidelobe value.
    x is a signed step function (array of +1/-1 values)."""
    # Autocorrelation via FFT
    n = 2 * N
    fft_x = np.fft.fft(x, n=n)
    acf = np.fft.ifft(np.abs(fft_x) ** 2)
    # acf has length n; the lags we care about are 1..N-1 (and symmetrically -1..-(N-1))
    # For real signal, acf is real. Take the max absolute value of sidelobes.
    # Sidelobes are indices 1..N-1 and n-(N-1)..n-1
    sidelobes = np.concatenate([acf[1:N], acf[n - (N - 1):n]])
    return np.max(np.abs(sidelobes))


def _try_flip(x, idx, current_val):
    """Try flipping x[idx] and return the new max sidelobe."""
    x[idx] = -x[idx]
    val = _max_sidelobe(x)
    x[idx] = current_val
    return val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Step 1: Initialize via spectral phase retrieval
    # Target: flat magnitude spectrum to minimize peak autocorrelation
    # We want |FFT(x)|^2 to be as flat as possible (except DC)
    
    # Start with random ±1
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Iterative spectral phase retrieval (Gerchberg-Saxton style)
    # We want the magnitude of FFT to be flat (all equal)
    target_mag = np.ones(N)
    target_mag[0] = 0  # Don't constrain DC too much, or we can keep it
    
    # Actually, for a ±1 sequence, the DC component is sum(x) which we can't control easily.
    # Let's just aim for flat magnitude in the non-DC bins.
    
    best_x = x.copy()
    best_val = _max_sidelobe(x)
    
    # Phase retrieval iterations
    for it in range(200):
        if time.time() > deadline - 30.0:
            break
        # Compute FFT
        fft_x = np.fft.fft(x)
        # Replace magnitude with target, keep phase
        mag = np.abs(fft_x)
        mag[mag == 0] = 1e-10
        phase = np.angle(fft_x)
        new_fft = target_mag * np.exp(1j * phase)
        # Inverse FFT and threshold to ±1
        x_new = np.fft.ifft(new_fft).real
        x = np.where(x_new >= 0, 1.0, -1.0)
        
        val = _max_sidelobe(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    x = best_x.copy()
    
    # Step 2: Simulated annealing refinement
    T = 0.1
    T_min = 0.0001
    cooling = 0.995
    
    while time.time() < deadline - 5.0:
        if T < T_min:
            T = 0.01  # Reheat
            # Random restart from best
            x = best_x.copy()
        
        # Pick random index to flip
        idx = np.random.randint(0, N)
        old_val = x[idx]
        x[idx] = -old_val
        
        new_val = _max_sidelobe(x)
        
        delta = new_val - best_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[idx] = old_val  # Revert
        
        T *= cooling
    
    return best_x
