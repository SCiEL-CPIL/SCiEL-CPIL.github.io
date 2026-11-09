"""Rewritten program for third autocorrelation inequality optimization."""
import numpy as np
import time
import scipy.optimize

N = 400
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds before deadline to stop

def _compute_autocorr(h):
    """Compute full autocorrelation array R[0..N-1] for h of length N."""
    n_fft = 1 << (2 * N - 1).bit_length()
    f = np.zeros(n_fft, dtype=np.float64)
    f[:N] = h
    F = np.fft.rfft(f)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:N]
    return acf

def _max_sidelobe(R):
    """Return max absolute value of R[1:]."""
    return np.max(np.abs(R[1:]))

def _generate_h_from_params(params, N, K):
    """
    Generate a signed sequence from sinusoidal parameters.
    params: array of length 2*K. [A1, phi1, A2, phi2, ...]
    x[n] = sum_{k=1}^K A_k * cos(2*pi*k*n/N + phi_k)
    h[n] = sign(x[n])
    """
    n = np.arange(N)
    x = np.zeros(N)
    for k in range(K):
        A = params[2*k]
        phi = params[2*k+1]
        x += A * np.cos(2 * np.pi * (k+1) * n / N + phi)
    
    # Handle zeros: if x[n] is 0, assign +1 or -1 arbitrarily (or based on previous)
    # sign(0) is 0 in numpy, so we map 0 to 1
    h = np.sign(x)
    h[h == 0] = 1.0
    return h

def _objective_sin(params, N, K):
    """
    Objective function for sinusoidal parameter optimization.
    Returns the max sidelobe of the generated signed sequence.
    """
    h = _generate_h_from_params(params, N, K)
    R = _compute_autocorr(h)
    return _max_sidelobe(R)

def _optimize_sinusoidal(N, K, seed, time_limit):
    """
    Optimize sinusoidal parameters using Nelder-Mead.
    """
    np.random.seed(seed)
    # Initial parameters: random amplitudes and phases
    # Amplitudes in [0.5, 1.5], phases in [0, 2*pi]
    A_init = np.random.uniform(0.5, 1.5, size=K)
    phi_init = np.random.uniform(0, 2*np.pi, size=K)
    params_init = np.concatenate([A_init, phi_init])
    
    # Define callback to check time
    start_time = time.time()
    def callback(xk):
        if time.time() - start_time > time_limit:
            return True
        return False

    try:
        res = scipy.optimize.minimize(
            _objective_sin, 
            params_init, 
            args=(N, K), 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4},
            callback=callback
        )
        best_params = res.x
        best_h = _generate_h_from_params(best_params, N, K)
        return best_h
    except Exception:
        return None

def construct():
    """
    Construct a signed step function with N intervals to minimize the
    upper bound on C3 (maximum autocorrelation sidelobe).
    
    Uses a hybrid approach:
    1. Coarse optimization using sinusoidal parameterization and Nelder-Mead.
    2. Fine-tuning using greedy local search with flips.
    """
    start_time = time.time()
    best_h = None
    best_val = np.inf
    
    def optimize_local(h, R, current_max, time_limit):
        """
        Perform greedy local search starting from h, R, current_max.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(N):
                if time.time() > time_limit:
                    return h, R, current_max
                
                delta = np.zeros(N)
                
                if i > 0:
                    delta[1:i+1] -= 2.0 * h[i] * h[:i][::-1]
                
                if i < N - 1:
                    delta[1:N-i] -= 2.0 * h[i] * h[i+1:]
                
                new_R = R - delta
                new_max = _max_sidelobe(new_R)
                
                if new_max < current_max - 1e-12:
                    h[i] *= -1.0
                    R = new_R
                    current_max = new_max
                    improved = True
                    
        return h, R, current_max

    # Strategy: Try different numbers of harmonics K
    # K=1 is square wave (poor), K=5, 10, 20
    K_values = [5, 10, 20]
    seed_counter = 0
    
    while time.time() < DEADLINE - SAFETY_MARGIN:
        seed_counter += 1
        K = K_values[seed_counter % len(K_values)]
        
        # Allocate time for coarse optimization
        # We want to leave time for local search
        # Let's say 10s for coarse, 10s for local?
        # Or just run coarse until a certain time, then local?
        # Better: Run coarse for a fixed small time, get candidate, then local.
        
        coarse_time_limit = 5.0 # seconds
        if time.time() + coarse_time_limit + 5.0 > DEADLINE - SAFETY_MARGIN:
            # Not enough time for a full cycle, break
            break
            
        h_candidate = _optimize_sinusoidal(N, K, seed_counter, coarse_time_limit)
        
        if h_candidate is None:
            h_candidate = np.random.choice([-1.0, 1.0], size=N)
        
        R = _compute_autocorr(h_candidate)
        current_max = _max_sidelobe(R)
        
        if current_max < best_val:
            best_val = current_max
            best_h = h_candidate.copy()
            
        # Local search polish
        remaining_time = DEADLINE - SAFETY_MARGIN - time.time()
        if remaining_time > 2.0:
            h_candidate, R, current_max = optimize_local(h_candidate, R, current_max, time.time() + remaining_time - 1.0)
            
            if current_max < best_val:
                best_val = current_max
                best_h = h_candidate.copy()
                
    return best_h if best_h is not None else np.ones(N)
