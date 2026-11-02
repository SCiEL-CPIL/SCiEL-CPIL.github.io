"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(s):
    """Compute max |autoconvolution| for signed sequence s (values in {-1,1}).
    Autoconvolution: (s * s)[k] = sum_i s[i] * s[i+k]
    We want to minimize the upper bound on C3, which relates to max |autocorrelation|.
    """
    # Use FFT for efficient computation
    n = len(s)
    # Zero-pad to avoid circular convolution issues
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(s, size)
    ac = np.fft.irfft(f * np.conj(f), size)[:2 * n - 1]
    # The autoconvolution values
    return np.max(np.abs(ac))


def _energy(s):
    """Energy function: max |autoconvolution|"""
    return _autoconv_max_abs(s)


def _greedy_polish(x, deadline):
    """Greedy coordinate descent: flip signs to reduce max abs autocorrelation."""
    x = x.copy()
    best_val = _autoconv_max_abs(x)
    no_improve = 0
    max_no_improve = 3
    
    while time.time() < deadline - 5.0:
        improved = False
        # Shuffle order to avoid bias
        order = np.random.permutation(N)
        for i in order:
            if time.time() > deadline - 5.0:
                return x, best_val
            # Try flipping x[i]
            old_val = x[i]
            x[i] = -old_val
            new_val = _autoconv_max_abs(x)
            if new_val < best_val - 1e-12:
                best_val = new_val
                improved = True
            else:
                x[i] = old_val  # revert
        
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                break
        else:
            no_improve = 0
    
    return x, best_val


def _anneal(x, start_time, deadline):
    """Simulated annealing with block flips to escape local minima."""
    x = x.copy()
    best_x = x.copy()
    best_val = _autoconv_max_abs(x)
    current_val = best_val
    
    T = 5.0  # initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() < deadline - 10.0 and T > T_min:
        # Choose block size: mix of small and large flips
        r = np.random.random()
        if r < 0.4:
            block_size = 1
        elif r < 0.7:
            block_size = np.random.randint(2, 10)
        elif r < 0.9:
            block_size = np.random.randint(10, 30)
        else:
            block_size = np.random.randint(30, 80)
        
        if block_size >= N:
            block_size = N - 1
        
        # Choose random position
        pos = np.random.randint(0, N - block_size + 1)
        
        # Try flipping the block
        old_block = x[pos:pos + block_size].copy()
        x[pos:pos + block_size] *= -1
        
        new_val = _autoconv_max_abs(x)
        delta = new_val - current_val
        
        # Accept if better, or with probability exp(-delta / T)
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = x.copy()
        else:
            x[pos:pos + block_size] = old_block
        
        # Cool down
        T *= cooling_rate
    
    return best_x, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    best_x = None
    best_val = np.inf
    
    # Multiple restarts with different initial conditions
    np.random.seed(42)
    
    # Initial candidates
    candidates = [
        np.ones(N),
    ]
    
    # Alternating pattern
    alt = np.ones(N)
    alt[1::2] = -1
    candidates.append(alt)
    
    # Sine patterns
    candidates.append(np.sign(np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))))
    candidates[-1][candidates[-1] == 0] = 1
    
    candidates.append(np.sign(np.cos(np.linspace(0, 2*np.pi, N, endpoint=False))))
    candidates[-1][candidates[-1] == 0] = 1
    
    # Random walk pattern
    rw = np.ones(N)
    for i in range(1, N):
        if np.random.random() < 0.3:
            rw[i] = -rw[i-1]
        else:
            rw[i] = rw[i-1]
    candidates.append(rw)
    
    # Add some random initializations
    for _ in range(5):
        candidates.append(np.random.choice([-1, 1], N))
    
    # Process each candidate
    for idx, init_x in enumerate(candidates):
        if time.time() > deadline - 20.0:
            break
        
        # Greedy polish first
        x, val = _greedy_polish(init_x, deadline)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
        
        # Annealing if time permits
        if time.time() < deadline - 30.0:
            x, val = _anneal(x, start_time, deadline)
            
            if val < best_val:
                best_val = val
                best_x = x.copy()
        
        # Final greedy polish after annealing
        if time.time() < deadline - 10.0:
            x, val = _greedy_polish(x, deadline)
            
            if val < best_val:
                best_val = val
                best_x = x.copy()
    
    # If we have time, do more random restarts
    while time.time() < deadline - 15.0:
        x = np.random.choice([-1, 1], N)
        x, val = _greedy_polish(x, deadline)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # Final polish with whatever time remains
    if best_x is not None and time.time() < deadline - 2.0:
        best_x, best_val = _greedy_polish(best_x, deadline)
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
