"""Rewritten for the third autocorrelation inequality task."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(f):
    """Compute max absolute value of autoconvolution (excluding self-lag 0)."""
    n = len(f)
    size = 2 * n
    F = np.fft.fft(f, n=size)
    conv = np.fft.ifft(F * F).real[:n]
    # Exclude lag 0 (self-convolution)
    return np.max(np.abs(conv[1:]))


def _block_flip(f, i, block_len):
    """Flip a block of length block_len starting at position i (cyclic)."""
    f_new = f.copy()
    end = (i + block_len) % N
    if i + block_len <= N:
        f_new[i:i + block_len] *= -1.0
    else:
        f_new[i:] *= -1.0
        f_new[:end] *= -1.0
    return f_new


def _single_flip(f, i):
    """Flip a single position i."""
    f_new = f.copy()
    f_new[i] *= -1.0
    return f_new


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s wall-clock limit
    
    rng = np.random.RandomState(12345)
    
    # Phase 1: Multi-start with block-flip simulated annealing
    # Try several different initializations and block sizes
    
    best_f = None
    best_score = np.inf
    
    # Different initializations and block configurations
    configs = [
        (42, 4, 50.0, 0.995),
        (123, 8, 40.0, 0.997),
        (456, 2, 30.0, 0.993),
        (789, 16, 60.0, 0.998),
        (101, 6, 45.0, 0.996),
    ]
    
    for cfg_idx, (seed, block_len, T_init, cooling) in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Remaining time allocation: give more time to early configs
        remaining = deadline - time.time()
        time_per_config = remaining / (len(configs) - cfg_idx) * 0.8
        
        rng_cfg = np.random.RandomState(seed)
        
        # Initialize with random ±1 values
        f = rng_cfg.choice([-1.0, 1.0], size=N)
        current_score = _autoconv_max_abs(f)
        
        T = T_init
        T_min = 0.5
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < start_time + time_per_config:
            # Pick a random position to start block flip
            i = rng_cfg.randint(N)
            
            # Occasionally try single flip for fine tuning
            if rng_cfg.rand() < 0.2:
                f_new = _single_flip(f, i)
            else:
                f_new = _block_flip(f, i, block_len)
            
            new_score = _autoconv_max_abs(f_new)
            
            if new_score < current_score:
                f = f_new
                current_score = new_score
                no_improve_count = 0
                
                if current_score < best_score:
                    best_score = current_score
                    best_f = f.copy()
            else:
                # Simulated annealing acceptance
                if T > T_min and rng_cfg.rand() < np.exp(-(new_score - current_score) / T):
                    f = f_new
                    current_score = new_score
                no_improve_count += 1
            
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Restart if stuck too long
            if no_improve_count > max_no_improve:
                # Random restart with small perturbation
                for _ in range(5):
                    i = rng_cfg.randint(N)
                    f = _block_flip(f, i, block_len)
                current_score = _autoconv_max_abs(f)
                no_improve_count = 0
                T = T_init * 0.5
        
    # Phase 2: Greedy single-flip refinement on the best solution
    if best_f is not None and time.time() < deadline:
        f = best_f.copy()
        current_score = _autoconv_max_abs(f)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                old_val = f[i]
                f[i] = -old_val
                new_score = _autoconv_max_abs(f)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    f[i] = old_val
        
        if current_score < best_score:
            best_score = current_score
            best_f = f.copy()
    
    # Phase 3: If we have time, try a few more random restarts with greedy
    if time.time() < deadline - 10.0:
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            
            rng_r = np.random.RandomState(1000 + restart * 137)
            f = rng_r.choice([-1.0, 1.0], size=N)
            current_score = _autoconv_max_abs(f)
            
            # Quick greedy refinement
            for _pass in range(3):
                if time.time() > deadline - 5.0:
                    break
                for i in range(N):
                    if time.time() > deadline - 5.0:
                        break
                    old_val = f[i]
                    f[i] = -old_val
                    new_score = _autoconv_max_abs(f)
                    if new_score < current_score:
                        current_score = new_score
                    else:
                        f[i] = old_val
            
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
    
    if best_f is None:
        return np.ones(N)
    
    return best_f
# EVOLVE-BLOCK-END
