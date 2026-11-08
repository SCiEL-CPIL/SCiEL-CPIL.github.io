"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconvolve(x):
    """Compute the circular autoconvolution of x using FFT."""
    n = len(x)
    fft_x = np.fft.fft(x)
    ac = np.fft.ifft(fft_x * np.conj(fft_x)).real
    return ac


def _compute_c3(x):
    """Compute the C3 upper bound: max of autoconvolution of sign(x)."""
    signs = np.sign(x)
    signs[signs == 0] = 1.0
    ac = _autoconvolve(signs)
    return np.max(ac)


def construct():
    deadline = time.time() + 140  # 150s wall-clock limit, 10s safety margin
    rng = np.random.default_rng(12345)
    
    best_x = None
    best_c3 = np.inf
    
    # Multiple random restarts with simulated annealing
    num_restarts = 50
    max_sa_steps = 3000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize: diverse starting points
        if restart == 0:
            x = np.ones(N)
        elif restart == 1:
            x = np.ones(N)
            x[::2] = -1.0
        elif restart == 2:
            x = np.ones(N)
            x[:N//2] = -1.0
        elif restart == 3:
            # Block pattern
            x = np.ones(N)
            block_size = N // 8
            for b in range(0, N, 2 * block_size):
                x[b:b+block_size] = -1.0
        else:
            # Random initialization
            x = rng.choice([-1.0, 1.0], size=N)
        
        current_c3 = _compute_c3(x)
        
        if current_c3 < best_c3:
            best_c3 = current_c3
            best_x = x.copy()
        
        # Simulated annealing
        temperature = 1.0
        cooling_rate = 0.9995
        min_temperature = 0.01
        
        for sa_step in range(max_sa_steps):
            if time.time() > deadline:
                break
            
            # Choose number of flips: mostly 1, sometimes 2-3
            r = rng.random()
            if r < 0.5:
                num_flips = 1
            elif r < 0.85:
                num_flips = 2
            else:
                num_flips = 3
            
            flips = rng.choice(N, size=num_flips, replace=False)
            
            x_trial = x.copy()
            x_trial[flips] *= -1.0
            
            c3_trial = _compute_c3(x_trial)
            delta_c3 = c3_trial - current_c3
            
            if delta_c3 < 0:
                # Accept improvement
                x = x_trial
                current_c3 = c3_trial
                if current_c3 < best_c3:
                    best_c3 = current_c3
                    best_x = x.copy()
            else:
                # Accept worse moves with probability exp(-delta/T)
                if temperature > min_temperature:
                    if rng.random() < np.exp(-delta_c3 / temperature):
                        x = x_trial
                        current_c3 = c3_trial
            
            # Cool down
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
        
        # Final polish: single-flip hill climbing
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                x_trial = x.copy()
                x_trial[i] *= -1.0
                c3_trial = _compute_c3(x_trial)
                if c3_trial < current_c3:
                    x = x_trial
                    current_c3 = c3_trial
                    improved = True
                    if current_c3 < best_c3:
                        best_c3 = current_c3
                        best_x = x.copy()
    
    # Global final polish on best solution found
    if best_x is not None:
        x_final = best_x.copy()
        c3_final = _compute_c3(x_final)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                x_trial = x_final.copy()
                x_trial[i] *= -1.0
                c3_trial = _compute_c3(x_trial)
                if c3_trial < c3_final:
                    x_final = x_trial
                    c3_final = c3_trial
                    improved = True
                    best_c3 = c3_final
                    best_x = x_final.copy()
    
    if best_x is None:
        return np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
