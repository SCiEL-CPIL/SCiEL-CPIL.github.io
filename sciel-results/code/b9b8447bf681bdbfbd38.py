"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import time
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Kill at 150s, use 145s as safety margin
    best_result = None
    best_score = float('inf')
    
    def autoconv_energy(heights):
        """Compute the L2 norm of the autoconvolution (as a proxy for C3)."""
        n_fft = 2 * N
        f_ext = np.zeros(n_fft)
        f_ext[:N] = heights
        F = np.fft.rfft(f_ext)
        power = F * np.conj(F)
        autocorr = np.fft.irfft(power, n=n_fft)[:N]
        # Minimize sum of squares of non-zero lags
        return np.sum(autocorr[1:N] ** 2)
    
    # Precompute indices for efficient evaluation
    indices = np.arange(N)
    
    def fast_flip_eval(h, idx):
        """Evaluate energy after flipping h[idx] without full recompute if possible.
        For now, just do the FFT approach but with pre-allocated arrays."""
        h[idx] *= -1
        n_fft = 2 * N
        f_ext = np.zeros(n_fft)
        f_ext[:N] = h
        F = np.fft.rfft(f_ext)
        power = F * np.conj(F)
        autocorr = np.fft.irfft(power, n=n_fft)[:N]
        energy = np.sum(autocorr[1:N] ** 2)
        h[idx] *= -1  # undo
        return energy
    
    rng = np.random.RandomState(42)
    
    # Strategy: Multiple restarts with Simulated Annealing + Hill Climbing polish
    num_restarts = 150
    max_sa_steps = 3000
    max_hc_steps = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        # Initialize with random sequence
        h = rng.choice([-1.0, 1.0], size=N)
        current_energy = autoconv_energy(h)
        
        if current_energy < best_score:
            best_score = current_energy
            best_result = h.copy()
        
        # Simulated Annealing
        T = 500.0
        T_min = 0.5
        cooling_rate = 0.9998
        
        for step in range(max_sa_steps):
            if time.time() > deadline - 2:
                break
                
            i = rng.randint(N)
            h[i] *= -1
            new_energy = autoconv_energy(h)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or (T > 0 and rng.rand() < np.exp(-delta_e / T)):
                current_energy = new_energy
                if current_energy < best_score:
                    best_score = current_energy
                    best_result = h.copy()
            else:
                h[i] *= -1
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        # Hill Climbing Polish (Greedy)
        improved = True
        hc_count = 0
        while improved and hc_count < max_hc_steps and time.time() < deadline - 2:
            improved = False
            i = rng.randint(N)
            h[i] *= -1
            new_energy = autoconv_energy(h)
            if new_energy < current_energy:
                current_energy = new_energy
                improved = True
                hc_count += 1
                if current_energy < best_score:
                    best_score = current_energy
                    best_result = h.copy()
            else:
                h[i] *= -1
                hc_count += 1
    
    # Final local search on the best result found
    if best_result is not None and time.time() < deadline - 1:
        h = best_result.copy()
        current_energy = best_score
        # Do a few more hill climbing steps
        for _ in range(500):
            if time.time() > deadline - 1:
                break
            i = rng.randint(N)
            h[i] *= -1
            new_energy = autoconv_energy(h)
            if new_energy < current_energy:
                current_energy = new_energy
                best_score = current_energy
                best_result = h.copy()
            else:
                h[i] *= -1
    
    if best_result is None:
        best_result = np.ones(N)
    
    return best_result
# EVOLVE-BLOCK-END
