"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400
import time
import numpy as np

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Kill at 150s, use 145s as safety margin
    
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

    best_result = None
    best_score = float('inf')
    
    rng = np.random.RandomState(42)
    
    # Strategy: Multiple restarts of Simulated Annealing + Hill Climbing polish
    num_restarts = 200
    max_sa_steps = 2000
    max_hc_steps = 500
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random sequence
        h = rng.choice([-1.0, 1.0], size=N)
        current_energy = autoconv_energy(h)
        
        # Simulated Annealing
        T = 1000.0
        T_min = 1.0
        cooling_rate = 0.9995
        
        for step in range(max_sa_steps):
            if time.time() > deadline:
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
        while improved and time.time() < deadline:
            improved = False
            for _ in range(max_hc_steps):
                if time.time() > deadline:
                    break
                # Try flipping each bit? Too slow.
                # Try random bits until one improves?
                # Or just a few random flips.
                i = rng.randint(N)
                h[i] *= -1
                new_energy = autoconv_energy(h)
                if new_energy < current_energy:
                    current_energy = new_energy
                    improved = True
                    if current_energy < best_score:
                        best_score = current_energy
                        best_result = h.copy()
                else:
                    h[i] *= -1
                    
    if best_result is None:
        best_result = np.ones(N)
        
    return best_result
# EVOLVE-BLOCK-END
