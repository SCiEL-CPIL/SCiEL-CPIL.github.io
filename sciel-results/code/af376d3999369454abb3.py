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
    
    # Strategy: Multiple restarts with Simulated Annealing + Hill Climbing polish
    # Increased restarts and steps, with faster convergence and better initialization
    num_restarts = 300
    max_sa_steps = 2000
    max_hc_steps = 1000
    
    # Precompute some structured initializations
    structured_inits = []
    
    # Alternating pattern
    h_alt = np.ones(N) * ((np.arange(N) % 2) * 4 - 1)
    structured_inits.append(h_alt)
    
    # Block pattern: +1 for first half, -1 for second
    h_block = np.ones(N)
    h_block[N//2:] = -1.0
    structured_inits.append(h_block)
    
    # Quarter blocks
    h_quarter = np.ones(N)
    h_quarter[N//4:3*N//4] = -1.0
    structured_inits.append(h_quarter)
    
    # Try each structured init first
    for h_init in structured_inits:
        if time.time() > deadline - 5:
            break
        current_energy = autoconv_energy(h_init)
        if current_energy < best_score:
            best_score = current_energy
            best_result = h_init.copy()
    
    for r in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        # Alternate between random and perturbed structured starts
        if r % 3 == 0 and r < len(structured_inits) * 3:
            base = structured_inits[r // 3 % len(structured_inits)]
            h = base.copy()
            # Perturb 10% of positions
            perturb_idx = rng.choice(N, size=N//10, replace=False)
            h[perturb_idx] *= -1
        else:
            h = rng.choice([-1.0, 1.0], size=N)
        
        current_energy = autoconv_energy(h)
        
        if current_energy < best_score:
            best_score = current_energy
            best_result = h.copy()
        
        # Simulated Annealing with adaptive temperature
        T = 500.0
        T_min = 0.1
        cooling_rate = 0.9999
        
        for step in range(max_sa_steps):
            if time.time() > deadline - 3:
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
        
        # Hill Climbing Polish (Greedy) - more aggressive
        improved = True
        hc_count = 0
        while improved and hc_count < max_hc_steps and time.time() < deadline - 3:
            improved = False
            # Try multiple random flips per iteration to find improvements faster
            for _ in range(5):
                if time.time() > deadline - 3:
                    break
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
        # Do a more thorough final polish
        for _ in range(2000):
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
