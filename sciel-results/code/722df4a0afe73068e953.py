"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # New Approach: Simulated Annealing to minimize Sum of Squares of Sidelobes (Energy)
    # This differs from the previous Hill Climbing on PSL.
    # We assume the metric is related to the energy of the autocorrelation sidelobes,
    # or that optimizing this energy leads to a better score than PSL minimization.
    
    np.random.seed(123) # Different seed
    f = np.random.choice([-1, 1], size=N)
    
    def calculate_energy(seq):
        """
        Calculate the sum of squares of the aperiodic autocorrelation sidelobes.
        E = sum_{k != 0} R(k)^2
        """
        ac = np.correlate(seq, seq, mode='full')
        # ac[N-1] is R(0) = N
        # We want sum of squares of all other terms
        # ac has length 2N-1
        # R(0) is at index N-1
        # Sum of squares of all elements minus R(0)^2
        return np.sum(ac**2) - ac[N-1]**2

    current_energy = calculate_energy(f)
    best_f = f.copy()
    best_energy = current_energy
    
    start_time = time.time()
    max_time = 100.0
    
    # Simulated Annealing Parameters
    T = 1000.0 # Initial temperature
    T_min = 0.1
    cooling_rate = 0.99995 # Tune to reach T_min near time limit
    
    # Estimate iterations:
    # Each iteration ~ 0.2ms (np.correlate + overhead)
    # 100s / 0.2ms = 500,000 iterations.
    # Cooling rate: T * (cooling_rate)^500000 = 0.1
    # (0.1/1000)^(1/500000) = 1e-3 ^ 2e-6 = exp(2e-6 * ln(1e-3)) = exp(-1.38e-5) = 1 - 1.38e-5
    # So cooling_rate = 1 - 1.4e-5 is roughly correct.
    
    while (time.time() - start_time) < max_time:
        # Flip a random bit
        i = np.random.randint(N)
        f[i] = -f[i]
        
        new_energy = calculate_energy(f)
        delta = new_energy - current_energy
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_f = f.copy()
        else:
            # Revert
            f[i] = -f[i]
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_f
# EVOLVE-BLOCK-END
