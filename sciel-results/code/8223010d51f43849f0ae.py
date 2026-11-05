import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    """
    Constructs a signed step function with 400 intervals to minimize the
    third autocorrelation coefficient upper bound using simulated annealing
    with a C3-focused objective and random restarts.
    
    The C3 upper bound relates to the maximum sidelobe of the autocorrelation.
    We minimize the L-infinity norm of sidelobes (max absolute value) to
    directly reduce the C3 bound, using simulated annealing to escape local minima.
    """
    
    def compute_c3_bound(s):
        """Compute the C3 upper bound: max |autocorrelation sidelobe|."""
        ac = np.correlate(s, s, mode='full')
        sidelobes = ac[N:]  # lags 1 to N-1
        return np.max(np.abs(sidelobes))
    
    def compute_sidelobe_energy(s):
        """Compute sum of squared sidelobes as secondary objective."""
        ac = np.correlate(s, s, mode='full')
        sidelobes = ac[N:]
        return np.sum(sidelobes**2)
    
    def combined_cost(s, weight=0.1):
        """Combined cost: primarily C3 bound, secondarily sidelobe energy."""
        c3 = compute_c3_bound(s)
        energy = compute_sidelobe_energy(s)
        return c3 + weight * energy / N
    
    def simulated_annealing(s, max_iter=50000, T_start=10.0, T_end=0.01, cooling=0.9998):
        """Simulated annealing to minimize the C3 bound."""
        current = s.copy()
        current_cost = combined_cost(current)
        best = current.copy()
        best_cost = current_cost
        T = T_start
        
        for i in range(max_iter):
            T = T_start * (cooling ** i)
            
            # Random flip
            idx = np.random.randint(N)
            new = current.copy()
            new[idx] = -new[idx]
            
            new_cost = combined_cost(new)
            delta = new_cost - current_cost
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current = new
                current_cost = new_cost
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost
        
        return best, best_cost
    
    # Multiple random restarts with different initializations
    best_signs = None
    best_score = np.inf
    
    rng = np.random.default_rng(42)
    
    for restart in range(8):
        if restart == 0:
            # Chirp initialization
            n = np.arange(N)
            phase = (n * (n - 1)) // 2
            signs = np.ones(N)
            signs[phase % 2 == 1] = -1.0
        elif restart == 1:
            # Pure random initialization
            signs = rng.choice([-1.0, 1.0], size=N)
        elif restart == 2:
            # Low-frequency sinusoidal sign pattern
            t = np.linspace(0, 4 * np.pi, N)
            signs = np.sign(np.sin(t) + 0.3 * np.sin(2 * t))
            signs[signs == 0] = 1.0
        elif restart == 3:
            # M-sequence style using linear feedback
            # Use a simple PRNG-based pattern
            signs = np.zeros(N)
            state = 1
            for i in range(N):
                signs[i] = 1.0 if (state & 1) == 1 else -1.0
                # Simple LFSR
                state = ((state << 1) ^ (state >> 7)) & 0xFF
        elif restart == 4:
            # Block pattern with varying block sizes
            signs = np.zeros(N)
            pos = 0
            block_sign = 1.0
            while pos < N:
                block_len = rng.integers(3, 15)
                signs[pos:pos+block_len] = block_sign
                pos += block_len
                block_sign = -block_sign
        elif restart == 5:
            # Random walk with occasional jumps
            signs = np.zeros(N)
            val = 1.0
            for i in range(N):
                signs[i] = val
                if rng.random() < 0.15:
                    val = -val
        elif restart == 6:
            # Fourier-based: sign of sum of a few random frequencies
            freqs = rng.uniform(0.01, 0.15, size=5)
            phases = rng.uniform(0, 2*np.pi, size=5)
            t = np.arange(N) / N
            signal = np.zeros(N)
            for f, p in zip(freqs, phases):
                signal += np.sin(2 * np.pi * f * t * N + p)
            signs = np.sign(signal)
            signs[signs == 0] = 1.0
        else:
            # Hybrid: chirp + random perturbation
            n = np.arange(N)
            phase = (n * (n - 1)) // 2
            signs = np.ones(N)
            signs[phase % 2 == 1] = -1.0
            # Flip 10% of signs randomly
            flip_mask = rng.random(N) < 0.1
            signs[flip_mask] = -signs[flip_mask]
        
        # Run simulated annealing
        result, score = simulated_annealing(signs, max_iter=30000)
        
        if score < best_score:
            best_score = score
            best_signs = result
    
    # Final local search refinement on the best solution
    current = best_signs.copy()
    current_cost = combined_cost(current)
    
    # Stochastic local search: flip one bit if it improves
    for _ in range(10000):
        idx = np.random.randint(N)
        new = current.copy()
        new[idx] = -new[idx]
        new_cost = combined_cost(new)
        if new_cost < current_cost:
            current = new
            current_cost = new_cost
    
    return current

# EVOLVE-BLOCK-END
