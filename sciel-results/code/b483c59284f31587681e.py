import numpy as np
from scipy.optimize import minimize

N = 400

def construct():
    """
    Constructs a signed step function with 400 intervals to minimize the
    third autocorrelation coefficient upper bound using FFT-based autocorrelation
    computation and a swap-based neighborhood simulated annealing.
    
    Uses efficient FFT autocorrelation, swap moves for better exploration,
    and multiple structured initializations.
    """
    
    def compute_ac_fft(s):
        """Compute full autocorrelation via FFT for speed."""
        n = len(s)
        # Zero-pad to avoid circular convolution issues
        size = 2 * n
        f = np.fft.rfft(s, n=size)
        ac = np.fft.irfft(f * np.conj(f), n=size)[:n]
        return ac
    
    def compute_c3_bound(s):
        """Compute the C3 upper bound: max |autocorrelation sidelobe|."""
        ac = compute_ac_fft(s)
        sidelobes = ac[1:]  # lags 1 to N-1
        return np.max(np.abs(sidelobes))
    
    def combined_cost(s):
        """Combined cost: primarily C3 bound."""
        ac = compute_ac_fft(s)
        sidelobes = ac[1:]
        c3 = np.max(np.abs(sidelobes))
        return c3
    
    def simulated_annealing_swap(s, max_iter=80000, T_start=5.0, T_end=0.001):
        """Simulated annealing with swap neighborhood."""
        current = s.copy()
        current_cost = combined_cost(current)
        best = current.copy()
        best_cost = current_cost
        
        # Precompute autocorrelation for current
        n = len(current)
        
        for i in range(max_iter):
            # Adaptive temperature
            T = T_start * (T_end / T_start) ** (i / max_iter)
            
            # Swap move: pick two random indices and swap their signs
            idx1, idx2 = np.random.randint(0, n, 2)
            if idx1 == idx2:
                continue
            
            new = current.copy()
            new[idx1], new[idx2] = current[idx2], current[idx1]
            
            new_cost = combined_cost(new)
            delta = new_cost - current_cost
            
            if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
                current = new
                current_cost = new_cost
                if current_cost < best_cost:
                    best = current.copy()
                    best_cost = current_cost
        
        return best, best_cost
    
    def local_search_swap(s, max_iter=5000):
        """Local search with swap moves."""
        current = s.copy()
        current_cost = combined_cost(current)
        n = len(current)
        
        for _ in range(max_iter):
            improved = False
            # Try a few random swaps
            for _ in range(10):
                idx1, idx2 = np.random.randint(0, n, 2)
                if idx1 == idx2:
                    continue
                new = current.copy()
                new[idx1], new[idx2] = current[idx2], current[idx1]
                new_cost = combined_cost(new)
                if new_cost < current_cost:
                    current = new
                    current_cost = new_cost
                    improved = True
                    break
            if not improved:
                break
        
        return current, current_cost
    
    best_signs = None
    best_score = np.inf
    
    rng = np.random.default_rng(123)
    
    initializations = []
    
    # 1. Zadoff-Chu-inspired: root of unity pattern
    root = 5  # coprime to N
    t = np.arange(N)
    zc = np.exp(2j * np.pi * root * t * (t + 1) // (2 * N))
    signs = np.sign(zc.real)
    signs[signs == 0] = 1.0
    initializations.append(signs)
    
    # 2. Chirp with different parameter
    n = np.arange(N)
    phase = (n * (n + 1)) // 2
    signs = np.ones(N)
    signs[phase % 3 == 0] = -1.0
    signs[phase % 3 == 1] = 1.0
    signs[phase % 3 == 2] = -1.0
    initializations.append(signs)
    
    # 3. Random
    initializations.append(rng.choice([-1.0, 1.0], size=N))
    
    # 4. Block pattern with golden-ratio-like block sizes
    signs = np.zeros(N)
    pos = 0
    block_sign = 1.0
    phi = (1 + np.sqrt(5)) / 2
    while pos < N:
        block_len = max(1, int(3 + 5 * (phi * (pos / N) % 1)))
        signs[pos:pos+block_len] = block_sign
        pos += block_len
        block_sign = -block_sign
    initializations.append(signs)
    
    # 5. Frequency comb: sign of sum of harmonics
    t = np.arange(N) / N
    signal = np.zeros(N)
    for k in [1, 3, 7, 13]:
        signal += np.sin(2 * np.pi * k * t * N / 4)
    signs = np.sign(signal)
    signs[signs == 0] = 1.0
    initializations.append(signs)
    
    # 6. Random walk
    signs = np.zeros(N)
    val = 1.0
    for i in range(N):
        signs[i] = val
        if rng.random() < 0.1:
            val = -val
    initializations.append(signs)
    
    # 7. LFSR-based
    signs = np.zeros(N)
    state = 1
    for i in range(N):
        signs[i] = 1.0 if (state & 1) == 1 else -1.0
        state = ((state << 1) ^ (state >> 7)) & 0xFF
    initializations.append(signs)
    
    # 8. Hybrid: mix of two patterns
    s1 = initializations[0].copy()
    s2 = initializations[2].copy()
    mix = rng.random(N)
    signs = np.where(mix < 0.5, s1, s2)
    initializations.append(signs)
    
    # Run simulated annealing on each initialization
    for init in initializations:
        result, score = simulated_annealing_swap(init, max_iter=60000)
        
        # Local search refinement
        result, score = local_search_swap(result, max_iter=3000)
        
        if score < best_score:
            best_score = score
            best_signs = result
    
    # Final refinement with flip-based local search
    current = best_signs.copy()
    current_cost = combined_cost(current)
    
    for _ in range(20000):
        idx = np.random.randint(N)
        new = current.copy()
        new[idx] = -new[idx]
        new_cost = combined_cost(new)
        if new_cost < current_cost:
            current = new
            current_cost = new_cost
    
    return current

# EVOLVE-BLOCK-END
