"""Program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    Uses a logistic map chaotic sequence generator with an irrational rotation
    offset for low-discrepancy sampling, followed by a multi-pass local search
    with a smarter ordering to minimize the C3 autocorrelation bound.
    """
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3: max absolute value of the
        third-order autocorrelation.
        """
        N_pad = 1 << (3 * N - 1).bit_length()
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))
    
    def gen_logistic_signs(seed):
        """Generate a sign sequence from logistic map at r=4."""
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
        return signs
    
    # Generate candidates using logistic map with several seeds
    # We mix pure logistic seeds with irrational rotation offsets
    seeds = []
    # Pure logistic seeds
    seeds.append(np.sqrt(2) % 1.0)
    seeds.append(np.sqrt(3) % 1.0)
    seeds.append(np.pi % 1.0)
    seeds.append((1 + np.sqrt(5)) / 2 % 1.0)
    seeds.append(np.e % 1.0)
    # Irrational rotation offsets combined with logistic
    for base_seed in [np.sqrt(2) % 1.0, np.pi % 1.0, np.e % 1.0]:
        for rot in [np.sqrt(5) % 1.0, np.sqrt(2) * np.pi % 1.0]:
            seeds.append((base_seed + rot) % 1.0)
    
    # Evaluate all candidates
    best_signs = None
    best_bound = float('inf')
    
    for seed in seeds:
        signs = gen_logistic_signs(seed)
        bound = compute_c3_bound(signs)
        if bound < best_bound:
            best_bound = bound
            best_signs = signs.copy()
    
    # Multi-pass local search with frequency-ordered flips
    # The idea: flip positions where the local autocorrelation contribution is largest
    signs = best_signs
    current_bound = best_bound
    max_evals = 20000
    evals = 0
    improved = True
    
    while improved and evals < max_evals:
        improved = False
        # Order positions by their current magnitude of influence (simple heuristic:
        # flip positions in a shuffled order each pass for diversity)
        order = np.arange(N)
        np.random.default_rng(42).shuffle(order)
        
        for i in order:
            if evals >= max_evals:
                break
            signs[i] *= -1.0
            new_bound = compute_c3_bound(signs)
            evals += 1
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
            else:
                signs[i] *= -1.0
    
    return signs
# EVOLVE-BLOCK-END
