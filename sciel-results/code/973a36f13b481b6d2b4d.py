# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Construct a signed step function with N equal intervals on [-1/4, 1/4].
    Uses Logistic Map (r=4) to generate initial sign sequences.
    Employs Steepest Descent local search with multiple restarts to minimize the C3 upper bound.
    """
    N = 400
    
    def compute_c3_bound(s):
        """
        Compute the upper bound on C3 (max absolute value of triple autocorrelation).
        """
        target_len = 3 * N - 2
        N_pad = 1 << (target_len - 1).bit_length()
        
        s_pad = np.zeros(N_pad)
        s_pad[:N] = s
        S = np.fft.fft(s_pad)
        C3 = np.fft.ifft(S ** 3).real
        return np.max(np.abs(C3))

    # Generate candidates using Logistic Map x_{n+1} = 4 x_n (1 - x_n)
    # We use a set of diverse irrational seeds.
    seeds = [
        np.sqrt(2) % 1.0,
        np.sqrt(3) % 1.0,
        np.pi % 1.0,
        (1 + np.sqrt(5)) / 2 % 1.0,
        np.e % 1.0,
        0.123456789,
        0.987654321,
        0.5 + (1/np.sqrt(2)) % 1.0,
        np.sqrt(1.1) % 1.0,
        np.sqrt(1.3) % 1.0,
        np.sqrt(1.7) % 1.0,
        np.sqrt(1.9) % 1.0
    ]
    
    candidates = []
    
    for seed in seeds:
        x = seed
        signs = np.zeros(N)
        for i in range(N):
            x = 4.0 * x * (1.0 - x)
            signs[i] = 1.0 if x > 0.5 else -1.0
        candidates.append(signs)
        
    # Select the best 5 candidates based on initial bound
    scored_candidates = []
    for s in candidates:
        bound = compute_c3_bound(s)
        scored_candidates.append((bound, s))
        
    scored_candidates.sort(key=lambda x: x[0])
    top_candidates = [s for _, s in scored_candidates[:5]]
    
    best_signs = top_candidates[0]
    best_bound = scored_candidates[0][0]
    
    # Steepest Descent Local Search on top candidates
    for start_signs in top_candidates:
        signs = start_signs.copy()
        current_bound = compute_c3_bound(signs)
        
        # Run local search
        for _ in range(20):  # 20 iterations
            best_flip_idx = -1
            best_new_bound = current_bound
            
            for i in range(N):
                signs[i] *= -1.0
                new_bound = compute_c3_bound(signs)
                
                if new_bound < best_new_bound:
                    best_new_bound = new_bound
                    best_flip_idx = i
                signs[i] *= -1.0
                
            if best_flip_idx != -1:
                signs[best_flip_idx] *= -1.0
                current_bound = best_new_bound
            else:
                break
                
        if current_bound < best_bound:
            best_bound = current_bound
            best_signs = signs.copy()
            
    return best_signs
# EVOLVE-BLOCK-END
