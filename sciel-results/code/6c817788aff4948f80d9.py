import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    N = 24
    M = 2**15  # FFT size
    sqrt_N = np.sqrt(N)

    # Precompute DFT matrix D where F = D @ c
    # F[k] = sum_{n=0}^{N-1} c_n exp(-2j pi k n / M)
    k = np.arange(M)
    n = np.arange(N)
    D = np.exp(-2j * np.pi * (k[:, None] * n[None, :] / M))

    def get_score(F):
        return np.max(np.abs(F)) / sqrt_N

    def flip_bit_F(F, c, idx):
        return F - 2.0 * c[idx] * D[:, idx]

    def flip_two_bits_F(F, c, idx1, idx2):
        return F - 2.0 * (c[idx1] * D[:, idx1] + c[idx2] * D[:, idx2])

    # Smooth surrogate: soft-max of |F| using log-sum-exp
    def soft_max_score(F, beta=50.0):
        abs_F = np.abs(F)
        max_val = np.max(abs_F)
        # log-sum-exp for numerical stability
        shifted = abs_F - max_val
        log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
        return (max_val + log_sum) / sqrt_N

    rng = np.random.RandomState(42)

    best_score = float('inf')
    best_c = None
    best_F = None

    # Known good starting points
    # Golay sequence of length 24
    golay_24 = np.array([1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1], dtype=float)
    
    # Try several known sequences and random starts
    initial_candidates = [golay_24.copy()]
    for _ in range(10):
        initial_candidates.append(np.random.choice([-1.0, 1.0], N))
    
    # Also try all-ones and alternating patterns
    initial_candidates.append(np.ones(N))
    initial_candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    for init_c in initial_candidates:
        if time.time() > deadline:
            break
        
        c = init_c.copy()
        F = D @ c
        current_score = get_score(F)
        
        # Simulated Annealing with smooth surrogate
        T = 0.05
        alpha = 0.9995
        T_min = 1e-7
        
        while time.time() < deadline - 10.0 and T > T_min:
            # Perturb
            if rng.rand() < 0.7:
                # 1 flip
                idx = rng.randint(N)
                F_new = flip_bit_F(F, c, idx)
                c_new = c.copy()
                c_new[idx] *= -1
            else:
                # 2 flip
                idx1, idx2 = rng.choice(N, 2, replace=False)
                F_new = flip_two_bits_F(F, c, idx1, idx2)
                c_new = c.copy()
                c_new[idx1] *= -1
                c_new[idx2] *= -1
            
            # Use smooth surrogate for acceptance decision
            new_smooth = soft_max_score(F_new, beta=50.0)
            current_smooth = soft_max_score(F, beta=50.0)
            delta_smooth = new_smooth - current_smooth
            
            if delta_smooth < 0 or rng.rand() < np.exp(-delta_smooth / T):
                F = F_new
                c = c_new
                current_score = get_score(F)
                
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                    best_F = F.copy()
            
            T *= alpha

    # Final Local Search on best_c
    if best_c is not None:
        c = best_c.copy()
        F = best_F.copy()
        current_score = best_score
        
        # 1-flip local search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_delta = 0
            best_idx = -1
            best_F_candidate = None
            for idx in range(N):
                F_candidate = flip_bit_F(F, c, idx)
                new_score = get_score(F_candidate)
                delta = new_score - current_score
                if delta < best_delta:
                    best_delta = delta
                    best_idx = idx
                    best_F_candidate = F_candidate
            
            if best_idx != -1:
                c[best_idx] *= -1
                F = best_F_candidate
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                    best_F = F.copy()
                improved = True
        
        # 2-flip local search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_delta = 0
            best_idx1 = -1
            best_idx2 = -1
            best_F_candidate = None
            for n1 in range(N):
                for n2 in range(n1+1, N):
                    F_candidate = flip_two_bits_F(F, c, n1, n2)
                    new_score = get_score(F_candidate)
                    delta = new_score - current_score
                    if delta < best_delta:
                        best_delta = delta
                        best_idx1 = n1
                        best_idx2 = n2
                        best_F_candidate = F_candidate
            
            if best_idx1 != -1:
                c[best_idx1] *= -1
                c[best_idx2] *= -1
                F = best_F_candidate
                current_score += best_delta
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                    best_F = F.copy()
                improved = True

    if best_c is None:
        return [1]*24
        
    return [int(x) for x in best_c]
