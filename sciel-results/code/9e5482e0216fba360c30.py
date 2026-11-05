import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    N = 24
    M = 2 ** 16  # FFT size for dense evaluation
    sqrt_N = np.sqrt(N)
    
    # Precompute DFT matrix D where F = D @ c
    # F[k] = sum_{n=0}^{N-1} c_n exp(-2j pi k n / M)
    k = np.arange(M)
    n_idx = np.arange(N)
    # phases = k[:, None] * n_idx[None, :]
    D = np.exp(-2j * np.pi * (k[:, None] * n_idx[None, :] / M))
    
    def get_score(F):
        return np.max(np.abs(F)) / sqrt_N

    def flip_bit_F(F, c, idx):
        # F_new = F - 2 * c[idx] * D[:, idx]
        return F - 2.0 * c[idx] * D[:, idx]

    def flip_two_bits_F(F, c, idx1, idx2):
        return F - 2.0 * (c[idx1] * D[:, idx1] + c[idx2] * D[:, idx2])

    def local_search_1flip(current_c, current_f, current_max):
        """Thorough 1-flip local search until no improvement."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_flip = -1
            best_max_val = current_max
            best_f_candidate = None
            for n in range(N):
                f_new = flip_bit_F(current_f, current_c, n)
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_val - 1e-12:
                    best_max_val = max_val
                    best_flip = n
                    best_f_candidate = f_new
            if best_flip != -1:
                current_c[best_flip] *= -1.0
                current_f = best_f_candidate
                current_max = best_max_val
                improved = True
        return current_c, current_f, current_max

    def local_search_2flip(current_c, current_f, current_max):
        """2-flip local search until no improvement."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_flip1 = -1
            best_flip2 = -1
            best_max_val = current_max
            best_f_candidate = None
            for n1 in range(N):
                for n2 in range(n1 + 1, N):
                    f_new = flip_two_bits_F(current_f, current_c, n1, n2)
                    max_val = np.max(np.abs(f_new))
                    if max_val < best_max_val - 1e-12:
                        best_max_val = max_val
                        best_flip1 = n1
                        best_flip2 = n2
                        best_f_candidate = f_new
            if best_flip1 != -1:
                current_c[best_flip1] *= -1.0
                current_c[best_flip2] *= -1.0
                current_f = best_f_candidate
                current_max = best_max_val
                improved = True
        return current_c, current_f, current_max

    # Initial candidates: Golay-Rudin-Shapiro truncated to 24
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    grs_coeffs = np.array(p[:24], dtype=np.float64)
    
    # Also try the known good candidate from inspiration 2 (if it exists in memory)
    # But let's stick to GRS and random restarts.
    
    best_c = grs_coeffs.copy()
    best_f = D @ best_c
    best_max = np.max(np.abs(best_f))
    
    # Initial local search on GRS
    best_c, best_f, best_max = local_search_1flip(best_c, best_f, best_max)
    best_c, best_f, best_max = local_search_2flip(best_c, best_f, best_max)
    
    # Simulated Annealing / Random Restarts
    rng = np.random.default_rng(42)
    
    # We will run a few restarts if time permits.
    # The search space is 2^24. GRS is a very strong starting point.
    # Let's try to improve with SA.
    
    T = 0.05
    alpha = 0.999
    T_min = 1e-6
    
    # Start SA from best found so far
    current_c = best_c.copy()
    current_f = best_f.copy()
    current_max = best_max
    
    while time.time() < deadline and T > T_min:
        # Perturb
        if rng.rand() < 0.7:
            # 1 flip
            idx = rng.randint(N)
            f_new = flip_bit_F(current_f, current_c, idx)
            c_new = current_c.copy()
            c_new[idx] *= -1
        else:
            # 2 flip
            idx1, idx2 = rng.choice(N, 2, replace=False)
            f_new = flip_two_bits_F(current_f, current_c, idx1, idx2)
            c_new = current_c.copy()
            c_new[idx1] *= -1
            c_new[idx2] *= -1
        
        new_max = np.max(np.abs(f_new))
        delta = new_max - current_max
        
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            current_f = f_new
            current_c = c_new
            current_max = new_max
            
            if current_max < best_max:
                best_max = current_max
                best_c = current_c.copy()
                best_f = current_f.copy()
                
        T *= alpha
        
        # Occasionally run local search to escape local minima more effectively
        if rng.rand() < 0.05:
            current_c, current_f, current_max = local_search_1flip(current_c, current_f, current_max)
            if current_max < best_max:
                best_max = current_max
                best_c = current_c.copy()
                best_f = current_f.copy()

    # Final polish on the best candidate found
    if time.time() < deadline - 5.0:
        best_c, best_f, best_max = local_search_1flip(best_c, best_f, best_max)
        best_c, best_f, best_max = local_search_2flip(best_c, best_f, best_max)
        
        # Try a few more random restarts if we have significant time left
        while time.time() < deadline - 10.0:
            c_rand = rng.choice([-1.0, 1.0], N)
            f_rand = D @ c_rand
            max_rand = np.max(np.abs(f_rand))
            
            c_rand, f_rand, max_rand = local_search_1flip(c_rand, f_rand, max_rand)
            c_rand, f_rand, max_rand = local_search_2flip(c_rand, f_rand, max_rand)
            
            if max_rand < best_max:
                best_max = max_rand
                best_c = c_rand.copy()
                best_f = f_rand.copy()

    return [int(c) for c in best_c]
