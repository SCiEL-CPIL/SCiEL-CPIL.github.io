"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Finds the optimal 24 coefficients in {+1, -1} minimizing the score.
    Uses spectral synthesis initialization, vectorized local search,
    and simulated annealing with golden-section refinement for accuracy.
    """
    N = 24
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute DFT matrix for fast evaluation
    M_grid = 16384
    theta_grid = 2 * np.pi * np.arange(M_grid) / M_grid
    E = np.exp(1j * np.outer(theta_grid, np.arange(N)))
    
    def get_supnorm_fast(coeffs):
        """Fast sup norm via dense grid."""
        P = E @ coeffs
        return float(np.max(np.abs(P)))
    
    def get_supnorm_accurate(coeffs):
        """Accurate sup norm with golden-section refinement."""
        M = 8192
        c_padded = np.zeros(M, dtype=complex)
        c_padded[:N] = coeffs
        P = M * np.fft.ifft(c_padded)
        mags = np.abs(P)
        
        K = 50
        idx = np.argsort(mags)[-K:][::-1]
        theta = 2 * np.pi * np.arange(M) / M
        
        max_val = 0.0
        for i in idx:
            w = 2 * np.pi / M
            a = theta[i] - w
            b = theta[i] + w
            gr = (5 ** 0.5 - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            def f(theta_val):
                z_val = np.exp(1j * theta_val)
                p_val = 0.0
                for coeff in reversed(coeffs):
                    p_val = p_val * z_val + coeff
                return np.abs(p_val)
            
            fc = f(c)
            fd = f(d)
            for _ in range(50):
                if fc < fd:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = f(d)
                else:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = f(c)
            peak_val = max(fc, fd)
            if peak_val > max_val:
                max_val = peak_val
        return max_val
    
    def local_search_vectorized(current_c, time_limit):
        """
        Fast local search using vectorized 1-flip and 2-flip evaluation.
        Optimized to avoid redundant computations.
        """
        current_val = get_supnorm_fast(current_c)
        best_c = current_c.copy()
        best_val = current_val
        basis = E.T  # Shape (N, M_grid)
        
        # Precompute all single-flip deltas
        single_deltas = [2 * current_c[i] * basis[i] for i in range(N)]
        
        max_iters = 5000
        for iteration in range(max_iters):
            if time.time() > time_limit:
                break
            
            P_curr = E @ current_c
            best_flip_val = current_val
            best_flip_idx = -1
            
            # 1-flip search
            for i in range(N):
                P_new = P_curr - single_deltas[i]
                val = np.max(np.abs(P_new))
                if val < best_flip_val - 1e-12:
                    best_flip_val = val
                    best_flip_idx = i
            
            if best_flip_idx >= 0:
                i = best_flip_idx
                current_c[i] *= -1
                # Update the delta for this index
                single_deltas[i] *= -1
                current_val = best_flip_val
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                continue
            
            # 2-flip search - vectorized
            best_2flip_val = current_val
            best_2flip = None
            
            # Precompute all 2-flip combinations
            for i in range(N - 1):
                for j in range(i + 1, N):
                    P_new = P_curr - single_deltas[i] - single_deltas[j]
                    val = np.max(np.abs(P_new))
                    if val < best_2flip_val - 1e-12:
                        best_2flip_val = val
                        best_2flip = (i, j)
            
            if best_2flip is not None:
                i, j = best_2flip
                current_c[i] *= -1
                current_c[j] *= -1
                # Update deltas
                single_deltas[i] *= -1
                single_deltas[j] *= -1
                current_val = best_2flip_val
                if current_val < best_val - 1e-12:
                    best_c = current_c.copy()
                    best_val = current_val
                continue
            
            # 3-flip search (less frequently, but more thorough)
            if iteration % 3 == 0:
                best_3flip_val = current_val
                best_3flip = None
                for i in range(N - 2):
                    for j in range(i + 1, N - 1):
                        for k in range(j + 1, N):
                            P_new = P_curr - single_deltas[i] - single_deltas[j] - single_deltas[k]
                            val = np.max(np.abs(P_new))
                            if val < best_3flip_val - 1e-12:
                                best_3flip_val = val
                                best_3flip = (i, j, k)
                
                if best_3flip is not None:
                    i, j, k = best_3flip
                    current_c[i] *= -1
                    current_c[j] *= -1
                    current_c[k] *= -1
                    # Update deltas
                    single_deltas[i] *= -1
                    single_deltas[j] *= -1
                    single_deltas[k] *= -1
                    current_val = best_3flip_val
                    if current_val < best_val - 1e-12:
                        best_c = current_c.copy()
                        best_val = current_val
                    continue
            
            break
        
        return best_c, best_val
    
    # --- Spectral Synthesis Initialization ---
    initial_candidates = []
    
    # 1. Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    initial_candidates.append(np.array(p[:N], dtype=float))
    
    # 2. Spectral synthesis: various spectrum shapes
    for _ in range(40):
        if time.time() > deadline - 30:
            break
        # Random phase, flat magnitude
        phases = np.random.uniform(0, 2 * np.pi, size=N)
        spectrum = np.exp(1j * phases)
        c_complex = np.fft.ifft(spectrum)
        c_real = np.real(c_complex)
        c_bin = np.sign(c_real)
        c_bin[c_bin == 0] = 1
        initial_candidates.append(c_bin)
        
        # Cosine-tapered magnitude
        taper = np.cos(np.pi * np.arange(N) / (N - 1)) ** 2
        spectrum2 = taper * np.exp(1j * phases)
        c_complex2 = np.fft.ifft(spectrum2)
        c_real2 = np.real(c_complex2)
        c_bin2 = np.sign(c_real2)
        c_bin2[c_bin2 == 0] = 1
        initial_candidates.append(c_bin2)
        
        # Gaussian-tapered magnitude
        centers = np.random.uniform(0, N, size=2)
        widths = np.random.uniform(2, 6, size=2)
        taper3 = np.zeros(N)
        for c, w in zip(centers, widths):
            taper3 += np.exp(-((np.arange(N) - c) ** 2) / (2 * w ** 2))
        spectrum3 = taper3 * np.exp(1j * phases)
        c_complex3 = np.fft.ifft(spectrum3)
        c_real3 = np.real(c_complex3)
        c_bin3 = np.sign(c_real3)
        c_bin3[c_bin3 == 0] = 1
        initial_candidates.append(c_bin3)
    
    # 3. Random starts
    for _ in range(20):
        if time.time() > deadline - 30:
            break
        c_rand = np.random.choice([-1, 1], size=N).astype(float)
        initial_candidates.append(c_rand)
    
    # Run local search on all candidates
    best_overall_c = None
    best_overall_val = float('inf')
    
    for c0 in initial_candidates:
        if time.time() > deadline - 10:
            break
        c_best, v_best = local_search_vectorized(c0, time.time() + 2.0)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best
    
    # Simulated annealing with adaptive cooling and diverse moves
    T = 0.5
    cooling_rate = 0.995
    min_T = 1e-6
    
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    triples = [(i, j, k) for i in range(N-2) for j in range(i+1, N-1) for k in range(j+1, N)]
    
    # Track best solution seen
    sa_best_c = best_overall_c.copy()
    sa_best_val = best_overall_val
    
    while T > min_T and time.time() < deadline - 15:
        # Pick random move with adaptive probabilities
        r = np.random.rand()
        if r < 0.5:
            # 2-flip move
            i, j = pairs[np.random.randint(len(pairs))]
            best_overall_c[i] *= -1
            best_overall_c[j] *= -1
            new_score = get_supnorm_fast(best_overall_c)
            delta = new_score - best_overall_val
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                best_overall_val = new_score
                if new_score < sa_best_val:
                    sa_best_val = new_score
                    sa_best_c = best_overall_c.copy()
            else:
                best_overall_c[i] *= -1
                best_overall_c[j] *= -1
        elif r < 0.8:
            # 1-flip move
            i = np.random.randint(N)
            best_overall_c[i] *= -1
            new_score = get_supnorm_fast(best_overall_c)
            delta = new_score - best_overall_val
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                best_overall_val = new_score
                if new_score < sa_best_val:
                    sa_best_val = new_score
                    sa_best_c = best_overall_c.copy()
            else:
                best_overall_c[i] *= -1
        elif r < 0.95:
            # 3-flip move
            i, j, k = triples[np.random.randint(len(triples))]
            best_overall_c[i] *= -1
            best_overall_c[j] *= -1
            best_overall_c[k] *= -1
            new_score = get_supnorm_fast(best_overall_c)
            delta = new_score - best_overall_val
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                best_overall_val = new_score
                if new_score < sa_best_val:
                    sa_best_val = new_score
                    sa_best_c = best_overall_c.copy()
            else:
                best_overall_c[i] *= -1
                best_overall_c[j] *= -1
                best_overall_c[k] *= -1
        else:
            # 4-flip move (rare, for large jumps)
            idxs = np.random.choice(N, size=4, replace=False)
            for idx in idxs:
                best_overall_c[idx] *= -1
            new_score = get_supnorm_fast(best_overall_c)
            delta = new_score - best_overall_val
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                best_overall_val = new_score
                if new_score < sa_best_val:
                    sa_best_val = new_score
                    sa_best_c = best_overall_c.copy()
            else:
                for idx in idxs:
                    best_overall_c[idx] *= -1
        
        T *= cooling_rate
        
        # Periodic local search - more frequent
        if time.time() % 3 < 0.3:
            best_overall_c, best_overall_val = local_search_vectorized(
                best_overall_c, time.time() + 0.8)
            if best_overall_val < sa_best_val:
                sa_best_val = best_overall_val
                sa_best_c = best_overall_c.copy()
        
        # Occasionally restart from best seen
        if time.time() % 10 < 0.5:
            best_overall_c = sa_best_c.copy()
            best_overall_val = sa_best_val
    
    # Final local search
    best_overall_c, best_overall_val = local_search_vectorized(
        best_overall_c, deadline - 5)
    
    # Final accurate evaluation and potential refinement
    final_val = get_supnorm_accurate(best_overall_c)
    
    # Try a few more local searches from perturbed versions
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        perturbed = best_overall_c.copy()
        for _ in range(np.random.randint(2, 6)):
            idx = np.random.randint(N)
            perturbed[idx] *= -1
        c_best, v_best = local_search_vectorized(perturbed, time.time() + 2.0)
        if v_best < best_overall_val:
            best_overall_val = v_best
            best_overall_c = c_best
    
    return [int(c) for c in best_overall_c]

# EVOLVE-BLOCK-END
