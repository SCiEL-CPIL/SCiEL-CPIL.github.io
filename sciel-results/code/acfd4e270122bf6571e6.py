import numpy as np
import time
import random

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    # Autoconvolution via FFT for efficiency
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    conv = np.fft.ifft(H * H).real
    # Autoconvolution of length N with itself has length 2N-1
    # The FFT gives circular conv, but for zero-padded it's linear
    conv_peak = np.max(conv[:2*N-1])
    if conv_peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_peak)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # 1. Initialize with a greedy Sidon-like set construction
    # Goal: Select indices such that pairwise sums are as unique as possible.
    # This minimizes the peak of the autoconvolution (which counts sum collisions).
    
    def greedy_sidon(n, max_iter=1000):
        """
        Greedy selection of indices in Z_n to maximize sum uniqueness.
        Returns a binary vector.
        """
        used_sums = set()
        selected = []
        
        # To be efficient, we can iterate through a shuffled list of candidates
        # or just try to add indices 0..n-1 if they don't create 'bad' collisions.
        # A true Sidon set in Z_n has size approx sqrt(n). For N=50, size ~ 7.
        # But we want to maximize the ratio, which might benefit from slightly denser sets
        # if the heights are optimized. Let's start with a sparse core.
        
        candidates = list(range(n))
        random.shuffle(candidates)
        
        for idx in candidates:
            # Check if adding idx creates too many sum collisions
            # Sums involving new idx: idx + j for all j in selected
            # Also idx + idx = 2*idx
            # We want to avoid sums that are already in used_sums if possible,
            # or at least minimize the count.
            
            potential_conflicts = 0
            new_sums = []
            
            for j in selected:
                s = (idx + j) % n
                if s in used_sums:
                    potential_conflicts += 1
                new_sums.append(s)
            
            s_double = (2 * idx) % n
            if s_double in used_sums:
                potential_conflicts += 1
            new_sums.append(s_double)
            
            # Heuristic: Accept if conflicts are low
            # As the set grows, conflicts will inevitably increase.
            # Let's aim for a set of size ~8-10 first.
            if potential_conflicts <= 1: 
                selected.append(idx)
                for s in new_sums:
                    used_sums.add(s)
                
                if len(selected) >= 10:
                    break
            
        h = np.zeros(n)
        for idx in selected:
            h[idx] = 1.0
        return h

    # Generate multiple random Sidon-like structures
    best_h = np.ones(N) # Fallback
    best_score = compute_ratio(best_h)
    
    # 2. Structural Search: Try different sparsity levels and structures
    candidates = []
    
    # A. Greedy Sidon Sets (Sparse)
    for _ in range(50):
        if time.time() > deadline:
            break
        h_sidon = greedy_sidon(N)
        if np.sum(h_sidon) > 0:
            candidates.append(h_sidon)
            
    # B. Block Diagonal / Sparse Blocks
    # Sometimes 2 blocks of size 2-3 separated by distance work well
    for _ in range(50):
        if time.time() > deadline:
            break
        h = np.zeros(N)
        # Place 2 to 4 small blocks
        num_blocks = random.randint(2, 4)
        for _ in range(num_blocks):
            start = random.randint(0, N-1)
            size = random.randint(1, 3)
            for k in range(size):
                h[(start+k) % N] = 1.0
        if np.sum(h) > 0:
            candidates.append(h)

    # C. Random Sparse Vectors
    for _ in range(50):
        if time.time() > deadline:
            break
        h = np.zeros(N)
        num_ones = random.randint(5, 15)
        indices = np.random.choice(N, num_ones, replace=False)
        h[indices] = 1.0
        candidates.append(h)

    # Evaluate candidates
    for h in candidates:
        if time.time() > deadline:
            break
        s = compute_ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()

    # 3. Spectral Refinement
    # The ratio is R = sum(h^2) / (sum(h) * max(conv))
    # In frequency domain, conv peak is related to the inverse transform of |H|^2.
    # We can adjust the magnitudes of H (keeping phases from the discrete structure)
    # to flatten the inverse transform, thereby reducing the peak.
    
    if time.time() < deadline:
        # We will optimize the weights of the selected non-zero indices
        # Or, more powerfully, we can treat the problem as optimizing the power spectrum P_k = |H_k|^2
        # subject to P_k >= 0 and the time-domain signal being real and non-negative.
        # However, enforcing non-negativity in time domain from frequency is hard.
        
        # Alternative: Local optimization of the amplitudes of the sparse set found above.
        # Let h = w * mask, where mask is the binary support from step 2.
        # We optimize w (weights) to minimize peak convolution while keeping sum(h) fixed (scale invariance).
        
        # Actually, let's just do a gradient-free ascent on the full vector h, 
        # but initialized with the sparse structure.
        # The "Sidon" structure provides a good initialization for the peak minimization.
        
        current_h = best_h.copy()
        current_score = best_score
        
        # Adaptive step size local search
        step_size = 0.1
        for _ in range(500):
            if time.time() > deadline:
                break
            
            improved = False
            # Try perturbing each element
            for i in range(N):
                if time.time() > deadline:
                    break
                
                # Try increasing
                h_try = current_h.copy()
                h_try[i] += step_size
                if h_try[i] < 0: h_try[i] = 0
                
                s_try = compute_ratio(h_try)
                if s_try > current_score:
                    current_score = s_try
                    current_h = h_try
                    improved = True
                    break # Restart search from new best
                    
                # Try decreasing
                h_try = current_h.copy()
                h_try[i] -= step_size
                if h_try[i] < 0: h_try[i] = 0
                
                s_try = compute_ratio(h_try)
                if s_try > current_score:
                    current_score = s_try
                    current_h = h_try
                    improved = True
                    break
            
            if not improved:
                # No improvement with current step size, reduce it
                step_size *= 0.8
                if step_size < 1e-4:
                    # Try a random jump to escape local minima
                    h_jump = current_h.copy()
                    idx = np.random.randint(0, N)
                    h_jump[idx] += np.random.uniform(-0.5, 0.5)
                    if h_jump[idx] < 0: h_jump[idx] = 0
                    s_jump = compute_ratio(h_jump)
                    if s_jump > current_score:
                        current_score = s_jump
                        current_h = h_jump
                        step_size = 0.1
                    else:
                        # Accept with small probability (Simulated Annealing style)
                        # Or just break if stuck for a long time
                        pass
            
            if current_score > best_score:
                best_score = current_score
                best_h = current_h.copy()

    # 4. Final Polish: Normalize and check
    # The ratio is scale invariant, so normalization doesn't change score, 
    # but ensures we return a valid non-negative sequence.
    if np.sum(best_h) > 0:
        # Optional: normalize to sum=1 for cleanliness, though not required by score
        # best_h = best_h / np.sum(best_h)
        pass
        
    return best_h.tolist()
