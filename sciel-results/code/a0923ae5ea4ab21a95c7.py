import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    # Target: every t in [128, 256] must be expressible as a+b with a<=b, a,b in A
    # Minimize max multiplicity over t in [128, 256]
    
    # Precompute FFT size for fast convolution
    fft_size = 512  # power of 2 >= 257
    
    def fast_convolve(v):
        """Compute convolution v*v using FFT."""
        V = np.fft.rfft(v, n=fft_size)
        conv = np.fft.irfft(V * V, n=fft_size)
        return np.round(conv).astype(np.int32)
    
    def compute_unordered_fast(conv, A_vec):
        """Compute unordered representation counts from convolution and vector."""
        # conv has length fft_size. We need indices 128 to 256.
        # ordered[t] = conv[t]
        # unordered[t] = (ordered[t] + (1 if t even and t/2 in A else 0)) // 2
        
        # Extract ordered counts for target range [128, 256]
        ordered = conv[128:257].astype(np.int32)
        
        # Compute diagonal contribution: 1 if t is even and t/2 is in A
        # t ranges from 128 to 256.
        # t is even if t % 2 == 0.
        # If t is even, mid = t // 2. We need to check if mid is in A.
        # mid ranges from 64 to 128.
        
        # Create a boolean mask for even t in the target range
        # t_values = np.arange(128, 257)
        # is_even = t_values % 2 == 0
        # mids = t_values // 2
        # diag_mask = is_even & A_vec[mids]  # A_vec is 0/1 array of length 129
        
        # Vectorized version:
        t_vals = np.arange(128, 257)
        is_even = t_vals % 2 == 0
        mids = t_vals // 2
        # A_vec is length 129 (indices 0..128). mids are 64..128.
        diag_contrib = (is_even & (A_vec[mids] > 0)).astype(np.int32)
        
        unordered = (ordered + diag_contrib) // 2
        return unordered
    
    def evaluate(A_set):
        """Return (max_mult, covered, target_counts) for the given set A."""
        v = np.zeros(129, dtype=np.float64)
        for x in A_set:
            v[x] = 1.0
        conv = fast_convolve(v)
        target = compute_unordered_fast(conv, v)
        max_mult = int(target.max()) if len(target) > 0 else 0
        covered = bool(target.min() >= 1) if len(target) > 0 else False
        return max_mult, covered, target
    
    def evaluate_vector(A_vec):
        """Fast evaluation using pre-allocated vector. Returns (max_mult, covered, target)."""
        conv = fast_convolve(A_vec)
        target = compute_unordered_fast(conv, A_vec)
        max_mult = int(target.max())
        covered = bool(target.min() >= 1)
        return max_mult, covered, target
    
    def build_initial_set():
        """Build a valid covering set using greedy approach."""
        A = set()
        A.add(0)
        A.add(128)
        
        # Greedy: for uncovered t, add most useful element
        for t in range(128, 257):
            covered_t = False
            for a in A:
                b = t - a
                if b in A and b >= a:
                    covered_t = True
                    break
            if not covered_t:
                # Try all possible a values
                candidates = []
                for a in range(max(0, t - 128), min(128, t // 2) + 1):
                    if a not in A:
                        candidates.append(a)
                if candidates:
                    best_a = None
                    best_count = -1
                    for a in candidates:
                        count = 0
                        for t2 in range(128, 257):
                            b2 = t2 - a
                            if b2 in A and b2 >= a:
                                count += 1
                        if count > best_count:
                            best_count = count
                            best_a = a
                    if best_a is not None:
                        A.add(best_a)
        
        return A
    
    A0 = build_initial_set()
    score0, covered0, target0 = evaluate(A0)
    
    if not covered0:
        A0 = set(range(0, 129, 2))
        score0, covered0, target0 = evaluate(A0)
        if not covered0:
            A0 = set(range(0, 129))
            score0, covered0, target0 = evaluate(A0)
    
    best_A = set(A0)
    best_score = score0
    best_covered = covered0
    
    # Convert to vector for fast SA
    current_vec = np.zeros(129, dtype=np.float64)
    for x in A0:
        current_vec[x] = 1.0
    
    current_score, current_covered, current_target = evaluate_vector(current_vec)
    
    T = 10.0
    T_min = 0.01
    cooling = 0.9995
    
    # Precompute indices for potential adds/removes to speed up random selection
    all_indices = np.arange(129)
    
    while time.time() < deadline - 5 and T > T_min:
        r = random.random()
        new_vec = current_vec.copy()
        
        if r < 0.4 and current_vec.sum() > 1:
            # Remove
            idxs = np.nonzero(current_vec)[0]
            if len(idxs) == 0: continue
            x = idxs[random.randint(0, len(idxs)-1)]
            new_vec[x] = 0.0
        elif r < 0.8:
            # Add
            idxs = np.nonzero(1.0 - current_vec)[0]
            if len(idxs) == 0: continue
            x = idxs[random.randint(0, len(idxs)-1)]
            new_vec[x] = 1.0
        else:
            # Swap
            idxs_in = np.nonzero(current_vec)[0]
            idxs_out = np.nonzero(1.0 - current_vec)[0]
            if len(idxs_in) == 0 or len(idxs_out) == 0: continue
            x_in = idxs_in[random.randint(0, len(idxs_in)-1)]
            x_out = idxs_out[random.randint(0, len(idxs_out)-1)]
            new_vec[x_in] = 0.0
            new_vec[x_out] = 1.0
        
        new_score, new_covered, new_target = evaluate_vector(new_vec)
        
        new_cost = new_score if new_covered else new_score + 10000
        current_cost = current_score if current_covered else current_score + 10000
        
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_vec = new_vec
            current_score = new_score
            current_covered = new_covered
            current_target = new_target
            
            if current_covered and (not best_covered or current_score < best_score):
                best_score = current_score
                best_covered = True
                # Update best_A from vector
                best_A = set(np.nonzero(current_vec)[0].tolist())
        
        T *= cooling
    
    # Final polish: try removing elements one by one
    if best_covered and time.time() < deadline - 1:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for x in sorted(best_A):
                if time.time() > deadline - 1:
                    break
                new_A = best_A - {x}
                s, c, t = evaluate(new_A)
                if c and s <= best_score:
                    best_A = new_A
                    best_score = s
                    improved = True
    
    # Ensure validity
    if not best_covered:
        best_A = set(range(0, 129))
    
    return sorted(best_A)
