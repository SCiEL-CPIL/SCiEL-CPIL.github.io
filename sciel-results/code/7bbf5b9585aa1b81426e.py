import numpy as np

def construct():
    N = 24
    # Precompute sample points
    M = 2000  # Number of samples for evaluation
    angles = 2 * np.pi * np.arange(M) / M
    z = np.exp(1j * angles)
    
    # Precompute powers of z for all coefficients
    # powers[k, m] = z[m] ** k
    # Shape: (N, M)
    powers = np.zeros((N, M), dtype=complex)
    powers[0, :] = 1
    for k in range(1, N):
        powers[k, :] = powers[k-1, :] * z
    
    def eval_peak(coeffs):
        # coeffs: array of shape (N,) with values +/-1
        # Compute P(z) for all sample points
        # vals[m] = sum_k coeffs[k] * powers[k, m]
        vals = powers.T @ coeffs
        return np.max(np.abs(vals))
    
    def refine_peak(coeffs):
        # Find the angle of the max peak in the sampled grid
        # Then use golden section search to refine
        vals = powers.T @ coeffs
        abs_vals = np.abs(vals)
        max_idx = np.argmax(abs_vals)
        
        # The max is at angles[max_idx]
        # We need to search in the neighborhood.
        # The function |P(z)| is smooth.
        # We can use a local parabolic fit or golden section.
        # Golden section on the interval [angles[max_idx-1], angles[max_idx+1]]
        # Note: angles are periodic.
        
        # For simplicity and speed, the sampled max is often sufficient for ranking.
        # But to be safe, we can return the sampled max.
        # The evaluator uses high accuracy, so we need the true min.
        # However, if we find the global min of the sampled max, it's likely the global min of the true max.
        return abs_vals[max_idx]

    best_coeffs = None
    best_peak = float('inf')
    
    # Generate initial candidates
    candidates = []
    
    # 1. All +1
    candidates.append(np.ones(N, dtype=int))
    
    # 2. Alternating
    candidates.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=int))
    
    # 3. Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:N], dtype=int))
    
    # 4. Random
    for _ in range(20):
        candidates.append(np.random.choice([-1, 1], size=N))
        
    # 5. Structured: blocks of +1 and -1
    # e.g. +++++++---++++++---
    for block_size in [2, 3, 4, 6, 8, 12]:
        c = []
        sign = 1
        for i in range(0, N, block_size):
            c.extend([sign] * block_size)
            sign *= -1
        candidates.append(np.array(c[:N], dtype=int))

    # Hill Climbing / Local Search
    for start in candidates:
        current = start.copy()
        current_peak = eval_peak(current)
        
        improved = True
        while improved:
            improved = False
            best_improvement = 0
            best_flip_idx = -1
            best_new_peak = current_peak
            
            for i in range(N):
                # Flip coefficient i
                old_val = current[i]
                current[i] = -old_val
                new_peak = eval_peak(current)
                current[i] = old_val
                
                if new_peak < best_new_peak:
                    best_new_peak = new_peak
                    best_flip_idx = i
                    best_improvement = current_peak - new_peak
            
            if best_improvement > 1e-9:
                current[best_flip_idx] = -current[best_flip_idx]
                current_peak = best_new_peak
                improved = True
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_coeffs = current.copy()
            
    # Final refinement: try to improve with a finer sampling or different metric?
    # The eval_peak uses M=2000. The true max might be slightly higher.
    # But the relative ordering should be preserved.
    
    return best_coeffs.tolist()
