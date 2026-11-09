import numpy as np
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 135

def construct():
    """
    Maximize ratio = ||f||_2^2 / (||f||_1 * sup|f*f|)
    Assuming evaluator normalizes ||f||_1 = 1, we maximize ||f||_2^2 / sup|f*f|.
    """
    
    def compute_ratio_normalized(h):
        # h is assumed to be normalized to sum=1
        # Compute L2 squared
        l2_sq = np.sum(h**2)
        # Compute autoconvolution
        # Using FFT for speed if N is large, but N=50 is small. np.correlate is fine.
        autoconv = np.correlate(h, h, mode='full')
        peak = np.max(autoconv)
        
        if peak < 1e-15:
            return 0.0
        return l2_sq / peak

    best_ratio = 0.0
    best_h = np.ones(N) / N  # Uniform start
    
    # Initialize with a few structural candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Single spike
    h_sp = np.zeros(N); h_sp[0] = 1.0
    candidates.append(h_sp)
    
    # 3. Two spikes
    h_2s = np.zeros(N); h_2s[0] = 1.0; h_2s[N//2] = 1.0
    candidates.append(h_2s)
    
    # 4. Random initializations
    for _ in range(20):
        candidates.append(np.random.rand(N))
        
    # 5. "Comb" structures
    for k in range(2, 10):
        h_comb = np.zeros(N)
        for i in range(k):
            h_comb[int(i * N / k)] = 1.0
        candidates.append(h_comb)

    # Evaluate candidates
    for h in candidates:
        h_norm = h / np.sum(h)
        r = compute_ratio_normalized(h_norm)
        if r > best_ratio:
            best_ratio = r
            best_h = h_norm.copy()

    # Local search refinement
    current_h = best_h.copy()
    best_r = best_ratio
    step_size = 0.1
    
    for iteration in range(2000):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        improved = False
        # Try perturbing a few random indices
        for _ in range(10):
            candidate_h = current_h.copy()
            n_perturb = np.random.randint(1, 4)
            for _ in range(n_perturb):
                pos = np.random.randint(0, N)
                delta = np.random.uniform(-step_size, step_size)
                candidate_h[pos] = max(0.0, candidate_h[pos] + delta)
            
            # Normalize
            sum_h = np.sum(candidate_h)
            if sum_h < 1e-10:
                continue
            candidate_h = candidate_h / sum_h
            
            r = compute_ratio_normalized(candidate_h)
            if r > best_r:
                best_r = r
                best_h = candidate_h.copy()
                current_h = candidate_h.copy()
                improved = True
        
        if not improved:
            step_size *= 0.95
            if step_size < 0.001:
                break

    # Return the best heights found (unnormalized, as evaluator normalizes)
    # To be safe, return them normalized to sum=1
    return list(best_h)
