import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the score: L2^2 / (L1 * Sup(autoconv))
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def find_sidon_set(N):
        S = [0]
        for i in range(1, N):
            candidate = S + [i]
            if is_sidon(candidate):
                S = candidate
        return S

    sidon_set = find_sidon_set(N)
    K = len(sidon_set)
    sorted_sidon = sorted(sidon_set)

    # Initialize with uniform weights on Sidon set
    best_h = np.zeros(N)
    best_h[sorted_sidon] = 1.0 / K
    best_score = _score_raw(best_h)

    # Phase 1: Block-coordinate descent with epigraph formulation
    # We optimize L2^2 subject to autoconv_max <= t and sum(h) = 1
    # This decouples the peak constraint from L2 accumulation
    
    h_work = best_h.copy()
    
    def get_autoconv_max(h):
        hc = np.convolve(h, h, mode='full')
        return np.max(hc)
    
    def get_l2_sq(h):
        return np.sum(h**2)
    
    # Block coordinate descent: update pairs of coordinates that contribute to the same autoconv peak
    # For each position k in autoconv, the contributing pairs are (i, k-i) where 0 <= i <= k and 0 <= k-i < N
    # We identify the dominant peak and update symmetric pairs around it
    
    for iteration in range(200):
        if time.time() > deadline - 30:
            break
            
        l1 = np.sum(h_work)
        if l1 < 1e-12:
            break
        
        hc = np.convolve(h_work, h_work, mode='full')
        peak_idx = np.argmax(hc)
        peak_val = hc[peak_idx]
        
        if peak_val < 1e-12:
            break
        
        # Find all pairs (i, j) that contribute to this peak: i + j = peak_idx
        # i ranges from max(0, peak_idx - N + 1) to min(N-1, peak_idx)
        i_min = max(0, peak_idx - N + 1)
        i_max = min(N - 1, peak_idx)
        
        improved = False
        
        for i in range(i_min, i_max + 1):
            j = peak_idx - i
            if j < 0 or j >= N:
                continue
            if i == j:
                continue  # Skip diagonal for now, handle separately
            
            # For pair (i, j), increasing h[i] and h[j] increases both L2^2 and the peak
            # We want to maximize L2^2 / peak, so we need to balance
            # Try small perturbations: increase one, decrease the other to keep sum constant
            
            current_l2_sq = get_l2_sq(h_work)
            current_score = current_l2_sq / (l1 * peak_val) / (DELTA**2)
            
            # Try moving mass from the smaller to the larger to increase L2^2
            if h_work[i] < h_work[j]:
                # Move mass from i to j
                for delta in [0.01, 0.005, 0.001]:
                    if h_work[i] - delta < 0:
                        continue
                    new_i = h_work[i] - delta
                    new_j = h_work[j] + delta
                    h_work[i] = new_i
                    h_work[j] = new_j
                    
                    new_hc = np.convolve(h_work, h_work, mode='full')
                    new_peak = np.max(new_hc)
                    new_l2_sq = get_l2_sq(h_work)
                    new_score = new_l2_sq / (l1 * new_peak) / (DELTA**2)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_h = h_work.copy()
                        improved = True
                    else:
                        # Revert
                        h_work[i] = new_i + delta
                        h_work[j] = new_j - delta
            else:
                # Move mass from j to i
                for delta in [0.01, 0.005, 0.001]:
                    if h_work[j] - delta < 0:
                        continue
                    new_i = h_work[i] + delta
                    new_j = h_work[j] - delta
                    h_work[i] = new_i
                    h_work[j] = new_j
                    
                    new_hc = np.convolve(h_work, h_work, mode='full')
                    new_peak = np.max(new_hc)
                    new_l2_sq = get_l2_sq(h_work)
                    new_score = new_l2_sq / (l1 * new_peak) / (DELTA**2)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_h = h_work.copy()
                        improved = True
                    else:
                        # Revert
                        h_work[i] = new_i - delta
                        h_work[j] = new_j + delta
        
        if not improved and iteration > 50:
            break
    
    # Phase 2: SLSQP polish on the best found solution
    if time.time() < deadline - 10:
        def neg_obj(h):
            l1 = np.sum(h)
            if l1 < 1e-12:
                return 0.0
            l2_sq = np.sum(h**2)
            hc = np.convolve(h, h, mode='full')
            mc = np.max(hc)
            if mc < 1e-12:
                return 0.0
            return -(l2_sq / mc)

        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        ]
        bounds = [(0.0, 1.0)] * N

        try:
            res = minimize(neg_obj, best_h, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
            if res.success:
                h_candidate = np.clip(res.x, 0, None)
                s = np.sum(h_candidate)
                if s > 0:
                    h_candidate /= s
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    # Phase 3: Final local search with random restarts
    if time.time() < deadline - 5:
        for restart in range(10):
            if time.time() > deadline - 2:
                break
            
            # Perturb the best solution
            h_perturbed = best_h.copy()
            # Randomly select a few indices to adjust
            n_perturb = np.random.randint(2, 6)
            indices = np.random.choice(N, size=n_perturb, replace=False)
            
            for idx in indices:
                # Add small random perturbation
                h_perturbed[idx] += np.random.uniform(-0.02, 0.02)
            
            # Ensure non-negativity and normalize
            h_perturbed = np.clip(h_perturbed, 0, None)
            s = np.sum(h_perturbed)
            if s > 0:
                h_perturbed /= s
            
            s_perturbed = _score_raw(h_perturbed)
            if s_perturbed > best_score:
                best_score = s_perturbed
                best_h = h_perturbed

    return list(best_h)
