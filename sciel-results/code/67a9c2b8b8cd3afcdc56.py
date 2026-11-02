"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_overlap_max(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    h_comp = 1.0 - h
    H = np.fft.fft(h)
    Hc = np.fft.fft(h_comp)
    corr = np.fft.ifft(H * np.conj(Hc)).real
    max_overlap = np.max(corr) * (2.0 / N)
    return max_overlap

def _overlap_vector(h):
    """Compute the overlap for all shifts."""
    h_comp = 1.0 - h
    H = np.fft.fft(h)
    Hc = np.fft.fft(h_comp)
    corr = np.fft.ifft(H * np.conj(Hc)).real
    return corr * (2.0 / N)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _compute_overlap_max(best_h)
    
    # Precompute indices for neighbors
    # We will use a greedy local search.
    # Objective: min_s I(s)
    # I(s) = (2/N) * sum_i h_i (1 - h_{i-s})
    # Let's work with the unnormalized correlation C(s) = sum_i h_i (1 - h_{i-s})
    # We want to minimize max_s C(s).
    
    # Initialize with a random pattern that sums to N/2
    # Using a "staircase" or random binary-like pattern might be better than uniform
    # Let's try a few random initializations
    
    np.random.seed(123)
    
    # Strategy: 
    # 1. Generate random h in [0,1]
    # 2. Normalize sum to N/2
    # 3. Local search: iterate over indices i, try changing h_i to h_i + delta or h_i - delta
    #    Or simply project h_i to 0 or 1? No, we need continuous [0,1].
    #    Actually, the problem is convex in h for a fixed s? No, max of linear functions is convex.
    #    So the objective f(h) = max_s (2/N) * sum_i h_i (1 - h_{i-s}) is convex in h?
    #    Wait. sum h_i (1-h_{i-s}) is quadratic (bilinear). It is not convex.
    #    So local search is appropriate.
    
    # Let's use a simple coordinate descent with step size adaptation.
    # Or even better: since we just need to return *a* sequence, and the evaluator measures the raw value,
    # and the seed is 1.0 (which corresponds to h=0.5 giving overlap 0.25? Let's check).
    
    # For h=0.5, 1-h=0.5.
    # C(s) = sum 0.5 * 0.5 = N * 0.25.
    # Overlap = (2/N) * N * 0.25 = 0.5.
    # Wait, the task says "score is the upper bound on C5".
    # And "search score is seed/value".
    # If seed program returns 1.0, and my current program scores 1.0514, that means my raw value is lower than the seed's raw value?
    # "1.0 equals the seed program and above 1.0 is an improvement"
    # So if score = 1.05, then 1.0 / value = 1.05 => value = 1/1.05 = 0.952.
    # This implies the seed's raw value is 1.0.
    # Let's check the seed's raw value.
    # If h=0.5, overlap = 0.5.
    # Maybe the "value" is not the overlap itself?
    # "score is the upper bound on C5"
    # In Erdos distinct distances or similar, C5 might be a specific constant.
    # But the prompt says "score is the upper bound on C5".
    # And "evaluator measures the raw value (lower is better)".
    # "search score is seed/value".
    # If seed returns 1.0 (search score), and seed value is V_seed, then 1.0 = 1.0 / V_seed => V_seed = 1.0.
    # So the seed program produces a raw value of 1.0.
    # My program produced a search score of 1.0514, so raw value = 1/1.0514 = 0.951.
    # So I am already beating the seed.
    # The goal is to beat 1.0514.
    
    # Let's implement a robust local search.
    
    def local_search(h, max_iters=5000):
        h = h.copy()
        # Normalize
        for _ in range(10):
            h += (N/2.0 - np.sum(h)) / N
            h = np.clip(h, 0.0, 1.0)
        
        current_overlap = _compute_overlap_max(h)
        overlaps = _overlap_vector(h)
        
        # Precompute the contribution of each h_i to each shift s?
        # C(s) = sum_i h_i (1 - h_{i-s})
        # Changing h_i affects C(s) for all s.
        # Specifically, term i in C(s) is h_i (1 - h_{i-s}).
        # Also term i+s in C(s) is h_{i+s} (1 - h_i).
        # This coupling makes coordinate descent tricky to update exactly without O(N) per step.
        # With N=100, O(N^2) per full pass is 10,000 operations.
        # 5000 iterations * 100 indices * 100 shifts = 50,000,000 ops.
        # This is very fast in Python/NumPy if vectorized, but loops are slow.
        # Let's try a simpler heuristic: random perturbations + acceptance if max overlap decreases.
        
        step_size = 0.1
        for it in range(max_iters):
            if time.time() > deadline:
                break
            i = np.random.randint(N)
            old_val = h[i]
            # Try a few deltas
            found_improvement = False
            for delta in [-step_size, -step_size/2, step_size/2, step_size]:
                new_val = old_val + delta
                if new_val < 0 or new_val > 1:
                    continue
                h[i] = new_val
                # Renormalize? No, that changes all other h_j.
                # The constraint is sum(h) = N/2.
                # If we change h_i, we must change another h_j to compensate?
                # Or just accept the violation temporarily and project?
                # Projecting is O(N).
                # Let's just project.
                h_proj = h.copy()
                h_proj += (N/2.0 - np.sum(h_proj)) / N
                h_proj = np.clip(h_proj, 0.0, 1.0)
                
                new_overlap = _compute_overlap_max(h_proj)
                if new_overlap < current_overlap:
                    h = h_proj
                    current_overlap = new_overlap
                    found_improvement = True
                    break
            if not found_improvement:
                # Reduce step size or restart?
                if step_size > 0.01:
                    step_size *= 0.9
                else:
                    # Random jump
                    h += np.random.uniform(-0.1, 0.1, N)
                    h += (N/2.0 - np.sum(h)) / N
                    h = np.clip(h, 0.0, 1.0)
                    current_overlap = _compute_overlap_max(h)
        
        return h, current_overlap

    # Run local search from multiple starts
    for restart in range(20):
        if time.time() > deadline:
            break
        
        if restart == 0:
            h_init = np.full(N, 0.5)
        elif restart == 1:
            # Binary-like
            h_init = np.random.choice([0, 1], N)
            h_init += (N/2.0 - np.sum(h_init)) / N
            h_init = np.clip(h_init, 0.0, 1.0)
        else:
            h_init = np.random.uniform(0, 1, N)
            h_init += (N/2.0 - np.sum(h_init)) / N
            h_init = np.clip(h_init, 0.0, 1.0)
        
        h_res, score_res = local_search(h_init, max_iters=2000)
        
        if score_res < best_score:
            best_score = score_res
            best_h = h_res

    return best_h
# EVOLVE-BLOCK-END
