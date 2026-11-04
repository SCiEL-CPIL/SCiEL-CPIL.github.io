import numpy as np

N = 400

def construct():
    # Target size for the support.
    # Theoretical max for Sidon set in Z_N is approx sqrt(N).
    M = 20
    
    # 1. Initialize with a random permutation of indices
    # We need a set of M distinct indices from [0, N)
    indices = np.random.permutation(N)
    support = set(indices[:M].tolist())
    
    # Helper function to calculate the peak autoconvolution (off-peak)
    # For a sparse vector, C_k = sum_{i in S} x_i x_{i+k}
    # With x_i = 1/sqrt(M), C_k = (1/M) * N_k
    # We want to minimize max |N_k| for k != 0.
    # Since N_k is integer, minimizing max N_k is equivalent.
    def get_peak_cost(support_set):
        diffs = {}
        support_list = list(support_set)
        for i in range(len(support_list)):
            for j in range(len(support_list)):
                if i == j: continue
                d = support_list[i] - support_list[j]
                # We only care about non-zero differences for off-peak
                # But for cost, we can just count occurrences of each diff
                if d in diffs:
                    diffs[d] += 1
                else:
                    diffs[d] = 1
        if not diffs:
            return 0
        # Max count of any difference
        # Note: In a Sidon set, all non-zero diffs are unique, so max count is 1.
        # If there are collisions, max count > 1.
        return max(diffs.values())
    
    current_cost = get_peak_cost(support)
    
    # 2. Local Search Optimization
    # Try to reduce the peak cost by swapping elements
    # We want to reach cost 1 (perfect Sidon) or as low as possible.
    
    # If we already have a perfect Sidon set, we can stop?
    # But maybe a different set has better properties for the "signed" aspect?
    # The problem says "signed step function".
    # If the metric is just peak magnitude, signs don't matter for Sidon sets.
    # If the metric is sum of magnitudes, signs don't matter.
    # I will assume peak magnitude is the goal.
    
    if current_cost == 1:
        pass # Already optimal for this M
    
    else:
        # Optimize support
        for _ in range(1000): # 1000 iterations of local search
            # Pick a random element from support to remove
            to_remove = np.random.choice(list(support))
            support.remove(to_remove)
            
            best_new_elem = None
            best_cost = current_cost
            
            # Try adding a random element from outside
            # To speed up, we can just try a few random candidates
            candidates = np.random.choice([i for i in range(N) if i not in support], size=min(50, N-M), replace=False)
            
            for c in candidates:
                support.add(c)
                new_cost = get_peak_cost(support)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_new_elem = c
                support.remove(c)
            
            if best_new_elem is not None:
                support.add(best_new_elem)
                current_cost = best_cost
                if current_cost == 1:
                    break
            else:
                # Revert
                support.add(to_remove)
    
    # 3. Construct the vector
    support_list = list(support)
    x = np.zeros(N)
    # Normalize to L2 = 1
    x[support_list] = 1.0 / np.sqrt(M)
    
    # 4. Assign signs
    # For a Sidon set, the autoconvolution values are +/- 1/M.
    # The signs of x determine the signs of C_k.
    # If the metric is max |C_k|, signs don't matter.
    # If the metric is something else, we might need to optimize signs.
    # Given the ambiguity, and that "signed" is in the prompt,
    # let's try to optimize signs if we have time/compute, 
    # but for now, all positive is a valid baseline.
    # However, if the "upper bound on C3" involves a sum of terms, 
    # random signs might reduce the sum if there's cancellation?
    # No, |sum| is minimized by random signs? No, |sum| is maximized by aligned signs.
    # If we want to minimize the upper bound, and the bound is on the magnitude,
    # random signs don't change the magnitude of individual terms.
    
    # Let's stick to all positive for now, as it's the standard for Sidon sets.
    
    return x
