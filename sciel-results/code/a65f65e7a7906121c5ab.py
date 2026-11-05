import time
from math import lcm
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe against 90s limit
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Precompute integer weights w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    # We want to find c_i in {-1, 0, 1} such that S = sum(c_i * w_i) is nonzero and |S| is minimized.
    # The actual value is S / L.
    
    # Split into two halves for Meet-in-the-Middle
    half_size = n // 2
    w1 = weights[:half_size]
    w2 = weights[half_size:]
    
    def generate_sums_and_coeffs(w_list):
        """Generate all possible (sum, coeffs) pairs iteratively."""
        sums_coeffs = [(0, [0] * len(w_list))]
        for i, w in enumerate(w_list):
            new_sums_coeffs = []
            for s, coeffs in sums_coeffs:
                # Try -1
                new_coeffs = coeffs[:]
                new_coeffs[i] = -1
                new_sums_coeffs.append((s - w, new_coeffs))
                # Try 0
                new_sums_coeffs.append((s, coeffs))
                # Try +1
                new_coeffs = coeffs[:]
                new_coeffs[i] = 1
                new_sums_coeffs.append((s + w, new_coeffs))
            sums_coeffs = new_sums_coeffs
        return sums_coeffs
    
    def generate_sums_only(w_list):
        """Generate all possible sums without storing coefficients."""
        sums = [0]
        for w in w_list:
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums
    
    def find_coeffs_iterative(weights, target_sum):
        """Iteratively find coefficients that sum to target_sum using DP."""
        n_w = len(weights)
        if n_w == 0:
            return [] if target_sum == 0 else None
        
        # Use memoization with a set of achievable (idx, sum) states
        # But that's too memory heavy for 15 elements.
        # Instead, use a different approach: work backwards.
        
        # Actually, let's use a smarter DFS with better pruning.
        # Sort weights by magnitude descending for better pruning.
        indexed_weights = [(w, i) for i, w in enumerate(weights)]
        indexed_weights.sort(key=lambda x: -abs(x[0]))
        
        suffix_max = [0] * (n_w + 1)
        for i in range(n_w - 1, -1, -1):
            suffix_max[i] = suffix_max[i+1] + abs(indexed_weights[i][0])
        
        result = [0] * n_w
        
        def dfs(idx, current_sum):
            if idx == n_w:
                return current_sum == target_sum
            
            w, orig_idx = indexed_weights[idx]
            
            # Pruning
            if current_sum + suffix_max[idx] < target_sum:
                return False
            if current_sum - suffix_max[idx] > target_sum:
                return False
            
            # Try 0, +1, -1 in order of closeness to target
            options = [0, 1, -1]
            # Sort by distance to target
            options.sort(key=lambda c: abs(target_sum - (current_sum + c * w)))
            
            for c in options:
                result[orig_idx] = c
                if dfs(idx + 1, current_sum + c * w):
                    return True
            
            return False
        
        if dfs(0, 0):
            return result
        return None
    
    # Generate sums for both halves (just sums, not coeffs, for memory efficiency)
    sums1 = generate_sums_only(w1)
    sums2 = generate_sums_only(w2)
    
    # Sort sums2 for binary search
    sums2.sort()
    
    best_abs_int_diff = float('inf')
    best_s1 = 0
    best_s2 = 0
    
    # Iterate through sums1
    # For each s1, we want s2 such that |s1 + s2| is minimized and non-zero.
    # Binary search in sums2 for -s1.
    
    len2 = len(sums2)
    import bisect
    
    for s1 in sums1:
        if time.time() - start_time > deadline - 10:
            break
            
        target = -s1
        # Binary search for target in sums2
        idx = bisect.bisect_left(sums2, target)
        
        # Check idx and idx-1
        candidates = []
        if idx < len2:
            candidates.append(sums2[idx])
        if idx > 0:
            candidates.append(sums2[idx-1])
            
        for s2 in candidates:
            diff = s1 + s2
            if diff != 0:
                abs_diff = abs(diff)
                if abs_diff < best_abs_int_diff:
                    best_abs_int_diff = abs_diff
                    best_s1 = s1
                    best_s2 = s2
                    if best_abs_int_diff == 1:
                        break
        if best_abs_int_diff == 1:
            break
            
    if best_abs_int_diff == float('inf'):
        # Fallback
        return [0] * 28 + [1, -1]
        
    # Reconstruct coefficients
    c1 = find_coeffs_iterative(w1, best_s1)
    c2 = find_coeffs_iterative(w2, best_s2)
    
    if c1 is None or c2 is None:
        return [0] * 28 + [1, -1]
        
    result = c1 + c2
    
    # Verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        return [0] * 28 + [1, -1]
        
    return result

# EVOLVE-BLOCK-END
