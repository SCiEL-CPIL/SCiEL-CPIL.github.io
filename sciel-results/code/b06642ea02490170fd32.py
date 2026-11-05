import numpy as np
import time
import random

def construct():
    n = 129 # 0 to 128
    target_min = 128
    target_max = 256
    targets = np.arange(target_min, target_max + 1)
    
    # Helper to compute representation counts for a given set A (as boolean array)
    def get_counts(x):
        conv = np.convolve(x, x, mode='full')
        # Unordered counts
        # R[k] = conv[k] / 2 if k odd
        # R[k] = (conv[k] + x[k//2]) / 2 if k even
        R = np.zeros(target_max + 1)
        
        # Vectorized calculation for target range
        conv_vals = conv[targets]
        
        even_mask = (targets % 2 == 0)
        odd_mask = ~even_mask
        
        R[targets[odd_mask]] = conv_vals[odd_mask] / 2.0
        
        mid_indices = targets[even_mask] // 2
        mid_vals = x[mid_indices]
        R[targets[even_mask]] = (conv_vals[even_mask] + mid_vals) / 2.0
        
        return R

    # Helper to check validity (all counts >= 1)
    def is_valid(R):
        return np.all(R >= 1.0)

    # Greedy Construction
    x = np.zeros(n, dtype=np.int8)
    # Start with a few random elements to break symmetry? Or just empty.
    # Let's start empty.
    
    # Precompute contribution of each element i to each target t
    # contrib[i, t] = 1 if i + j = t for some j in A? No, greedy adds one element at a time.
    # When we add i, it forms pairs (i, j) for all j in A.
    # So the new count for target t increases by number of j in A such that t-j = i => j = t-i.
    # Basically, adding i increases count of t by x[t-i] (if 0<=t-i<n).
    
    current_counts = np.zeros(target_max + 1)
    
    for _ in range(n): # At most n elements
        best_gain = -1
        best_idx = -1
        
        # Try adding each unadded element
        for i in range(n):
            if x[i] == 1:
                continue
            
            # Calculate gain: how many new targets are covered or how much deficit is reduced?
            # To ensure coverage, we prioritize filling gaps.
            # Gain = sum of (1 / (current_count[t] + 1)) for t in targets where t-i is in A?
            # Simpler: Gain = number of targets t in [128, 256] such that current_count[t] == 0 
            #         and (t-i) is in A.
            # Actually, if we just want to minimize max, we should balance.
            # But first, we need validity.
            # Let's use a score: 
            # 1. If adding i covers a currently 0-count target, big bonus.
            # 2. Else, small bonus for reducing max?
            
            # Efficient check:
            # For each target t, if x[t-i] == 1, then count[t] increases.
            # We want to maximize the number of zero-count targets that become non-zero.
            
            zero_targets_mask = (current_counts == 0)
            # For a candidate i, the targets it affects are t such that t-i is in [0, n-1] and x[t-i]==1.
            # Let J be the set of indices in A.
            # New pairs are (i, j) for j in J. Sums are i+j.
            # So affected targets are i + J.
            
            # This loop is O(n^2) per step, total O(n^3) = 129^3 ~ 2M ops. Very fast.
            
            # Optimization: Precompute which targets are currently zero.
            zero_indices = np.where(zero_targets_mask)[0]
            
            if len(zero_indices) == 0:
                # All covered. Now optimize for max count?
                # In greedy phase, just pick random or one that minimizes max increase.
                pass
            
            # For each candidate i, count how many zero targets it hits.
            # i + j = t => j = t - i.
            # j must be in A (x[j]==1).
            
            best_score = -1
            for i in range(n):
                if x[i] == 1:
                    continue
                
                # Find j in A such that t = i+j is a zero target.
                # j = t - i.
                # t is in [128, 256].
                # j is in [0, 128].
                # So t-i must be in [0, 128].
                
                # Vectorized check for this i:
                # valid_js = x (boolean array)
                # We want to check if any (t - i) where t in zero_indices is in A.
                
                # Let's just iterate zero_indices? No, too slow if many zeros.
                # Instead:
                # For a fixed i, the targets it can cover are t = i + j for j in A.
                # We want to count how many of these t are in zero_indices.
                
                # Get all j in A:
                # Since we are in a loop, we can maintain a list of active indices.
                
                # Let's maintain a list `active_indices`.
                # For candidate i:
                # potential_targets = i + active_indices
                # valid_potential_targets = potential_targets[(potential_targets >= target_min) & (potential_targets <= target_max)]
                # hits = np.sum(zero_targets_mask[valid_potential_targets])
                
                # This is efficient enough.
                
                pass # Placeholder, will implement properly below

        # Re-implementing the inner loop properly for clarity and speed
        active_indices = np.nonzero(x)[0]
        
        best_i = -1
        best_hits = -1
        best_max_increase = np.inf
        
        if len(active_indices) == 0:
             # First element, any is fine. Pick middle?
             best_i = 64
        else:
            for i in range(n):
                if x[i] == 1:
                    continue
                
                # Potential sums: i + active_indices
                sums = i + active_indices
                # Filter to target range
                mask = (sums >= target_min) & (sums <= target_max)
                valid_sums = sums[mask]
                
                if len(valid_sums) == 0:
                    continue
                
                # Count how many of these sums are currently 0
                hits = np.sum(current_counts[valid_sums] == 0)
                
                # Tie-breaker: minimize the max count increase?
                # The increase for a target t is 1 if t-i is in A.
                # Since we only add 1 element, each affected target count increases by 1.
                # So max increase is 1 for any hit.
                # If hits are equal, we might prefer i that doesn't create high counts?
                # For now, just maximize hits.
                
                if hits > best_hits:
                    best_hits = hits
                    best_i = i
                elif hits == best_hits and best_hits > 0:
                    # Tie break: choose i that results in lower peak?
                    # Hard to compute without simulating. Just pick first.
                    pass

        if best_i != -1:
            x[best_i] = 1
            # Update current_counts
            active_indices = np.nonzero(x)[0]
            # Recalculate counts? Or update incrementally.
            # Incremental: for each j in active_indices (excluding best_i), sum = best_i + j.
            # If sum in range, current_counts[sum] += 1.
            # Note: active_indices now includes best_i.
            # Pairs are unordered. (i, j) and (j, i) are same.
            # When adding i, we form pairs (i, j) for all j in A_old.
            # So we iterate j in A_old.
            old_active = active_indices[active_indices != best_i]
            sums = best_i + old_active
            mask = (sums >= target_min) & (sums <= target_max)
            current_counts[sums[mask]] += 1
            
            # Also, if best_i + best_i is in range?
            # No, we only add pairs (i, j) where j is in A_OLD.
            # The pair (i, i) is formed only if i was already in A? No, i is new.
            # Wait. The set is A. We add i to A.
            # New pairs are (i, j) for j in A_OLD.
            # Does (i, i) count?
            # The problem says "sum of two elements of A".
            # If A={1}, sum is 2. Pair (1,1).
            # If we add 1 to empty set, A={1}. Pair (1,1) exists.
            # My incremental update `sums = best_i + old_active` misses (best_i, best_i).
            # If best_i + best_i is in [128, 256], we must increment count[2*best_i] by 1.
            
            double_sum = 2 * best_i
            if target_min <= double_sum <= target_max:
                current_counts[double_sum] += 1

    # Check validity
    R = get_counts(x)
    if not is_valid(R):
        # Fallback: if greedy failed (unlikely), add random elements until valid
        while not is_valid(R):
            idx = random.randint(0, n-1)
            if x[idx] == 0:
                x[idx] = 1
                R = get_counts(x)
    
    # Local Search to minimize max count
    # Cost function: Max(R) * 1000 + Sum(R^2)
    # We want to minimize the peak.
    
    def get_cost(R):
        max_r = np.max(R)
        sq_sum = np.sum(R**2)
        return max_r * 1000.0 + sq_sum

    current_R = get_counts(x)
    current_cost = get_cost(current_R)
    
    start_time = time.time()
    max_time = 55.0 # Leave 5s buffer
    
    # Local search: flip bits
    # Try to remove elements that don't break validity and reduce max.
    # Try to add elements that reduce max? (Dilute the peaks)
    
    # Strategy: 
    # 1. Try removing each element. If valid, accept if cost decreases.
    # 2. Try adding each element. If cost decreases, accept.
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        n_targets = len(current_R)
        
        # Phase 1: Removals
        for i in range(n):
            if x[i] == 1:
                x[i] = 0
                new_R = get_counts(x)
                if is_valid(new_R):
                    new_cost = get_cost(new_R)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        current_R = new_R
                        improved = True
                    else:
                        x[i] = 1
                else:
                    x[i] = 1
        
        # Phase 2: Additions
        if (time.time() - start_time) > max_time: break
        
        for i in range(n):
            if x[i] == 0:
                x[i] = 1
                new_R = get_counts(x)
                new_cost = get_cost(new_R)
                if new_cost < current_cost:
                    current_cost = new_cost
                    current_R = new_R
                    improved = True
                else:
                    x[i] = 0
    
    # Final check
    final_R = get_counts(x)
    if not is_valid(final_R):
        # Emergency repair
        while not is_valid(final_R):
            # Find a target with 0 count
            missing = np.where(final_R == 0)[0]
            if len(missing) == 0: break
            t = missing[0]
            # Find a pair (a, b) in [0, 128] summing to t that is not in A
            # Just pick random a, b
            for _ in range(100):
                a = random.randint(0, 128)
                b = t - a
                if 0 <= b <= 128:
                    if x[a] == 0 or x[b] == 0:
                        x[a] = 1
                        x[b] = 1
                        final_R = get_counts(x)
                        if is_valid(final_R):
                            break
                    else:
                        # Both already in A? Then count should be > 0. 
                        # This shouldn't happen if logic is correct.
                        pass

    result = np.nonzero(x)[0].tolist()
    if not result:
        return [64]
    return result
