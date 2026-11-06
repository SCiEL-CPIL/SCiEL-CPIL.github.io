"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    if not hasattr(construct, '_cached_answers'):
        start_time = time.time()
        deadline = start_time + 85.0  # Leave 5s margin for 90s limit

        # 1. Generate smooth numbers up to 128
        smooth = []
        for x in range(1, 129):
            y = x
            for p in (2, 3, 5):
                while y % p == 0:
                    y //= p
            if y == 1:
                smooth.append(x)
        
        n_smooth = len(smooth)
        smooth_set = set(smooth)

        # 2. Precompute incompatibility bitmask
        # incompat[i] has bit j set if smooth[i] and smooth[j] are incompatible (one divides the other)
        incompat = [0] * n_smooth
        for i in range(n_smooth):
            for j in range(n_smooth):
                if i != j:
                    a = smooth[i]
                    b = smooth[j]
                    if a % b == 0 or b % a == 0:
                        incompat[i] |= (1 << j)

        # Precompute suffix sums for min/max pruning?
        # min_suffix_sum[i][k] = sum of k smallest numbers from i to end
        # max_suffix_sum[i][k] = sum of k largest numbers from i to end
        # This might be memory heavy if k is large, but k is small (<= 30).
        # Actually, we can just calculate on the fly or use simple bounds.
        # Let's use a helper to check if 'remaining' is possible with 'k' items from 'start_idx'
        
        # To optimize DFS, we can precompute the list of valid next steps? No, incompatibility depends on chosen set.
        # We will use a standard DFS with bitmask for incompatibility.

        def find_min_antichain(n):
            # Try k=1
            if n in smooth_set:
                return [n]
            
            # Try k=2, 3, ...
            # We need to search for a set of size k summing to n.
            # We use DFS with index tracking.
            
            # Optimization: If n is small, k is small. If n is large, k might be larger.
            # Max k is 30.
            
            # We'll iterate k from 2 to 30
            for k in range(2, 31):
                if time.time() > deadline:
                    break
                
                # We need to find i1 < i2 < ... < ik such that sum(smooth[ij]) = n and all are compatible.
                
                # Pruning check:
                # The minimum sum of k distinct smooth numbers is sum(smooth[0..k-1])
                # The maximum sum of k distinct smooth numbers is sum(smooth[n_smooth-k..n_smooth-1])
                # If n < min_sum or n > max_sum, skip this k.
                
                min_sum_k = sum(smooth[:k])
                max_sum_k = sum(smooth[n_smooth-k:])
                
                if n < min_sum_k or n > max_sum_k:
                    continue

                found = False
                chosen_vals = []
                
                # DFS function
                # start_idx: next index to consider
                # remaining: sum needed
                # current_k: how many more numbers to pick
                # chosen_idx: list of indices chosen so far (for incompatibility check)
                
                def dfs(start_idx, remaining, current_k, chosen_idx, chosen_vals):
                    nonlocal found
                    if found:
                        return True
                    
                    if current_k == 0:
                        if remaining == 0:
                            found = True
                            return True
                        else:
                            return False
                    
                    # Pruning:
                    # 1. Not enough numbers left
                    if start_idx + current_k > n_smooth:
                        return False
                    
                    # 2. Sum bounds
                    # Min sum: sum of smallest current_k numbers from start_idx
                    # Max sum: sum of largest current_k numbers from start_idx to end
                    
                    # Efficient min/max sum calculation:
                    # Since smooth is sorted, min sum is sum(smooth[start_idx : start_idx+current_k])
                    # Max sum is sum(smooth[n_smooth-current_k : n_smooth]) ... wait, must be >= start_idx.
                    # Actually, the largest current_k numbers available are the last current_k numbers in the list 
                    # provided their indices are >= start_idx.
                    
                    # Let's just use a quick check:
                    # If remaining < smooth[start_idx] * current_k (approx, actually sum of smallest)
                    # Let's precompute prefix sums for smooth to get sum of any range quickly?
                    # No, let's just iterate.
                    
                    # Better prune:
                    # The smallest possible sum for current_k numbers starting at start_idx is:
                    min_possible = 0
                    for i in range(start_idx, start_idx + current_k):
                        if i >= n_smooth: break
                        min_possible += smooth[i]
                    
                    if remaining < min_possible:
                        return False
                        
                    # The largest possible sum for current_k numbers starting at start_idx is:
                    # We pick the largest current_k numbers from [start_idx, n_smooth-1]
                    max_possible = 0
                    # Indices: n_smooth - 1, n_smooth - 2, ...
                    # We need indices >= start_idx.
                    # So we pick from min(n_smooth-1, start_idx + n_smooth - 1 - start_idx) ...
                    # Basically, the largest current_k numbers in smooth[start_idx:]
                    # These are smooth[n_smooth-current_k] to smooth[n_smooth-1] IF n_smooth-current_k >= start_idx
                    # Otherwise, we are constrained by start_idx.
                    
                    # Let's just iterate backwards for max possible
                    count = 0
                    for i in range(n_smooth - 1, start_idx - 1, -1):
                        if count == current_k:
                            break
                        max_possible += smooth[i]
                        count += 1
                    
                    if remaining > max_possible:
                        return False

                    # Iterate next index
                    # We can bound the loop:
                    # smooth[idx] must be <= remaining
                    # Also, we need at least current_k - 1 numbers left after idx.
                    # So idx can go up to n_smooth - current_k
                    
                    for idx in range(start_idx, n_smooth - current_k + 1):
                        if found:
                            return True
                        
                        val = smooth[idx]
                        if val > remaining:
                            break # Since smooth is sorted, further vals are larger
                        
                        # Check compatibility with all chosen
                        compatible = True
                        for ci in chosen_idx:
                            if incompat[ci] & (1 << idx):
                                compatible = False
                                break
                        
                        if not compatible:
                            continue
                        
                        chosen_idx.append(idx)
                        chosen_vals.append(val)
                        
                        if dfs(idx + 1, remaining - val, current_k - 1, chosen_idx, chosen_vals):
                            return True
                        
                        chosen_idx.pop()
                        chosen_vals.pop()
                    
                    return False

                # Start DFS for this k
                # We can add a global time check inside DFS if needed, but k loop is outer.
                dfs(0, n, k, [], [])
                
                if found:
                    return chosen_vals

            # If not found with k <= 30, return a fallback (should not happen for n<=128)
            # Emergency fallback: try to find ANY representation, even if not minimal
            # This should rarely be hit.
            return [n] if n in smooth_set else [1, n-1]

        # Build final answers
        answers = []
        for n in range(1, 129):
            if time.time() > deadline:
                # Fallback for remaining targets if time runs out
                # Use a simple greedy or just [n] if smooth
                if n in smooth_set:
                    answers.append([n])
                else:
                    # Try to find a quick representation
                    # This is a safety net
                    rem = n
                    chosen = []
                    for s in reversed(smooth):
                        if s <= rem:
                            # Check compatibility
                            ok = True
                            for c in chosen:
                                if c % s == 0 or s % c == 0:
                                    ok = False
                                    break
                            if ok:
                                chosen.append(s)
                                rem -= s
                                if rem == 0:
                                    break
                    if rem == 0 and chosen:
                        answers.append(chosen)
                    else:
                        answers.append([n] if n in smooth_set else [1])
                continue
            
            ans = find_min_antichain(n)
            answers.append(ans)
            
        construct._cached_answers = answers

    return construct._cached_answers

# EVOLVE-BLOCK-END
