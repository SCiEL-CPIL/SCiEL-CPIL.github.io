"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Precompute divisors of 5040
    n = 5040
    target = 4979
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    # Filter: only divisors that could help (<= target)
    divisors = [d for d in divisors if d <= target]
    divisors.sort(reverse=True)
    m = len(divisors)
    
    # Precompute suffix sums for pruning: suffix_sum[i] = sum(divisors[i:])
    suffix_sum = [0] * (m + 1)
    for i in range(m - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
    
    best_result = None
    
    # Try increasing number of terms: 1, 2, 3, ...
    for k in range(1, min(m, 256) + 1):
        if time.time() > deadline:
            break
        
        if k <= 1:
            # k=1: just check if target is a divisor
            if target in divisors:
                return [target]
            continue
        
        if k <= 6:
            # Recursive search with pruning
            def search(start, remaining_k, remaining_sum, chosen):
                nonlocal best_result
                if time.time() > deadline:
                    return False
                if remaining_k == 0:
                    if remaining_sum == 0:
                        best_result = list(chosen)
                        return True
                    return False
                # Not enough divisors left
                if start + remaining_k > m:
                    return False
                # Max possible sum from remaining divisors
                if suffix_sum[start] < remaining_sum:
                    return False
                # Min possible sum: we must pick remaining_k from divisors[start:]
                # The smallest remaining_k divisors from start are at the end
                # Actually divisors is sorted descending, so smallest are at end
                # We need to pick remaining_k items from divisors[start:]
                # The minimum sum is sum of the last remaining_k items available
                # But we can only go up to m, so min sum = sum(divisors[m-remaining_k:m]) if start <= m-remaining_k
                # Actually more carefully: we pick from indices start..m-1, need remaining_k items
                # Min sum = sum of the remaining_k smallest in divisors[start:] = sum(divisors[m-remaining_k:m])
                # But only if m - remaining_k >= start, i.e., remaining_k <= m - start
                # Which we already checked above
                min_sum = suffix_sum[m - remaining_k] - suffix_sum[m]  # This isn't right
                # Let me recompute: suffix_sum[i] = sum(divisors[i:])
                # Sum of last remaining_k elements = suffix_sum[m - remaining_k] - suffix_sum[m]
                # suffix_sum[m] = 0, so min_sum = suffix_sum[m - remaining_k]
                # Wait, that's the sum of divisors[m-remaining_k:] which is the last remaining_k elements
                # But we're choosing from divisors[start:], so the minimum is actually the sum of the 
                # remaining_k smallest elements in divisors[start:], which are the last remaining_k 
                # elements of the whole array IF start <= m - remaining_k
                # Since divisors is sorted descending, the smallest elements are at the end
                # So min sum from divisors[start:] picking remaining_k = sum(divisors[m-remaining_k:m])
                # This is valid as long as m - remaining_k >= start
                # Which is equivalent to remaining_k <= m - start (already checked)
                min_possible = suffix_sum[m - remaining_k]
                if min_possible > remaining_sum:
                    return False
                
                # Prune: even if we take the largest possible, can we reach remaining_sum?
                # Max sum from divisors[start:] picking remaining_k = sum(divisors[start:start+remaining_k])
                max_possible = suffix_sum[start] - suffix_sum[start + remaining_k]
                if max_possible < remaining_sum:
                    return False
                
                # Try each divisor from start
                # Upper bound: we need remaining_sum, and we have remaining_k items
                # The current item must be <= remaining_sum
                # Also, the current item must be >= some lower bound
                # Lower bound: remaining_sum - (max sum of remaining_k-1 items after i)
                # = remaining_sum - (suffix_sum[i+1] - suffix_sum[i+1 + remaining_k - 1])
                # Upper bound: remaining_sum - (min sum of remaining_k-1 items after i)
                # = remaining_sum - (suffix_sum[m - (remaining_k - 1)] - suffix_sum[m])
                # = remaining_sum - suffix_sum[m - remaining_k + 1]
                
                for i in range(start, m - remaining_k + 1):
                    if time.time() > deadline:
                        return False
                    d = divisors[i]
                    if d > remaining_sum:
                        continue
                    # Lower bound check: can we fill remaining_sum - d with remaining_k-1 items from divisors[i+1:]?
                    rem_k = remaining_k - 1
                    rem_s = remaining_sum - d
                    if rem_k == 0:
                        if rem_s == 0:
                            chosen.append(d)
                            best_result = list(chosen)
                            chosen.pop()
                            return True
                        continue
                    if i + 1 + rem_k > m:
                        continue
                    # Max possible from divisors[i+1:] picking rem_k
                    max_rem = suffix_sum[i + 1] - suffix_sum[i + 1 + rem_k]
                    if max_rem < rem_s:
                        # d is too small; since divisors is descending, smaller d's will also fail
                        # Actually no: d is decreasing, so rem_s = remaining_sum - d is increasing
                        # So we can't break, we should continue
                        continue
                    # Min possible from divisors[i+1:] picking rem_k
                    min_rem = suffix_sum[m - rem_k]
                    if min_rem > rem_s:
                        # d is too large; since divisors is descending, larger d's (earlier) were already skipped
                        # Actually d is decreasing, so if min_rem > rem_s, then for smaller d, rem_s is larger,
                        # so min_rem > rem_s might still hold. We can't break easily.
                        # But if min_rem > rem_s, it means even the smallest rem_k sum is too big,
                        # so we need a larger d to reduce rem_s. But d is decreasing, so earlier items are larger.
                        # We already passed them. So this d and all subsequent (smaller) d's will have
                        # rem_s >= current rem_s, and min_rem is fixed, so min_rem > rem_s will continue.
                        # We can break.
                        break
                    
                    chosen.append(d)
                    if search(i + 1, rem_k, rem_s, chosen):
                        return True
                    chosen.pop()
            
            if search(0, k, target, []):
                if best_result is not None:
                    return best_result
        else:
            # For k > 6, use greedy as fallback
            break
    
    # Fallback: greedy
    remainder = target
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    
    assert remainder == 0, f"Greedy failed, remainder={remainder}"
    return chosen
# EVOLVE-BLOCK-END
