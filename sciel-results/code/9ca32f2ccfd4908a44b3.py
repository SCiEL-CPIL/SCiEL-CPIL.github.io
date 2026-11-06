import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    target_mask = 0
    for d in range(1, 31):
        target_mask |= (1 << d)
        
    def solve_for_K_M(K, M):
        """
        Find a Sidon set of size K in [0, M] covering 1..30.
        """
        def dfs(current_set, used_diffs_mask, covered_mask, depth):
            # Check if all targets are covered
            if (covered_mask & target_mask) == target_mask:
                return current_set[:]
            
            # If we run out of elements allowed
            if len(current_set) >= K:
                return None
                
            # Time check
            if time.time() > deadline:
                return None
                
            last = current_set[-1] if current_set else -1
            
            # Pruning: 
            # uncovered = target_mask & ~covered_mask
            uncovered = target_mask & ~covered_mask
            if uncovered:
                # Find smallest uncovered difference
                min_unc = (uncovered & -uncovered).bit_length() - 1
                
                # If we place x, it covers some diffs.
                # The remaining uncovered diffs must be covered by future elements.
                # Future elements are > x.
                # If a difference d_rem is not covered by x, it must be formed by v - u where v, u > x.
                # This requires v >= u + d_rem > x + d_rem.
                # So M must be at least x + d_rem.
                # If x + d_rem > M, then d_rem CANNOT be covered by future elements.
                # Thus, d_rem MUST be covered by x.
                
                # We will check this for each candidate x.
                
            valid_xs = []
            
            for x in range(last + 1, M + 1):
                # Check Sidon constraint and coverage
                conflict = False
                new_covered = 0
                for a in current_set:
                    d = x - a
                    if used_diffs_mask & (1 << d):
                        conflict = True
                        break
                    if d <= 30:
                        if covered_mask & (1 << d) == 0:
                            new_covered |= (1 << d)
                
                if conflict:
                    continue
                
                # Check if remaining uncovered diffs can be formed by future elements
                # remaining_uncovered = uncovered & ~new_covered
                remaining_uncovered = uncovered & ~new_covered
                
                if remaining_uncovered:
                    # Find smallest remaining uncovered diff
                    min_rem = (remaining_uncovered & -remaining_uncovered).bit_length() - 1
                    
                    # This diff min_rem must be formed by two future elements u, v > x.
                    # So v - u = min_rem.
                    # Since u > x, v > x + min_rem.
                    # So we need M >= x + min_rem.
                    if x + min_rem > M:
                        continue
                        
                    # Also, we need enough slots to form this diff?
                    # We need at least 2 slots left to form ANY diff between future elements.
                    # Slots left after placing x: K - (depth + 1)
                    slots_left = K - (depth + 1)
                    if slots_left < 2:
                        # We can't form any diff between future elements.
                        # So remaining_uncovered must be empty.
                        # But it's not. So this x is invalid.
                        continue
                        
                    # If slots_left >= 2, it's *possible* to form min_rem.
                    # But we also need to check if OTHER remaining uncovered diffs can be formed.
                    # This is harder. The min_rem check is a necessary condition.
                    # Is it sufficient? No. But it prunes a lot.
                    
                    # Additional check:
                    # The largest remaining uncovered diff?
                    # max_rem = remaining_uncovered.bit_length() - 1
                    # v - u = max_rem.
                    # v <= M, u > x.
                    # M - (x+1) >= max_rem ?
                    # M - x - 1 >= max_rem ?
                    # x + max_rem + 1 <= M ?
                    max_rem = remaining_uncovered.bit_length() - 1
                    if x + max_rem + 1 > M:
                        continue

                valid_xs.append((x, new_covered))
            
            if not valid_xs:
                return None
                
            # Sort candidates
            valid_xs.sort(key=lambda t: (-bin(t[1]).count('1'), t[0]))
            
            for x, new_cov in valid_xs:
                # Update state
                new_used_mask = used_diffs_mask
                for a in current_set:
                    d = x - a
                    new_used_mask |= (1 << d)
                
                new_cov_mask = covered_mask | new_cov
                
                res = dfs(current_set + [x], new_used_mask, new_cov_mask, depth + 1)
                if res is not None:
                    return res
                    
            return None
            
        return dfs([0], 0, 0, 0)

    # Search for minimal M
    # M must be at least 36 (for K=9)
    # We iterate M, and for each M, we try K=9, 10, ...
    
    for M in range(36, 100):
        if time.time() > deadline - 5:
            break
            
        # Max K for this M: K(K-1)/2 <= M
        # K^2 < 2M + 1 => K < sqrt(2M+1)
        max_k = int((2*M + 1) ** 0.5)
        
        # Try K from 9 to max_k
        for K in range(9, max_k + 1):
            if time.time() > deadline - 5:
                break
                
            result = solve_for_K_M(K, M)
            if result is not None:
                return result

    # Fallback
    result = [0]
    used = set()
    for x in range(1, 1000):
        valid = True
        for a in result:
            d = x - a
            if d in used:
                valid = False
                break
        if valid:
            result.append(x)
            for a in result[:-1]:
                used.add(x - a)
            if len(result) >= 32:
                break
    return result
