"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin from 90s limit
    
    def update_dp(dp, x):
        """Update DP table by adding element x. dp[k] is bitmask of sums of size k."""
        n = len(dp)
        new_dp = [0] * (n + 1)
        new_dp[0] = 1
        for k in range(1, n + 1):
            val = dp[k] if k < n else 0
            if k > 0:
                val |= (dp[k-1] << x)
            new_dp[k] = val
        return new_dp

    def get_violations(selected, dp_without):
        """
        Calculate the number of violations.
        Violation: a in S is mean of subset of S\{a} size >= 2.
        Returns count of violating elements.
        """
        n = len(selected)
        if n < 3:
            return 0
        
        violations = 0
        for i, a in enumerate(selected):
            others = selected[:i] + selected[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            dp_others = dp_without[a]
            
            for k in range(2, m + 1):
                target = k * a
                if target > 512 * k:
                    continue
                if (dp_others[k] >> target) & 1:
                    violations += 1
                    break
        
        return violations

    def greedy_incremental(order, max_size=48):
        selected = []
        n = 0
        dp_all = [1]
        dp_without = {}
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= max_size:
                break
            
            valid = True
            # Check 1: x mean of S
            for k in range(2, n + 1):
                target = k * x
                if target > 512 * k:
                    continue
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: a in S becomes invalid
            for a in selected:
                for k in range(2, n + 2):
                    target_sum = k * a - x
                    if target_sum < 0:
                        continue
                    if target_sum > 512 * (k - 1):
                        continue
                    if k - 1 < len(dp_without[a]):
                        if (dp_without[a][k-1] >> target_sum) & 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if not valid:
                continue
            
            selected.append(x)
            n += 1
            old_dp_all = dp_all
            dp_all = update_dp(dp_all, x)
            
            for a in selected[:-1]:
                dp_without[a] = update_dp(dp_without[a], x)
            dp_without[x] = old_dp_all
        
        return selected, dp_all, dp_without

    def validate_set(s):
        if len(s) < 3:
            return True
        dp_all = [1]
        dp_without = {}
        for x in s:
            old_dp_all = dp_all
            dp_all = update_dp(dp_all, x)
            for a in s[:-1]:
                if a in dp_without:
                    dp_without[a] = update_dp(dp_without[a], x)
            dp_without[x] = old_dp_all
            
        for i, a in enumerate(s):
            others = s[:i] + s[i+1:]
            m = len(others)
            if m < 2:
                continue
            dp_others = dp_without[a]
            for k in range(2, m + 1):
                target = k * a
                if target > 512 * k:
                    continue
                if (dp_others[k] >> target) & 1:
                    return False
        return True

    # 1. Greedy Initialization with 2-adic valuation prioritization
    best_set = []
    best_dp_all = [1]
    best_dp_without = {}
    best_len = 0
    
    def two_adic_valuation(n):
        if n == 0: return 0
        v = 0
        while n % 2 == 0:
            v += 1
            n //= 2
        return v

    # Sort candidates by highest 2-adic valuation (powers of 2 first)
    candidates_by_valuation = sorted(range(1, 513), key=lambda x: -two_adic_valuation(x))
    
    # Try a few random shuffles for greedy, but start with the valuation-sorted one
    # and a few randomized variants that preserve some valuation structure
    
    # First try: pure valuation sorted
    res, dp_a, dp_w = greedy_incremental(candidates_by_valuation)
    if len(res) > best_len:
        best_set = res
        best_dp_all = dp_a
        best_dp_without = dp_w
        best_len = len(res)
        
    if best_len >= 48:
        return best_set

    # Then try randomized variants that bias towards high valuation
    for seed in range(30):
        if time.time() - start_time > deadline:
            break
        random.seed(seed)
        candidates = list(range(1, 513))
        # Weight by valuation: higher valuation = higher weight in shuffle
        # Simple approach: sort by -valuation + random noise
        vals = [two_adic_valuation(x) for x in candidates]
        # Create pairs (noise - val, x) and sort
        noisy = [(random.random() * 10 - v, x) for v, x in zip(vals, candidates)]
        noisy.sort(key=lambda t: t[0])
        candidates = [x for _, x in noisy]
        
        res, dp_a, dp_w = greedy_incremental(candidates)
        if len(res) > best_len:
            best_set = res
            best_dp_all = dp_a
            best_dp_without = dp_w
            best_len = len(res)
            
        if best_len >= 48:
            break

    if not best_set:
        return []

    # 2. Local Search: Swap low-valuation elements for high-valuation ones
    # State: S (list), dp_all, dp_without
    S = best_set[:]
    dp_all = best_dp_all[:]
    dp_without = {k: v[:] for k, v in best_dp_without.items()}
    
    def calc_energy(S, dp_all, dp_without):
        n = len(S)
        if n < 3:
            return -n
        violations = get_violations(S, dp_without)
        return violations * 100 - n

    current_energy = calc_energy(S, dp_all, dp_without)
    
    # Local search: try to swap elements to increase size or reduce violations
    # Focus on: if we have violations, remove violators. If not, try to add more.
    # Also try swaps to find better configurations.
    
    improved = True
    while improved and time.time() - start_time < deadline:
        improved = False
        
        # If we have violations, try removing violators
        if current_energy > 0:
            violations = get_violations(S, dp_without)
            if violations > 0:
                # Find a violating element and remove it
                for a in S:
                    dp_others = dp_without[a]
                    m = len(S) - 1
                    for k in range(2, m + 1):
                        target = k * a
                        if target > 512 * k:
                            continue
                        if (dp_others[k] >> target) & 1:
                            # Remove a
                            idx = S.index(a)
                            S.pop(idx)
                            # Rebuild DP
                            dp_all = [1]
                            dp_without = {}
                            for y in S:
                                old_dp_all = dp_all
                                dp_all = update_dp(dp_all, y)
                                for b in S[:-1]:
                                    if b in dp_without:
                                        dp_without[b] = update_dp(dp_without[b], y)
                                dp_without[y] = old_dp_all
                            
                            current_energy = calc_energy(S, dp_all, dp_without)
                            improved = True
                            break
                    if improved:
                        break
        
        # Try to add elements if we have space and no violations
        if len(S) < 48 and current_energy <= 0:
            added = False
            # Try adding elements in order of decreasing valuation
            candidates = [x for x in range(1, 513) if x not in S]
            candidates.sort(key=lambda x: -two_adic_valuation(x))
            
            for x in candidates:
                if time.time() - start_time > deadline:
                    break
                
                valid = True
                # Check 1: x mean of S
                for k in range(2, len(S) + 1):
                    target = k * x
                    if target > 512 * k:
                        continue
                    if (dp_all[k] >> target) & 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Check 2: a in S becomes invalid
                for a in S:
                    for k in range(2, len(S) + 2):
                        target_sum = k * a - x
                        if target_sum < 0:
                            continue
                        if target_sum > 512 * (k - 1):
                            continue
                        if k - 1 < len(dp_without[a]):
                            if (dp_without[a][k-1] >> target_sum) & 1:
                                valid = False
                                break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Add x
                S.append(x)
                old_dp_all = dp_all
                dp_all = update_dp(dp_all, x)
                for a in S[:-1]:
                    dp_without[a] = update_dp(dp_without[a], x)
                dp_without[x] = old_dp_all
                
                current_energy = calc_energy(S, dp_all, dp_without)
                improved = True
                added = True
                break
            
            if added:
                continue
        
        # Try swaps: replace a low-valuation element with a high-valuation one
        if len(S) > 0 and time.time() - start_time < deadline:
            # Find element with lowest valuation in S
            if S:
                min_val_idx = min(range(len(S)), key=lambda i: two_adic_valuation(S[i]))
                x_remove = S[min_val_idx]
                
                # Try adding high-valuation elements not in S
                candidates = [x for x in range(1, 513) if x not in S]
                candidates.sort(key=lambda x: -two_adic_valuation(x))
                
                swapped = False
                for x_add in candidates[:20]:  # Limit to top 20 for speed
                    if time.time() - start_time > deadline:
                        break
                    
                    # Try removing x_remove and adding x_add
                    S_test = S[:]
                    S_test.pop(min_val_idx)
                    S_test.append(x_add)
                    
                    if validate_set(S_test):
                        # Accept swap
                        S = S_test
                        dp_all = [1]
                        dp_without = {}
                        for y in S:
                            old_dp_all = dp_all
                            dp_all = update_dp(dp_all, y)
                            for b in S[:-1]:
                                if b in dp_without:
                                    dp_without[b] = update_dp(dp_without[b], y)
                            dp_without[y] = old_dp_all
                        
                        current_energy = calc_energy(S, dp_all, dp_without)
                        improved = True
                        swapped = True
                        break
                
                if swapped:
                    continue

    # Final Validation
    if not validate_set(S):
        # Repair
        valid_set = []
        for x in S:
            test = valid_set + [x]
            if validate_set(test):
                valid_set.append(x)
        S = valid_set

    return S

# EVOLVE-BLOCK-END
