import time
import random

def construct():
    deadline = time.time() + 55.0
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    def compute_dp(S):
        """Compute subset-sum DP table with cardinality for set S."""
        n = len(S)
        if n == 0:
            return [1]
        dp = [0] * (n + 1)
        dp[0] = 1
        for val in S:
            # Iterate backwards to avoid using the same element multiple times in one step
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << val)
        return dp

    def precompute_removal_dps(S):
        """
        Precompute DP tables for S \ {a} for all a in S.
        Returns a list of (a, dp_removed, sum_removed)
        """
        removals = []
        n = len(S)
        for i, a in enumerate(S):
            others = S[:i] + S[i+1:]
            dp_o = compute_dp(others)
            sum_o = sum(others)
            removals.append((a, dp_o, sum_o))
        return removals

    def is_valid_with_precomputed(S, x, dp_S, sum_S, removal_dps):
        """
        Checks if S U {x} is valid, given that S is valid.
        S is a list of integers.
        x is the new integer to add.
        dp_S: DP table for S.
        sum_S: Sum of S.
        removal_dps: List of (a, dp_{S\{a}}, sum_{S\{a}}) for all a in S.
        """
        if x in S:
            return False
        
        n = len(S)
        
        # 1. Check if x is the mean of a subset of S
        # k*x must be a subset sum of S of size k, for k >= 2
        for k in range(2, n + 1):
            target = k * x
            if target > sum_S:
                break
            if (dp_S[k] >> target) & 1:
                return False

        # 2. Check if any a in S becomes invalid due to x
        # For each a in S, check if k*a is a subset sum of (S \ {a}) U {x} for any k >= 2
        # This implies sum(T) + x = k*a for some T subset of S \ {a}, |T| = k-1 >= 1
        # So sum(T) = k*a - x.
        # We check if k*a - x is a subset sum of S \ {a} of size k-1.
        
        for a, dp_o, sum_o in removal_dps:
            m = n - 1  # size of S \ {a}
            if m < 1:
                continue
            
            # k ranges from 2 to m+1 (subset size in original is k, so T size is k-1)
            # Max k is m+1 (using all elements of S\{a} plus x)
            for k in range(2, m + 2):
                target = k * a - x
                if target < 0:
                    continue
                if target > sum_o:
                    break
                
                j = k - 1
                if j <= m:
                    if (dp_o[j] >> target) & 1:
                        return False
                        
        return True

    def calculate_conflict_score(S, x):
        """
        Heuristic score: number of 3-term arithmetic progressions involving x and two elements from S.
        Lower is better.
        """
        count = 0
        S_set = set(S)
        # x, a, b form AP if x+a=2b, x+b=2a, a+b=2x
        # Case 1: x is the mean. (Checked in validity, but for score we want to avoid x being mean? No, we want to avoid x CAUSING others to be means.)
        # The validity check handles exact constraints. This score is a heuristic for "density".
        # Let's count how many pairs (a,b) in S such that x, a, b are in AP.
        for a in S:
            # If x is mean: 2x = a + b => b = 2x - a
            b = 2 * x - a
            if b in S_set and b != a:
                count += 1
            # If a is mean: 2a = x + b => b = 2a - x
            b = 2 * a - x
            if b in S_set and b != x and b != a:
                count += 1
            # If b is mean: 2b = x + a => b = (x+a)/2
            if (x + a) % 2 == 0:
                b = (x + a) // 2
                if b in S_set and b != x and b != a:
                    count += 1
        return count

    def try_greedy_best_fit(candidates):
        current = []
        dp_current = None
        sum_current = 0
        removals_current = []
        
        # Filter candidates to those not in current (initially all)
        available = list(candidates)
        
        while len(current) < 48 and time.time() < deadline:
            if not available:
                break
            
            valid_candidates = []
            for x in available:
                if x in current:
                    continue
                # Quick check: x must not be in current
                # Use precomputed removals for current set
                if is_valid_with_precomputed(current, x, dp_current, sum_current, removals_current):
                    score = calculate_conflict_score(current, x)
                    valid_candidates.append((score, x))
            
            if not valid_candidates:
                break
            
            # Pick the candidate with the lowest conflict score
            # If tie, pick randomly or smallest? Let's pick randomly among lowest to add diversity
            min_score = min(s for s, _ in valid_candidates)
            best_candidates = [x for s, x in valid_candidates if s == min_score]
            x_to_add = rng.choice(best_candidates)
            
            current.append(x_to_add)
            available.remove(x_to_add)
            sum_current += x_to_add
            dp_current = compute_dp(current)
            removals_current = precompute_removal_dps(current)
            
        return current

    def local_search(S):
        S = list(S)
        S_set = set(S)
        available = [x for x in range(1, 513) if x not in S_set]
        dp_S = compute_dp(S)
        sum_S = sum(S)
        removals_S = precompute_removal_dps(S)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(available)
            
            # Try adding
            # We already have removals_S and dp_S for the current S
            added = False
            for x in available:
                if len(S) >= 48:
                    break
                if is_valid_with_precomputed(S, x, dp_S, sum_S, removals_S):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    sum_S += x
                    dp_S = compute_dp(S)
                    removals_S = precompute_removal_dps(S)
                    improved = True
                    added = True
                    break
            
            if len(S) >= 48:
                break
            
            # Try swapping
            if not improved:
                rng.shuffle(S)
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    removed = S[i]
                    S_temp = S[:i] + S[i+1:]
                    sum_temp = sum_S - removed
                    dp_temp = compute_dp(S_temp)
                    removals_temp = precompute_removal_dps(S_temp)
                    
                    for x in available:
                        if x in S_temp:
                            continue
                        if is_valid_with_precomputed(S_temp, x, dp_temp, sum_temp, removals_temp):
                            S = S_temp + [x]
                            S_set = set(S)
                            available.remove(x)
                            available.append(removed)
                            sum_S = sum_temp + x
                            dp_S = compute_dp(S)
                            removals_S = precompute_removal_dps(S)
                            improved = True
                            break
                    if improved:
                        break
        return S

    # 1. Structured Greedy Passes with Best-Fit
    # We will run a few passes with different orderings or just use the best-fit on a shuffled list
    # The best-fit greedy inherently handles ordering by picking the "best" available at each step.
    
    # Pass 1: Shuffled candidates
    rng_init = random.Random(1)
    candidates = list(range(1, 513))
    rng_init.shuffle(candidates)
    res = try_greedy_best_fit(candidates)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
        
    if best_size == 48:
        return best_set

    # Pass 2: Descending
    res = try_greedy_best_fit(list(range(512, 0, -1)))
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
        
    if best_size == 48:
        return best_set

    # Pass 3: Ascending
    res = try_greedy_best_fit(list(range(1, 513)))
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
        
    if best_size == 48:
        return best_set

    # 2. Local Search
    if best_set and time.time() < deadline:
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
        if best_size == 48:
            return best_set

    # 3. More Local Search with restarts
    if best_set and time.time() < deadline:
        for _ in range(10): # Increase restarts if time permits
            if time.time() > deadline:
                break
            perturbed = list(best_set)
            rng_restart = random.Random(time.time_ns())
            if perturbed:
                idx = rng_restart.randrange(len(perturbed))
                perturbed.pop(idx)
            improved = local_search(perturbed)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
            if best_size == 48:
                return best_set

    return best_set
