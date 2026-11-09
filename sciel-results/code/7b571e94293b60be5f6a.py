import time
import random

def construct():
    deadline = time.time() + 55.0
    MAX_N = 512
    MAX_K = 48
    MAX_SUM = MAX_N * MAX_K  # 512 * 48 = 24576

    def compute_dp(S):
        """Compute subset-sum DP table with cardinality for set S."""
        n = len(S)
        dp = [0] * (n + 1)
        dp[0] = 1
        for val in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << val)
        return dp

    def is_valid_with_dp(S, x, dp_S=None, sum_S=None):
        """
        Checks if S U {x} is valid, given that S is valid.
        """
        if x in S:
            return False
        
        n = len(S)
        if sum_S is None:
            sum_S = sum(S)
        if dp_S is None:
            dp_S = compute_dp(S)
        
        # 1. Check if x is the mean of a subset of S
        for k in range(2, n + 1):
            target = k * x
            if target > sum_S:
                break
            if (dp_S[k] >> target) & 1:
                return False

        # 2. Check if any a in S becomes invalid due to x
        # For each a in S, check if k*a is a subset sum of (S \ {a}) U {x} for any k >= 2
        # This means sum(T) + x = k*a for some T subset of S \ {a}, |T| >= 1
        # So sum(T) = k*a - x.
        # We need to check if k*a - x is a subset sum of S \ {a} for any k >= 2 and |T| >= 1.
        
        for i, a in enumerate(S):
            others = S[:i] + S[i+1:]
            m = len(others)
            if m < 1:
                continue
            
            # Compute DP for others
            dp_o = compute_dp(others)
            max_sum_o = sum_S - a
            
            # k*a - x must be a sum of a subset of 'others' of size j >= 1.
            # Total subset size is j + 1 (for x).
            # Condition: subset size >= 2 => j + 1 >= 2 => j >= 1.
            # Also k >= 2.
            
            for k in range(2, m + 2):
                target = k * a - x
                if target < 0:
                    continue
                if target > max_sum_o:
                    break
                
                j = k - 1
                if j <= m:
                    if (dp_o[j] >> target) & 1:
                        return False
                        
        return True

    def try_greedy(candidates):
        current = []
        dp_current = None
        sum_current = 0
        for x in candidates:
            if len(current) >= MAX_K:
                break
            if time.time() > deadline:
                break
            if is_valid_with_dp(current, x, dp_current, sum_current):
                current.append(x)
                sum_current += x
                dp_current = compute_dp(current)
        return current

    def local_search(S):
        S = list(S)
        S_set = set(S)
        available = [x for x in range(1, MAX_N + 1) if x not in S_set]
        dp_S = compute_dp(S)
        sum_S = sum(S)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(available)
            
            # Try adding
            for x in available:
                if len(S) >= MAX_K:
                    break
                if is_valid_with_dp(S, x, dp_S, sum_S):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    sum_S += x
                    dp_S = compute_dp(S)
                    improved = True
                    break
            
            if len(S) >= MAX_K:
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
                    for x in available:
                        if x in S_temp:
                            continue
                        if is_valid_with_dp(S_temp, x, dp_temp, sum_temp):
                            S = S_temp + [x]
                            S_set = set(S)
                            available.remove(x)
                            available.append(removed)
                            sum_S = sum_temp + x
                            dp_S = compute_dp(S)
                            improved = True
                            break
                    if improved:
                        break
        return S

    # 1. Structured Greedy Passes
    # Descending
    desc = list(range(MAX_N, 0, -1))
    res = try_greedy(desc)
    best_set = res
    best_size = len(res)
    
    # Ascending
    asc = list(range(1, MAX_N + 1))
    res = try_greedy(asc)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
        
    # Random Passes
    rng = random.Random(42)
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        candidates = list(range(1, MAX_N + 1))
        rng.shuffle(candidates)
        res = try_greedy(candidates)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
        if best_size == MAX_K:
            return best_set

    # 2. Local Search
    if best_set and time.time() < deadline:
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
        if best_size == MAX_K:
            return best_set

    # 3. More Local Search with restarts
    if best_set and time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            perturbed = list(best_set)
            rng = random.Random(time.time_ns())
            if perturbed:
                idx = rng.randrange(len(perturbed))
                perturbed.pop(idx)
            improved = local_search(perturbed)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
            if best_size == MAX_K:
                return best_set

    return best_set
