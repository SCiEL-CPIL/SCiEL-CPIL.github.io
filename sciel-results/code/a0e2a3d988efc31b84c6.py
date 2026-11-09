import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55.0
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    # Precompute pairwise conflict graph: x conflicts with y if x = (y+z)/2 or similar
    # Actually, the constraint is: no element is the mean of a subset of >=2 OTHER elements
    # So for a set S, element x is invalid if exists subset T of S\{x}, |T|>=2, such that sum(T) = k*x for some k>=2
    
    # Precompute which pairs (a,b) create direct 2-element mean conflicts
    # a is mean of {b,c} iff 2a = b+c, i.e., b+c = 2a
    # For larger subsets, we need subset-sum checks
    
    def compute_dp_numpy(S):
        """Compute subset-sum DP using numpy for speed. Returns list of bitsets by cardinality."""
        n = len(S)
        if n == 0:
            return [0]
        # Use Python ints for bitsets (they handle arbitrary precision)
        dp = [0] * (n + 1)
        dp[0] = 1  # sum 0 with 0 elements
        for val in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << val)
        return dp
    
    def is_valid_addition(S, x, dp_S, sum_S):
        """
        Check if adding x to S keeps the set valid.
        S is already valid. Need to check:
        1. x is not the mean of any subset of S (size >= 2)
        2. No element in S becomes the mean of a subset involving x
        """
        if x in S:
            return False
        
        n = len(S)
        
        # Check 1: x is not mean of any subset of S of size k >= 2
        # sum(subset) = k*x for some k in [2, n]
        for k in range(2, n + 1):
            target = k * x
            if target > sum_S:
                break
            if (dp_S[k] >> target) & 1:
                return False
        
        # Check 2: For each a in S, check if a is mean of subset of (S\{a}) U {x}
        # subset = T U {x} where T is subset of S\{a}, |T| >= 1
        # sum(T) + x = k*a, k >= 2
        # sum(T) = k*a - x
        # |T| = j, total size = j+1 >= 2 => j >= 1
        # k >= 2
        
        for i, a in enumerate(S):
            others_sum = sum_S - a
            m = n - 1  # size of S\{a}
            
            # Compute DP for S\{a} efficiently
            # We can reuse dp_S by removing a's contribution
            # But it's complex. For now, compute fresh for small sets.
            # Optimization: only check if k*a - x is in reasonable range
            for k in range(2, m + 2):
                target = k * a - x
                if target < 0:
                    continue
                if target > others_sum:
                    break
                j = k - 1
                if j > m:
                    continue
                # Need to check if target is a subset sum of S\{a} with exactly j elements
                # Compute dp for S\{a}
                S_without_a = S[:i] + S[i+1:]
                dp_o = compute_dp_numpy(S_without_a)
                if (dp_o[j] >> target) & 1:
                    return False
            
        return True
    
    def try_greedy(candidates):
        current = []
        current_set = set()
        dp_current = [0]
        sum_current = 0
        for x in candidates:
            if len(current) >= 48:
                break
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            if is_valid_addition(current, x, dp_current, sum_current):
                current.append(x)
                current_set.add(x)
                sum_current += x
                dp_current = compute_dp_numpy(current)
        return current
    
    def local_search(S):
        S = list(S)
        S_set = set(S)
        available = [x for x in range(1, 513) if x not in S_set]
        dp_S = compute_dp_numpy(S)
        sum_S = sum(S)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(available)
            
            # Try adding
            for x in available:
                if len(S) >= 48:
                    break
                if is_valid_addition(S, x, dp_S, sum_S):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    sum_S += x
                    dp_S = compute_dp_numpy(S)
                    improved = True
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
                    dp_temp = compute_dp_numpy(S_temp)
                    for x in available:
                        if x in S_temp:
                            continue
                        if is_valid_addition(S_temp, x, dp_temp, sum_temp):
                            S = S_temp + [x]
                            S_set = set(S)
                            available.remove(x)
                            available.append(removed)
                            sum_S = sum_temp + x
                            dp_S = compute_dp_numpy(S)
                            improved = True
                            break
                    if improved:
                        break
        return S
    
    # 1. Structured Greedy Passes
    # Descending
    desc = list(range(512, 0, -1))
    res = try_greedy(desc)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
        
    # Ascending
    asc = list(range(1, 513))
    res = try_greedy(asc)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
        
    # Random Passes with weighted ordering
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        candidates = list(range(1, 513))
        # Weight toward numbers that are less likely to be means
        # Powers of 2, primes, etc. tend to be safer
        weights = []
        for x in candidates:
            # Lower weight for numbers that divide many others
            divisors = sum(1 for d in range(1, 513) if d % x == 0)
            weights.append(1.0 / (1 + divisors))
        total_w = sum(weights)
        for i in range(len(candidates) - 1, 0, -1):
            r = rng.random() * sum(weights[:i+1])
            cumulative = 0
            for j in range(i + 1):
                cumulative += weights[j]
                if r <= cumulative:
                    candidates[i], candidates[j] = candidates[j], candidates[i]
                    weights[i], weights[j] = weights[j], weights[i]
                    break
        res = try_greedy(candidates)
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
            if best_size == 48:
                return best_set
    
    return best_set
