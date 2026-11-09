import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 57.0
    n_max = 512
    target_size = 48

    # Precompute conflict edges using subset-sum DP
    # For each element x, check if x is mean of any subset of size >= 2 from [1..n_max]\{x}
    # We'll build a bitset-based DP for each candidate set

    def build_conflict_set(x, universe):
        """
        Check if x is the arithmetic mean of any subset of universe of size >= 2.
        universe is a sorted list of positive integers not containing x.
        Returns True if conflict exists.
        """
        n = len(universe)
        if n < 2:
            return False
        
        max_sum = sum(universe)
        max_k = min(n, max_sum // x)
        if max_k < 2:
            return False
        
        # DP: dp[k] is a bitset where bit s is 1 if sum s is achievable with exactly k elements
        dp = [0] * (n + 1)
        dp[0] = 1  # sum 0 with 0 elements
        
        for val in universe:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << val)
        
        for k in range(2, max_k + 1):
            target = k * x
            if (dp[k] >> target) & 1:
                return True
        return False

    # Strategy: Use a greedy approach with conflict checking
    # But first, let's try to find a good initial set using a different ordering strategy
    
    def greedy_with_conflicts(order):
        """Greedy build checking conflicts incrementally."""
        S = []
        S_set = set()
        for x in order:
            if len(S) >= target_size:
                break
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            # Check if x conflicts with any subset of S
            others = [s for s in S if s != x]
            if build_conflict_set(x, others):
                continue
            # Also check: for each a in S, is a the mean of a subset of S\{a} U {x}?
            valid = True
            for a in S:
                if a == x:
                    continue
                # Check if a is mean of subset of (S\{a}) U {x}
                # Subset T of (S\{a}) U {x}, |T| >= 2, mean(T) = a
                # Case 1: T doesn't contain x -> T subset of S\{a}, already valid since S is valid
                # Case 2: T contains x -> T = {x} U M where M subset of S\{a}
                #   sum(M) + x = |M|+1 times a
                #   sum(M) = (|M|+1)*a - x
                others_no_a = [s for s in S if s != a and s != x]
                m = len(others_no_a)
                if m >= 1:
                    # Check if there exists M subset of others_no_a with sum(M) = (|M|+1)*a - x
                    # Let j = |M|, then sum(M) = (j+1)*a - x
                    max_sum_m = sum(others_no_a)
                    max_j = min(m, max_sum_m // max(1, a))
                    dp_m = [0] * (m + 1)
                    dp_m[0] = 1
                    for val in others_no_a:
                        for k in range(m - 1, -1, -1):
                            if dp_m[k]:
                                dp_m[k+1] |= (dp_m[k] << val)
                    for j in range(1, m + 1):
                        target_sum = (j + 1) * a - x
                        if target_sum < 0 or target_sum > max_sum_m:
                            continue
                        if (dp_m[j] >> target_sum) & 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                S.append(x)
                S_set.add(x)
        return S

    best_set = []
    best_size = 0

    # Try multiple orderings
    orderings = [
        list(range(n_max, 0, -1)),
        list(range(1, n_max + 1)),
    ]
    
    rng = random.Random(42)
    for i in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        orderings.append(order)

    for order in orderings:
        if time.time() > deadline:
            break
        S = greedy_with_conflicts(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)

    # Local search: try to add or swap elements
    if best_size < target_size and time.time() < deadline:
        S = list(best_set)
        S_set = set(S)
        available = [x for x in range(1, n_max + 1) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(available)
            
            # Try to add
            for x in available:
                if len(S) >= target_size:
                    break
                others = [s for s in S if s != x]
                if not build_conflict_set(x, others):
                    # Check reverse conflicts
                    valid = True
                    for a in S:
                        others_no_a = [s for s in S if s != a and s != x]
                        m = len(others_no_a)
                        if m >= 1:
                            max_sum_m = sum(others_no_a)
                            dp_m = [0] * (m + 1)
                            dp_m[0] = 1
                            for val in others_no_a:
                                for k in range(m - 1, -1, -1):
                                    if dp_m[k]:
                                        dp_m[k+1] |= (dp_m[k] << val)
                            for j in range(1, m + 1):
                                target_sum = (j + 1) * a - x
                                if 0 <= target_sum <= max_sum_m:
                                    if (dp_m[j] >> target_sum) & 1:
                                        valid = False
                                        break
                        if not valid:
                            break
                    if valid:
                        S.append(x)
                        S_set.add(x)
                        available.remove(x)
                        improved = True
                        break
            
            if len(S) >= target_size:
                break
            
            # Try swaps
            if not improved:
                rng.shuffle(S)
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    old = S[i]
                    S_temp = S[:i] + S[i+1:]
                    
                    for x in available:
                        if x in S_set:
                            continue
                        # Check if x can replace old in S_temp
                        others = [s for s in S_temp if s != x]
                        if not build_conflict_set(x, others):
                            valid = True
                            for a in S_temp:
                                others_no_a = [s for s in S_temp if s != a and s != x]
                                m = len(others_no_a)
                                if m >= 1:
                                    max_sum_m = sum(others_no_a)
                                    dp_m = [0] * (m + 1)
                                    dp_m[0] = 1
                                    for val in others_no_a:
                                        for k in range(m - 1, -1, -1):
                                            if dp_m[k]:
                                                dp_m[k+1] |= (dp_m[k] << val)
                                    for j in range(1, m + 1):
                                        target_sum = (j + 1) * a - x
                                        if 0 <= target_sum <= max_sum_m:
                                            if (dp_m[j] >> target_sum) & 1:
                                                valid = False
                                                break
                                if not valid:
                                    break
                            if valid:
                                S = S_temp + [x]
                                S_set = set(S)
                                available.remove(x)
                                available.append(old)
                                improved = True
                                break
                    if improved:
                        break

    return best_set
