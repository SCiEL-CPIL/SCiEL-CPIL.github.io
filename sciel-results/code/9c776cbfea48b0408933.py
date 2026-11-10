import time
import random
import numpy as np

def construct():
    deadline = time.time() + 85.0  # safety margin under 90s
    N = 256
    MAX_SIZE = 60

    def check_valid(vals):
        """Check if vals satisfy the cardinality-separated subset sum property.
        Returns True if valid. Uses optimized bitset DP per cardinality."""
        if not vals:
            return True
        n = len(vals)
        if n == 1:
            return True
        max_sum = sum(vals)
        # sums[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
        # Use Python integers as bitsets (arbitrary precision)
        sums = [0] * (n + 1)
        sums[0] = 1  # sum 0 with 0 elements
        
        for v in vals:
            # Process from high k to low k to avoid using v multiple times
            for k in range(min(n - 1, n - 1), -1, -1):
                if sums[k]:
                    sums[k + 1] |= (sums[k] << v)
        
        # Check pairwise disjointness of sums[k] for different k
        # Optimization: accumulate union and check overlap incrementally
        union = 0
        for i in range(n + 1):
            if sums[i] & union:
                return False
            union |= sums[i]
        return True

    def check_intervals(vals):
        """Fast sufficient check: min/max sum intervals for different cardinalities are disjoint.
        Returns True if valid by this sufficient condition."""
        if not vals:
            return True
        n = len(vals)
        if n == 1:
            return True
        sorted_vals = sorted(vals)
        
        # M[k] = sum of smallest k elements
        # X[k] = sum of largest k elements
        
        # We need X[i] < M[j] for all i < j.
        # It suffices to check X[k] < M[k+1] for all k=1..n-1.
        
        # M[k]
        M = [0] * (n + 1)
        for i in range(1, n + 1):
            M[i] = M[i-1] + sorted_vals[i-1]
            
        # X[k]
        X = [0] * (n + 1)
        for i in range(1, n + 1):
            X[i] = X[i-1] + sorted_vals[n-i]
            
        for k in range(1, n):
            if not (X[k] < M[k+1]):
                return False
        return True

    def check_partial_dp(vals, max_card=12):
        """Faster check using DP only up to max_card cardinality.
        If we only need to verify that no collisions occur up to max_card,
        this is much faster than full DP for large sets."""
        if not vals:
            return True
        n = len(vals)
        if n <= 2:
            return True
        
        mc = min(n, max_card)
        sums = [0] * (mc + 1)
        sums[0] = 1
        
        for v in vals:
            for k in range(mc - 1, -1, -1):
                if sums[k]:
                    sums[k + 1] |= (sums[k] << v)
        
        # Check disjointness for the computed cardinalities
        union = 0
        for i in range(mc + 1):
            if sums[i] & union:
                return False
            union |= sums[i]
        return True

    def try_greedy_descending(pool, max_size, fast_check=False):
        """Try to build a valid set by greedily adding elements from largest to smallest."""
        result = []
        for v in sorted(pool, reverse=True):
            if len(result) >= max_size:
                break
            candidate = result + [v]
            if fast_check:
                if check_intervals(candidate):
                    result = candidate
            else:
                if check_valid(candidate):
                    result = candidate
        return result

    def try_greedy_ascending(pool, max_size, fast_check=False):
        """Try to build a valid set by greedily adding elements from smallest to largest."""
        result = []
        for v in sorted(pool):
            if len(result) >= max_size:
                break
            candidate = result + [v]
            if fast_check:
                if check_intervals(candidate):
                    result = candidate
            else:
                if check_valid(candidate):
                    result = candidate
        return result

    def try_grow(initial, all_vals, fast_check=False):
        """Grow a valid initial set by trying to add remaining values (smallest first)."""
        cur = list(initial)
        cur_set = set(cur)
        for v in sorted(all_vals):
            if v in cur_set:
                continue
            cand = cur + [v]
            if fast_check:
                if check_intervals(cand):
                    cur = cand
                    cur_set.add(v)
            else:
                if check_valid(cand):
                    cur = cand
                    cur_set.add(v)
        return cur

    def try_grow_with_dp(initial, all_vals, dp_max_card=15):
        """Grow using partial DP check which is less conservative than intervals."""
        cur = list(initial)
        cur_set = set(cur)
        for v in sorted(all_vals):
            if v in cur_set:
                continue
            cand = cur + [v]
            if check_partial_dp(cand, dp_max_card):
                cur = cand
                cur_set.add(v)
        return cur

    def try_randomized(pool, max_size, iterations, rng, fast_check=False):
        """Try randomized orderings to find larger valid sets."""
        best = []
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break
            shuffled = pool[:]
            rng.shuffle(shuffled)
            result = []
            for v in shuffled:
                if len(result) >= max_size:
                    break
                candidate = result + [v]
                if fast_check:
                    if check_intervals(candidate):
                        result = candidate
                else:
                    if check_valid(candidate):
                        result = candidate
            if len(result) > len(best):
                best = result
                if len(best) >= max_size:
                    break
        return best

    def try_randomized_dp(pool, max_size, iterations, rng, dp_max_card=15):
        """Try randomized orderings using partial DP check."""
        best = []
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break
            shuffled = pool[:]
            rng.shuffle(shuffled)
            result = []
            for v in shuffled:
                if len(result) >= max_size:
                    break
                candidate = result + [v]
                if check_partial_dp(candidate, dp_max_card):
                    result = candidate
            if len(result) > len(best):
                best = result
                if len(best) >= max_size:
                    break
        return best

    def try_local_search(initial, pool, max_size, iterations, rng):
        """Local search: try to swap elements to improve the set."""
        cur = list(initial)
        cur_set = set(cur)
        best = list(cur)
        
        for _ in range(iterations):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try removing one element and adding two others
            if len(cur) >= 2:
                for i in range(len(cur)):
                    if time.time() > deadline - 3:
                        break
                    candidate_remove = cur[:i] + cur[i+1:]
                    if not check_intervals(candidate_remove):
                        continue
                    
                    # Try adding two new elements
                    candidates = [v for v in pool if v not in cur_set]
                    rng.shuffle(candidates)
                    for a in candidates[:20]:
                        for b in candidates[:20]:
                            if a == b:
                                continue
                            cand = candidate_remove + [a, b]
                            if len(set(cand)) != len(cand):
                                continue
                            if check_intervals(cand):
                                if len(cand) > len(best):
                                    best = list(cand)
                                cur = list(cand)
                                cur_set = set(cand)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Try adding one element if possible
            if not improved and len(cur) < max_size:
                candidates = [v for v in pool if v not in cur_set]
                rng.shuffle(candidates)
                for v in candidates[:30]:
                    cand = cur + [v]
                    if check_intervals(cand):
                        cur = list(cand)
                        cur_set = set(cand)
                        if len(cur) > len(best):
                            best = list(cur)
                        improved = True
                        break
        
        return best

    def try_local_search_dp(initial, pool, max_size, iterations, rng, dp_max_card=15):
        """Local search using partial DP check instead of intervals."""
        cur = list(initial)
        cur_set = set(cur)
        best = list(cur)
        
        for _ in range(iterations):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try removing one element and adding two others
            if len(cur) >= 2:
                for i in range(len(cur)):
                    if time.time() > deadline - 3:
                        break
                    candidate_remove = cur[:i] + cur[i+1:]
                    if not check_partial_dp(candidate_remove, dp_max_card):
                        continue
                    
                    # Try adding two new elements
                    candidates = [v for v in pool if v not in cur_set]
                    rng.shuffle(candidates)
                    for a in candidates[:15]:
                        for b in candidates[:15]:
                            if a == b:
                                continue
                            cand = candidate_remove + [a, b]
                            if len(set(cand)) != len(cand):
                                continue
                            if check_partial_dp(cand, dp_max_card):
                                if len(cand) > len(best):
                                    best = list(cand)
                                cur = list(cand)
                                cur_set = set(cand)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # Try adding one element if possible
            if not improved and len(cur) < max_size:
                candidates = [v for v in pool if v not in cur_set]
                rng.shuffle(candidates)
                for v in candidates[:25]:
                    cand = cur + [v]
                    if check_partial_dp(cand, dp_max_card):
                        cur = list(cand)
                        cur_set = set(cand)
                        if len(cur) > len(best):
                            best = list(cur)
                        improved = True
                        break
        
        return best

    pool = list(range(1, N + 1))
    best_exact = []
    
    # Strategy 1: Superincreasing baseline (Exact)
    sup = [1]
    for i in range(1, 60):
        nxt = sum(sup) + 1
        if nxt <= N:
            sup.append(nxt)
        else:
            break
    if check_valid(sup):
        best_exact = list(sup)
    
    # Strategy 2: Descending greedy (Fast)
    desc_fast = try_greedy_descending(pool, MAX_SIZE, fast_check=True)
    if check_valid(desc_fast) and len(desc_fast) > len(best_exact):
        best_exact = desc_fast
        
    # Strategy 3: Ascending greedy (Fast)
    asc_fast = try_greedy_ascending(pool, MAX_SIZE, fast_check=True)
    if check_valid(asc_fast) and len(asc_fast) > len(best_exact):
        best_exact = asc_fast

    # Strategy 4: Grow from superincreasing (Fast)
    grown_fast = try_grow(sup, pool, fast_check=True)
    if check_valid(grown_fast) and len(grown_fast) > len(best_exact):
        best_exact = grown_fast

    # Strategy 5: Grow from superincreasing using partial DP (less conservative)
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        grown_dp = try_grow_with_dp(sup, pool, dp_max_card=15)
        if check_valid(grown_dp) and len(grown_dp) > len(best_exact):
            best_exact = grown_dp

    # Strategy 6: Randomized attempts (Fast)
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        rng = random.Random(42)
        iterations = min(500, int(remaining_time * 50))
        rand_fast = try_randomized(pool, MAX_SIZE, iterations, rng, fast_check=True)
        if check_valid(rand_fast) and len(rand_fast) > len(best_exact):
            best_exact = rand_fast

    # Strategy 7: Randomized attempts using partial DP
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        rng = random.Random(777)
        iterations = min(300, int(remaining_time * 30))
        rand_dp = try_randomized_dp(pool, MAX_SIZE, iterations, rng, dp_max_card=15)
        if check_valid(rand_dp) and len(rand_dp) > len(best_exact):
            best_exact = rand_dp

    # Strategy 8: Random small seeds then grow (Fast)
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        rng = random.Random(12345)
        for trial in range(min(200, int(remaining_time * 20))):
            if time.time() > deadline - 1:
                break
            size = rng.randint(1, 8)
            seed_vals = rng.sample(range(1, N + 1), size)
            if not check_intervals(seed_vals):
                continue
            cur = try_grow(sorted(seed_vals), pool, fast_check=True)
            if check_valid(cur) and len(cur) > len(best_exact):
                best_exact = cur

    # Strategy 9: Local search from best found so far (interval-based)
    remaining_time = deadline - time.time()
    if remaining_time > 3 and len(best_exact) > 0:
        rng = random.Random(999)
        ls_result = try_local_search(best_exact, pool, MAX_SIZE, min(50, int(remaining_time * 5)), rng)
        if check_valid(ls_result) and len(ls_result) > len(best_exact):
            best_exact = ls_result

    # Strategy 10: Local search using partial DP (less conservative)
    remaining_time = deadline - time.time()
    if remaining_time > 3 and len(best_exact) > 0:
        rng = random.Random(888)
        ls_dp = try_local_search_dp(best_exact, pool, MAX_SIZE, min(40, int(remaining_time * 4)), rng, dp_max_card=15)
        if check_valid(ls_dp) and len(ls_dp) > len(best_exact):
            best_exact = ls_dp

    # Strategy 11: Try to grow the best result further using exact check
    remaining_time = deadline - time.time()
    if remaining_time > 5 and len(best_exact) > 0 and len(best_exact) < MAX_SIZE:
        cur = list(best_exact)
        cur_set = set(cur)
        for v in sorted(pool):
            if time.time() > deadline - 2:
                break
            if v in cur_set:
                continue
            cand = cur + [v]
            if check_valid(cand):
                cur = cand
                cur_set.add(v)
        if len(cur) > len(best_exact):
            best_exact = cur

    # Final verification
    if not check_valid(best_exact):
        # Fallback to superincreasing if something went wrong
        if check_valid(sup):
            best_exact = sup
        else:
            best_exact = [1] # At least 1 is valid

    # Ensure we return a list of Python ints
    best_exact = [int(x) for x in best_exact]
    return best_exact
