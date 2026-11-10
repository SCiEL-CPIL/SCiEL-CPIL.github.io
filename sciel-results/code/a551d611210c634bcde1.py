import time
import random

def construct():
    deadline = time.time() + 85.0  # safety margin under 90s
    N = 256
    MAX_SIZE = 60

    def check_valid(vals):
        """Check if vals satisfy the cardinality-separated subset sum property.
        Returns True if valid. Uses bitset DP per cardinality."""
        if not vals:
            return True
        n = len(vals)
        max_sum = sum(vals)
        # sums[k] is a bitset where bit s is set if sum s is achievable with exactly k elements
        sums = [0] * (n + 1)
        sums[0] = 1  # sum 0 with 0 elements
        for v in vals:
            for k in range(n - 1, -1, -1):
                if sums[k]:
                    sums[k + 1] |= (sums[k] << v)
        
        # Check pairwise disjointness of sums[k] for different k
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums[i] & sums[j]:
                    return False
        return True

    def check_intervals(vals):
        """Fast sufficient check: min/max sum intervals for different cardinalities are disjoint.
        Returns True if valid by this sufficient condition."""
        if not vals:
            return True
        n = len(vals)
        sorted_vals = sorted(vals)
        
        # Precompute prefix sums for min sums
        # M[k] = sum of smallest k elements
        # X[k] = sum of largest k elements
        
        # We need X[i] < M[j] for all i < j.
        # It suffices to check X[k] < M[k+1] for all k=1..n-1.
        
        # M[k]
        M = [0] * (n + 1)
        for i in range(1, n + 1):
            M[i] = M[i-1] + sorted_vals[i-1]
            
        # X[k]
        # X[k] = sum of largest k elements = TotalSum - sum of smallest n-k elements
        total_sum = M[n]
        # Alternatively compute directly
        X = [0] * (n + 1)
        for i in range(1, n + 1):
            X[i] = X[i-1] + sorted_vals[n-i]
            
        for k in range(1, n):
            if not (X[k] < M[k+1]):
                return False
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

    # Strategy 5: Randomized attempts (Fast)
    remaining_time = deadline - time.time()
    if remaining_time > 2:
        rng = random.Random(42)
        # Estimate how many iterations we can afford
        # Fast check is O(N^2) ~ 3600 ops. 
        # Exact check is much slower.
        # We can do many more fast iterations.
        iterations = min(500, int(remaining_time * 50))
        rand_fast = try_randomized(pool, MAX_SIZE, iterations, rng, fast_check=True)
        if check_valid(rand_fast) and len(rand_fast) > len(best_exact):
            best_exact = rand_fast

    # Strategy 6: Random small seeds then grow (Fast)
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
