import time
import random
import sys
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 58  # Safety margin for 60s limit

    # Precompute smallest prime factor for fast divisibility checks
    MAX_VAL = 5120  # max possible sum of 30 elements each <= 256
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def factorize(n):
        """Return dict of prime -> exponent for n."""
        if n <= 1:
            return {}
        factors = {}
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            factors[p] = cnt
        return factors

    def divides(a, b):
        """Check if a divides b (a != 0)."""
        if a == 0 or b == 0:
            return False
        return b % a == 0

    def compute_subset_sums(S):
        """Compute all subset sums (including 0) as a sorted list."""
        sums = [0]
        for x in S:
            sums = sums + [s + x for s in sums]
        return sums

    def is_valid_full(S):
        """Full validation: all non-empty subset sums distinct and no two divide each other."""
        if not S:
            return True
        sums = compute_subset_sums(S)
        non_empty = [s for s in sums if s != 0]
        if len(set(non_empty)) != len(non_empty):
            return False
        non_empty.sort()
        n = len(non_empty)
        for i in range(n):
            u = non_empty[i]
            for j in range(i + 1, n):
                v = non_empty[j]
                if v % u == 0:
                    return False
        return True

    def can_add(S, x, sums_cache):
        """Check if x can be added to valid set S with cached subset sums."""
        new_sums = [s + x for s in sums_cache]
        # Check distinctness: new sums must not overlap with old non-zero sums
        old_set = set(s for s in sums_cache if s != 0)
        for ns in new_sums:
            if ns != 0 and ns in old_set:
                return False
        # Check divisibility among new sums
        non_empty_new = [s for s in new_sums if s != 0]
        non_empty_new.sort()
        for i in range(len(non_empty_new)):
            u = non_empty_new[i]
            for j in range(i + 1, len(non_empty_new)):
                v = non_empty_new[j]
                if v % u == 0:
                    return False
        # Cross check: new vs old
        non_empty_old = [s for s in sums_cache if s != 0]
        for u in non_empty_new:
            for v in non_empty_old:
                if v % u == 0 or u % v == 0:
                    return False
        return True

    def greedy_build(order, sums_cache_init=None):
        """Build a set greedily following the given order."""
        S = []
        if sums_cache_init is not None:
            sums_cache = list(sums_cache_init)
        else:
            sums_cache = [0]
        for x in order:
            if time.time() - start_time > time_limit - 1:
                break
            if can_add(S, x, sums_cache):
                S.append(x)
                sums_cache = sums_cache + [s + x for s in sums_cache]
        return S

    def local_search_2opt(S, all_candidates):
        """Try 2-opt: remove one element, try to add a different one, or swap."""
        if not S:
            return S
        S = list(S)
        improved = True
        while improved and time.time() - start_time < time_limit - 3:
            improved = False
            n = len(S)
            for i in range(n):
                if time.time() - start_time > time_limit - 3:
                    break
                S_without_i = S[:i] + S[i+1:]
                sums_without_i = compute_subset_sums(S_without_i)
                for x in all_candidates:
                    if x in S:
                        continue
                    if can_add(S_without_i, x, sums_without_i):
                        S_new = S_without_i + [x]
                        if is_valid_full(S_new):
                            S = S_new
                            improved = True
                            break
                if improved:
                    break
        return S

    def prime_superincreasing_build():
        """
        Structural approach: use primes as base multipliers.
        Choose a sequence of primes p1, p2, ..., and set elements as:
        a_i = p_i * k_i where k_i is chosen to maintain superincreasing property
        and avoid divisibility conflicts.
        
        Key idea: if all elements are multiples of distinct primes, then any subset sum
        has a specific prime factorization. Two sums divide each other only if their
        prime factorizations are compatible.
        """
        # Use first N primes
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                  53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
        
        best_set = []
        best_size = 0
        
        # Try different subsets of primes and multipliers
        for start_idx in range(0, 10):
            for length in range(4, 25):
                if time.time() - start_time > time_limit - 5:
                    return best_set, best_size
                if start_idx + length > len(primes):
                    continue
                
                # Try building with these primes
                S = []
                sums_cache = [0]
                for idx in range(start_idx, start_idx + length):
                    p = primes[idx]
                    # Try different multipliers for this prime
                    found = False
                    for k in range(1, 11):
                        x = p * k
                        if x > 256 or x in S:
                            continue
                        if can_add(S, x, sums_cache):
                            S.append(x)
                            sums_cache = sums_cache + [s + x for s in sums_cache]
                            found = True
                            break
                    if not found:
                        break
                
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
        
        return best_set, best_size

    def random_superincreasing_build():
        """
        Random superincreasing construction: pick elements in increasing order
        where each new element > sum of all previous. This guarantees distinct
        subset sums. Then check divisibility.
        """
        best_set = []
        best_size = 0
        
        for trial in range(200):
            if time.time() - start_time > time_limit - 5:
                break
            
            S = []
            current_sum = 0
            remaining = list(range(1, 257))
            random.shuffle(remaining)
            
            for x in remaining:
                if x <= current_sum:
                    continue  # Must be superincreasing
                if can_add(S, x, compute_subset_sums(S)):
                    S.append(x)
                    current_sum += x
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        return best_set, best_size

    def mixed_strategy_build():
        """
        Mixed strategy: start with some small primes, then fill with larger numbers.
        This balances the distinctness guarantee with the ability to pack more elements.
        """
        best_set = []
        best_size = 0
        
        for trial in range(100):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Start with a few small distinct primes
            small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            random.shuffle(small_primes)
            S = []
            sums_cache = [0]
            
            # Add 1-3 small primes
            num_small = random.randint(1, 3)
            for p in small_primes[:num_small]:
                if can_add(S, p, sums_cache):
                    S.append(p)
                    sums_cache = sums_cache + [s + p for s in sums_cache]
            
            # Fill with larger numbers
            candidates = [x for x in range(30, 257) if x not in S]
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() - start_time > time_limit - 5:
                    break
                if can_add(S, x, sums_cache):
                    S.append(x)
                    sums_cache = sums_cache + [s + x for s in sums_cache]
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
        
        return best_set, best_size

    # Phase 1: Try multiple structural strategies
    best_set = [31, 32, 34]
    best_size = 3

    # Strategy 1: Prime superincreasing
    s1, size1 = prime_superincreasing_build()
    if size1 > best_size:
        best_size = size1
        best_set = s1[:]

    # Strategy 2: Random superincreasing
    s2, size2 = random_superincreasing_build()
    if size2 > best_size:
        best_size = size2
        best_set = s2[:]

    # Strategy 3: Mixed strategy
    s3, size3 = mixed_strategy_build()
    if size3 > best_size:
        best_size = size3
        best_set = s3[:]

    # Phase 2: Random greedy builds with various orderings
    all_candidates = list(range(1, 257))
    for restart in range(300):
        if time.time() - start_time > time_limit - 8:
            break
        random.shuffle(all_candidates)
        S = greedy_build(all_candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    # Phase 3: Local search from best set
    if time.time() - start_time < time_limit - 5:
        improved_set = local_search_2opt(best_set, all_candidates)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set[:]

    # Phase 4: Try to extend best set with targeted additions
    if time.time() - start_time < time_limit - 3 and best_size < 20:
        S = list(best_set)
        sums_cache = compute_subset_sums(S)
        # Try adding elements in order of "niceness"
        for x in sorted(all_candidates, reverse=True):
            if time.time() - start_time > time_limit - 3:
                break
            if x in S:
                continue
            if can_add(S, x, sums_cache):
                S.append(x)
                sums_cache = sums_cache + [s + x for s in sums_cache]
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]

    # Final verification
    if not is_valid_full(best_set):
        best_set = [31, 32, 34]
        best_size = 3

    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
    print(f"Size: {len(res)}")
