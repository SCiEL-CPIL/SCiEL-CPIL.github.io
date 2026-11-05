import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute primes up to 20000 (max sum is 10000+9999=19999)
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]

    primes_up_to = sieve(20000)

    # Precompute prime factors for all numbers up to 20000
    # pf[x] = frozenset of prime factors of x (use frozenset for hashing)
    pf = [frozenset() for _ in range(20001)]
    pf_list = [set() for _ in range(20001)]
    for p in primes_up_to:
        for multiple in range(p, 20001, p):
            pf_list[multiple].add(p)
    for i in range(20001):
        pf[i] = frozenset(pf_list[i])

    def get_primes_for_sums(nums):
        """Get union of all prime factors of pairwise sums. Returns a set."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                s = nums[i] + nums[j]
                all_primes.update(pf[s])
        return all_primes

    best_count = float('inf')
    best_set = None

    # M-multiple strategy:
    # If all a_i = M * k_i, then a_i + a_j = M * (k_i + k_j)
    # Prime factors of (a_i + a_j) = pf[M] ∪ pf[k_i + k_j]
    # Total union = pf[M] ∪ (union over all pairs of pf[k_i + k_j])
    #
    # We want to choose M and 12 distinct k_i (with M*k_i <= 10000)
    # to minimize |pf[M] ∪ union of pf[k_i+k_j]|
    #
    # Key: choose M to be a product of small primes, so pf[M] is small
    # and M absorbs those primes from the sums.
    # Then we need to find k_i such that k_i+k_j only involve a few primes.

    # Try various M values that are products of small primes
    # M should be large enough that 12*M <= 10000 (so at least 12 multiples available)
    # M <= 833

    # Candidate M values: products of subsets of {2,3,5,7,11,13,...}
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Generate candidate M values
    candidate_Ms = set()
    for r in range(1, 5):  # up to 4 prime factors
        for combo in combinations(small_primes, r):
            m = 1
            for p in combo:
                m *= p
            if m <= 833:
                candidate_Ms.add(m)
    
    # Also add some structured M values
    for p in small_primes:
        for power in range(1, 6):
            m = p ** power
            if m <= 833:
                candidate_Ms.add(m)
    
    candidate_Ms = sorted(candidate_Ms)
    
    # For each M, we need to find 12 distinct k_i in [1, floor(10000/M)]
    # such that the union of pf[k_i + k_j] over all pairs is minimized
    # Total = pf[M] ∪ (that union)
    
    # Precompute for each possible M, the "cost" of choosing a set of k values
    # Strategy: for each M, try to find k_i such that all k_i + k_j are "smooth"
    # (have few prime factors)
    
    # Better approach: for each M, greedily build a set of k values
    # that minimize the number of distinct primes in all pairwise sums
    
    def solve_for_M(M, pf_M, max_k):
        """Find 12 distinct k_i in [1, max_k] minimizing |pf_M ∪ union pf[k_i+k_j]|"""
        if max_k < 12:
            return None, float('inf')
        
        # Greedy approach: start with a candidate set, then try to improve
        # Initial guess: k_i = i for i in 1..12 (if max_k >= 12)
        if max_k < 12:
            return None, float('inf')
        
        # Try several initial configurations
        best_k_set = None
        best_prime_count = float('inf')
        
        # Configuration 1: k = 1, 2, ..., 12
        if max_k >= 12:
            k_init = list(range(1, 13))
            primes_union = set(pf_M)
            for i in range(12):
                for j in range(i + 1, 12):
                    s = k_init[i] + k_init[j]
                    primes_union.update(pf[s])
            count = len(primes_union)
            if count < best_prime_count:
                best_prime_count = count
                best_k_set = k_init[:]
        
        # Configuration 2: k = 1, 3, 5, ..., 23 (odd numbers)
        if max_k >= 23:
            k_init = list(range(1, 24, 2))[:12]
            primes_union = set(pf_M)
            for i in range(12):
                for j in range(i + 1, 12):
                    s = k_init[i] + k_init[j]
                    primes_union.update(pf[s])
            count = len(primes_union)
            if count < best_prime_count:
                best_prime_count = count
                best_k_set = k_init[:]
        
        # Configuration 3: k = 2, 4, 6, ..., 24 (even numbers)
        if max_k >= 24:
            k_init = list(range(2, 25, 2))[:12]
            primes_union = set(pf_M)
            for i in range(12):
                for j in range(i + 1, 12):
                    s = k_init[i] + k_init[j]
                    primes_union.update(pf[s])
            count = len(primes_union)
            if count < best_prime_count:
                best_prime_count = count
                best_k_set = k_init[:]
        
        # Configuration 4: k = 1, 4, 7, 10, ... (arithmetic progression mod 3)
        if max_k >= 1 + 3*11:
            k_init = [1 + 3*i for i in range(12)]
            primes_union = set(pf_M)
            for i in range(12):
                for j in range(i + 1, 12):
                    s = k_init[i] + k_init[j]
                    primes_union.update(pf[s])
            count = len(primes_union)
            if count < best_prime_count:
                best_prime_count = count
                best_k_set = k_init[:]
        
        # Configuration 5: k = 1, 5, 9, 13, ... (mod 4)
        if max_k >= 1 + 4*11:
            k_init = [1 + 4*i for i in range(12)]
            primes_union = set(pf_M)
            for i in range(12):
                for j in range(i + 1, 12):
                    s = k_init[i] + k_init[j]
                    primes_union.update(pf[s])
            count = len(primes_union)
            if count < best_prime_count:
                best_prime_count = count
                best_k_set = k_init[:]
        
        # Configuration 6: k = 2, 6, 10, 14, ... (mod 4, even)
        if max_k >= 2 + 4*11:
            k_init = [2 + 4*i for i in range(12)]
            primes_union = set(pf_M)
            for i in range(12):
                for j in range(i + 1, 12):
                    s = k_init[i] + k_init[j]
                    primes_union.update(pf[s])
            count = len(primes_union)
            if count < best_prime_count:
                best_prime_count = count
                best_k_set = k_init[:]
        
        # Local search on best configuration
        if best_k_set is not None and time.time() < deadline - 10:
            current = best_k_set[:]
            current_primes = set(pf_M)
            for i in range(12):
                for j in range(i + 1, 12):
                    current_primes.update(pf[current[i] + current[j]])
            current_count = len(current_primes)
            
            # Try to improve by replacing individual k values
            for iteration in range(50):
                if time.time() > deadline - 10:
                    break
                improved = False
                for idx in range(12):
                    old_k = current[idx]
                    # Try replacing with other values
                    for new_k in range(1, min(max_k + 1, old_k + 30)):
                        if new_k == old_k or new_k in current:
                            continue
                        test = current[:]
                        test[idx] = new_k
                        test_primes = set(pf_M)
                        for i in range(12):
                            for j in range(i + 1, 12):
                                test_primes.update(pf[test[i] + test[j]])
                        if len(test_primes) < current_count:
                            current = test
                            current_primes = test_primes
                            current_count = len(test_primes)
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    break
            
            if current_count < best_prime_count:
                best_prime_count = current_count
                best_k_set = current[:]
        
        return best_k_set, best_prime_count

    # Try each candidate M
    for M in candidate_Ms:
        if time.time() > deadline - 15:
            break
        
        max_k = 10000 // M
        if max_k < 12:
            continue
        
        pf_M = pf[M]
        k_set, prime_count = solve_for_M(M, pf_M, max_k)
        
        if k_set is not None and prime_count < best_count:
            best_count = prime_count
            best_set = [M * k for k in k_set]
    
    # Fallback: try the original modular approach if M-multiple didn't find anything good
    if best_set is None or best_count > 15:
        # All ≡ 1 mod 210
        candidate_mod210 = [1 + 210 * i for i in range(12)]
        primes_c1 = get_primes_for_sums(candidate_mod210)
        if len(primes_c1) < best_count:
            best_count = len(primes_c1)
            best_set = list(candidate_mod210)
        
        # All ≡ 1 mod 105
        candidate_mod105 = [1 + 105 * i for i in range(12)]
        primes_c6 = get_primes_for_sums(candidate_mod105)
        if len(primes_c6) < best_count:
            best_count = len(primes_c6)
            best_set = list(candidate_mod105)
        
        # All ≡ 1 mod 30
        candidate_mod30 = [1 + 30 * i for i in range(12)]
        primes_c2 = get_primes_for_sums(candidate_mod30)
        if len(primes_c2) < best_count:
            best_count = len(primes_c2)
            best_set = list(candidate_mod30)
        
        # Powers of 2
        powers_of_2 = [2**i for i in range(12)]
        if all(x <= 10000 for x in powers_of_2):
            primes_c7 = get_primes_for_sums(powers_of_2)
            if len(primes_c7) < best_count:
                best_count = len(primes_c7)
                best_set = list(powers_of_2)
        
        # Multiples of 64
        multiples_64 = [64 * (i + 1) for i in range(12)]
        primes_c8 = get_primes_for_sums(multiples_64)
        if len(primes_c8) < best_count:
            best_count = len(primes_c8)
            best_set = list(multiples_64)
    
    # Final local search on best_set if we have time
    if best_set is not None and time.time() < deadline - 20:
        current = list(best_set)
        current_primes = get_primes_for_sums(current)
        current_count = len(current_primes)
        used = set(current)
        
        for iteration in range(200):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for idx in range(12):
                old_val = current[idx]
                # Try several replacement values
                candidates = []
                for offset in range(-50, 51):
                    nv = old_val + offset
                    if 1 <= nv <= 10000 and nv != old_val and nv not in used:
                        candidates.append(nv)
                # Also try some random-ish values
                for seed in range(10):
                    nv = 1 + (iteration * 137 + idx * 31 + seed * 17) % 10000
                    if 1 <= nv <= 10000 and nv != old_val and nv not in used:
                        candidates.append(nv)
                
                for new_val in candidates:
                    if new_val in used:
                        continue
                    test_set = list(current)
                    test_set[idx] = new_val
                    test_primes = get_primes_for_sums(test_set)
                    
                    if len(test_primes) < current_count:
                        current = test_set
                        current_primes = test_primes
                        current_count = len(test_primes)
                        used.discard(old_val)
                        used.add(new_val)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        if current_count < best_count:
            best_count = current_count
            best_set = current
    
    # Final check: ensure validity
    if best_set is None:
        # Absolute fallback
        best_set = list(range(1, 13))
    
    assert len(set(best_set)) == 12
    assert all(1 <= x <= 10000 for x in best_set)
    
    return best_set
