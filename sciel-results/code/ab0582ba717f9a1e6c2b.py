import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute prime factors for all numbers up to 20000 using sieve
    def build_pf(limit):
        pf = [set() for _ in range(limit + 1)]
        for p in range(2, limit + 1):
            if not pf[p]:  # p is prime if no factor found yet
                for m in range(p, limit + 1, p):
                    pf[m].add(p)
        return pf

    pf = build_pf(20000)

    def get_primes_for_sums(nums):
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(pf[nums[i] + nums[j]])
        return all_primes

    # Precomputed candidate
    candidate = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]
    best_set = list(candidate)
    best_count = len(get_primes_for_sums(candidate))

    def smooth_numbers(primes, max_val):
        """Generate all numbers with only prime factors from 'primes' up to max_val."""
        result = set([1])
        for p in primes:
            new = set()
            for x in result:
                v = x
                while v * p <= max_val:
                    v *= p
                    new.add(v)
            result |= new
        return result

    def try_smooth_solution(prime_set):
        """Try to find 12 distinct numbers whose pairwise sums are all S-smooth."""
        ps = list(prime_set)
        max_sum = 20000
        smooth_sums = smooth_numbers(ps, max_sum)
        
        # Candidate numbers: 1..200 (small numbers more likely to have smooth sums)
        candidates = list(range(1, 201))
        
        # Greedy with backtracking-lite: try to pick 12
        chosen = []
        for x in candidates:
            if len(chosen) >= 12:
                break
            ok = True
            for c in chosen:
                if (x + c) not in smooth_sums:
                    ok = False
                    break
            if ok:
                chosen.append(x)
        
        if len(chosen) == 12:
            return chosen
        return None

    # Try 2-prime sets with more small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    if time.time() < deadline - 15:
        for i in range(len(small_primes)):
            for j in range(i + 1, len(small_primes)):
                if time.time() > deadline - 15:
                    break
                ps = (small_primes[i], small_primes[j])
                sol = try_smooth_solution(ps)
                if sol is not None:
                    c = len(get_primes_for_sums(sol))
                    if c < best_count:
                        best_count = c
                        best_set = sol
                    if best_count <= 2:
                        break
            if best_count <= 2:
                break

    # Also try 3-prime sets if we haven't found a 2-prime solution
    if best_count > 2 and time.time() < deadline - 15:
        triplets = [(2, 3, 5), (2, 3, 7), (2, 3, 11), (2, 5, 7), (3, 5, 7)]
        for ps in triplets:
            if time.time() > deadline - 15:
                break
            sol = try_smooth_solution(ps)
            if sol is not None:
                c = len(get_primes_for_sums(sol))
                if c < best_count:
                    best_count = c
                    best_set = sol
                if best_count <= 3:
                    break

    # Enhanced local search with incremental updates
    def local_search(start, max_time):
        current = list(start)
        current_primes = get_primes_for_sums(current)
        current_count = len(current_primes)
        
        improved = True
        iterations = 0
        while improved and time.time() < max_time:
            improved = False
            iterations += 1
            for idx in range(12):
                if time.time() > max_time:
                    break
                old_val = current[idx]
                # Compute primes contributed by pairs involving idx
                old_pair_primes = set()
                for j in range(12):
                    if j != idx:
                        old_pair_primes.update(pf[old_val + current[j]])
                
                # Try replacements
                for offset in range(1, 500):
                    for new_val in [old_val + offset, old_val - offset]:
                        if new_val < 1 or new_val > 10000:
                            continue
                        if new_val == old_val:
                            continue
                        # Check distinctness
                        if any(new_val == current[j] for j in range(12) if j != idx):
                            continue
                        
                        # Compute new pair primes
                        new_pair_primes = set()
                        for j in range(12):
                            if j != idx:
                                new_pair_primes.update(pf[new_val + current[j]])
                        
                        # New total primes = (current_primes - old_pair_primes) | new_pair_primes
                        remaining = current_primes - old_pair_primes
                        test_primes = remaining | new_pair_primes
                        
                        if len(test_primes) < current_count:
                            current[idx] = new_val
                            current_primes = test_primes
                            current_count = len(test_primes)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return current, current_count

    if time.time() < deadline - 5:
        best_set, best_count = local_search(best_set, deadline - 5)

    # Second pass local search with wider neighborhood
    if time.time() < deadline - 3 and best_count > 2:
        def wider_local_search(start, max_time):
            current = list(start)
            current_primes = get_primes_for_sums(current)
            current_count = len(current_primes)
            
            improved = True
            while improved and time.time() < max_time:
                improved = False
                for idx in range(12):
                    if time.time() > max_time:
                        break
                    old_val = current[idx]
                    old_pair_primes = set()
                    for j in range(12):
                        if j != idx:
                            old_pair_primes.update(pf[old_val + current[j]])
                    
                    # Try a wider set of replacements
                    candidates = list(range(1, min(old_val + 1000, 10001)))
                    # Shuffle for diversity
                    import random
                    random.seed(idx * 1000 + int(time.time() * 1000))
                    random.shuffle(candidates)
                    
                    for new_val in candidates:
                        if new_val == old_val:
                            continue
                        if any(new_val == current[j] for j in range(12) if j != idx):
                            continue
                        
                        new_pair_primes = set()
                        for j in range(12):
                            if j != idx:
                                new_pair_primes.update(pf[new_val + current[j]])
                        
                        remaining = current_primes - old_pair_primes
                        test_primes = remaining | new_pair_primes
                        
                        if len(test_primes) < current_count:
                            current[idx] = new_val
                            current_primes = test_primes
                            current_count = len(test_primes)
                            improved = True
                            break
                    if improved:
                        break
            
            return current, current_count
        
        best_set, best_count = wider_local_search(best_set, deadline - 3)

    # Final validation
    assert len(set(best_set)) == 12, f"Not 12 distinct: {best_set}"
    assert all(1 <= x <= 10000 for x in best_set), f"Out of range: {best_set}"

    return best_set
