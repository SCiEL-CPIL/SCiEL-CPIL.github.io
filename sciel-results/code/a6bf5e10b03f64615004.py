import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

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
    prime_set = set(primes_up_to)

    # Precompute prime factors for all numbers up to 20000
    # pf[x] = set of prime factors of x
    pf = [set() for _ in range(20001)]
    for p in primes_up_to:
        for multiple in range(p, 20001, p):
            pf[multiple].add(p)

    def get_primes_for_sums(nums):
        """Get union of all prime factors of pairwise sums."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                s = nums[i] + nums[j]
                all_primes.update(pf[s])
        return all_primes

    # KEY INSIGHT: Find 12 distinct numbers where ALL pairwise sums are S-smooth
    # for a small prime set S. This bounds the answer to |S| distinct primes.
    
    # Strategy: For a given prime set S, find all numbers n <= 10000 such that
    # for every other chosen number m, n+m is S-smooth.
    # 
    # Simpler approach: If all numbers are in a set A where A+A subset S-smooth numbers,
    # then any pairwise sum is S-smooth.
    #
    # Even simpler: if all numbers are multiples of some base B, and B*(a+b) is S-smooth
    # for all pairs, we need {a+b} to only introduce primes in S.
    
    # BEST APPROACH: Use the fact that if all numbers are powers of 2 times small constants,
    # or better: find 12 numbers where all pairwise sums have prime factors from {2,3} only.
    
    # Numbers n such that n+m is 2,3-smooth for all m in our set.
    # 2,3-smooth numbers (Hamming numbers) up to 20000:
    # 2^a * 3^b for a,b >= 0, value <= 20000
    
    smooth_23 = set()
    for a in range(15):  # 2^14 = 16384
        val = 1 << a
        if val > 20000:
            break
        for b in range(10):  # 3^9 = 19683
            v = val * (3 ** b)
            if v > 20000:
                break
            smooth_23.add(v)
    
    # Now find 12 distinct numbers n in [1,10000] such that for all pairs, n_i + n_j in smooth_23
    # This is a constraint satisfaction problem. Let's try greedy.
    
    def find_smooth_set(smooth_set, n_needed=12, max_val=10000):
        """Find n_needed distinct numbers in [1,max_val] where all pairwise sums are in smooth_set."""
        if time.time() > deadline - 5:
            return None
        
        # Try to build the set greedily
        chosen = []
        # Start with small numbers that are themselves differences of smooth numbers
        # n_i + n_j must be smooth, so n_i = smooth1 - n_j
        
        # Alternative: pick numbers from a structured family
        # If all numbers are of form k*2^a, sums are 2^min * (k1*2^(a1-min) + k2*2^(a2-min))
        
        # Let's try: all numbers are powers of 2: 1,2,4,8,...
        # Sum: 2^a + 2^b = 2^min * (1 + 2^|a-b|)
        # For this to be 2,3-smooth, need 1+2^k to be 3-smooth (only prime factor 3)
        # 1+2^1=3 ✓, 1+2^2=5 ✗, 1+2^3=9=3^2 ✓, 1+2^4=17 ✗
        # So powers of 2 with differences 1 or 3 work for the odd part.
        
        # Better: just search directly
        # Candidate pool: numbers n where n + m is smooth for many m
        # 
        # Let's try a different angle: use numbers that are all small multiples
        # of a common base, so sums stay in a controlled range.
        
        # Try: numbers = [c * s for s in some smooth-friendly set]
        # If c=1: need all pairwise sums of our 12 numbers to be 2,3-smooth
        # 2,3-smooth numbers up to 20000 are relatively dense near the bottom.
        
        # Let's just brute-force: try all subsets of size 12 from a candidate pool
        # Candidate pool: numbers n in [1, 200] (small numbers have more smooth sums)
        
        pool = list(range(1, 201))
        
        # Precompute: for each pair (i,j) in pool, is pool[i]+pool[j] in smooth_set?
        # Use bitset or just check
        
        best = None
        best_count = 999
        
        # Greedy approach: start with smallest, add next number that maintains smoothness
        for start in [1, 2, 3]:
            current = [start]
            for _ in range(n_needed - 1):
                # Find best next number to add
                best_next = None
                best_score = 999
                for cand in pool:
                    if cand in current:
                        continue
                    # Check if cand + all existing are smooth
                    ok = True
                    for existing in current:
                        if (cand + existing) not in smooth_set:
                            ok = False
                            break
                    if ok:
                        # Score: prefer numbers that leave more options (heuristic: smaller is better for now)
                        score = cand
                        if score < best_score:
                            best_score = score
                            best_next = cand
                if best_next is None:
                    break
                current.append(best_next)
            
            if len(current) == n_needed:
                # Verify
                valid = True
                for i in range(n_needed):
                    for j in range(i+1, n_needed):
                        if (current[i] + current[j]) not in smooth_set:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    primes = get_primes_for_sums(current)
                    if len(primes) < best_count:
                        best_count = len(primes)
                        best = list(current)
        
        return best
    
    # Try finding a 2,3-smooth solution
    best_set = None
    best_count = 999
    
    result_23 = find_smooth_set(smooth_23)
    if result_23 is not None:
        primes_23 = get_primes_for_sums(result_23)
        best_set = result_23
        best_count = len(primes_23)
    
    # Also try 2,3,5-smooth
    smooth_235 = set()
    for a in range(15):
        val = 1 << a
        if val > 20000:
            break
        for b in range(10):
            v = val * (3 ** b)
            if v > 20000:
                break
            for c in range(8):
                w = v * (5 ** c)
                if w > 20000:
                    break
                smooth_235.add(w)
    
    if best_count > 3:  # Only try if 2,3 didn't work
        result_235 = find_smooth_set(smooth_235)
        if result_235 is not None:
            primes_235 = get_primes_for_sums(result_235)
            if len(primes_235) < best_count:
                best_count = len(primes_235)
                best_set = result_235
    
    # Fallback to original heuristic approaches if smooth search failed
    if best_set is None:
        candidate_mod210 = [1 + 210 * i for i in range(12)]
        primes_c1 = get_primes_for_sums(candidate_mod210)
        best_count = len(primes_c1)
        best_set = list(candidate_mod210)
        
        candidate_mod30 = [1 + 30 * i for i in range(12)]
        primes_c2 = get_primes_for_sums(candidate_mod30)
        if len(primes_c2) < best_count:
            best_count = len(primes_c2)
            best_set = list(candidate_mod30)
    
    # Local search to improve
    if time.time() < deadline - 10 and best_set is not None:
        def local_search(start, budget=30):
            if time.time() > deadline - 5:
                return list(start), best_count
            
            current = list(start)
            current_primes = get_primes_for_sums(current)
            current_count = len(current_primes)
            
            for _ in range(budget):
                if time.time() > deadline - 5:
                    break
                
                improved = False
                for idx in range(12):
                    old_val = current[idx]
                    for new_val in range(1, min(10001, old_val + 500)):
                        if new_val == old_val:
                            continue
                        if new_val in current:
                            continue
                        
                        test_set = list(current)
                        test_set[idx] = new_val
                        test_primes = get_primes_for_sums(test_set)
                        
                        if len(test_primes) < current_count:
                            current = test_set
                            current_primes = test_primes
                            current_count = len(test_primes)
                            improved = True
                            break
                    
                    if improved:
                        break
                
                if not improved:
                    break
        
        best_set, best_count = local_search(best_set, budget=20)
    
    # Final check: ensure validity
    assert len(set(best_set)) == 12
    assert all(1 <= x <= 10000 for x in best_set)
    
    return best_set
