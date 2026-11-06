import time
import random
import math

# Precompute Smallest Prime Factor (SPF) up to 20000
MAX_VAL = 10000
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM ** 0.5) + 1):
    if spf[i] == i:
        for j in range(i * i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

# Precompute bitmask of prime factors for each number up to MAX_SUM
# Use first 64 primes as bitmask universe
primes_list = []
for i in range(2, MAX_SUM + 1):
    if spf[i] == i:
        primes_list.append(i)
        if len(primes_list) >= 64:
            break

prime_to_bit = {p: (1 << i) for i, p in enumerate(primes_list)}

pf_mask = [0] * (MAX_SUM + 1)
for i in range(2, MAX_SUM + 1):
    mask = 0
    n = i
    while n > 1:
        p = spf[n]
        if p in prime_to_bit:
            mask |= prime_to_bit[p]
        while n % p == 0:
            n //= p
    pf_mask[i] = mask


def count_bits(x):
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


def construct():
    start_time = time.time()
    deadline = start_time + 82.0

    # Key insight: use nums[i] = M * k[i] where M is a product of small primes
    # This makes all pairwise sums divisible by M
    # Then the prime factors of sums = primes(M) ∪ primes(k[i]+k[j])
    # To minimize distinct primes, we want primes(k[i]+k[j]) to be a subset of primes(M)
    
    # Try different multipliers M
    # M = 2: sums even, only prime 2 guaranteed
    # M = 6: sums divisible by 6, primes {2,3} guaranteed
    # M = 30: sums divisible by 30, primes {2,3,5} guaranteed
    # M = 210: sums divisible by 210, primes {2,3,5,7} guaranteed
    # M = 6: best balance - small enough for large k values, covers 2 primes
    
    # Strategy: for M=6, find 12 distinct k values such that all k[i]+k[j] 
    # have prime factors only from {2,3,5,7} (i.e., are 7-smooth)
    # Then the answer is 4 distinct primes (2,3,5,7)
    
    # Precompute 7-smooth numbers up to 2*MAX_K where MAX_K = 10000//6 = 1666
    # 7-smooth numbers are of the form 2^a * 3^b * 5^c * 7^d
    
    MAX_K = MAX_VAL // 6  # 1666
    MAX_K_SUM = 2 * MAX_K  # 3332
    
    smooth_set = set()
    for a in range(0, 13):
        v2 = 1
        for _ in range(a):
            v2 *= 2
        for b in range(0, 9):
            v3 = 1
            for _ in range(b):
                v3 *= 3
            for c in range(0, 6):
                v5 = 1
                for _ in range(c):
                    v5 *= 5
                for d in range(0, 5):
                    v7 = 1
                    for _ in range(d):
                        v7 *= 7
                    s = v2 * v3 * v5 * v7
                    if s <= MAX_K_SUM:
                        smooth_set.add(s)
    
    smooth_list = sorted(smooth_set)
    
    # We need to find 12 distinct k values such that all pairwise sums are in smooth_set
    # This is equivalent to finding 12 numbers whose pairwise differences are all even
    # (to keep sums in a manageable range) and whose pairwise sums are 7-smooth
    
    # Let's try: k[i] = 2 * m[i] where m[i] are distinct positive integers
    # Then k[i] + k[j] = 2*(m[i]+m[j]) which is 7-smooth iff m[i]+m[j] is 7-smooth/2
    # Actually, 2*(m[i]+m[j]) is 7-smooth iff m[i]+m[j] is of form 2^a*3^b*5^c*7^d
    
    # Alternative: just search directly for k values
    # Use local search with the constraint that we want smooth sums
    
    def is_smooth(s):
        return s in smooth_set
    
    def calc_smooth_penalty(k_list):
        """Count how many pairwise sums are NOT 7-smooth."""
        penalty = 0
        n = len(k_list)
        for i in range(n):
            for j in range(i + 1, n):
                if not is_smooth(k_list[i] + k_list[j]):
                    penalty += 1
        return penalty
    
    def calc_actual_score(nums):
        """Calculate actual score (number of distinct primes in all pairwise sums)."""
        mask = 0
        n = len(nums)
        for i in range(n):
            xi = nums[i]
            for j in range(i + 1, n):
                mask |= pf_mask[xi + nums[j]]
        return count_bits(mask)
    
    best_score = float('inf')
    best_set = None
    
    # Phase 1: Try to find a set where all pairwise k-sums are 7-smooth
    # Start with structured initializations
    
    # Try k values that are multiples of small smooth numbers
    # k[i] = 2 * 3^i for i=0..11: 2, 6, 18, 54, ... too large
    # k[i] = 2^i for i=1..12: 2,4,8,16,32,64,128,256,512,1024,2048,4096 - exceeds MAX_K
    # k[i] = 2*i: 2,4,...,24 - check if all sums are 7-smooth
    #   2+4=6 (smooth), 2+6=8 (smooth), 2+8=10 (smooth), ... 22+24=46 (not smooth, 46=2*23)
    
    # Let's try k[i] = 2 * i where i is chosen from smooth numbers
    # Actually, let's just do a targeted search
    
    # Candidate k values: small even numbers that are "smooth-friendly"
    candidate_ks = [k for k in range(2, MAX_K + 1, 2) if k <= 500]
    
    # Phase 1a: Try specific structured sets
    structured_sets = [
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 30],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28],
        [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72],
        [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46],
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
    ]
    
    for k_set in structured_sets:
        if len(set(k_set)) == 12 and all(0 < k <= MAX_K for k in k_set):
            nums = [6 * k for k in k_set]
            sc = calc_actual_score(nums)
            if sc < best_score:
                best_score = sc
                best_set = nums[:]
    
    # Phase 1b: Local search to minimize smooth penalty, then actual score
    def local_search(k_init, max_iter=5000):
        """Local search to find k values with all smooth pairwise sums."""
        k_list = k_init[:]
        current_penalty = calc_smooth_penalty(k_list)
        current_score = calc_actual_score([6 * k for k in k_list])
        
        best_k = k_list[:]
        best_local_penalty = current_penalty
        best_local_score = current_score
        
        no_improve = 0
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if no_improve > 500:
                break
            
            i = random.randint(0, 11)
            old_k = k_list[i]
            
            # Generate new k value - bias towards small even numbers
            if random.random() < 0.7:
                new_k = random.choice([k for k in range(2, 100, 2)])
            else:
                new_k = random.choice([k for k in range(2, 500, 2)])
            
            if new_k == old_k or new_k in k_list:
                continue
            
            k_list[i] = new_k
            new_penalty = calc_smooth_penalty(k_list)
            
            if new_penalty <= current_penalty:
                new_score = calc_actual_score([6 * k for k in k_list])
                if new_score < best_local_score or (new_penalty < best_local_penalty and new_score <= best_local_score):
                    best_local_score = new_score
                    best_local_penalty = new_penalty
                    best_k = k_list[:]
                current_penalty = new_penalty
                no_improve = 0
            else:
                k_list[i] = old_k
                no_improve += 1
        
        return best_k, best_local_score
    
    # Run local search from multiple starting points
    start_points = [
        list(range(2, 26, 2)),  # 2,4,...,24
        [2 * i for i in range(1, 13)],
        [6 * i for i in range(1, 13)],
        [2 * i for i in range(2, 14)],  # 4,6,...,26
    ]
    
    for sp in start_points:
        if len(set(sp)) == 12 and all(0 < k <= MAX_K for k in sp):
            best_k, best_sc = local_search(sp)
            if best_sc < best_score:
                best_score = best_sc
                best_set = [6 * k for k in best_k]
                if best_score <= 4:
                    break
    
    # Phase 2: If we haven't found score <= 5, try more aggressive search
    if best_score > 5:
        # Try random restarts with hill climbing
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization with bias towards small even numbers
            k_init = []
            while len(k_init) < 12:
                if random.random() < 0.8:
                    k = random.choice([k for k in range(2, 60, 2)])
                else:
                    k = random.choice([k for k in range(2, 300, 2)])
                if k not in k_init:
                    k_init.append(k)
            
            best_k, best_sc = local_search(k_init, max_iter=2000)
            if best_sc < best_score:
                best_score = best_sc
                best_set = [6 * k for k in best_k]
                if best_score <= 4:
                    break
    
    # Phase 3: Final polish - try to improve the best solution found
    if best_set is not None:
        k_best = [x // 6 for x in best_set]
        
        # Try swapping pairs to reduce score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(12):
                for j in range(i + 1, 12):
                    for new_k in range(2, 200, 2):
                        if new_k in k_best:
                            continue
                        old_ks = [k_best[i], k_best[j]]
                        k_best[i] = new_k
                        new_score = calc_actual_score([6 * k for k in k_best])
                        if new_score < best_score:
                            best_score = new_score
                            best_set = [6 * k for k in k_best]
                            improved = True
                        else:
                            k_best[i] = old_ks[0]
        
        # Single-element swaps
        for i in range(12):
            if time.time() > deadline:
                break
            old_k = k_best[i]
            for new_k in range(2, 500, 2):
                if new_k in k_best:
                    continue
                k_best[i] = new_k
                new_score = calc_actual_score([6 * k for k in k_best])
                if new_score < best_score:
                    best_score = new_score
                    best_set = [6 * k for k in k_best]
                    old_k = new_k
                else:
                    k_best[i] = old_k
    
    # Ensure we always return a valid answer
    if best_set is None or len(best_set) != 12:
        # Fallback: use multiples of 6
        best_set = [6 * i for i in range(1, 13)]
    
    # Verify and clean up
    best_set = list(set(best_set))
    if len(best_set) < 12:
        # Fill with additional values
        i = 13
        while len(best_set) < 12 and i <= 100:
            if 6 * i not in best_set:
                best_set.append(6 * i)
            i += 1
        best_set = best_set[:12]
    
    return best_set
