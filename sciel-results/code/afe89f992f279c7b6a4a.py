"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations, product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute smallest prime factor for fast factorization up to 20000
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def prime_factors_fast(n):
        """Return set of prime factors using SPF table."""
        if n <= 1:
            return frozenset()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return frozenset(factors)

    # Precompute prime factors for all values 2..20000
    pf_cache = [frozenset() for _ in range(MAX_SUM + 1)]
    for v in range(2, MAX_SUM + 1):
        pf_cache[v] = prime_factors_fast(v)

    def count_primes_fast(nums):
        """Count distinct primes dividing all pairwise sums using cache."""
        primes = frozenset()
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                s = nums[i] + nums[j]
                primes = primes | pf_cache[s]
        return len(primes)

    def count_primes_incremental(nums, i, old_val, new_val):
        """Compute new prime count after changing nums[i] from old_val to new_val."""
        primes = set()
        n = len(nums)
        for a in range(n):
            if a == i:
                continue
            for b in range(a+1, n):
                if b == i:
                    continue
                s = nums[a] + nums[b]
                primes.update(pf_cache[s])
        for j in range(n):
            if j == i:
                continue
            s = new_val + nums[j]
            primes.update(pf_cache[s])
        return len(primes)

    best_count = float('inf')
    best_nums = None

    def try_update(nums):
        nonlocal best_count, best_nums
        c = count_primes_fast(nums)
        if c < best_count:
            best_count = c
            best_nums = nums[:]
        return c

    # Strategy 1: Backtracking search for small prime sets
    # Target prime sets to try in order of increasing size
    target_prime_sets = [
        frozenset([2, 3]),
        frozenset([2, 3, 5]),
        frozenset([2, 5]),
        frozenset([3, 5]),
        frozenset([2, 3, 5, 7]),
    ]
    
    for target_p in target_prime_sets:
        if time.time() > deadline - 10:
            break
            
        # Precompute valid sums for this target prime set
        valid_sums = set()
        for v in range(2, MAX_SUM + 1):
            if pf_cache[v] <= target_p:
                valid_sums.add(v)
        
        # Search for 12 distinct integers from 1..10000 such that all pairwise sums are in valid_sums
        # We restrict search space to 1..5000 initially for speed, then expand if needed
        # But 12 numbers is small, so we can try 1..1000 first
        
        found = False
        for max_candidate in [100, 500, 1000, 2000, 5000, 10000]:
            if time.time() > deadline - 5:
                break
                
            candidates = list(range(1, min(max_candidate, 10000) + 1))
            # Filter candidates: u must have at least 11 valid sums with other candidates?
            # Or just build graph and find clique
            
            # Build adjacency list for candidates
            # adj[u] = set of v > u such that u+v in valid_sums
            # To save memory, we can just check on the fly in backtracking
            
            def backtrack(current_set, next_start_idx):
                nonlocal found
                if len(current_set) == 12:
                    found = True
                    return True
                if time.time() > deadline - 5:
                    return False
                
                # Heuristic: try to pick candidates that have high connectivity
                # Or just iterate
                # To speed up, we can precompute for each candidate how many valid sums it has with other candidates
                # But that's O(N^2). N=10000 is 10^8, too slow.
                
                # Let's just iterate through candidates starting from next_start_idx
                # And check compatibility with all in current_set
                
                # Optimization: if current_set is empty, just pick 1
                # Actually, let's pick candidates in a smart order
                
                for i in range(next_start_idx, len(candidates)):
                    cand = candidates[i]
                    # Check if cand is compatible with all in current_set
                    ok = True
                    for existing in current_set:
                        if (cand + existing) not in valid_sums:
                            ok = False
                            break
                    if ok:
                        current_set.append(cand)
                        if backtrack(current_set, i + 1):
                            return True
                        current_set.pop()
                        
                return False

            # Start backtracking
            # We can try different starting points to find diverse solutions
            # But simple DFS from 1 is a good start
            # To avoid getting stuck, we can shuffle candidates or try multiple starts
            
            # Let's try starting with 1
            if backtrack([1], 1):
                try_update(current_set)
                found = True
                break
            
            # If not found, try starting with 2, 3, etc.
            if not found:
                for start_val in range(2, 20):
                    if time.time() > deadline - 5:
                        break
                    if backtrack([start_val], candidates.index(start_val) + 1):
                        try_update(current_set)
                        found = True
                        break
                if found:
                    break
            
            if found:
                break

    # If we found a solution with a small prime set, we might still improve
    # But if we found size 2, it's likely optimal or near optimal.
    # If best_count is 2, we can stop early?
    # Theoretical min is 1, which is impossible for n=12.
    # So 2 is a very good target.
    
    if best_count == 2:
        return best_nums

    # Strategy 2: Local improvement with incremental evaluation
    if best_nums is None:
        # Initialize with a good heuristic
        # Use arithmetic progression with smooth difference
        d = 6
        a = 1
        best_nums = [a + i*d for i in range(12)]
        best_count = count_primes_fast(best_nums)

    if time.time() < deadline - 10:
        nums = best_nums[:]
        c = best_count
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline:
                    break
                old_val = nums[i]
                # Try wider range of deltas
                for delta in list(range(-50, 51)) + list(range(100, 201, 5)) + list(range(-200, -100, 5)):
                    new_val = old_val + delta
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in nums:
                        continue
                    nums[i] = new_val
                    c2 = count_primes_incremental(nums, i, old_val, new_val)
                    if c2 < c:
                        c = c2
                        if c < best_count:
                            best_count = c
                            best_nums = nums[:]
                        improved = True
                    else:
                        nums[i] = old_val

    # Strategy 3: Random restarts with incremental local improvement
    if time.time() < deadline - 15:
        for attempt in range(300):
            if time.time() > deadline - 5:
                break
            # Generate random structured candidate
            mode = random.randint(0, 3)
            if mode == 0:
                # Multiples of smooth base
                base = random.choice([2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,21,24,28,30,32,35,36,40,42,45,48,50,54,56,60,63,64,70,72,75,80,84,90,96,100,105,108,112,120,126,128,135,140,144,150,160,168,180,192,200,210,216,224,240,252,256,270,280,288,300,315,320,336,360,375,384,400,420,432,448,450,480,500,504,512,540,560,576,600,630,640,672,700,720,750,768,800,840,864,900,960,1000,1008,1024,1050,1080,1120,1152,1200,1260,1280,1344,1400,1440,1500,1536,1600,1680,1728,1800,1920,2000,2016,2048,2100,2160,2240,2304,2400,2520,2560,2688,2800,2880,3000,3072,3200,3360,3456,3600,3840,4000,4032,4096,4200,4320,4480,4608,4800,5040,5120,5376,5600,5760,6000,6144,6400,6720,6912,7200,7680,8000,8064,8192,8400,8640,8960,9216,9600,10000])
                max_q = 10000 // base
                if max_q < 12:
                    continue
                qs = random.sample(range(1, max_q + 1), 12)
                nums = [q * base for q in qs]
            elif mode == 1:
                # Arithmetic progression
                d = random.choice([2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,21,24,28,30,32,35,36,40,42,45,48,50,54,56,60,63,64,70,72,75,80,84,90,96,100,105,108,112,120,126,128,135,140,144,150,160,168,180,192,200,210,216,224,240,252,256,270,280,288,300,315,320,336,360,375,384,400,420,432,448,450,480,500,504,512,540,560,576,600,630,640,672,700,720,750,768,800,840,864,900,960,1000,1008,1024,1050,1080,1120,1152,1200,1260,1280,1344,1400,1440,1500,1536,1600,1680,1728,1800,1920,2000,2016,2048,2100,2160,2240,2304,2400,2520,2560,2688,2800,2880,3000,3072,3200,3360,3456,3600,3840,4000,4032,4096,4200,4320,4480,4608,4800,5040,5120,5376,5600,5760,6000,6144,6400,6720,6912,7200,7680,8000,8064,8192,8400,8640,8960,9216,9600,10000])
                max_a = 10000 - 11 * d
                if max_a < 1:
                    continue
                a = random.randint(1, max_a)
                nums = [a + i*d for i in range(12)]
            elif mode == 2:
                # Congruence class
                M = random.choice([4,6,8,10,12,14,16,18,20,24,28,30,32,36,40,42,48,50,54,56,60,64,70,72,80,84,90,96,100,108,112,120,128,140,144,150,160,168,180,192,200,210,216,224,240,250,252,256,270,280,288,300,320,336,350,360,375,384,400,420,432,448,450,480,500,504,512,540,560,576,600,630,640,672,700,720,750,768,800,840,864,900,960,1000])
                r = M // 2
                if r < 1:
                    continue
                last = r + 11 * M
                if last > 10000:
                    continue
                start_k = random.randint(0, max(0, (10000 - r) // M - 11))
                nums = [r + (start_k + k)*M for k in range(12)]
            else:
                nums = random.sample(range(1, 10001), 12)
            
            c = count_primes_fast(nums)
            if c <= best_count + 2:
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for i in range(12):
                        if time.time() > deadline - 5:
                            break
                        old_val = nums[i]
                        for delta in range(-20, 21):
                            if delta == 0:
                                continue
                            new_val = old_val + delta
                            if new_val < 1 or new_val > 10000:
                                continue
                            if new_val in nums:
                                continue
                            nums[i] = new_val
                            c2 = count_primes_incremental(nums, i, old_val, new_val)
                            if c2 < c:
                                c = c2
                                if c < best_count:
                                    best_count = c
                                    best_nums = nums[:]
                                improved = True
                            else:
                                nums[i] = old_val

    if best_nums is None:
        return list(range(1, 13))
    
    return best_nums
# EVOLVE-BLOCK-END
