import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    def prime_factors(n):
        """Return set of prime factors of n."""
        if n <= 1:
            return set()
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    def is_smooth(n, allowed_primes_set):
        """Check if all prime factors of n are in allowed_primes_set."""
        if n <= 1:
            return True
        temp = n
        for p in sorted(allowed_primes_set):
            while temp % p == 0:
                temp //= p
        return temp == 1
    
    def generate_smooth_numbers(max_val, allowed_primes):
        """Generate all numbers <= max_val whose prime factors are in allowed_primes."""
        allowed_set = set(allowed_primes)
        result = []
        for i in range(1, max_val + 1):
            if is_smooth(i, allowed_set):
                result.append(i)
        return result
    
    def find_max_clique_12(smooth_nums, allowed_set, max_time):
        """Find 12 numbers from smooth_nums where all pairwise sums are also smooth."""
        n = len(smooth_nums)
        if n < 12:
            return None
        
        # Precompute which pairs have smooth sums
        # smooth_nums is sorted, sums are in range [3, 19999]
        # Check if sum is smooth
        def sum_is_smooth(a, b):
            return is_smooth(a + b, allowed_set)
        
        # Build adjacency: for each pair (i,j), check if smooth_nums[i] + smooth_nums[j] is smooth
        # We need a clique of size 12 in this graph
        
        # Use a greedy + backtracking approach
        # First, try to find a good starting point using greedy
        # Then refine
        
        # For efficiency, precompute a boolean matrix for which pairs work
        # But n could be large, so let's be smart
        
        # Try different strategies based on time remaining
        best_found = None
        best_count = 0
        
        # Greedy approach: start with a number, try to add others
        # Sort by how many compatible partners they have
        compat_count = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if sum_is_smooth(smooth_nums[i], smooth_nums[j]):
                    compat_count[i] += 1
                    compat_count[j] += 1
        
        # Order by compatibility
        order = sorted(range(n), key=lambda x: -compat_count[x])
        
        # Greedy: pick the most compatible, then try to build around it
        for start_idx in order[:20]:  # Try top 20 starting points
            if time.time() > max_time:
                break
            chosen = [start_idx]
            # Try to add compatible numbers
            candidates = [j for j in order if j != start_idx and sum_is_smooth(smooth_nums[start_idx], smooth_nums[j])]
            candidates.sort(key=lambda x: -compat_count[x])
            
            for cand in candidates:
                if len(chosen) >= 12:
                    break
                if time.time() > max_time:
                    break
                # Check if cand is compatible with all already chosen
                compatible = True
                for ch in chosen:
                    if not sum_is_smooth(smooth_nums[cand], smooth_nums[ch]):
                        compatible = False
                        break
                if compatible:
                    chosen.append(cand)
            
            if len(chosen) > best_count:
                best_count = len(chosen)
                best_found = chosen[:]
                if best_count >= 12:
                    break
        
        if best_found and len(best_found) >= 12:
            return [smooth_nums[i] for i in best_found[:12]]
        return None
    
    # Try different small prime sets
    prime_sets_to_try = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
    ]
    
    best_result = None
    best_prime_count = float('inf')
    
    for primes in prime_sets_to_try:
        if time.time() > deadline:
            break
        
        allowed_set = set(primes)
        smooth_nums = generate_smooth_numbers(10000, allowed_set)
        
        if len(smooth_nums) < 12:
            continue
        
        result = find_max_clique_12(smooth_nums, allowed_set, deadline)
        
        if result and len(result) == 12:
            # Verify all distinct and <= 10000
            if len(set(result)) == 12 and all(1 <= x <= 10000 for x in result):
                # Count distinct primes in all pairwise sums
                all_primes = set()
                for i in range(12):
                    for j in range(i + 1, 12):
                        all_primes |= prime_factors(result[i] + result[j])
                if len(all_primes) < best_prime_count:
                    best_prime_count = len(all_primes)
                    best_result = result
                # If we found with 2 primes, that's likely optimal
                if best_prime_count <= 2:
                    break
    
    # If no good result found, fall back to a reasonable set
    if best_result is None:
        # Use numbers that are powers of 2: 1,2,4,8,16,32,64,128,256,512,1024,2048
        # Sums: 1+2=3, 1+4=5, etc. - these introduce many primes
        # Better: use multiples of a highly composite number
        # Let's try: 60, 120, 180, ... (multiples of 60)
        # Sums are multiples of 60, so prime factors subset of {2,3,5}
        best_result = [60 * i for i in range(1, 13)]
        # Verify: 60*i + 60*j = 60*(i+j), prime factors of 60*(i+j)
        # 60 = 2^2 * 3 * 5, so factors are {2,3,5} plus factors of (i+j)
        # i+j ranges from 3 to 23, so factors could include 7,11,13,17,19,23
        # That's not great. Let me use a better base.
        
        # Use numbers that are all even and multiples of a smooth number
        # Try: all numbers = k * 2^a * 3^b * 5^c such that sums work
        # Simplest: use 12 numbers all equal to... no, must be distinct
        # Use: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24 (even numbers)
        # Sums are even, so factor 2 is always there. Other factors vary.
        
        # Let's just use a set and compute
        best_result = list(range(1, 13))
    
    return best_result
