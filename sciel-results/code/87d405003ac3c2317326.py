import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N_MIN = 10000
    N_MAX = 1000000
    
    best_n = N_MIN
    best_lpf = None
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lpf = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                lpf = d
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            lpf = n
        return lpf
    
    # Precompute LPF for the best candidate
    best_lpf = largest_prime_factor(best_n * (best_n + 1))
    
    # For each prime P, generate P-smooth numbers in [N_MIN, N_MAX]
    # and check for consecutive pairs
    
    # Get a list of primes to use as P values
    # We'll try P from small to large
    
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Generate primes up to some limit
    primes = []
    for i in range(2, 200):
        if is_prime(i):
            primes.append(i)
    
    # For each candidate P (prime), generate P-smooth numbers
    for P in primes:
        if time.time() > deadline - 5:
            break
        
        # Generate all P-smooth numbers in [N_MIN, N_MAX]
        # Use a priority queue / Dijkstra-like approach
        # Start with 1, multiply by primes <= P
        
        from heapq import heappush, heappop
        
        smooth_set = set()
        heap = [(1, 0)]  # (value, index into primes)
        visited = set()
        
        while heap:
            val, idx = heappop(heap)
            if val in visited:
                continue
            visited.add(val)
            
            if N_MIN <= val <= N_MAX:
                smooth_set.add(val)
            
            # Don't expand if val > N_MAX (since all primes >= 2)
            if val > N_MAX:
                continue
            
            for j in range(idx, len(primes)):
                p = primes[j]
                if p > P:
                    break
                new_val = val * p
                if new_val > N_MAX:
                    break
                heappush(heap, (new_val, j))
        
        # Check for consecutive pairs
        smooth_sorted = sorted(smooth_set)
        for i in range(len(smooth_sorted) - 1):
            if smooth_sorted[i + 1] - smooth_sorted[i] == 1:
                n_candidate = smooth_sorted[i]
                lpf_candidate = largest_prime_factor(n_candidate * (n_candidate + 1))
                if best_lpf is None or lpf_candidate < best_lpf:
                    best_lpf = lpf_candidate
                    best_n = n_candidate
        
        # If we found a pair where the largest prime factor equals P, 
        # we can stop since we're going in increasing order of P
        # But actually we should continue to make sure we have the global min
        # However, since we're increasing P, the first valid pair at P gives lpf <= P
        # and any later pair at P' > P gives lpf <= P', so the first found is optimal
        # Wait, not exactly - a pair at P might have lpf < P if both numbers are P-smooth
        # but their actual largest prime factor is smaller. But we're checking exact LPF.
        # The key insight: if both n and n+1 are P-smooth, then LPF(n(n+1)) <= P.
        # So the minimum possible LPF is the smallest P for which such a pair exists.
        # Once we find a pair at P, no pair at P' < P exists (we checked all smaller P).
        # So the first P where we find a consecutive pair gives the optimal LPF = that P
        # (or possibly less, but we compute it exactly).
        
        if smooth_set:
            # We found smooth numbers at this P, check if we have a consecutive pair
            # If yes, this P is the answer (since all smaller P had no consecutive pair)
            found_pair = False
            for i in range(len(smooth_sorted) - 1):
                if smooth_sorted[i + 1] - smooth_sorted[i] == 1:
                    found_pair = True
                    break
            if found_pair:
                # This is the optimal P
                break
    
    # Verify the result
    final_lpf = largest_prime_factor(best_n * (best_n + 1))
    return best_n
