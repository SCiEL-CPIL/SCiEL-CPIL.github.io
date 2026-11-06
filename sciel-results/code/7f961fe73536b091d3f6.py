"""Smooth two-summand decomposition of 123457 using binary search on smoothness bound."""
import time
import heapq

def construct():
    deadline = time.time() + 85.0
    TARGET = 123457

    def get_primes_up_to(limit):
        """Sieve of Eratosthenes to get all primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_prime[i]:
                is_prime[i*i:limit+1:i] = bytearray(b'\x00' * len(is_prime[i*i:limit+1:i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    # Get all primes up to TARGET for the binary search
    all_primes = get_primes_up_to(TARGET)
    n_primes = len(all_primes)

    def generate_b_smooth(B_index, max_val):
        """Generate all B-smooth numbers up to max_val, where B is all_primes[B_index].
        Returns a sorted list of such numbers."""
        if B_index < 0:
            return []
        B = all_primes[B_index]
        # Small primes up to B
        small_primes = [p for p in all_primes if p <= B]
        
        smooth = set()
        heap = [1]
        smooth.add(1)
        
        while heap:
            x = heapq.heappop(heap)
            if x > max_val:
                continue
            for p in small_primes:
                nx = x * p
                if nx > max_val:
                    break
                if nx not in smooth:
                    smooth.add(nx)
                    heapq.heappush(heap, nx)
        
        return sorted(smooth)

    def exists_pair(B_index):
        """Check if there exist a, b with a+b=TARGET where both are B-smooth
        (i.e., all prime factors <= all_primes[B_index])."""
        if time.time() > deadline - 3.0:
            return False, None
        
        smooth_list = generate_b_smooth(B_index, TARGET)
        smooth_set = set(smooth_list)
        
        for a in smooth_list:
            b = TARGET - a
            if b < 1:
                continue
            if b in smooth_set:
                return True, (a, b)
        
        return False, None

    # Binary search on the index in all_primes
    # We want the smallest B such that exists_pair(B) is True
    lo = 0
    hi = n_primes - 1
    best_pair = (1, TARGET - 1)
    best_B_index = hi  # fallback to largest prime
    
    # First check if even the largest prime works (it always should since any number is <= TARGET-smooth)
    # Actually, for B = TARGET, every number up to TARGET is B-smooth, so a=1, b=TARGET-1 works.
    
    while lo <= hi:
        if time.time() > deadline - 5.0:
            break
        mid = (lo + hi) // 2
        found, pair = exists_pair(mid)
        if found:
            best_pair = pair
            best_B_index = mid
            hi = mid - 1
        else:
            lo = mid + 1

    a, b = best_pair
    return [a, b]
