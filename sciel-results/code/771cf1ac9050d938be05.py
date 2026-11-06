import math

def construct():
    N = 123457
    
    # Sieve of Eratosthenes to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # Binary search on the index of primes
    low = 0
    high = len(primes) - 1
    best_pair = None
    
    # Threshold for switching strategy
    # If B is large, use complement set
    threshold = N // 3
    
    def find_pair(B):
        if B < threshold:
            # Generate smooth numbers (S_good)
            smooths = {1}
            # Get primes <= B
            # We can use bisect to find index
            idx = bisect.bisect_right(primes, B)
            small_primes = primes[:idx]
            
            for p in small_primes:
                current_smooths = list(smooths)
                for s in current_smooths:
                    val = s
                    while val * p <= N:
                        val *= p
                        smooths.add(val)
                        
            # Check for pair
            for a in smooths:
                if N - a in smooths:
                    return (a, N - a)
            return None
        else:
            # Generate non-smooth numbers (S_bad)
            # Numbers with a prime factor > B
            # Since B >= N/3, max power of such prime is 1.
            # So x = p or x = 2p (if 2p <= N)
            bad = set()
            idx = bisect.bisect_right(primes, B)
            # Primes > B
            for p in primes[idx:]:
                if p > N: break
                bad.add(p)
                if 2 * p <= N:
                    bad.add(2 * p)
            
            # Check for pair: a and N-a must NOT be in bad
            for a in range(1, N // 2 + 1):
                if a not in bad and (N - a) not in bad:
                    return (a, N - a)
            return None

    import bisect

    while low <= high:
        mid = (low + high) // 2
        B = primes[mid]
        pair = find_pair(B)
        if pair:
            best_pair = pair
            high = mid - 1
        else:
            low = mid + 1
            
    if best_pair is None:
        # Fallback
        return [1, N-1]
        
    return [best_pair[0], best_pair[1]]
