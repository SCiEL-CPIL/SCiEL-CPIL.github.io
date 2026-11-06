"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    import heapq
    
    N = 123457
    
    # Generate all primes up to N using sieve
    def sieve(n):
        if n < 2:
            return []
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = 0
        is_prime[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(N)
    
    # Generate B-smooth numbers up to N using a heap
    def generate_B_smooth(N, B, primes):
        relevant_primes = [p for p in primes if p <= B]
        if not relevant_primes:
            return {1}
        
        smooth_set = set([1])
        heap = [1]
        
        while heap:
            val = heapq.heappop(heap)
            for p in relevant_primes:
                new_val = val * p
                if new_val > N:
                    break
                if new_val not in smooth_set:
                    smooth_set.add(new_val)
                    heapq.heappush(heap, new_val)
        
        return smooth_set
    
    # Try increasing B from smallest primes upward
    for B in primes:
        smooth_set = generate_B_smooth(N, B, primes)
        
        # Check for complementary pairs
        for a in smooth_set:
            b = N - a
            if b > 0 and b in smooth_set:
                return [a, b]
    
    # Fallback
    return [1, N - 1]

# EVOLVE-BLOCK-END
