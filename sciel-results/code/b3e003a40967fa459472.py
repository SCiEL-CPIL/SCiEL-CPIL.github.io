"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 123457
    
    # Generate primes up to N using a simple sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    best_a, best_b = 1, N - 1
    best_score = float('inf')
    
    # Iterate through candidate max prime factors P
    # We only need to check primes up to N, but likely very small ones will work
    for P in primes:
        if time.time() - start_time > DEADLINE:
            break
            
        # Generate all P-smooth numbers <= N
        # A number is P-smooth if all its prime factors are in primes[:k] where primes[k-1] <= P
        # Since we iterate P as primes, we use primes up to P inclusive.
        
        # Find index of P in primes list
        # Since P is from the primes list, we can just use a counter or binary search
        # But it's easier to just pass the list of primes <= P
        # Let's optimize: we can just pass the full primes list but stop recursion when p > P?
        # No, the recursive generator needs to know which primes are allowed.
        
        # Let's find the index of P
        # We can maintain a list of primes considered so far
        # Actually, the loop variable P is the current limit.
        # We need the list of primes <= P.
        
        # Efficient way:
        # We know primes is sorted.
        # Let's just use a helper that takes the list of allowed primes.
        
        # To avoid slicing every time, we can note that primes are small.
        # Slicing is O(k) which is small.
        
        allowed_primes = []
        for p in primes:
            if p > P:
                break
            allowed_primes.append(p)
            
        smooth_numbers = set()
        
        def gen_smooth(idx, current):
            if current > N:
                return
            smooth_numbers.add(current)
            for i in range(idx, len(allowed_primes)):
                p = allowed_primes[i]
                if current * p > N:
                    break
                gen_smooth(i, current * p)
                
        gen_smooth(0, 1)
        
        # Check for a, b in smooth_numbers such that a + b = N
        # We can iterate through the set
        found = False
        for a in smooth_numbers:
            b = N - a
            if b in smooth_numbers:
                # Calculate actual score to be precise, though it should be <= P
                # P(a) and P(b) are <= P
                # We want to minimize max(P(a), P(b)).
                # Since we are iterating P in increasing order, the first valid P
                # gives the minimal possible max prime factor.
                # However, it's possible that for a larger P, we find a pair
                # with smaller max prime factor? No, because if a pair has max prime factor M,
                # it would have been found when P=M was tested (since M is in allowed_primes for P=M).
                # So the first P that yields a solution is the optimal max prime factor.
                
                # Just in case, we can verify the actual max prime factor
                # But returning immediately is safe and faster.
                return [a, b]
                
    # If no solution found within deadline (unlikely), return best found or default
    # Fallback to a reasonable guess if search fails
    # Try a quick random or heuristic search if sieve approach is too slow?
    # The sieve approach is very fast for small P.
    
    # If we reach here, it means no P-smooth pair was found for P < deadline.
    # This is highly unlikely for N=123457.
    # Let's return a default valid pair to avoid crashing
    return [1, N - 1]

# EVOLVE-BLOCK-END
