import time
import sys

def construct():
    TARGET = 123457
    deadline = time.time() + 50.0  # Safety margin
    
    # List of small primes to try as max prime factor K
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def generate_smooth_numbers(limit, max_prime):
        """
        Generate all numbers <= limit with prime factors <= max_prime.
        Returns a set of such numbers.
        """
        # Find primes up to max_prime
        # Since max_prime is small in our loop, we can just use a hardcoded list or simple sieve
        # For simplicity and speed, we assume the 'primes' list passed or derived is sufficient.
        # Actually, the function takes max_prime, so we should generate primes up to max_prime.
        
        # Simple sieve for small primes
        def get_primes_up_to(n):
            if n < 2:
                return []
            sieve = [True] * (n + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(n**0.5) + 1):
                if sieve[i]:
                    for j in range(i*i, n+1, i):
                        sieve[j] = False
            return [i for i, is_prime in enumerate(sieve) if is_prime]
            
        ps = get_primes_up_to(max_prime)
        
        smooths = set()
        
        # Recursive generation
        def gen(idx, current):
            if current > limit:
                return
            smooths.add(current)
            if idx >= len(ps):
                return
            p = ps[idx]
            # Multiply by p repeatedly
            next_val = current
            while next_val <= limit:
                next_val *= p
                if next_val > limit:
                    break
                gen(idx, next_val)
                if time.time() > deadline:
                    return
        # Start with 1
        gen(0, 1)
        
        return smooths

    # We want to minimize max(P(a), P(b)).
    # This is equivalent to finding the smallest K such that there exists a, b 
    # with a+b=TARGET, P(a)<=K, P(b)<=K.
    
    best_solution = [1, 123456] # Fallback to seed
    best_score = 293 # P(123456)
    
    # Try increasing K
    # We can just iterate through the primes list.
    # For each K, check if a solution exists.
    
    for K in primes:
        if time.time() > deadline:
            break
            
        smooths = generate_smooth_numbers(TARGET - 1, K)
        
        # Check for pair
        found = False
        # Optimization: iterate through smaller half of smooths?
        # Just iterate all
        for a in smooths:
            b = TARGET - a
            if b >= 1 and b in smooths:
                # Found a valid pair
                # The score is max(P(a), P(b)) which is <= K
                # Since we are iterating K in increasing order, the first K found is optimal
                # Note: P(a) could be less than K, but max(P(a),P(b)) <= K.
                # Is it possible that a solution exists with max(P(a),P(b)) < K?
                # Yes, but we would have found it in a previous iteration with smaller K.
                # So the first K that yields a solution gives the minimal max(P(a),P(b)).
                return [a, b]
                
    # If no solution found with small primes, fallback
    return best_solution
