"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 90  # seconds
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Start with all primes
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in candidates if is_prime(x)]
    current_set = set(primes)
    
    # Composites to consider adding
    composites = [x for x in candidates if not is_prime(x)]
    
    # Function to check if adding x to S is valid
    def can_add(x, S):
        # Check a=x, b,c in S
        # x | b*c for any distinct b,c in S
        for b in S:
            if b == x: continue
            for c in S:
                if c == x or c == b: continue
                if (b * c) % x == 0:
                    return False
        # Check a in S, b=x, c in S
        # a | x*c
        for a in S:
            if a == x: continue
            for c in S:
                if c == x or c == a: continue
                if (x * c) % a == 0:
                    return False
        # Check a in S, b in S, c=x
        # a | b*x
        for a in S:
            if a == x: continue
            for b in S:
                if b == x or b == a: continue
                if (b * x) % a == 0:
                    return False
        return True

    # Greedily add composites
    # Sort composites by something? Maybe larger ones first or smaller?
    # Larger numbers are less likely to divide products of others?
    # Actually, if x is large, x | bc is harder.
    # If x is small, x | bc is easier.
    # So we should try to add LARGE composites first?
    # Or small?
    # If we add small composites, they might divide products of many others.
    # Let's try adding composites in descending order.
    
    for x in sorted(composites, reverse=True):
        if time.time() > deadline - 5:
            break
        if can_add(x, current_set):
            current_set.add(x)
            
    # Try to remove some primes to add more composites?
    # This is complex. Let's just return the current best.
    
    # Ensure types are correct
    result = list(current_set)
    result.sort()
    return result

# EVOLVE-BLOCK-END
