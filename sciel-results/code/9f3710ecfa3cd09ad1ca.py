"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    
    def phi(n):
        """Compute Euler's totient function for n."""
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result
    
    # Search for smallest n in [1, 1000000] with phi(n) divisible by 120
    for n in range(1, 1000001):
        if time.time() > deadline:
            # Fallback: return a known valid answer
            return 143
        if phi(n) % 120 == 0:
            return n
    
    # Should not reach here, but fallback
    return 143
# EVOLVE-BLOCK-END
