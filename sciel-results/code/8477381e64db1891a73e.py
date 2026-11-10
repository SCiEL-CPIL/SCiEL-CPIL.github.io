"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 50  # safety margin under 60s
    
    def totient(n):
        """Compute Euler's totient function phi(n)."""
        if n == 1:
            return 1
        result = n
        x = n
        p = 2
        while p * p <= x:
            if x % p == 0:
                while x % p == 0:
                    x //= p
                result -= result // p
            p += 1
        if x > 1:
            result -= result // x
        return result
    
    # Scan from n=1 upward to find minimum n with phi(n) % 120 == 0
    for n in range(1, 1000001):
        if time.time() > deadline:
            break
        if totient(n) % 120 == 0:
            return n
    
    # Fallback: return a known valid value (241: phi(241)=240)
    return 241
# EVOLVE-BLOCK-END
