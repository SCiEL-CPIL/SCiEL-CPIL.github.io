"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120."""
    # We need to find the minimum n where 120 divides phi(n).
    # The totient function phi(n) for n = p1^a1 * p2^a2 * ... is n * prod(1 - 1/pi).
    # For phi(n) to be divisible by 120 = 8 * 3 * 5, it needs to have these factors.
    
    # Let's use a direct search with an efficient totient calculation using a sieve-like approach
    # or simply trial division for each n up to a reasonable limit.
    # Since we want to minimize n, we can just iterate from 1 upward and check each n.
    
    # Efficient totient calculation for a single n:
    # phi(n) = n * prod_{p|n} (1 - 1/p)
    
    def totient(n):
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
    
    # Search from 1 upward
    for n in range(1, 1000001):
        if totient(n) % 120 == 0:
            return n
    
    # Fallback (should not be reached since n=143 works)
    return 143
# EVOLVE-BLOCK-END
