"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120."""
    import time
    deadline = time.time() + 85.0  # 5 second safety margin before 90s kill
    
    # We need the minimum n such that 120 | phi(n)
    # 120 = 2^3 * 3 * 5
    # phi(n) = n * prod_{p|n} (1 - 1/p)
    
    # From the knowledge: n=143 works (phi(143) = 120)
    # Let's verify: 143 = 11 * 13, phi(143) = 143 * (10/11) * (12/13) = 143 * 120/143 = 120
    # So 120 | phi(143). 
    # 
    # But is there a smaller n? Let's check all n from 1 to 142.
    # We'll do a direct search with efficient totient computation.
    
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
    
    # Search from 1 upward for the smallest n with 120 | phi(n)
    # We know n=143 works, so we only need to check up to 143
    # But let's be safe and search up to a reasonable bound
    best = None
    for n in range(1, 144):
        if time.time() > deadline:
            break
        if totient(n) % 120 == 0:
            best = n
            break
    
    if best is None:
        # Fallback: continue searching if needed
        for n in range(144, 1000001):
            if time.time() > deadline:
                break
            if totient(n) % 120 == 0:
                best = n
                break
    
    if best is None:
        best = 143  # Known valid answer
    
    return best
# EVOLVE-BLOCK-END
