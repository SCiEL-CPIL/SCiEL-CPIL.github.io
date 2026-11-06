"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return integer n in [1,1000000] with phi(n) divisible by 120.
    Minimize n.
    
    We know analytically that n=143 is the minimum.
    phi(143) = phi(11*13) = 10*12 = 120.
    To ensure robustness and align with search-based inspirations, we perform a 
    very fast verification search for n in [1, 143] to confirm minimality, 
    then return the result. This is computationally negligible.
    """
    target = 120
    
    def phi(n):
        """Compute Euler's totient function for a small integer n."""
        result = n
        p = 2
        m = n
        while p * p <= m:
            if m % p == 0:
                while m % p == 0:
                    m //= p
                result -= result // p
            p += 1
        if m > 1:
            result -= result // m
        return result

    # Search from 1 to 143 (known upper bound for the minimum)
    # This is fast enough to be instantaneous.
    for n in range(1, 144):
        if phi(n) % target == 0:
            return n
            
    # Fallback: The analytical minimum is 143.
    return 143

# EVOLVE-BLOCK-END
