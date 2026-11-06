"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    We use a direct search approach: iterate n from 1 upwards, compute phi(n) on the fly,
    and return the first n where phi(n) % 120 == 0.
    
    The minimal n is expected to be small (in the hundreds or low thousands), so this
    terminates quickly.
    """
    def phi(n):
        """Compute Euler's totient function phi(n) by trial division."""
        if n == 1:
            return 1
        result = n
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result
    
    # Search from 1 to 1000000 for the smallest n with phi(n) divisible by 120
    for n in range(1, 1000001):
        if phi(n) % 120 == 0:
            return n
    
    # Should not reach here, but return a fallback
    return -1
# EVOLVE-BLOCK-END
