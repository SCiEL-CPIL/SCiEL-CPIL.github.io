"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Return integer n in [1,1000000] with phi(n) divisible by 120.
    Minimize n.
    """
    # The smallest integer n such that phi(n) is divisible by 120 is 143.
    # phi(143) = phi(11 * 13) = phi(11) * phi(13) = 10 * 12 = 120.
    # Any n < 143 cannot have phi(n) divisible by 120 because:
    # 1. phi(n) must be divisible by 5, implying n has a prime factor p == 1 (mod 5)
    #    or n is divisible by 25.
    # 2. Smallest prime p == 1 (mod 5) is 11.
    # 3. If 11 divides n, n = 11*k. phi(n) = 10*phi(k) (if gcd(11,k)=1).
    #    We need 120 | 10*phi(k) => 12 | phi(k).
    #    Smallest k with 12 | phi(k) is 13 (phi(13)=12).
    #    So n = 11 * 13 = 143.
    # 4. If 25 divides n, n = 25*k. phi(n) = 20*phi(k).
    #    We need 120 | 20*phi(k) => 6 | phi(k).
    #    Smallest k with 6 | phi(k) is 7 (phi(7)=6).
    #    So n = 25 * 7 = 175.
    # 143 < 175, so 143 is the minimum.
    return 143
# EVOLVE-BLOCK-END
