def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Mathematical derivation:
    We seek the minimal n such that 120 | phi(n).
    Since phi(n) < n for n > 2, any n with phi(n) = 120 must satisfy n > 120.
    The only multiple of 120 strictly less than 143 is 120 itself.
    Therefore, we check if there exists n < 143 such that phi(n) = 120.
    The integer 143 = 11 * 13 is prime-free composite.
    phi(143) = phi(11) * phi(13) = 10 * 12 = 120.
    Thus phi(143) is divisible by 120.
    
    We verify that no n < 143 has phi(n) divisible by 120.
    If n < 143, phi(n) < 143. The only possible value for phi(n) divisible by 120 in this range is 120.
    The equation phi(n) = 120 has solutions, and the smallest one is n = 143.
    Hence, n = 143 is the minimum.
    
    This hardcoded value is returned to eliminate computational overhead, as per the directive.
    """
    return 143
