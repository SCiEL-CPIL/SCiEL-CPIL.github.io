def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    Mathematical Derivation:
    1. We seek the minimum n such that 120 | phi(n).
    2. Since phi(n) <= n for all n >= 1, and phi(n) is a positive integer, 
       if 120 | phi(n), then phi(n) >= 120, which implies n >= 120.
    3. The possible values for phi(n) are 120, 240, 360, ...
       - If phi(n) = 120, n can be as small as ~143.
       - If phi(n) = 240, n must be at least 240.
       - Thus, the global minimum n must be a solution to phi(n) = 120, 
         provided that the minimum such n is less than 240.
    4. We find the minimum n such that phi(n) = 120.
       Let n = p1^e1 * ... * pk^ek.
       phi(n) = n * prod(1 - 1/pi) = 120.
       This requires pi - 1 to divide 120 for all prime factors pi.
       Possible primes pi are {2, 3, 5, 7, 11, 13, 31, 41, 61}.
       
       Checking combinations of distinct primes (square-free n) first:
       - n = 11 * 13 = 143. 
         phi(143) = (11-1)*(13-1) = 10 * 12 = 120.
       - n = 5 * 31 = 155.
         phi(155) = (5-1)*(31-1) = 4 * 30 = 120.
       - n = 3 * 61 = 183.
         phi(183) = (3-1)*(61-1) = 2 * 60 = 120.
       - n = 3 * 7 * 11 = 231.
         phi(231) = 2 * 6 * 10 = 120.
       
       The smallest of these is 143.
       
       Checking if any n < 143 has phi(n) = 120:
       Since 143 is the smallest product of distinct primes yielding 120, 
       and no prime power p^k yields phi(p^k)=120 (verified: 2^k, 3^k, etc. 
       do not produce 120 exactly without mixing primes), 143 is the minimum.
       
       Any n with phi(n) >= 240 must have n >= 240 > 143.
       
    5. Therefore, 143 is the global minimum.
    """
    return 143
