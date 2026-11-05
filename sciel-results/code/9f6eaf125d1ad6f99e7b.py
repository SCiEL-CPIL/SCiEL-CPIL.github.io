def _phi(n):
    """Compute Euler's totient function for a given integer n."""
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

def construct():
    """
    Return the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    This implementation iterates through integers starting from 1, calculating the
    totient function for each candidate using trial division. Since the minimal
    solution is small (143), this approach is efficient and robust.
    """
    target_divisor = 120
    # The minimal n is known to be small, but we scan up to 10^6 as per contract.
    # The loop will break early once the first match is found.
    for n in range(1, 1000001):
        if _phi(n) % target_divisor == 0:
            return n
    # Fallback if no solution found in range (should not happen for 120)
    return -1
