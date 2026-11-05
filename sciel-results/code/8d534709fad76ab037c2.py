import math
import time

def _phi(n):
    """
    Compute Euler's totient of n efficiently using trial division.
    """
    if n == 1:
        return 1

    result = n
    p = 2

    # Handle factor 2
    if n % p == 0:
        result -= result // p
        while n % p == 0:
            n //= p

    # Check odd factors
    p = 3
    while p * p <= n:
        if n % p == 0:
            result -= result // p
            while n % p == 0:
                n //= p
        p += 2

    # If n is still greater than 1, then it is a prime factor
    if n > 1:
        result -= result // n

    return result

def construct():
    """
    Find the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a deadline-guarded search to verify the known minimum (143) and
    returns the best valid candidate found within the time budget.
    """
    deadline = time.time() + 85  # 85s margin against 90s kill
    best_n = 143  # known mathematical minimum

    # Verify 143 is valid (phi(143)=120, 120%120==0)
    if _phi(best_n) % 120 != 0:
        # Fallback: should never happen, but search from 1
        for n in range(1, 1000001):
            if time.time() > deadline:
                break
            if _phi(n) % 120 == 0:
                best_n = n
                break
        return best_n

    # 143 is valid. Now search from 1 to 142 to confirm no smaller n exists.
    # This is a bounded search (at most 142 iterations), well within budget.
    for n in range(1, best_n):
        if time.time() > deadline:
            break
        if _phi(n) % 120 == 0:
            best_n = n
            break

    return best_n
