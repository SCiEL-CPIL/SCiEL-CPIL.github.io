def _phi(n):
    if n == 1:
        return 1
    result = n
    p = 2
    nn = n
    while p * p <= nn:
        if nn % p == 0:
            while nn % p == 0:
                nn //= p
            result -= result // p
        p += 1
    if nn > 1:
        result -= result // nn
    return result

def construct():
    # Find the smallest n >= 1 such that phi(n) % 120 == 0
    # We know the answer is small.
    for n in range(1, 1000):
        if _phi(n) % 120 == 0:
            return n
    # Fallback to a known valid number if loop fails (should not happen)
    return 143
