<<<<<<< SEARCH
def _bitmask_kernel(n):
    """Compute the bitmask of the squarefree kernel of n."""
    if n <= 1:
        return 0
    result = 0
    f = n
    for i, p in enumerate(_PRIMES):
        if p * p > f:
            break
        if f % p == 0:
            exp = 0
            while f % p == 0:
                f //= p
                exp += 1
            if exp % 2 == 1:
                result |= (1 << i)
    if f > 1:
        result |= (1 << _PRIME_INDEX[f])
    return result
=======
def _bitmask_kernel(n):
    """Compute the bitmask of the squarefree kernel of n."""
    if n <= 1:
        return 0
    result = 0
    f = n
    for i, p in enumerate(_PRIMES):
        if p * p > f:
            break
        if f % p == 0:
            exp = 0
            while f % p == 0:
                f //= p
                exp += 1
            if exp % 2 == 1:
                result |= (1 << i)
    if f > 1:
        result |= (1 << _PRIME_INDEX[f])
    return result


# Precompute all kernels
_KER_MASK = [0] * (MAX_VAL + 1)
for i in range(211, MAX_VAL + 1):
    _KER_MASK[i] = _bitmask_kernel(i)

_TARGET_MASK = _bitmask_kernel(210)
>>>>>>> REPLACE

Wait, I have a duplicate `_KER_MASK` and `_TARGET_MASK` definition. Let me fix that by removing the duplicate.

