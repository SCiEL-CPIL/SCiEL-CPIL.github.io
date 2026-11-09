"""
Find the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.

The answer is 143:
  143 = 11 * 13
  phi(143) = phi(11) * phi(13) = 10 * 12 = 120
  120 % 120 == 0

No n < 143 satisfies the condition (verified by exhaustive check below).
We perform a fast constructive search using the property that if n = prod p_i^k_i,
then phi(n) = prod p_i^(k_i-1) * (p_i - 1). For phi(n) to be divisible by 120 = 2^3*3*5,
we need the product of the phi-components to cover these prime powers.

For efficiency and correctness, we simply verify by direct computation up to a small bound.
Since the answer is known to be 143, we return it directly after a quick sanity check.
"""

def _phi(n):
    """Compute Euler's totient function for n."""
    if n == 1:
        return 1
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
    The minimum is 143. We verify this quickly and return it.
    """
    # Quick verification that 143 works
    if _phi(143) % 120 == 0:
        # Verify no smaller n works by checking up to 142
        # This is fast enough (142 iterations, each O(sqrt(n)) worst case)
        for n in range(1, 143):
            if _phi(n) % 120 == 0:
                return n
        return 143
    # Fallback: search up to 1000000 if 143 somehow fails (should not happen)
    for n in range(1, 1000001):
        if _phi(n) % 120 == 0:
            return n
    return 143
