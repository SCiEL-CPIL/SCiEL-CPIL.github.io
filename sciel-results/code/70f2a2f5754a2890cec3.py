import time
from math import gcd

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs a, b in the set.
    Maximizes cardinality.
    """
    start_time = time.time()
    timeout = 55  # Safety margin before 60s limit

    N = 255

    # The optimal strategy for N=255 (which is 3*5*17) is to take all multiples
    # of the smallest prime factor, 3. This yields a clique of size floor(255/3) = 85.
    #
    # Why this is optimal:
    # - 255 must be in the set.
    # - Every element in the set must share a factor with 255, so each element
    #   must be divisible by at least one of {3, 5, 17}.
    # - If we include any element that is NOT divisible by 3 (but is divisible
    #   by 5 or 17), then every other element in the set must share a factor
    #   with it. This restricts which multiples of 3 we can include.
    # - The set of all multiples of 3 is a valid clique (any two multiples of 3
    #   have gcd >= 3 > 1).
    # - Any attempt to mix in numbers not divisible by 3 forces exclusion of
    #   many multiples of 3 that are coprime to those numbers, reducing total size.
    # - Therefore, the maximum clique containing 255 is exactly the set of
    #   all multiples of 3 in [1, 255].

    # Construct the set of multiples of 3 in [1, 255]
    S = list(range(3, N + 1, 3))

    # Verify 255 is in the set (it should be, since 255 % 3 == 0)
    if 255 not in S:
        S.append(255)
        S.sort()

    # Ensure non-empty
    if not S:
        return [255]

    return S
