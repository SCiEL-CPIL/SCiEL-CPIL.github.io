import math

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _pairwise_coprime(vals):
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if math.gcd(vals[i], vals[j]) > 1:
                return False
    return True

def _validate(A, B):
    if len(A) != 3 or len(B) != 3:
        return 0
    if len(set(A)) != 3 or len(set(B)) != 3:
        return 0
    for x in A:
        if x < 1 or x > 128:
            return 0
    for x in B:
        if x < 1 or x > 128:
            return 0
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    if not _pairwise_coprime(distinct):
        return 0
    return len(distinct)

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    
    # Optimal solution: A=[4, 16, 40], B=[1, 3, 7]
    # Sums: 5, 7, 11, 17, 19, 23, 41, 43, 47
    # All are distinct primes, thus pairwise coprime.
    # Cardinality 9 is the maximum possible for 3x3 sets.
    return [[4, 16, 40], [1, 3, 7]]
