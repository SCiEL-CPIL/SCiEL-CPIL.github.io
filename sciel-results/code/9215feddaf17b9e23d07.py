import math

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    Strategy:
    The number 255 = 3 * 5 * 17.
    Any element x in the set must share a factor with 255, so x is divisible by 3, 5, or 17.
    Let S be the set.
    If S contains any element x that is divisible by 3 but not by 5 or 17 (a "pure 3" multiple),
    then for any y in S, gcd(x, y) > 1 implies y must be divisible by 3 (since x is not div by 5 or 17).
    Thus, if S contains a pure 3 multiple, S must be a subset of the multiples of 3.
    The set of multiples of 3 in [1, 255] has size floor(255/3) = 85.
    
    If S contains NO pure 3 multiples, then every element in S that is a multiple of 3 is also a multiple of 5 or 17.
    Thus S is a subset of (Multiples of 5) U (Multiples of 17).
    The size of this union is |M5| + |M17| - |M85| = 51 + 15 - 3 = 63.
    
    Since 85 > 63, the maximum cardinality is 85, achieved by taking all multiples of 3.
    """
    # Generate all multiples of 3 in [1, 255]
    # This set is valid because any two multiples of 3 have gcd >= 3.
    # 255 is a multiple of 3, so it is included.
    return list(range(3, 256, 3))

if __name__ == "__main__":
    # Quick self-test
    s = construct()
    assert 255 in s
    assert len(s) > 0
    # Verify pairwise gcd
    valid = True
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if math.gcd(s[i], s[j]) == 1:
                valid = False
                break
        if not valid:
            break
    assert valid, "Set is not pairwise non-coprime"
    print(f"Constructed set size: {len(s)}")
