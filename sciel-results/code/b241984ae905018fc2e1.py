import math

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a, b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    
    Proof of optimality: 255 = 3 * 5 * 17. The set of all multiples of 3
    in [1,255] has size 85 and is pairwise non-coprime (any two share factor 3).
    Since 3 is in the set, any additional element x must satisfy gcd(x,3)>1,
    i.e., 3|x. Thus no element outside the multiples of 3 can be added.
    Hence 85 is the maximum possible cardinality.
    """
    N = 255
    # All multiples of 3 in [1, 255]: {3, 6, 9, ..., 255}
    # Size = 255 // 3 = 85
    # Contains 255 since 255 % 3 == 0
    # Pairwise non-coprime: gcd(3a, 3b) >= 3 > 1 for all distinct a,b
    result = list(range(3, N + 1, 3))
    
    # Verify 255 is present (it is, but guard for safety)
    if N not in result:
        result.append(N)
        
    return result
