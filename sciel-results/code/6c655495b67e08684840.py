def construct():
    """
    Return a nonempty subset of [1, 255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    The maximum cardinality is 85, achieved by the set of all multiples of 3 in [1, 255].
    Any attempt to include elements not divisible by 3 requires removing significantly more
    multiples of 3 (those coprime to the added element) to maintain the pairwise non-coprime property.
    """
    return list(range(3, 256, 3))
