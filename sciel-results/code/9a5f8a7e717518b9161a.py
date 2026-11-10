def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255, 
    with gcd(a, b) > 1 for all distinct pairs. 
    Maximizes cardinality.
    
    The optimal set is the set of all multiples of the smallest prime factor of 255, which is 3.
    Size: floor(255/3) = 85.
    """
    return list(range(3, 256, 3))
