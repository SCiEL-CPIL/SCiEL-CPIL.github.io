def construct():
    """
    Returns a subset of [1, 256] satisfying:
    For every a < b <= c in the set, a does NOT divide (b + c).
    Maximize cardinality.
    
    Strategy:
    The set [171, 256] is a valid solution of size 86.
    Proof: For any a in [171, 256], the smallest possible value of b+c 
    (given b > a) is (a+1) + (a+1) = 2a+2.
    The multiples of a in the range [2a+2, 512] are:
    3a, 4a, ...
    Since a >= 171, 3a >= 513 > 512.
    Thus, there are no multiples of a in the valid range for b+c.
    So the condition holds trivially.
    
    Can we do better?
    If we try to include a number x < 171, we must ensure that for all 
    pairs (b, c) in the set with b > x, c >= b, x does not divide b+c.
    
    Consider adding 170 to [171, 256].
    We need 170 to not divide b+c for all 170 < b <= c in [171, 256].
    Possible values for b+c are in [171+171, 256+256] = [342, 512].
    Multiples of 170 in this range: 340 (too small), 510.
    So we must avoid b+c = 510.
    Pairs (b, c) in [171, 256] such that b+c=510 and 170 < b <= c:
    b can range from max(171, 510-256)=254 to min(256, 510/2)=255.
    Pairs: (254, 256), (255, 255).
    To fix this, we must remove at least one element from each pair.
    Removing 256 breaks (254, 256).
    Removing 255 breaks (255, 255).
    If we remove 255 and 256, the set becomes [170, 254]. Size 254-170+1 = 85.
    This is smaller than 86.
    
    What if we remove only 254? Then (255, 255) remains. 170 divides 510. Invalid.
    What if we remove only 255? Then (254, 256) remains. 170 divides 510. Invalid.
    So adding 170 requires removing at least 2 elements from the top, reducing size to 85.
    
    Generally, adding smaller elements creates more constraints (more multiples 
    to avoid) and requires removing more elements from the high end to satisfy 
    the divisibility constraints, leading to a net decrease in cardinality.
    
    Therefore, [171, 256] is optimal among contiguous intervals starting near the top.
    Non-contiguous sets might exist, but the density loss from removing elements 
    to satisfy divisibility for smaller a's typically outweighs the gain of 
    including those smaller a's.
    
    The interval [171, 256] is the largest subset of the form [k, 256] that is valid.
    Any valid set must have max element <= 256.
    If min element is m, then 3m > 512 is a sufficient condition for validity 
    if the set is dense.
    m > 512/3 = 170.66, so m >= 171.
    Thus [171, 256] is the maximal interval.
    """
    return list(range(171, 257))
