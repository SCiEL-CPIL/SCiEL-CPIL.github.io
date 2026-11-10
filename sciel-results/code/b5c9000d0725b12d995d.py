import time

def construct():
    """
    Constructs a set of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    Maximizes the set size.
    """
    # We search for the largest n such that an Arithmetic Progression
    # {a, a+d, ..., a+(n-1)d} fits in 1..256 and satisfies the
    # cardinality-separated subset sum condition.
    
    # The condition is: MinSum(k) > MaxSum(k-1) for all k in 1..n
    # For an AP with start a and diff d:
    # MinSum(k) = k*a + d*k*(k-1)//2
    # MaxSum(k-1) = (k-1)*a + d * sum_{j=n-k+1}^{n-1} j
    # sum_{j=n-k+1}^{n-1} j = (n-k+1 + n-1)*(k-1)//2 = (2n-k)*(k-1)//2
    #
    # Inequality:
    # k*a + d*k*(k-1)//2 > (k-1)*a + d*(2n-k)*(k-1)//2
    # a > d * [ (2n-k)*(k-1) - k*(k-1) ] / (2*k)
    # a > d * [ (k-1)(2n - 2k) ] / (2*k)
    # a > d * (k-1)(n-k) / k
    
    # We want to find the largest n.
    # We iterate n from 256 down to 1.
    # For each n, we try d=1 first (most dense).
    
    for n in range(256, 0, -1):
        # Try small d values. d=1 is optimal for maximizing n.
        for d in range(1, 5):
            # Calculate the minimum required 'a'
            # a must be > d * (k-1)(n-k)/k for all k in 1..n
            # The term (k-1)(n-k)/k is maximized around k = sqrt(n).
            # We can just iterate k to be safe, n <= 256 so it's fast.
            
            max_required_ratio = 0
            for k in range(1, n + 1):
                if k == 1:
                    ratio = 0
                else:
                    ratio = (k - 1) * (n - k) / k
                if ratio > max_required_ratio:
                    max_required_ratio = ratio
            
            # a > d * max_required_ratio
            # Since a must be integer, a >= floor(d * max_required_ratio) + 1
            # Note: if ratio is integer, a > integer means a >= integer + 1.
            # If ratio is not integer, a > x means a >= floor(x) + 1.
            min_a = int(d * max_required_ratio) + 1
            
            # a must also be >= 1
            if min_a < 1:
                min_a = 1
                
            # a must be such that the last element a + (n-1)*d <= 256
            max_a = 256 - (n - 1) * d
            
            if min_a <= max_a:
                # We found a valid AP
                # Construct the set
                return [min_a + i * d for i in range(n)]
                
    # Fallback: n=1 is always valid
    return [1]
