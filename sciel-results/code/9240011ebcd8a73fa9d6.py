def construct():
    """
    Returns the smallest integer n in [1, 1000000] such that phi(n) is divisible by 120.
    
    The minimal such integer is 143.
    phi(143) = phi(11 * 13) = phi(11) * phi(13) = 10 * 12 = 120.
    120 is divisible by 120.
    No n < 143 satisfies the condition.
    """
    return 143
