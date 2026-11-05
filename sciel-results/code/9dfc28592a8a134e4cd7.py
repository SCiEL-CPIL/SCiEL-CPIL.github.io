def construct():
    """
    Constructs a set of 20 distinct positive integers <= 10000
    that minimizes max(|A+A|, |A*A|).
    
    Strategy: Use a grid of exponents for bases 2 and 3.
    A = {2^i * 3^j | 0 <= i <= 4, 0 <= j <= 3}
    This ensures |A*A| is small (63) due to structured exponent addition.
    The small values of the elements promote additive collisions, reducing |A+A|.
    """
    A = []
    for i in range(5):
        for j in range(4):
            val = (2 ** i) * (3 ** j)
            A.append(val)
    return A
