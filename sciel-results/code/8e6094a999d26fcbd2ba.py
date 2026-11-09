import time

def construct():
    """
    Returns a list of 2 mutually orthogonal Latin squares of order 10.
    """
    n = 10
    sq1 = [[0 for _ in range(n)] for _ in range(n)]
    sq2 = [[0 for _ in range(n)] for _ in range(n)]
    
    for r in range(n):
        for c in range(n):
            # Decompose r and c into (x, u) and (y, v)
            # r = 5*x + u, c = 5*y + v
            x, u = divmod(r, 5)
            y, v = divmod(c, 5)
            
            # Square 1
            # High bit: (x + y) % 2
            # Low bit: (u + v) % 5
            h1 = (x + y) % 2
            l1 = (u + v) % 5
            sq1[r][c] = 5 * h1 + l1
            
            # Square 2
            # High bit: (x + 2*y) % 2 which is x % 2 (since 2y is 0 mod 2)
            # Wait, x is 0 or 1. So h2 = x.
            # Low bit: (2*u + v) % 5
            h2 = x % 2
            l2 = (2 * u + v) % 5
            sq2[r][c] = 5 * h2 + l2
            
    # Ensure types are strictly Python int
    # The operations above produce Python ints.
    # Ensure outer and inner structures are lists.
    return [sq1, sq2]
