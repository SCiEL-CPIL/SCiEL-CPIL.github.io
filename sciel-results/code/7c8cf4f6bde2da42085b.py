import time

# Set a deadline to ensure we don't timeout, although this construction is O(1)
# We use a margin to be safe.
_DEADLINE = time.time() + 145

def construct():
    """
    Returns a rank-35 exact tensor decomposition for multiplying a 2x2 matrix 
    by a 2x10 matrix.
    
    This rank is believed to be optimal for the given dimensions, as it corresponds
    to 5 independent Strassen rank-7 decompositions for the 2x2 column blocks.
    Attempts to use field extensions (e.g., roots of unity) to share bilinear terms
    across blocks do not reduce the rank over the rationals for arbitrary matrices.
    
    Dimensions:
    n = 2, m = 2, p = 10
    Rank = 35
    """
    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5
    rank = 7 * num_blocks # 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        base = 7 * b
        
        # M1 = (a + d)(e + h)
        col = base
        U[0][col] = 1  # a
        U[3][col] = 1  # d
        V[2*b][col] = 1    # e
        V[10 + 2*b + 1][col] = 1 # h
        W[4*b][col] = 1    # C11
        W[4*b+3][col] = 1  # C22
        
        # M2 = (c + d)e
        col = base + 1
        U[2][col] = 1  # c
        U[3][col] = 1  # d
        V[2*b][col] = 1    # e
        W[4*b+1][col] = 1  # C21
        W[4*b+3][col] = -1 # C22
        
        # M3 = a(f - h)
        col = base + 2
        U[0][col] = 1  # a
        V[2*b+1][col] = 1    # f
        V[10 + 2*b + 1][col] = -1 # h
        W[4*b+2][col] = 1   # C12
        W[4*b+3][col] = 1   # C22
        
        # M4 = d(g - e)
        col = base + 3
        U[3][col] = 1  # d
        V[10 + 2*b][col] = 1   # g
        V[2*b][col] = -1       # e
        W[4*b][col] = 1        # C11
        W[4*b+1][col] = 1      # C21
        
        # M5 = (a + b)h
        col = base + 4
        U[0][col] = 1  # a
        U[1][col] = 1  # b
        V[10 + 2*b + 1][col] = 1 # h
        W[4*b][col] = -1         # C11
        W[4*b+2][col] = 1        # C12
        
        # M6 = (c - a)(e + f)
        col = base + 5
        U[2][col] = 1   # c
        U[0][col] = -1  # a
        V[2*b][col] = 1   # e
        V[2*b+1][col] = 1 # f
        W[4*b+3][col] = 1 # C22
        
        # M7 = (b - d)(g + h)
        col = base + 6
        U[1][col] = 1   # b
        U[3][col] = -1  # d
        V[10 + 2*b][col] = 1   # g
        V[10 + 2*b + 1][col] = 1 # h
        W[4*b][col] = 1        # C11

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
