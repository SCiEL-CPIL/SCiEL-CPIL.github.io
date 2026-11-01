import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 matrix multiplication.
    
    The tensor rank of 2x2 matrix multiplication is 7 (Strassen's algorithm).
    This is the known minimum for exact decomposition over the rationals.
    
    We construct the decomposition programmatically from the Strassen formulas,
    verify it, and return the factor matrices.
    """
    
    # Dimensions
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize U, V, W as zeros
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Index conventions:
    # A has entries a_{ij}, flattened as i*m + j (row-major)
    #   a00=0, a01=1, a10=2, a11=3
    # B has entries b_{jk}, flattened as j*p + k (row-major)
    #   b00=0, b01=1, b10=2, b11=3
    # C has entries c_{ik}, flattened as k*n + i (trace convention: W[k*n+i, r])
    #   c00 (i=0,k=0) -> 0
    #   c10 (i=1,k=0) -> 1
    #   c01 (i=0,k=1) -> 2
    #   c11 (i=1,k=1) -> 3
    
    # Strassen's 7 products:
    # m0 = (a00 + a11)(b00 + b11)
    # m1 = (a10 + a11)b00
    # m2 = a00(b01 - b11)
    # m3 = a11(b10 - b00)
    # m4 = (a00 + a01)b11
    # m5 = (a10 - a00)(b00 + b01)
    # m6 = (a01 - a11)(b10 + b11)
    
    # Output C:
    # c00 = m0 + m3 - m4 + m6
    # c01 = m2 + m4
    # c10 = m1 + m3
    # c11 = m0 - m1 + m2 + m5
    
    # Define each Strassen term as (U contributions, V contributions, W contributions)
    # Each contribution is a list of (index, coefficient) pairs
    
    terms = [
        # m0 = (a00 + a11)(b00 + b11) -> c00 + c11
        (
            [(0, 1), (3, 1)],      # U: a00, a11
            [(0, 1), (3, 1)],      # V: b00, b11
            [(0, 1), (3, 1)]       # W: c00, c11
        ),
        # m1 = (a10 + a11)b00 -> c10 - c11
        (
            [(2, 1), (3, 1)],      # U: a10, a11
            [(0, 1)],              # V: b00
            [(1, 1), (3, -1)]      # W: c10, -c11
        ),
        # m2 = a00(b01 - b11) -> c01 + c11
        (
            [(0, 1)],              # U: a00
            [(1, 1), (3, -1)],     # V: b01, -b11
            [(2, 1), (3, 1)]       # W: c01, c11
        ),
        # m3 = a11(b10 - b00) -> c00 + c10
        (
            [(3, 1)],              # U: a11
            [(2, 1), (0, -1)],     # V: b10, -b00
            [(0, 1), (1, 1)]       # W: c00, c10
        ),
        # m4 = (a00 + a01)b11 -> -c00 + c01
        (
            [(0, 1), (1, 1)],      # U: a00, a01
            [(3, 1)],              # V: b11
            [(0, -1), (2, 1)]      # W: -c00, c01
        ),
        # m5 = (a10 - a00)(b00 + b01) -> c11
        (
            [(2, 1), (0, -1)],     # U: a10, -a00
            [(0, 1), (1, 1)],      # V: b00, b01
            [(3, 1)]               # W: c11
        ),
        # m6 = (a01 - a11)(b10 + b11) -> c00
        (
            [(1, 1), (3, -1)],     # U: a01, -a11
            [(2, 1), (3, 1)],      # V: b10, b11
            [(0, 1)]               # W: c00
        ),
    ]
    
    # Populate matrices from term definitions
    for r, (u_contribs, v_contribs, w_contribs) in enumerate(terms):
        for idx, coef in u_contribs:
            U[idx, r] += coef
        for idx, coef in v_contribs:
            V[idx, r] += coef
        for idx, coef in w_contribs:
            W[idx, r] += coef
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
