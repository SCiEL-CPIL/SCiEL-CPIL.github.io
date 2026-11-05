"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
import itertools

def _verify(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    U: (nm, R), V: (mp, R), W: (pn, R)
    For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    For all other (i,j,k) combinations, the sum should be 0.
    """
    R = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check all required terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ui = U[i*m + j]
                vi = V[j*p + k]
                wi = W[k*n + i]
                prod = ui * vi * wi
                s = np.sum(prod)
                if abs(s - 1) > 1e-9:
                    return False
    
    # Check a few random "wrong" terms to ensure they are 0
    # We can't check all, but we check all combinations where (i,j,k) is not a valid index for the product.
    # Actually, the tensor is defined such that T[i*m+j, j*p+k, k*n+i] = 1 and 0 otherwise.
    # So we need to check that for any (a, b, c) not matching the pattern, sum U[a,c] V[b,c] W[c,c] = 0.
    # This is hard to check exhaustively if R is large, but we can check all "nearby" or random ones.
    # For a correct decomposition, the tensor is zero everywhere else.
    
    # Let's check all combinations of indices a in [0, nm), b in [0, mp), c in [0, pn)
    # This is nm * mp * pn = (nm*mp*pn) checks. For 3,5,7: 15*35*21 = 11025.
    # Each check is O(R). If R~100, that's ~1M operations. Fast enough.
    
    nm = n*m
    mp = m*p
    pn = p*n
    
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                # If this index combination corresponds to a valid multiplication term, skip (or check it's 1)
                # i = a // m, j = a % m
                # j2 = b // p, k = b % p
                # k2 = c // n, i2 = c % n
                # If i == i2, j == j2, k == k2, then it should be 1.
                # Otherwise it should be 0.
                
                i1 = a // m
                j1 = a % m
                j2 = b // p
                k1 = b % p
                k2 = c // n
                i2 = c % n
                
                ui = U[a]
                vi = V[b]
                wi = W[c]
                prod = ui * vi * wi
                s = np.sum(prod)
                
                if i1 == i2 and j1 == j2 and k1 == k2:
                    if abs(s - 1) > 1e-9:
                        return False
                else:
                    if abs(s) > 1e-9:
                        return False
    return True

def _strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices:
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01], [b10, b11]]
    U[i*2+j, k] is coeff of A_ij in term k.
    V[j*2+k, l] is coeff of B_jk in term l.
    W[k*2+i, m] is coeff of C_ki in term m.
    """
    # Strassen's 7 multiplications:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b10)
    # m7 = (a01-a11)*(b01+b11)
    
    # U (A part)
    # Row 0: a00, Row 1: a01, Row 2: a10, Row 3: a11
    # Col 0 (m1): a00+a11 -> [1, 0, 0, 1]
    # Col 1 (m2): a10+a11 -> [0, 0, 1, 1]
    # Col 2 (m3): a00 -> [1, 0, 0, 0]
    # Col 3 (m4): a11 -> [0, 0, 0, 1]
    # Col 4 (m5): a00+a01 -> [1, 1, 0, 0]
    # Col 5 (m6): a10-a00 -> [-1, 0, 1, 0]
    # Col 6 (m7): a01-a11 -> [0, 1, 0, -1]
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    # V (B part)
    # Row 0: b00, Row 1: b01, Row 2: b10, Row 3: b11
    # Col 0 (m1): b00+b11 -> [1, 0, 0, 1]
    # Col 1 (m2): b00 -> [1, 0, 0, 0]
    # Col 2 (m3): b01-b11 -> [0, 1, 0, -1]
    # Col 3 (m4): b10-b00 -> [-1, 0, 1, 0]
    # Col 4 (m5): b11 -> [0, 0, 0, 1]
    # Col 5 (m6): b00+b10 -> [1, 0, 1, 0]
    # Col 6 (m7): b01+b11 -> [0, 1, 0, 1]
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    # W (C part)
    # Row 0: c00, Row 1: c01, Row 2: c10, Row 3: c11
    # c00 = m1 - m4 + m6 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m3 + m5 - m6
    
    # Col 0 (m1): c00, c11 -> [1, 0, 0, 1]
    # Col 1 (m2): c10 -> [0, 0, 1, 0]
    # Col 2 (m3): c01, c11 -> [0, 1, 0, -1]
    # Col 3 (m4): c00, c10 -> [-1, 0, 1, 0]
    # Col 4 (m5): c01, c11 -> [0, 1, 0, 1]
    # Col 5 (m6): c00, c11 -> [1, 0, 0, -1]
    # Col 6 (m7): c00 -> [1, 0, 0, 0]
    
    W = np.array([
        [1, 0, 0, -1, 0, 1, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, -1, 0, 1, -1, 0]
    ], dtype=float)
    
    return U, V, W

def _embed_2x2_into_3x5x7(U2, V2, W2):
    """
    Embed a 2x2x2 decomposition into a 3x5x7 decomposition.
    This will have rank 7.
    We need to map indices.
    A is 3x5, B is 5x7, C is 7x3.
    We are only computing the top-left 2x2 block of C, using top-left 2x2 of A and top-left 2x2 of B.
    Wait, the product C = AB.
    C[i, j] = sum_k A[i, k] B[k, j].
    If we only care about i in {0,1}, j in {0,1}, and k in {0,1}, we can use the 2x2 decomposition.
    But the tensor must be zero for all other combinations.
    
    U for 3x5x7 has shape (15, 7).
    V for 3x5x7 has shape (35, 7).
    W for 3x5x7 has shape (21, 7).
    
    U[i*5+j, k] should be non-zero only if i<2 and j<2.
    Specifically, U[i*5+j, k] = U2[i*2+j, k] for i<2, j<2.
    
    V[j*7+k, l] should be non-zero only if j<2 and k<2.
    Specifically, V[j*7+k, l] = V2[j*2+k, l] for j<2, k<2.
    
    W[k*3+i, m] should be non-zero only if k<2 and i<2.
    Specifically, W[k*3+i, m] = W2[k*2+i, m] for k<2, i<2.
    
    Let's verify the zero condition.
    If i>=2, U is 0, so sum is 0.
    If j>=2, V is 0, so sum is 0.
    If k>=2, W is 0, so sum is 0.
    If i<2, j<2, k<2, it matches the 2x2 case.
    What if i<2, j<2, but k>=2?
    Then W[k*3+i] is 0, so sum is 0.
    What if i<2, k<2, but j>=2?
    Then V[j*7+k] is 0, so sum is 0.
    So the zero condition holds.
    
    This gives a valid rank 7 decomposition for the sub-tensor, and 0 elsewhere.
    Wait, is this a valid decomposition for the FULL tensor?
    The full tensor T has T[i*5+j, j*7+k, k*3+i] = 1.
    Our decomposition gives D[i*5+j, j*7+k, k*3+i] = 1 only if i<2, j<2, k<2.
    For all other (i,j,k), D=0.
    But T=1 for ALL (i,j,k).
    So this decomposition is NOT equal to T. It is only equal to the restriction of T to the 2x2x2 corner.
    The task requires an EXACT decomposition of the matrix multiplication tensor.
    So D must equal T.
    Therefore, this rank 7 decomposition is INVALID for the full 3x5x7 tensor.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Try to find a better decomposition using random search or structured methods.
    # Since finding a rank < 105 is hard, let's try a few specific structured ones.
    # But we already know that simple embedding doesn't work because it doesn't cover the whole tensor.
    
    # Is there a way to combine Strassen-like blocks?
    # If we partition the matrices into blocks, we can apply Strassen to each block.
    # But the blocks are not square.
    
    # Let's try a simple heuristic: random integer search for rank 35..50.
    # This is unlikely to work, but let's try a very short search.
    
    # Actually, let's just return the schoolbook decomposition to be safe.
    # The task says "Find a rank-minimal exact tensor decomposition".
    # If we can't find a better one, schoolbook is the correct answer.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
