import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for j_prime in range(m):
                for k_prime in range(p):
                    expected = 1.0 if (j == j_prime) else 0.0
                    # For each output position (I, K), check
                    for I in range(n):
                        for K in range(p):
                            lhs = 0.0
                            for c in range(rank):
                                lhs += U[i*m+j][c] * V[j_prime*p+k_prime][c] * W[K*n+I][c]
                            rhs = 1.0 if (i == I and j_prime == j and k_prime == K) else 0.0
                            if abs(lhs - rhs) > 1e-9:
                                return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_block(p):
    """
    Return Strassen decomposition for 2x2 matrix multiplication by 2xp matrix.
    This computes C = A * B where A is 2x2 and B is 2xp.
    Uses rank 7 for the 2x2 part.
    """
    # Strassen for 2x2 * 2xp
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12, ..., b1p], [b21, b22, ..., b2p]]
    # C = [[c11, c12, ..., c1p], [c21, c22, ..., c2p]]
    #
    # Strassen formulas for 2x2 * 2xp:
    # M1 = (a11 + a22) * (b11 + b21, b12 + b22, ..., b1p + b2p)
    # M2 = (a21 + a22) * b11, b12, ..., b1p
    # M3 = a11 * (b21 - b11, b22 - b12, ..., b2p - b1p)
    # M4 = a22 * (b11, b12, ..., b1p)
    # M5 = (a11 + a12) * b21, b22, ..., b2p
    # M6 = (a21 - a11) * (b11, b12, ..., b1p)
    # M7 = (a12 - a22) * (b21, b22, ..., b2p)
    #
    # c11 = M1 + M3 - M4 - M7
    # c12 = M1 + M3 - M4 - M7 (for each column)
    # c21 = M2 + M4
    # c22 = M2 + M4 (for each column)
    #
    # Wait, let me be more careful. For 2x2 * 2xp:
    # C[0,k] = a11*b1[k] + a12*b2[k]
    # C[1,k] = a21*b1[k] + a22*b2[k]
    #
    # Strassen:
    # M1 = (a11+a22) * (b1+b2)  -> vector of length p
    # M2 = (a21+a22) * b1       -> vector of length p
    # M3 = a11 * (b2-b1)        -> vector of length p
    # M4 = a22 * b1             -> vector of length p
    # M5 = (a11+a12) * b2       -> vector of length p
    # M6 = (a21-a11) * b1       -> vector of length p
    # M7 = (a12-a22) * b2       -> vector of length p
    #
    # C[0,k] = M1[k] + M3[k] - M4[k] - M7[k]
    # C[1,k] = M2[k] + M4[k]
    
    rank = 7 * p  # 7 Strassen products, each of length p
    nm = 2 * 2  # 2x2 input
    mp = 2 * p  # 2xp input
    pn = p * 2  # px2 output
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    # M1: (a11+a22) * (b1+b2)
    # U[0,0] = 1, U[1,1] = 1 (a11 and a22)
    # V[0,0] = 1, V[1,0] = 1 (b1 and b2)
    for k in range(p):
        col = k  # M1, k-th component
        U[0][col] = 1  # a11
        U[3][col] = 1  # a22
        V[0][col] = 1  # b1
        V[2][col] = 1  # b2
        W[k*2+0][col] = 1  # C[0,k]
        W[k*2+1][col] = 1  # C[1,k]
    
    # M2: (a21+a22) * b1
    for k in range(p):
        col = p + k  # M2, k-th component
        U[2][col] = 1  # a21
        U[3][col] = 1  # a22
        V[0][col] = 1  # b1
        W[k*2+1][col] = 1  # C[1,k]
    
    # M3: a11 * (b2-b1)
    for k in range(p):
        col = 2*p + k  # M3, k-th component
        U[0][col] = 1  # a11
        V[0][col] = -1  # -b1
        V[2][col] = 1   # b2
        W[k*2+0][col] = 1  # C[0,k]
    
    # M4: a22 * b1
    for k in range(p):
        col = 3*p + k  # M4, k-th component
        U[3][col] = 1  # a22
        V[0][col] = 1  # b1
        W[k*2+0][col] = -1  # C[0,k]
        W[k*2+1][col] = 1   # C[1,k]
    
    # M5: (a11+a12) * b2
    for k in range(p):
        col = 4*p + k  # M5, k-th component
        U[0][col] = 1  # a11
        U[1][col] = 1  # a12
        V[2][col] = 1  # b2
        # Not directly used in final output, but needed for Strassen
        # Actually, M5 is not used in the 2x2*2xp case above. Let me reconsider.
        # For 2x2 * 2xp, we only need 4 products, not 7!
        # C[0,k] = a11*b1[k] + a12*b2[k]
        # C[1,k] = a21*b1[k] + a22*b2[k]
        # This is just 4 independent dot products, so rank is 4, not 7.
        # Strassen doesn't help here because the second matrix is 2xp, not 2x2.
        pass
    
    # Actually, for 2x2 * 2xp, the minimal rank is 4 (schoolbook).
    # Strassen only helps when both matrices are square.
    # So for the 2x2 block, we use schoolbook: rank = 2*2*p = 4p
    # For the 2x1 block, we use schoolbook: rank = 2*1*p = 2p
    # Total: 6p = 60 for p=10
    
    # Let me reconsider the problem. We have 2x3 * 3x10.
    # We can split the 3 columns into 2+1.
    # A = [A1 | A2] where A1 is 2x2 and A2 is 2x1
    # B = [B1; B2] where B1 is 2x10 and B2 is 1x10
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: 2x2 * 2x10, rank = 2*2*10 = 40 (schoolbook)
    # A2*B2: 2x1 * 1x10, rank = 2*1*10 = 20 (schoolbook)
    # Total: 60
    #
    # Can we do better? The tensor for 2x2 * 2x10 has rank at most 40.
    # The tensor for 2x1 * 1x10 has rank at most 20.
    # But maybe we can share some structure?
    #
    # Actually, the full tensor is the sum of two tensors. The rank of a sum is at most
    # the sum of the ranks, but could be less if there's overlap.
    #
    # For now, let's just return the schoolbook solution.
    return None  # This function is not used as-is

def _structured_decomposition(n, m, p):
    """
    Try to find a rank-minimal decomposition using structured linear algebra.
    For 2x3 * 3x10, we can potentially reduce rank by combining terms.
    
    The schoolbook rank is 2*3*10 = 60.
    
    Let's think of it as:
    C[0,k] = a00*b0k + a01*b1k + a02*b2k
    C[1,k] = a10*b0k + a11*b1k + a12*b2k
    
    We can try to use fewer than 30 products per output row by sharing computations.
    But since the a's and b's are independent variables, we likely need at least
    the "border rank" or "tensor rank" of this computation.
    
    For now, return schoolbook.
    """
    if n != 2 or m != 3 or p != 10:
        return None
    
    result = _schoolbook(n, m, p)
    return result

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Try structured decomposition
    structured = _structured_decomposition(n, m, p)
    if structured is not None:
        # Verify
        if _verify(structured["U"], structured["V"], structured["W"], n, m, p):
            return structured
    
    return result
# EVOLVE-BLOCK-END
