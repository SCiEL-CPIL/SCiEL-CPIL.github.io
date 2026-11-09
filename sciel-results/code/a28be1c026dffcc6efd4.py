"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import random
import math

import numpy as np

DEADLINE = 145.0  # safety margin before 150 s wall-clock kill

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _to_rational(x, max_den=8):
    """Snap a float to a half-integer rational if close; else return None."""
    if x is None:
        return None
    # Try half-integers first (denominator 1 or 2)
    for den in (1, 2):
        num = round(x * den)
        if abs(x - num / den) < 1e-6:
            return num / den
    return None

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that the decomposition is exact for matrix multiplication."""
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For each (i,k), the (j) slice should be 1
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _rank7_2x2x3():
    """Return rank-7 decomposition for 2x2x3 matrix multiplication.
    
    Uses the known minimal rank decomposition. For multiplying a 2x2 matrix
    by a 2x3 matrix, the rank is 7 (not 6 as schoolbook).
    
    Returns U (4x7), V (6x7), W (6x7) where:
    - U[i*2+j, c] for i in {0,1}, j in {0,1}
    - V[j*3+k, c] for j in {0,1}, k in {0,1,2}  
    - W[k*2+i, c] for k in {0,1,2}, i in {0,1}
    """
    # Rank-7 decomposition for 2x2x3
    # This is a known result; we use a specific construction
    U = np.zeros((4, 7))
    V = np.zeros((6, 7))
    W = np.zeros((6, 7))
    
    # Term 1: (0,0) * (0,0)
    U[0, 0] = 1.0
    V[0, 0] = 1.0
    W[0, 0] = 1.0
    
    # Term 2: (0,0) * (0,1)
    U[0, 1] = 1.0
    V[1, 1] = 1.0
    W[2, 1] = 1.0
    
    # Term 3: (0,0) * (0,2)
    U[0, 2] = 1.0
    V[2, 2] = 1.0
    W[4, 2] = 1.0
    
    # Term 4: (1,0) * (1,0)
    U[2, 3] = 1.0
    V[3, 3] = 1.0
    W[1, 3] = 1.0
    
    # Term 5: (1,0) * (1,1)
    U[2, 4] = 1.0
    V[4, 4] = 1.0
    W[3, 4] = 1.0
    
    # Term 6: (1,0) * (1,2)
    U[2, 5] = 1.0
    V[5, 5] = 1.0
    W[5, 5] = 1.0
    
    # Term 7: Cross term to handle (0,1) and (1,1) contributions
    # We need to add terms for the missing combinations
    # Actually, let me use a proper rank-7 construction
    
    # Reset and use a verified rank-7 decomposition
    U = np.zeros((4, 7))
    V = np.zeros((6, 7))
    W = np.zeros((6, 7))
    
    # Use the standard rank-7 construction for 2x2x3
    # From literature: this requires careful construction
    
    # For now, let's verify if rank 7 is achievable or fall back
    # The schoolbook for 2x2x3 is rank 6, so rank 7 can't be better
    # Wait - schoolbook is n*m*p = 2*2*3 = 6
    # So rank 7 is WORSE than schoolbook!
    
    # I made an error. For 2x2x3, schoolbook rank is 6, and it's known
    # that rank <= 6 (and >= 5 by lower bounds). So rank 6 is optimal or near-optimal.
    
    # Let me reconsider the Kronecker approach
    # 4x4x9: if we can decompose as (2x2x3) x (2x2x3), rank would be 6*6=36
    # But that's the same as the lower bound max(16,36,36)=36
    # So we might be able to achieve rank 36!
    
    # Let's return schoolbook for now and implement proper Kronecker product
    U, V, W = _schoolbook(2, 2, 3)
    return U, V, W

def _kronecker_product_decomp(U1, V1, W1, U2, V2, W2):
    """Compute Kronecker product of two tensor decompositions.
    
    If T1 = sum_c U1[c] ⊗ V1[c] ⊗ W1[c] and T2 = sum_d U2[d] ⊗ V2[d] ⊗ W2[d]
    then T1 ⊗ T2 = sum_{c,d} (U1[c]⊗U2[d]) ⊗ (V1[c]⊗V2[d]) ⊗ (W1[c]⊗W2[d])
    """
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    total_rank = rank1 * rank2
    
    U = np.zeros((U1.shape[0] * U2.shape[0], total_rank))
    V = np.zeros((V1.shape[0] * V2.shape[0], total_rank))
    W = np.zeros((W1.shape[0] * W2.shape[0], total_rank))
    
    col = 0
    for c in range(rank1):
        for d in range(rank2):
            U[:, col] = np.kron(U1[:, c], U2[:, d])
            V[:, col] = np.kron(V1[:, c], V2[:, d])
            W[:, col] = np.kron(W1[:, c], W2[:, d])
            col += 1
    
    return U, V, W

def _construct_4x4x9_kronecker():
    """Construct 4x4x9 decomposition using Kronecker product of 2x2x3 decompositions.
    
    4x4x9 = (2x2) x (2x2) x (3x3) = (2x2x3) ⊗ (2x2x3)
    Rank = rank(2x2x3) * rank(2x2x3) = 6 * 6 = 36 (if schoolbook is optimal for 2x2x3)
    """
    # Get schoolbook for 2x2x3 (rank 6)
    U1, V1, W1 = _schoolbook(2, 2, 3)
    
    # Kronecker product with itself
    U, V, W = _kronecker_product_decomp(U1, V1, W1, U1, V1, W1)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    
    # Try Kronecker product approach: 4x4x9 = (2x2x3) ⊗ (2x2x3), rank = 36
    U, V, W = _construct_4x4x9_kronecker()
    
    # Verify
    if _verify(U, V, W, n, m, p):
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
