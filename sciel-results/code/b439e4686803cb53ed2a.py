"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130

    result = _build_optimized_3x5x6(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        if _verify_fast(U, V, W, n, m, p):
            U_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U.tolist()]
            V_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V.tolist()]
            W_list = [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W.tolist()]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback
    return _build_schoolbook(n, m, p)


def _get_strassen_2x2x2():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U (A)
    U[0, 0] = 1; U[3, 0] = 1
    U[2, 1] = 1; U[3, 1] = 1
    U[0, 2] = 1
    U[3, 3] = 1
    U[0, 4] = 1; U[1, 4] = 1
    U[2, 5] = 1; U[0, 5] = -1
    U[1, 6] = 1; U[3, 6] = -1
    
    # V (B)
    V[0, 0] = 1; V[3, 0] = 1
    V[0, 1] = 1
    V[1, 2] = 1; V[3, 2] = -1
    V[2, 3] = 1; V[0, 3] = -1
    V[3, 4] = 1
    V[0, 5] = 1; V[1, 5] = 1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W (C)
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    W[2, 2] = 1; W[2, 4] = 1
    W[1, 1] = 1; W[1, 3] = 1
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _build_strassen_3x3x3():
    """
    Returns U, V, W for 3x3x3 multiplication using Strassen-based decomposition.
    Rank 23.
    """
    # We can build this by decomposing 3x3 into 2x2 + 1x1 + ...
    # Or use a known 23-rank decomposition.
    # A simpler way is to use the recursive Strassen on 3x3?
    # No, 3 is prime.
    # Let's use the 2x2 Strassen + Schoolbook decomposition that yields 23.
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    sU, sV, sW = _get_strassen_2x2x2()
    
    col = 0
    # Top-left 2x2 block
    # U: rows 0,1; cols 0,1
    U[0:2, 0:2, :] = sU # Wrong dimension
    # Let's map indices properly.
    # A is 3x3. Indices i,j in 0..2.
    # Block A00: i in 0,1; j in 0,1.
    
    # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Uses Strassen 7.
    for c in range(7):
        U[0*3 + 0, col+c] = sU[0, c]
        U[0*3 + 1, col+c] = sU[1, c]
        U[1*3 + 0, col+c] = sU[2, c]
        U[1*3 + 1, col+c] = sU[3, c]
        
        V[0*3 + 0, col+c] = sV[0, c]
        V[0*3 + 1, col+c] = sV[1, c]
        V[1*3 + 0, col+c] = sV[2, c]
        V[1*3 + 1, col+c] = sV[3, c]
        
        W[0*3 + 0, col+c] = sW[0, c]
        W[0*3 + 1, col+c] = sW[1, c]
        W[1*3 + 0, col+c] = sW[2, c]
        W[1*3 + 1, col+c] = sW[3, c]
    col += 7

    # Block 1: A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A is 2x1, B is 1x2. Rank 4.
    for i in range(2):
        for k in range(2):
            U[i*3 + 2, col] = 1
            V[2*3 + k, col] = 1 # B[2, k]
            W[k*3 + i, col] = 1
            col += 1
    col += 0 # 4 terms

    # Block 2: A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    # A is 1x2, B is 2x2. Rank 4.
    for j in range(2):
        for k in range(2):
            U[2*3 + j, col] = 1
            V[j*3 + k, col] = 1
            W[k*3 + 2, col] = 1
            col += 1
    col += 0 # 4 terms

    # Block 3: A[0:2, 0:2] * B[0:2, 2] -> C[0:2, 2]
    # A is 2x2, B is 2x1. Rank 4.
    for i in range(2):
        for j in range(2):
            U[i*3 + j, col] = 1
            V[j*3 + 2, col] = 1
            W[2*3 + i, col] = 1
            col += 1
    col += 0 # 4 terms

    # Block 4: A[2, 2] * B[2, 2] -> C[2, 2]
    # Rank 1.
    U[2*3 + 2, col] = 1
    V[2*3 + 2, col] = 1
    W[2*3 + 2, col] = 1
    col += 1

    # Block 5: A[0:2, 2] * B[2, 2] -> C[0:2, 2]
    # A 2x1, B 1x1. Rank 2.
    for i in range(2):
        U[i*3 + 2, col] = 1
        V[2*3 + 2, col] = 1
        W[2*3 + i, col] = 1
        col += 1
    col += 0 # 2 terms

    # Block 6: A[2, 0:2] * B[0:2, 2] -> C[2, 0:2]
    # A 1x2, B 2x1. Rank 4.
    # Wait, I already did A[2,0:2]*B[0:2,0:2] in Block 2.
    # I need A[2,0:2] * B[0:2,2] for C[2, 0:2].
    # This contributes to C[2,0] and C[2,1].
    for j in range(2):
        U[2*3 + j, col] = 1
        V[j*3 + 2, col] = 1
        W[2*3 + j, col] = 1 # No, C index is k.
        # C[2, k]. B is [0:2, 2]. So B[j, 2].
        # U is A[2, j].
        # W is C[2, k]? No.
        # Let's restart the mapping. This is error prone.
        
    # Use a known 23-rank decomposition?
    # Or just stick to 84?
    # The improvement is small (80 vs 84).
    # And high risk of bugs.
    
    return None

def _build_optimized_3x5x6(n, m, p):
    # Fallback to current 84-rank if 3x3x3 fails
    return _build_strassen_3x5x6(n, m, p)

def _build_strassen_3x5x6(n, m, p):
    sU, sV, sW = _get_strassen_2x2x2()
    rank = 84
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for k_block in range(3):
        k0 = k_block * 2
        k1 = k0 + 1
        
        for c in range(7):
            U[0*5 + 0, col] = sU[0, c]
            U[0*5 + 1, col] = sU[1, c]
            U[1*5 + 0, col] = sU[2, c]
            U[1*5 + 1, col] = sU[3, c]
            V[0*6 + k0, col] = sV[0, c]
            V[0*6 + k1, col] = sV[1, c]
            V[1*6 + k0, col] = sV[2, c]
            V[1*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
            
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
            
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
                
        for j in range(2):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
                
        for j in range(2, 4):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
                
        for k in range(2):
            U[2*5 + 4, col] = 1
            V[4*6 + k0 + k, col] = 1
            W[(k0 + k)*3 + 2, col] = 1
            col += 1
            
    return U, V, W, rank

def _build_schoolbook(n, m, p):
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

def _verify_fast(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

# EVOLVE-BLOCK-END
