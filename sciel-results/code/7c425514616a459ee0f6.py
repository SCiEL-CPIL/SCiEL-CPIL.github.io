"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """Fast verification using numpy."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
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

def _block_decomposition(n, m, p):
    """
    Constructive block decomposition.
    Split the multiplication into blocks.
    For M_{4,4,9}, we can view it as multiplying a 4x4 matrix by a 4x9 matrix.
    We can split the 4x9 matrix into two 4x4 blocks and one 4x1 block? No, 9 = 4 + 4 + 1.
    Or 4x9 = [4x4, 4x4, 4x1].
    
    Let A be 4x4, B be 4x9.
    B = [B1, B2, B3] where B1, B2 are 4x4 and B3 is 4x1.
    C = A * B = [A*B1, A*B2, A*B3].
    
    Rank of A*B1 (4x4x4) is at most 16 (schoolbook) or 7 (Strassen) or 14?
    Actually, the rank of M_{4,4,4} is known to be 14? No, lower bound is 28.
    Wait, the rank of M_{n,n,n} for n=4 is 14? No.
    The rank of M_{2,2,2} is 7.
    The rank of M_{4,4,4} is at most 16 (schoolbook).
    Actually, using Strassen on 2x2 blocks:
    4x4 = 2x2 blocks of size 2.
    Strassen reduces 8 multiplications of 2x2 to 7.
    Each 2x2 multiplication has rank 7.
    So total rank is 7 * 7 = 49? No, that's if we decompose the 2x2 multiplication.
    Wait, Strassen is a decomposition of M_{4,4,4} into 7 "sub-problems" of size 2x2x2?
    No, Strassen computes 7 products of 2x2 matrices.
    If we then compute each 2x2 product using schoolbook (rank 8), total rank is 7*8 = 56.
    If we compute each 2x2 product using Strassen (rank 7), total rank is 7*7 = 49.
    
    But there is a better decomposition for M_{4,4,4}?
    Actually, the rank of M_{4,4,4} is 14? No, that's impossible.
    Lower bound for M_{4,4,4} is 2*4^2 - 4 = 28.
    So rank is at least 28.
    
    Let's just use schoolbook for the blocks to be safe and fast to implement correctly.
    Block 1: 4x4 * 4x4 -> Rank 16
    Block 2: 4x4 * 4x4 -> Rank 16
    Block 3: 4x4 * 4x1 -> Rank 4
    
    Total Rank = 16 + 16 + 4 = 36.
    
    This is much better than 144.
    """
    # We are multiplying A (4x4) by B (4x9).
    # B is split into B1 (4x4), B2 (4x4), B3 (4x1).
    # C1 = A * B1 (4x4)
    # C2 = A * B2 (4x4)
    # C3 = A * B3 (4x1)
    
    n, m, p = 4, 4, 9
    
    # Block 1: A (4x4) * B1 (4x4)
    # U1: 4x4, V1: 4x4, W1: 4x4
    # Schoolbook rank 16
    U1 = [[0]*16 for _ in range(16)]
    V1 = [[0]*16 for _ in range(16)]
    W1 = [[0]*16 for _ in range(16)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = i*4 + j
                vidx = j*4 + k
                widx = k*4 + i
                col = i*16 + j*4 + k
                U1[idx][col] = 1
                V1[vidx][col] = 1
                W1[widx][col] = 1
                
    # Block 2: A (4x4) * B2 (4x4)
    # Same structure, just different columns in B and C
    U2 = [[0]*16 for _ in range(16)]
    V2 = [[0]*16 for _ in range(16)]
    W2 = [[0]*16 for _ in range(16)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                idx = i*4 + j
                vidx = j*4 + k
                widx = k*4 + i
                col = i*16 + j*4 + k
                U2[idx][col] = 1
                V2[vidx][col] = 1
                W2[widx][col] = 1
                
    # Block 3: A (4x4) * B3 (4x1)
    # U3: 4x4, V3: 4x1, W3: 1x4
    # Rank 4
    U3 = [[0]*4 for _ in range(16)]
    V3 = [[0]*4 for _ in range(4)]
    W3 = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(1):
                idx = i*4 + j
                vidx = j*1 + k
                widx = k*4 + i
                col = i*4 + j*1 + k
                U3[idx][col] = 1
                V3[vidx][col] = 1
                W3[widx][col] = 1
                
    # Combine
    # U: 16 rows (4x4)
    # V: 36 rows (4x9)
    # W: 36 rows (9x4)
    
    rank = 36
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Fill U
    for i in range(16):
        for j in range(16):
            U[i][j] = U1[i][j]
        for j in range(16, 32):
            U[i][j] = U2[i][j-16]
        for j in range(32, 36):
            U[i][j] = U3[i][j-32]
            
    # Fill V
    # B1 is rows 0-3 of V (4x4)
    # B2 is rows 4-7 of V (4x4)
    # B3 is rows 8-11 of V (4x1) -> Wait, V is 4x9, so 36 rows.
    # B is 4x9.
    # B1 is cols 0-3 of B.
    # B2 is cols 4-7 of B.
    # B3 is col 8 of B.
    # V rows are indexed by (j, k) where j is row of A (0-3) and k is col of B (0-8).
    # So V row index = j*9 + k.
    
    # V1 corresponds to B1 (k=0..3)
    for i in range(16): # i is (j, k) in B1, j=0..3, k=0..3
        j = i // 4
        k = i % 4
        vidx = j*9 + k
        for c in range(16):
            V[vidx][c] = V1[i][c]
            
    # V2 corresponds to B2 (k=4..7)
    for i in range(16): # i is (j, k') in B2, j=0..3, k'=0..3
        j = i // 4
        k_prime = i % 4
        k = k_prime + 4
        vidx = j*9 + k
        for c in range(16):
            V[vidx][16 + c] = V2[i][c]
            
    # V3 corresponds to B3 (k=8)
    for i in range(4): # i is (j, k'') in B3, j=0..3, k''=0
        j = i
        k = 8
        vidx = j*9 + k
        for c in range(4):
            V[vidx][32 + c] = V3[i][c]
            
    # Fill W
    # W rows are indexed by (k, i) where k is col of B (0-8) and i is row of C (0-3).
    # W row index = k*4 + i.
    
    # W1 corresponds to C1 (k=0..3)
    for i in range(16): # i is (k, i') in C1, k=0..3, i'=0..3
        k = i // 4
        i_prime = i % 4
        widx = k*4 + i_prime
        for c in range(16):
            W[widx][c] = W1[i][c]
            
    # W2 corresponds to C2 (k=4..7)
    for i in range(16): # i is (k', i') in C2, k'=0..3, i'=0..3
        k_prime = i // 4
        i_prime = i % 4
        k = k_prime + 4
        widx = k*4 + i_prime
        for c in range(16):
            W[widx][16 + c] = W2[i][c]
            
    # W3 corresponds to C3 (k=8)
    for i in range(4): # i is (k'', i') in C3, k''=0, i'=0..3
        k_prime = i
        i_prime = i
        k = 8
        widx = k*4 + i_prime
        for c in range(4):
            W[widx][32 + c] = W3[i][c]
            
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Try block decomposition first
    best = _block_decomposition(n, m, p)
    best_rank = 36
    
    # Verify it
    if _verify_fast(best["U"], best["V"], best["W"], n, m, p):
        pass # Good
    else:
        # Fallback to schoolbook if block decomposition is wrong
        best = _schoolbook(n, m, p)
        best_rank = n * m * p
        
    return best
# EVOLVE-BLOCK-END
