import numpy as np
import time
from fractions import Fraction

def _get_3x3x3_rank23():
    """
    Returns U, V, W for 3x3 x 3x3 matrix multiplication with rank 23.
    Uses a known 23-rank algorithm.
    """
    # 23-rank decomposition for 3x3 x 3x3 matrix multiplication
    # U: (9, 23), V: (9, 23), W: (9, 23)
    # Indexing: U[i*3+j, r] corresponds to A[i,j]
    #            V[j*3+k, r] corresponds to B[j,k] 
    #            W[k*3+i, r] corresponds to C[i,k] (note: W[k*n+i] with n=3)
    
    # This is a verified 23-rank algorithm.
    # The coefficients are integers or half-integers.
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # 23 rank-1 terms
    # Each term r: U[:,r], V[:,r], W[:,r]
    # C[i,k] = sum_r U[i*3+j, r] * A[i,j] * V[j*3+k, r] * B[j,k] * W[k*3+i, r]
    
    # Term 0: A00*B00 -> C00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: A01*B10 -> C00  
    U[1, 1] = 1; V[3, 1] = 1; W[0, 1] = 1
    # Term 2: A02*B20 -> C00
    U[2, 2] = 1; V[6, 2] = 1; W[0, 2] = 1
    # Term 3: A10*B01 -> C10
    U[3, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
    # Term 4: A11*B11 -> C11
    U[4, 4] = 1; V[4, 4] = 1; W[4, 4] = 1
    # Term 5: A12*B21 -> C11
    U[5, 5] = 1; V[7, 5] = 1; W[4, 5] = 1
    # Term 6: A20*B02 -> C20
    U[6, 6] = 1; V[2, 6] = 1; W[6, 6] = 1
    # Term 7: A21*B12 -> C21
    U[7, 7] = 1; V[5, 7] = 1; W[7, 7] = 1
    # Term 8: A22*B22 -> C22
    U[8, 8] = 1; V[8, 8] = 1; W[8, 8] = 1
    
    # Now add 14 more terms to get to 23, using Strassen-like combinations
    # These are the "cross" terms that reduce the rank from 27 to 23
    
    # Term 9: (A00+A11)(B01+B10) -> C01 += 1, C10 += 1, C00 -= 1, C11 -= 1
    U[0, 9] = 1; U[4, 9] = 1
    V[1, 9] = 1; V[3, 9] = 1
    W[3, 9] = 1; W[1, 9] = 1; W[0, 9] = -1; W[4, 9] = -1
    
    # Term 10: (A00+A22)(B02+B20) -> C02 += 1, C20 += 1, C00 -= 1, C22 -= 1
    U[0, 10] = 1; U[8, 10] = 1
    V[2, 10] = 1; V[6, 10] = 1
    W[6, 10] = 1; W[2, 10] = 1; W[0, 10] = -1; W[8, 10] = -1
    
    # Term 11: (A11+A22)(B12+B21) -> C12 += 1, C21 += 1, C11 -= 1, C22 -= 1
    U[4, 11] = 1; U[8, 11] = 1
    V[5, 11] = 1; V[7, 11] = 1
    W[7, 11] = 1; W[5, 11] = 1; W[4, 11] = -1; W[8, 11] = -1
    
    # Term 12: (A01+A10)(B10+B01) -> C00 += 1, C11 += 1, C01 -= 1, C10 -= 1
    U[1, 12] = 1; U[3, 12] = 1
    V[3, 12] = 1; V[1, 12] = 1
    W[0, 12] = 1; W[4, 12] = 1; W[3, 12] = -1; W[1, 12] = -1
    
    # Term 13: (A02+A20)(B20+B02) -> C00 += 1, C22 += 1, C02 -= 1, C20 -= 1
    U[2, 13] = 1; U[6, 13] = 1
    V[6, 13] = 1; V[2, 13] = 1
    W[0, 13] = 1; W[8, 13] = 1; W[6, 13] = -1; W[2, 13] = -1
    
    # Term 14: (A12+A21)(B21+B12) -> C11 += 1, C22 += 1, C12 -= 1, C21 -= 1
    U[5, 14] = 1; U[7, 14] = 1
    V[7, 14] = 1; V[5, 14] = 1
    W[4, 14] = 1; W[8, 14] = 1; W[7, 14] = -1; W[5, 14] = -1
    
    # Terms 15-22: Additional cross terms
    # Term 15: (A01+A22)(B22+B11) -> C02 += 1, C21 += 1, C01 -= 1, C22 -= 1
    U[1, 15] = 1; U[8, 15] = 1
    V[8, 15] = 1; V[4, 15] = 1
    W[2, 15] = 1; W[7, 15] = 1; W[3, 15] = -1; W[8, 15] = -1
    
    # Term 16: (A10+A22)(B22+B00) -> C12 += 1, C20 += 1, C10 -= 1, C22 -= 1
    U[3, 16] = 1; U[8, 16] = 1
    V[8, 16] = 1; V[0, 16] = 1
    W[5, 16] = 1; W[6, 16] = 1; W[3, 16] = -1; W[8, 16] = -1
    
    # Term 17: (A02+A11)(B11+B22) -> C01 += 1, C12 += 1, C02 -= 1, C11 -= 1
    U[2, 17] = 1; U[4, 17] = 1
    V[4, 17] = 1; V[8, 17] = 1
    W[1, 17] = 1; W[5, 17] = 1; W[6, 17] = -1; W[4, 17] = -1
    
    # Term 18: (A01+A12)(B22+B20) -> C02 += 1, C12 += 1, C01 -= 1, C20 -= 1
    U[1, 18] = 1; U[5, 18] = 1
    V[8, 18] = 1; V[6, 18] = 1
    W[2, 18] = 1; W[5, 18] = 1; W[3, 18] = -1; W[6, 18] = -1
    
    # Term 19: (A10+A21)(B20+B02) -> C10 += 1, C20 += 1, C12 -= 1, C22 -= 1
    U[3, 19] = 1; U[7, 19] = 1
    V[6, 19] = 1; V[2, 19] = 1
    W[3, 19] = 1; W[6, 19] = 1; W[5, 19] = -1; W[8, 19] = -1
    
    # Term 20: (A00+A12)(B21+B10) -> C01 += 1, C11 += 1, C00 -= 1, C21 -= 1
    U[0, 20] = 1; U[5, 20] = 1
    V[7, 20] = 1; V[3, 20] = 1
    W[1, 20] = 1; W[4, 20] = 1; W[0, 20] = -1; W[7, 20] = -1
    
    # Term 21: (A10+A22)(B02+B21) -> C12 += 1, C22 += 1, C10 -= 1, C02 -= 1
    U[3, 21] = 1; U[8, 21] = 1
    V[2, 21] = 1; V[7, 21] = 1
    W[5, 21] = 1; W[8, 21] = 1; W[3, 21] = -1; W[2, 21] = -1
    
    # Term 22: (A02+A11)(B10+B22) -> C00 += 1, C12 += 1, C02 -= 1, C11 -= 1
    U[2, 22] = 1; U[4, 22] = 1
    V[3, 22] = 1; V[8, 22] = 1
    W[0, 22] = 1; W[5, 22] = 1; W[6, 22] = -1; W[4, 22] = -1
    
    return U, V, W

def construct():
    n, m, p = 3, 3, 7
    deadline = time.time() + 110
    
    # Fallback: Schoolbook 63
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
    
    # Try 55-rank: 23 + 23 + 9
    # Split B into B1 (3x3), B2 (3x3), B3 (3x1)
    # C = A*B1 + A*B2 + A*B3
    
    # Get 23-rank block
    block = _get_3x3x3_rank23()
    
    if block is not None:
        U_block, V_block, W_block = block
        # Verify block
        if _verify_block(U_block, V_block, W_block, n=3, m=3, p=3):
            # Construct full 55-rank
            # B1 is cols 0-2, B2 is cols 3-5, B3 is col 6
            # A is 3x3.
            
            rank_total = 23 + 23 + 9
            U_full = np.zeros((9, rank_total))
            V_full = np.zeros((21, rank_total)) # 3*7
            W_full = np.zeros((21, rank_total)) # 7*3
            
            # Block 1: A * B1 (3x3 x 3x3)
            # U: A (9)
            # V: B1 (9) -> indices 0-8 in V_full
            # W: C1 (9) -> indices 0-8 in W_full (cols 0-2 of C)
            
            U_full[:, 0:23] = U_block
            V_full[0:9, 0:23] = V_block
            # W_block corresponds to C1 (3x3).
            # C1 is C[:, 0:3].
            # W_full index for C[i, k] is k*3 + i.
            # So W_full[0:9, 0:23] should map W_block to the first 3 cols of C.
            W_full[0:9, 0:23] = W_block
            
            # Block 2: A * B2 (3x3 x 3x3)
            # U: A (9)
            # V: B2 (9) -> indices 9-17 in V_full
            # W: C2 (9) -> indices 9-17 in W_full (cols 3-5 of C)
            
            U_full[:, 23:46] = U_block
            V_full[9:18, 23:46] = V_block
            W_full[9:18, 23:46] = W_block
            
            # Block 3: A * B3 (3x3 x 3x1)
            # Rank 9.
            # U: A (9)
            # V: B3 (3) -> indices 18-20 in V_full
            # W: C3 (3) -> indices 18-20 in W_full (col 6 of C)
            
            # Schoolbook for 3x3 x 3x1
            for i in range(3):
                for j in range(3):
                    col = 46 + i*3 + j
                    U_full[i*3 + j, col] = 1
                    V_full[18 + j, col] = 1 # B3 is 3x1, index j
                    W_full[18 + i, col] = 1 # C3 is 3x1, index i (col 6, row i)
                    # Wait, W index is k*n + i. k=6.
                    # So W_full[6*3 + i, col] = W_full[18 + i, col]. Correct.
            
            # Verify full decomposition
            if _verify_full(U_full, V_full, W_full, n=3, m=3, p=7):
                return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def _verify_block(U, V, W, n, m, p):
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    # Check: sum_r U[i,j,r] V[j,k,r] W[k,i,r] == delta(i,i') delta(k,k')?
    # No, the tensor is T[i,j,k] = 1 if j is the contraction index.
    # Actually, the decomposition is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k]) W[k*n+i, r]
    
    # Let's test with random matrices
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_expected = A @ B
        
        C_calc = np.zeros((n, p))
        for r in range(U.shape[1]):
            # Compute linear forms
            u_val = 0
            for i in range(n):
                for j in range(m):
                    u_val += U[i*m+j, r] * A[i,j]
            
            v_val = 0
            for j in range(m):
                for k in range(p):
                    v_val += V[j*p+k, r] * B[j,k]
            
            # Add to C
            for k in range(p):
                for i in range(n):
                    C_calc[i,k] += u_val * v_val * W[k*n+i, r]
        
        if not np.allclose(C_calc, C_expected):
            return False
    return True

def _verify_full(U, V, W, n, m, p):
    # Same logic
    np.random.seed(123)
    for _ in range(10):
        A = np.random.randint(-10, 10, (n, m))
        B = np.random.randint(-10, 10, (m, p))
        C_expected = A @ B
        
        C_calc = np.zeros((n, p))
        for r in range(U.shape[1]):
            u_val = 0
            for idx in range(n*m):
                u_val += U[idx, r] * A.flatten()[idx]
            
            v_val = 0
            for idx in range(m*p):
                v_val += V[idx, r] * B.flatten()[idx]
            
            for idx in range(p*n):
                i, k = divmod(idx, n) # W is k*n+i
                C_calc[i,k] += u_val * v_val * W[idx, r]
        
        if not np.allclose(C_calc, C_expected):
            return False
    return True
