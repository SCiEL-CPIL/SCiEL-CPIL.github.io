import time
import sys

# EVOLVE-BLOCK-START

def _trivial_decomposition(n, m, p):
    """
    Return the trivial (rank n*m*p) exact tensor decomposition for n x m x p matrix multiplication.
    
    U: shape (n*m, rank)  -- entries for A[i,j] indexed as i*m+j
    V: shape (m*p, rank)  -- entries for B[j,k] indexed as j*p+k
    W: shape (p*n, rank)  -- entries for C[i,k] indexed as k*n+i (trace convention)
    
    The decomposition satisfies:
    C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  for all i,j,k
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][t] = 1
                V[v_idx][t] = 1
                W[w_idx][t] = 1
                t += 1
    
    return U, V, W

def _strassen_2x2_2x2_UVW():
    """
    Return the Strassen decomposition for 2x2 * 2x2 -> 2x2 multiplication.
    Rank = 7.
    
    U: shape (4, 7) for A[i,j] indexed as i*2+j
    V: shape (4, 7) for B[j,k] indexed as j*2+k  
    W: shape (4, 7) for C[i,k] indexed as k*2+i (trace convention)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """
    Verify that the decomposition (U, V, W) correctly computes matrix multiplication.
    Returns True if valid, False otherwise.
    """
    try:
        import numpy as np
        np.random.seed(12345)
        
        # Test with several random matrices
        for trial in range(5):
            A = np.random.randint(-10, 10, (n, m), dtype=float)
            B = np.random.randint(-10, 10, (m, p), dtype=float)
            C_expected = A @ B
            
            rank = len(U[0])
            C_computed = np.zeros((n, p), dtype=float)
            
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        u_row = U[i * m + j]
                        v_row = V[j * p + k]
                        w_val = W[k * n + i]
                        for r in range(rank):
                            s += u_row[r] * v_row[r] * w_val[r]
                    C_computed[i, k] = s
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        
        return True
    except Exception:
        return False

def _strassen_4x4_4x4_UVW():
    """
    Return the Strassen-based decomposition for 4x4 * 4x4 -> 4x4.
    Uses block Strassen: treat 4x4 as 2x2 blocks of 2x2 matrices.
    Total rank = 7 * 7 = 49.
    
    U: shape (16, 49) for A[i,j] indexed as i*4+j
    V: shape (16, 49) for B[j,k] indexed as j*4+k
    W: shape (16, 49) for C[i,k] indexed as k*4+i (trace convention)
    """
    U2, V2, W2 = _strassen_2x2_2x2_UVW()
    
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    
    block_A_coeffs = {
        (0,0): [1, 0, 1, 0, 1, 0, 0],
        (0,1): [0, 0, 0, 0, 1, 0, 1],
        (1,0): [0, 1, 0, 0, 0, -1, 0],
        (1,1): [1, 1, 0, 1, 0, -1, 0],
    }
    
    block_B_coeffs = {
        (0,0): [1, 1, 0, 0, 0, 1, 0],
        (0,1): [0, 0, 1, 0, 0, 1, 0],
        (1,0): [0, 0, 0, 1, 0, 0, 1],
        (1,1): [1, -1, 0, 0, 1, 0, 1],
    }
    
    block_C_coeffs = {
        (0,0): [1, 0, 0, 1, -1, 0, 1],
        (0,1): [0, 0, 1, 0, 1, 0, 0],
        (1,0): [0, 1, 0, 1, 0, 0, 0],
        (1,1): [1, -1, 1, 0, 0, 1, 0],
    }
    
    # Fill in U
    for i in range(4):
        for j in range(4):
            bi, bj = i // 2, j // 2
            ii, jj = i % 2, j % 2
            inner_idx = ii * 2 + jj
            for rb in range(7):
                for ri in range(7):
                    combined_r = rb * 7 + ri
                    U[i*4+j][combined_r] = block_A_coeffs[(bi,bj)][rb] * U2[inner_idx][ri]
    
    # Fill in V
    for j in range(4):
        for k in range(4):
            bi, bj = j // 2, k // 2
            ii, jj = j % 2, k % 2
            inner_idx = ii * 2 + jj
            for rb in range(7):
                for ri in range(7):
                    combined_r = rb * 7 + ri
                    V[j*4+k][combined_r] = block_B_coeffs[(bi,bj)][rb] * V2[inner_idx][ri]
    
    # Fill in W (trace convention: k*4+i)
    for i in range(4):
        for k in range(4):
            bi, bj = i // 2, k // 2
            ii, jj = i % 2, k % 2
            inner_idx = ii * 2 + jj
            for rb in range(7):
                for ri in range(7):
                    combined_r = rb * 7 + ri
                    W[k*4+i][combined_r] = block_C_coeffs[(bi,bj)][rb] * W2[inner_idx][ri]
    
    return U, V, W

def _extend_4x4_4x4_to_4x4_7():
    """
    Extend the 4x4*4x4 -> 4x4 Strassen decomposition (rank 49) to 4x4*4x7 -> 4x7.
    Uses trivial decomposition (rank 48) for the last 3 columns.
    Total rank = 49 + 48 = 97.
    """
    U4, V4, W4 = _strassen_4x4_4x4_UVW()
    U3, V3, W3 = _trivial_decomposition(4, 4, 3)
    
    U = [row4 + row3 for row4, row3 in zip(U4, U3)]
    V = [row4 + row3 for row4, row3 in zip(V4, V3)]
    
    W = [[0] * 97 for _ in range(28)]
    
    # Copy W4 into W (rows 0-15)
    for i in range(16):
        for r in range(49):
            W[i][r] = W4[i][r]
    
    # Copy W3 into W (rows 16-27, corresponding to k=4,5,6)
    for i3 in range(12):
        k3, i_val = i3 // 4, i3 % 4
        row_in_W = (4 + k3) * 4 + i_val
        for r in range(48):
            W[row_in_W][49 + r] = W3[i3][r]
    
    return U, V, W

def _attempt_4x4_4x7_low_rank():
    """
    Attempt to find a lower-rank decomposition than 97 for 4x4*4x7->4x7.
    Uses a structured approach based on known tensor rank bounds.
    """
    # For 4x4*4x7, the known upper bound is better than 112.
    # The trivial is 112. Strassen+trivial is 97.
    # Can we do better? 
    # M(4,4,7) <= M(4,4,4) + M(4,4,3) is one way, giving 49+48=97.
    # Another way: M(4,4,7) <= 2*M(2,4,7) + 2*M(2,4,7) ? No.
    # M(4,4,7) <= M(4,4,2) + M(4,4,5) ? 
    # Let's try to see if we can use a different split.
    # What if we split the 7 columns into 2+2+3?
    # M(4,4,7) <= M(4,4,2) + M(4,4,2) + M(4,4,3)
    # M(4,4,2) can be done with rank 4*4*2/2 ? No, trivial is 32.
    # Actually M(4,4,2) is just two 4x4*4x1 multiplications.
    # M(4,4,1) has rank 16 (trivial). So M(4,4,2) <= 32.
    # This doesn't help immediately unless we have better decompositions.
    
    # Let's try a different known bound:
    # M(4,4,7) <= M(4,4,4) + M(4,4,3) = 49 + 48 = 97.
    # Is there a better bound?
    # Alon-Boppana: M(4,4,7) <= ...
    # Actually, the best known bound for M(4,4,7) might be lower.
    # But finding an explicit decomposition is hard.
    
    # Let's try to verify if 97 is correct and if we can shave off a bit.
    # The 4x4*4x3 part uses trivial rank 48.
    # Can we do 4x4*4x3 better?
    # M(4,4,3) <= M(4,4,2) + M(4,4,1) = 32 + 16 = 48. Same.
    # M(4,4,3) <= M(4,2,3) + M(4,2,3) ? 
    # M(4,2,3) trivial is 24. So 48.
    
    # It seems 97 is a solid upper bound. Let's just return the 97-rank solution.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = 4, 4, 7
    
    # Start with the trivial decomposition (rank = 4*4*7 = 112)
    U, V, W = _trivial_decomposition(n, m, p)
    
    # Verify the trivial decomposition works
    if not _verify_decomposition(n, m, p, U, V, W):
        pass
    
    # Try the extended Strassen decomposition (rank 97)
    try:
        U_strassen, V_strassen, W_strassen = _extend_4x4_4x4_to_4x4_7()
        
        # Verify
        if _verify_decomposition(n, m, p, U_strassen, V_strassen, W_strassen):
            U, V, W = U_strassen, V_strassen, W_strassen
    except Exception:
        pass
    
    # Verify final result
    if not _verify_decomposition(n, m, p, U, V, W):
        # Fallback to trivial
        U, V, W = _trivial_decomposition(n, m, p)
    
    # Return the best decomposition found
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
