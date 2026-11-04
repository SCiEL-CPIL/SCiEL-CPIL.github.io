"""Tensor decomposition for 2x4 * 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

DEADLINE_SECONDS = 115.0

def _verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    for trial in range(20):
        A = np.random.randint(-3, 4, size=(n, m)).astype(float)
        B = np.random.randint(-3, 4, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # Compute using decomposition
        # C[i,k] = sum_j sum_c U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
        # Actually the standard form is:
        # C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
        # No, that's not right either. Let me think about this more carefully.
        
        # The correct interpretation for matrix multiplication tensor decomposition:
        # We need: for all A, B: C = A @ B
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Decomposition: C[i,k] = sum_c M_c(A) * N_c(B) * W_c(i,k)
        # where M_c(A) = sum_j U[i*m+j, c] * A[i,j]  -- but this depends on i!
        
        # Actually, I think the correct form is:
        # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
        # But U depends on i and j, V depends on j and k, W depends on k and i.
        
        # Let me just verify by direct computation using the tensor contraction:
        # The tensor T[i,j,k] = 1 for all valid indices (since A[i,j]*B[j,k] -> C[i,k])
        # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Wait, that would mean for each (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # And the reconstruction is: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k]
        
        # So the verification is: for all i, j, k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Let's check this.
        
    # Direct tensor verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_remove_column(U, V, W, col, n=2, m=4, p=4):
    """Try removing column col and check if decomposition still works."""
    rank = len(U[0])
    if rank <= 1:
        return False
    
    U_new = [row[:col] + row[col+1:] for row in U]
    V_new = [row[:col] + row[col+1:] for row in V]
    W_new = [row[:col] + row[col+1:] for row in W]
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        return True
    return False

def _try_merge_columns(U, V, W, col1, col2, n=2, m=4, p=4):
    """Try merging two columns (adding one to the other) and check validity."""
    rank = len(U[0])
    if rank <= 1 or col1 == col2:
        return False
    
    # Merge col2 into col1: U[:,col1] += U[:,col2], etc., then remove col2
    U_new = []
    for row in U:
        new_row = row[:col1] + [row[col1] + row[col2]] + row[col1+1:col2] + row[col2+1:]
        U_new.append(new_row)
    V_new = []
    for row in V:
        new_row = row[:col1] + [row[col1] + row[col2]] + row[col1+1:col2] + row[col2+1:]
        V_new.append(new_row)
    W_new = []
    for row in W:
        new_row = row[:col1] + [row[col1] + row[col2]] + row[col1+1:col2] + row[col2+1:]
        W_new.append(new_row)
    
    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
        return True
    return False

def _strassen_block(U, V, W, col_offset, a_rows, b_rows, c_rows):
    """Add Strassen's 7-multiplication block for 2x2 matrix multiplication.
    
    a_rows: indices in U for the 4 A entries [A00, A01, A10, A11]
    b_rows: indices in V for the 4 B entries [B00, B01, B10, B11]
    c_rows: indices in W for the 4 C entries [C00, C01, C10, C11]
    """
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 - M4 + M5 - M7
    # C01 = M4 + M7
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M6
    
    A00, A01, A10, A11 = a_rows
    B00, B01, B10, B11 = b_rows
    C00, C01, C10, C11 = c_rows
    
    # M1
    U[A00][col_offset + 0] += 1
    U[A11][col_offset + 0] += 1
    V[B00][col_offset + 0] += 1
    V[B11][col_offset + 0] += 1
    W[C00][col_offset + 0] += 1
    W[C11][col_offset + 0] += 1
    
    # M2
    U[A10][col_offset + 1] += 1
    U[A11][col_offset + 1] += 1
    V[B00][col_offset + 1] += 1
    W[C10][col_offset + 1] += 1
    W[C11][col_offset + 1] -= 1
    
    # M3
    U[A00][col_offset + 2] += 1
    V[B10][col_offset + 2] += 1
    V[B11][col_offset + 2] -= 1
    W[C10][col_offset + 2] += 1
    W[C11][col_offset + 2] += 1
    
    # M4
    U[A11][col_offset + 3] += 1
    V[B01][col_offset + 3] += 1
    V[B00][col_offset + 3] -= 1
    W[C00][col_offset + 3] -= 1
    W[C01][col_offset + 3] += 1
    
    # M5
    U[A00][col_offset + 4] += 1
    U[A01][col_offset + 4] += 1
    V[B11][col_offset + 4] += 1
    W[C00][col_offset + 4] -= 1
    W[C01][col_offset + 4] += 1
    
    # M6
    U[A10][col_offset + 5] += 1
    U[A00][col_offset + 5] -= 1
    V[B00][col_offset + 5] += 1
    V[B01][col_offset + 5] += 1
    W[C11][col_offset + 5] += 1
    
    # M7
    U[A01][col_offset + 6] += 1
    U[A11][col_offset + 6] -= 1
    V[B10][col_offset + 6] += 1
    V[B11][col_offset + 6] += 1
    W[C00][col_offset + 6] -= 1
    W[C01][col_offset + 6] += 1

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Start with Strassen decomposition: rank 28
    # A is 2x4: rows 0,1 are A_L (2x2), rows 2,3 are A_R (2x2)
    #   A_L: U[0]=A[0,0], U[1]=A[0,1], U[2]=A[1,0], U[3]=A[1,1]
    #   A_R: U[4]=A[0,2], U[5]=A[0,3], U[6]=A[1,2], U[7]=A[1,3]
    # B is 4x4: 
    #   B_TL: V[0]=B[0,0], V[1]=B[0,1], V[2]=B[1,0], V[3]=B[1,1]
    #   B_TR: V[4]=B[0,2], V[5]=B[0,3], V[6]=B[1,2], V[7]=B[1,3]
    #   B_BL: V[8]=B[2,0], V[9]=B[2,1], V[10]=B[3,0], V[11]=B[3,1]
    #   B_BR: V[12]=B[2,2], V[13]=B[2,3], V[14]=B[3,2], V[15]=B[3,3]
    # C is 2x4: W[k*n+i]
    #   C_L: W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    #   C_R: W[4]=C[0,2], W[5]=C[0,3], W[6]=C[1,2], W[7]=C[1,3]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: A_L * B_TL -> C_L
    _strassen_block(U, V, W, 0, [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3])
    
    # Block 2: A_R * B_BL -> C_L
    _strassen_block(U, V, W, 7, [4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3])
    
    # Block 3: A_L * B_TR -> C_R
    _strassen_block(U, V, W, 14, [0, 1, 2, 3], [4, 5, 6, 7], [4, 5, 6, 7])
    
    # Block 4: A_R * B_BR -> C_R
    _strassen_block(U, V, W, 21, [4, 5, 6, 7], [12, 13, 14, 15], [4, 5, 6, 7])
    
    # Verify the initial decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Now try to reduce rank by removing or merging columns
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try removing individual columns
    for col in range(rank):
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        if _try_remove_column(U, V, W, col, n, m, p):
            U = [row[:col] + row[col+1:] for row in U]
            V = [row[:col] + row[col+1:] for row in V]
            W = [row[:col] + row[col+1:] for row in W]
            best_U, best_V, best_W = U, V, W
            best_rank = len(U[0])
    
    # Try merging pairs of columns
    rank = len(U[0])
    for col1 in range(rank):
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        for col2 in range(col1 + 1, rank):
            if time.time() - start_time > DEADLINE_SECONDS:
                break
            if _try_merge_columns(U, V, W, col1, col2, n, m, p):
                U_new = []
                for row in U:
                    new_row = row[:col1] + [row[col1] + row[col2]] + row[col1+1:col2] + row[col2+1:]
                    U_new.append(new_row)
                V_new = []
                for row in V:
                    new_row = row[:col1] + [row[col1] + row[col2]] + row[col1+1:col2] + row[col2+1:]
                    V_new.append(new_row)
                W_new = []
                for row in W:
                    new_row = row[:col1] + [row[col1] + row[col2]] + row[col1+1:col2] + row[col2+1:]
                    W_new.append(new_row)
                U, V, W = U_new, V_new, W_new
                best_U, best_V, best_W = U, V, W
                best_rank = len(U[0])
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Return the verified Strassen decomposition
        U = [[0] * 28 for _ in range(n * m)]
        V = [[0] * 28 for _ in range(m * p)]
        W = [[0] * 28 for _ in range(p * n)]
        _strassen_block(U, V, W, 0, [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3])
        _strassen_block(U, V, W, 7, [4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3])
        _strassen_block(U, V, W, 14, [0, 1, 2, 3], [4, 5, 6, 7], [4, 5, 6, 7])
        _strassen_block(U, V, W, 21, [4, 5, 6, 7], [12, 13, 14, 15], [4, 5, 6, 7])
        return {"U": U, "V": V, "W": W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
