"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

import time
import random
import sys

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] == delta_{i,j,k}
    where delta=1 if the corresponding product term exists (always true for
    matrix mult tensor), i.e., the tensor entry T[i,j,k] = 1 for all i,j,k
    in the standard matrix multiplication tensor.
    
    Actually, for matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
    (since C[i,k] = sum_j A[i,j]*B[j,k], each (i,j,k) contributes 1).
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for s in range(rank):
                    total += U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _strassen_2x2x2_block(n_block, m_block, p_block, offset_i, offset_j, offset_k):
    """
    Generate Strassen decomposition for a sub-block of size n_block x m_block x p_block
    where all are 2. Returns local U, V, W entries (lists of 7 columns) for
    the 7 Strassen products.
    
    Strassen for 2x2*2x2 -> 2x2:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B00)
    M4 = A11(B01-B11)
    M5 = (A00+A10)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 - M4 + M5 + M7
    C01 = M3 + M4
    C10 = M2 + M3
    C11 = M1 - M2 + M6 + M7
    """
    # U: rows are (i,j) pairs for A, V: rows are (j,k) pairs for B, W: rows are (k,i) pairs for C
    # 7 columns for the 7 Strassen products
    U = [[0.0]*7 for _ in range(4)]  # n*m = 4
    V = [[0.0]*7 for _ in range(4)]  # m*p = 4
    W = [[0.0]*7 for _ in range(4)]  # p*n = 4
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1.0  # A00
    U[3][0] = 1.0  # A11
    V[0][0] = 1.0  # B00
    V[3][0] = 1.0  # B11
    W[0][0] = 1.0  # C00
    W[3][0] = 1.0  # C11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1.0  # A10
    U[3][1] = 1.0  # A11
    V[0][1] = 1.0  # B00
    W[0][1] = -1.0  # C11: -M2
    # C10 gets +M2
    W[2][1] = 1.0  # Wait, C indices: W[k*n+i] where k is row of C, i is col of C
    # Let me re-index: W[k*n + i] for C[k][i]
    # C10 = C[1][0] -> W[1*2+0] = W[2]
    # C11 = C[1][1] -> W[1*2+1] = W[3]
    W[2][1] = 1.0   # C10 += M2
    W[3][1] = -1.0  # C11 -= M2
    
    # M3 = A00(B10-B00)
    U[0][2] = 1.0  # A00
    V[2][2] = 1.0  # B10 (j=1,k=0 -> j*p+k = 1*2+0 = 2)
    V[0][2] = -1.0 # B00 (j=0,k=0 -> 0)
    W[2][2] = 1.0  # C10 += M3
    W[1][2] = 1.0  # C01 += M3
    
    # M4 = A11(B01-B11)
    U[3][3] = 1.0  # A11
    V[1][3] = 1.0  # B01 (j=0,k=1 -> 0*2+1 = 1)
    V[3][3] = -1.0 # B11 (j=1,k=1 -> 1*2+1 = 3)
    W[0][3] = -1.0 # C00 -= M4
    W[1][3] = 1.0  # C01 += M4
    
    # M5 = (A00+A10)B11
    U[0][4] = 1.0  # A00
    U[2][4] = 1.0  # A10
    V[3][4] = 1.0  # B11
    W[0][4] = 1.0  # C00 += M5
    
    # M6 = (A10-A00)(B00+B10)
    U[2][5] = 1.0  # A10
    U[0][5] = -1.0 # A00
    V[0][5] = 1.0  # B00
    V[2][5] = 1.0  # B10
    W[3][5] = 1.0  # C11 += M6
    
    # M7 = (A01-A11)(B01+B11)
    U[1][6] = 1.0  # A01
    U[3][6] = -1.0 # A11
    V[1][6] = 1.0  # B01
    V[3][6] = 1.0  # B11
    W[0][6] = 1.0  # C00 += M7
    W[3][6] = 1.0  # C11 += M7
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 2, 2, 6
    
    # We can decompose this as three independent 2x2 x 2x2 multiplications
    # (since p=6 = 3*2, we can split B into three 2x2 column blocks)
    # Each 2x2 x 2x2 multiplication can be done with Strassen rank 7
    # Total rank = 3 * 7 = 21
    
    rank = 21
    U = [[0.0]*rank for _ in range(n*m)]   # 4 x 21
    V = [[0.0]*rank for _ in range(m*p)]   # 12 x 21
    W = [[0.0]*rank for _ in range(p*n)]   # 12 x 21
    
    # For each of the 3 blocks (column block t = 0, 1, 2 of B, each 2x2)
    # B's columns: for block t, columns 2t and 2t+1
    # A is 2x2, B_block is 2x2, C_block is 2x2
    # In the global indexing:
    #   A[i,j] -> U[i*m+j] = U[i*2+j]
    #   B[j, 2t+k] -> V[j*p + (2t+k)] = V[j*6 + 2t + k]
    #   C[2t+k, i] -> W[(2t+k)*n + i] = W[(2t+k)*2 + i]
    
    for t in range(3):
        block_offset = t * 7  # column offset in rank
        local_U, local_V, local_W = _strassen_2x2x2_block(2, 2, 2, 0, 0, 0)
        
        for s in range(7):
            for i in range(2):
                for j in range(2):
                    idx_u = i*2 + j
                    U[idx_u][block_offset + s] = local_U[idx_u][s]
            
            for j in range(2):
                for k in range(2):
                    idx_v_local = j*2 + k
                    idx_v_global = j*6 + (2*t + k)
                    V[idx_v_global][block_offset + s] = local_V[idx_v_local][s]
            
            for k in range(2):
                for i in range(2):
                    idx_w_local = k*2 + i
                    idx_w_global = (2*t + k)*2 + i
                    W[idx_w_global][block_offset + s] = local_W[idx_w_local][s]
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0]*rank for _ in range(n * m)]
        V = [[0]*rank for _ in range(m * p)]
        W = [[0]*rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to integers if possible
    def to_int_list(lst_of_lists):
        result = []
        for row in lst_of_lists:
            new_row = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    new_row.append(int(round(val)))
                else:
                    new_row.append(val)
            result.append(new_row)
        return result
    
    U = to_int_list(U)
    V = to_int_list(V)
    W = to_int_list(W)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
