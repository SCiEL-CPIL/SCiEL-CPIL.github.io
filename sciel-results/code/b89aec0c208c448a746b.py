import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # We split the problem into 4 blocks:
    # 1. 2x2x4 (j in 0,1; k in 0,1) -> 2x2x2 Strassen
    # 2. 2x2x4 (j in 0,1; k in 2,3) -> 2x2x2 Strassen
    # 3. 2x1x4 (j in 2; k in 0,1) -> Schoolbook
    # 4. 2x1x4 (j in 2; k in 2,3) -> Schoolbook
    # Wait, p=8. k is 0..7.
    # Split k into 4 blocks of size 2?
    # Or 2 blocks of size 4?
    # My previous logic:
    # Split k into two halves of 4.
    # For each half, split j into 2 and 1.
    # j in {0,1} uses 2x2x4.
    # j in {2} uses 2x1x4.
    
    # So:
    # Block 1: j in {0,1}, k in {0,1,2,3}. (2x2x4)
    # Block 2: j in {2}, k in {0,1,2,3}. (2x1x4)
    # Block 3: j in {0,1}, k in {4,5,6,7}. (2x2x4)
    # Block 4: j in {2}, k in {4,5,6,7}. (2x1x4)
    
    # 2x2x4 decomposes into two 2x2x2.
    # 2x1x4 decomposes into four 2x1x1 (or just 4 scalar mults).
    
    # Total Rank:
    # 2 * (2 * 7) + 2 * (4) = 28 + 8 = 36?
    # Wait. 2x2x4 rank 14.
    # 2x1x4 rank 8.
    # Total: 14 + 8 + 14 + 8 = 44.
    
    def add_strassen(i_offset, j_offset, k_offset, start_col):
        # i_offset: 0 or 1 (i is 0..1, so offset 0)
        # j_offset: 0 or 2 (j index in A/B)
        # k_offset: 0, 2, 4, 6 (k index in B/C)
        # This handles a 2x2x2 block
        # A indices: (i, j_offset), (i, j_offset+1)
        # B indices: (j_offset, k_offset), (j_offset+1, k_offset) ...
        
        # M0 = (A[i, j] + A[i+1, j+1]) * (B[j, k] + B[j+1, k+1])
        # Note: A is 2x3. j_offset is 0 or 1?
        # In our split, j is 0,1 for the first part.
        # So j_offset=0.
        
        # U indices for A:
        # u1 = i * m + j_offset
        # u2 = (i+1) * m + (j_offset+1)
        
        # V indices for B:
        # v1 = j_offset * p + k_offset
        # v2 = (j_offset+1) * p + (k_offset+1)
        
        # W indices for C:
        # w1 = k_offset * n + i
        # w2 = (k_offset+1) * n + (i+1)
        
        # The 7 multiplications:
        
        # M0
        U[u1][start_col] += 1
        U[u2][start_col] += 1
        V[v1][start_col] += 1
        V[v2][start_col] += 1
        # W contributions for M0:
        # c00 += M0, c11 += M0
        W[w1][start_col] += 1
        W[w2][start_col] += 1
        col = start_col + 1
        
        # M1
        u1 = i * m + (j_offset+1)
        u2 = (i+1) * m + (j_offset+1)
        v1 = j_offset * p + k_offset
        # M1 = (A[i, j+1] + A[i+1, j+1]) * B[j, k]
        U[u1][col] += 1
        U[u2][col] += 1
        V[v1][col] += 1
        # c00 -= M1, c10 += M1
        W[w1][col] -= 1
        W[k_offset * n + (i+1)][col] += 1
        col += 1
        
        # M2
        u1 = i * m + j_offset
        v1 = (j_offset+1) * p + (k_offset+1)
        v2 = (j_offset+1) * p + k_offset
        # M2 = A[i, j] * (B[j+1, k+1] - B[j+1, k])
        # Wait, standard M2 = A[i,j] * (B[j+1, k+1] - B[j+1, k])?
        # My previous M2 = a00 (b01 - b11).
        # Here j_offset=0. b01 is B[0,1]. b11 is B[1,1].
        # So v1 should be B[0, k+1] and v2 B[1, k+1]?
        # Let's stick to the explicit indices.
        # M2 = A[i, j] * (B[j, k+1] - B[j+1, k+1]) ?
        # Let's use the verified formulas from before:
        # M2 = a00 (b01 - b11).
        # a00 is A[i, j].
        # b01 is B[j, k+1].
        # b11 is B[j+1, k+1].
        # So:
        U[i * m + j_offset][col] += 1
        V[j_offset * p + (k_offset+1)][col] += 1
        V[(j_offset+1) * p + (k_offset+1)][col] -= 1
        # c01 += M2, c11 -= M2
        W[(k_offset+1) * n + i][col] += 1
        W[w2][col] -= 1
        col += 1
        
        # M3
        # M3 = a11 (b10 - b00)
        # a11 is A[i+1, j+1]
        # b10 is B[j+1, k]
        # b00 is B[j, k]
        U[(i+1) * m + (j_offset+1)][col] += 1
        V[(j_offset+1) * p + k_offset][col] += 1
        V[j_offset * p + k_offset][col] -= 1
        # c00 += M3, c10 += M3, c11 += M3
        W[w1][col] += 1
        W[k_offset * n + (i+1)][col] += 1
        W[w2][col] += 1
        col += 1
        
        # M4
        # M4 = (a00 + a01) b11
        # a00 is A[i, j], a01 is A[i, j+1]
        # b11 is B[j+1, k+1]
        U[i * m + j_offset][col] += 1
        U[i * m + (j_offset+1)][col] += 1
        V[(j_offset+1) * p + (k_offset+1)][col] += 1
        # c01 += M4, c10 -= M4, c11 += M4
        W[(k_offset+1) * n + i][col] += 1
        W[k_offset * n + (i+1)][col] -= 1
        W[w2][col] += 1
        col += 1
        
        # M5
        # M5 = (a01 - a10) (b00 + b01)
        # a01 is A[i, j+1], a10 is A[i+1, j]
        # b00 is B[j, k], b01 is B[j, k+1]
        U[i * m + (j_offset+1)][col] += 1
        U[(i+1) * m + j_offset][col] -= 1
        V[j_offset * p + k_offset][col] += 1
        V[j_offset * p + (k_offset+1)][col] += 1
        # c00 -= M5, c01 += M5, c10 += M5
        W[w1][col] -= 1
        W[(k_offset+1) * n + i][col] += 1
        W[k_offset * n + (i+1)][col] += 1
        col += 1
        
        # M6
        # M6 = (a10 - a11) (b10 + b11)
        # a10 is A[i+1, j], a11 is A[i+1, j+1]
        # b10 is B[j+1, k], b11 is B[j+1, k+1]
        U[(i+1) * m + j_offset][col] += 1
        U[(i+1) * m + (j_offset+1)][col] -= 1
        V[(j_offset+1) * p + k_offset][col] += 1
        V[(j_offset+1) * p + (k_offset+1)][col] += 1
        # c00 += M6, c01 -= M6, c10 += M6
        W[w1][col] += 1
        W[(k_offset+1) * n + i][col] -= 1
        W[k_offset * n + (i+1)][col] += 1
        col += 1
        
        return col

    def add_2x1x2(i_offset, j_idx, k_offset, start_col):
        # 2x1x2 block
        # A is 2x1, B is 1x2
        # C is 2x2
        # M0 = A[i, j] * B[j, k]
        # M1 = A[i, j] * B[j, k+1]
        # M2 = A[i+1, j] * B[j, k]
        # M3 = A[i+1, j] * B[j, k+1]
        
        col = start_col
        # M0
        U[i_offset * m + j_idx][col] += 1
        V[j_idx * p + k_offset][col] += 1
        W[k_offset * n + i_offset][col] += 1
        col += 1
        
        # M1
        U[i_offset * m + j_idx][col] += 1
        V[j_idx * p + (k_offset+1)][col] += 1
        W[(k_offset+1) * n + i_offset][col] += 1
        col += 1
        
        # M2
        U[(i_offset+1) * m + j_idx][col] += 1
        V[j_idx * p + k_offset][col] += 1
        W[k_offset * n + (i_offset+1)][col] += 1
        col += 1
        
        # M3
        U[(i_offset+1) * m + j_idx][col] += 1
        V[j_idx * p + (k_offset+1)][col] += 1
        W[(k_offset+1) * n + (i_offset+1)][col] += 1
        col += 1
        
        return col

    # Block 1: j in {0,1}, k in {0,1,2,3}
    # Two 2x2x2 blocks: k in {0,1} and k in {2,3}
    col = add_strassen(0, 0, 0, col) # i=0, j=0, k=0
    col = add_strassen(0, 0, 2, col) # i=0, j=0, k=2
    
    # Block 2: j in {2}, k in {0,1,2,3}
    # Two 2x1x2 blocks: k in {0,1} and k in {2,3}
    col = add_2x1x2(0, 2, 0, col)
    col = add_2x1x2(0, 2, 2, col)
    
    # Block 3: j in {0,1}, k in {4,5,6,7}
    col = add_strassen(0, 0, 4, col)
    col = add_strassen(0, 0, 6, col)
    
    # Block 4: j in {2}, k in {4,5,6,7}
    col = add_2x1x2(0, 2, 4, col)
    col = add_2x1x2(0, 2, 6, col)
    
    return {"U": U, "V": V, "W": W}
