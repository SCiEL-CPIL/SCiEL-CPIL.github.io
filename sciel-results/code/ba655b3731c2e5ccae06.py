import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Strategy:
    # Use Strassen's algorithm (rank 7) for all 2x2x2 blocks.
    # Split m=5 into j_pairs: (0,1), (2,3), and remaining j=4.
    # Split p=6 into k_pairs: (0,1), (2,3), (4,5).
    
    # Blocks:
    # 1. j in {0,1}, k in {0,1} -> 2x2x2 (Strassen)
    # 2. j in {0,1}, k in {2,3} -> 2x2x2 (Strassen)
    # 3. j in {0,1}, k in {4,5} -> 2x2x2 (Strassen)
    # 4. j in {2,3}, k in {0,1} -> 2x2x2 (Strassen)
    # 5. j in {2,3}, k in {2,3} -> 2x2x2 (Strassen)
    # 6. j in {2,3}, k in {4,5} -> 2x2x2 (Strassen)
    # 7. j=4, k in {0,1} -> 2x1x2 (Rank 1? No, rank 2 for separate i, but rank 1 for combined? 
    #    Wait, T_{i,4,k} = 1. This is a 2x1x2 tensor.
    #    As derived before, rank 1 is sufficient if we can combine i and k?
    #    T_{i,4,k} = U_{i,4} V_{4,k} W_{k,i}.
    #    If U=[1,1], V=[1,1], W=[1,1,1,1].
    #    T_{0,4,0} = 1*1*1 = 1
    #    T_{0,4,1} = 1*1*1 = 1
    #    T_{1,4,0} = 1*1*1 = 1
    #    T_{1,4,1} = 1*1*1 = 1
    #    Yes, rank 1 works for 2x1x2.
    #    So 3 blocks of 2x1x2 for j=4, k in {0,1}, {2,3}, {4,5}.
    #    Total rank = 6 * 7 + 3 * 1 = 42 + 3 = 45.
    
    rank = 45
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper: Add Strassen 2x2x2 block
    # i_offset, j_offset, k_offset
    def add_strassen_block(i_offset, j_offset, k_offset, start_col):
        # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        # Global: gi = i_offset + i, gj = j_offset + j, gk = k_offset + k
        
        # Strassen 7 terms
        # 1. (A00+A11)(B00+B11)
        # U: A00, A11
        # V: B00, B01, B10, B11
        # W: C00, C01, C10, C11
        
        # Term 1
        U[(i_offset+0)*m + (j_offset+0), start_col] += 1
        U[(i_offset+1)*m + (j_offset+1), start_col] += 1
        V[(j_offset+0)*p + (k_offset+0), start_col] += 1
        V[(j_offset+0)*p + (k_offset+1), start_col] += 1
        V[(j_offset+1)*p + (k_offset+0), start_col] += 1
        V[(j_offset+1)*p + (k_offset+1), start_col] += 1
        W[(k_offset+0)*n + (i_offset+0), start_col] += 1
        W[(k_offset+1)*n + (i_offset+0), start_col] += 1
        W[(k_offset+0)*n + (i_offset+1), start_col] += 1
        W[(k_offset+1)*n + (i_offset+1), start_col] += 1
        start_col += 1
        
        # Term 2: (A10+A11)B00
        U[(i_offset+1)*m + (j_offset+0), start_col] += 1
        U[(i_offset+1)*m + (j_offset+1), start_col] += 1
        V[(j_offset+0)*p + (k_offset+0), start_col] += 1
        W[(k_offset+0)*n + (i_offset+1), start_col] += 1
        W[(k_offset+1)*n + (i_offset+1), start_col] -= 1
        start_col += 1
        
        # Term 3: A00(B01-B11)
        U[(i_offset+0)*m + (j_offset+0), start_col] += 1
        V[(j_offset+0)*p + (k_offset+1), start_col] += 1
        V[(j_offset+1)*p + (k_offset+1), start_col] -= 1
        W[(k_offset+1)*n + (i_offset+0), start_col] += 1
        W[(k_offset+1)*n + (i_offset+1), start_col] += 1
        start_col += 1
        
        # Term 4: A11(B10-B00)
        U[(i_offset+1)*m + (j_offset+1), start_col] += 1
        V[(j_offset+1)*p + (k_offset+0), start_col] += 1
        V[(j_offset+0)*p + (k_offset+0), start_col] -= 1
        W[(k_offset+0)*n + (i_offset+0), start_col] += 1
        W[(k_offset+0)*n + (i_offset+1), start_col] += 1
        start_col += 1
        
        # Term 5: (A00+A01)B11
        U[(i_offset+0)*m + (j_offset+0), start_col] += 1
        U[(i_offset+0)*m + (j_offset+1), start_col] += 1
        V[(j_offset+1)*p + (k_offset+1), start_col] += 1
        W[(k_offset+0)*n + (i_offset+0), start_col] -= 1
        W[(k_offset+1)*n + (i_offset+0), start_col] += 1
        start_col += 1
        
        # Term 6: (A10-A00)(B00+B10)
        U[(i_offset+1)*m + (j_offset+0), start_col] += 1
        U[(i_offset+0)*m + (j_offset+0), start_col] -= 1
        V[(j_offset+0)*p + (k_offset+0), start_col] += 1
        V[(j_offset+1)*p + (k_offset+0), start_col] += 1
        W[(k_offset+1)*n + (i_offset+1), start_col] += 1
        start_col += 1
        
        # Term 7: (A01-A11)(B01+B11)
        U[(i_offset+0)*m + (j_offset+1), start_col] += 1
        U[(i_offset+1)*m + (j_offset+1), start_col] -= 1
        V[(j_offset+0)*p + (k_offset+1), start_col] += 1
        V[(j_offset+1)*p + (k_offset+1), start_col] += 1
        W[(k_offset+0)*n + (i_offset+0), start_col] += 1
        start_col += 1
        
        return start_col

    # Helper: Add 2x1x2 block (Rank 1)
    def add_2x1x2_block(i_offset, j_offset, k_offset, start_col):
        # i in {0,1}, j fixed, k in {0,1}
        gi0 = i_offset
        gi1 = i_offset + 1
        gj = j_offset
        gk0 = k_offset
        gk1 = k_offset + 1
        
        # U: both i's
        U[gi0 * m + gj, start_col] += 1
        U[gi1 * m + gj, start_col] += 1
        
        # V: both k's
        V[gj * p + gk0, start_col] += 1
        V[gj * p + gk1, start_col] += 1
        
        # W: all 4 combinations
        W[gk0 * n + gi0, start_col] += 1
        W[gk0 * n + gi1, start_col] += 1
        W[gk1 * n + gi0, start_col] += 1
        W[gk1 * n + gi1, start_col] += 1
        
        return start_col + 1

    # Apply Strassen blocks
    # j pairs: (0,1), (2,3)
    # k pairs: (0,1), (2,3), (4,5)
    
    for j0, j1 in [(0, 1), (2, 3)]:
        for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
            col = add_strassen_block(0, j0, k0, col)
            
    # Apply 2x1x2 blocks for j=4
    for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
        col = add_2x1x2_block(0, 4, k0, col)
        
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for alpha in range(col):
                    s += U[i * m + j, alpha] * V[j * p + k, alpha] * W[k * n + i, alpha]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    
    if valid:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to standard rank
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
