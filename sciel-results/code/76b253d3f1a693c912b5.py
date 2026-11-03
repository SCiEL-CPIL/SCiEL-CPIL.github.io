import time
import sys

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will build lists of coefficients for each rank column
    # Actually, it's easier to just initialize arrays and fill them.
    # But rank is not known a priori in the loop, so we accumulate in lists of tuples or dict of lists.
    # Let's use a list of (row, coeff) for U, V, W for each rank index.
    
    # To be efficient, we'll just build the matrices as lists of lists and append columns.
    # Or just use the known rank 44.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # i_off: row offset for A and C (0 or 1 for the 2x2 block)
        # j_off: col offset for A and row offset for B (0 or 2)
        # k_off: col offset for B and C (0 or 2)
        
        # The block is 2x2x2
        # A indices: i in {i_off, i_off+1}, j in {j_off, j_off+1}
        # B indices: j in {j_off, j_off+1}, k in {k_off, k_off+1}
        # C indices: i in {i_off, i_off+1}, k in {k_off, k_off+1}
        
        # Strassen terms:
        # P1: (A00 + A11)(B00 + B11) -> C00 += 1, C11 += 1
        # P2: (A10 + A11)B00       -> C10 += 1, C11 -= 1
        # P3: A00(B01 - B11)       -> C00 -= 1, C01 += 1
        # P4: A11(B10 - B00)       -> C00 += 1, C10 += 1
        # P5: (A00 + A01)B11       -> C00 -= 1, C01 += 1
        # P6: (A10 - A00)(B00 + B01) -> C11 += 1
        # P7: (A01 - A11)(B10 + B11) -> C00 += 1, C11 -= 1
        
        # Mapping to global indices:
        # u_idx(i, j) = i * 4 + j
        # v_idx(j, k) = j * 4 + k
        # w_idx(k, i) = k * 3 + i
        
        # P1
        U[i_off * 4 + j_off][col] += 1
        U[(i_off+1) * 4 + (j_off+1)][col] += 1
        V[j_off * 4 + k_off][col] += 1
        V[(j_off+1) * 4 + (k_off+1)][col] += 1
        W[k_off * 3 + i_off][col] += 1
        W[(k_off+1) * 3 + (i_off+1)][col] += 1
        col += 1
        
        # P2
        U[(i_off+1) * 4 + j_off][col] += 1
        U[(i_off+1) * 4 + (j_off+1)][col] += 1
        V[j_off * 4 + k_off][col] += 1
        W[k_off * 3 + (i_off+1)][col] += 1
        W[(k_off+1) * 3 + (i_off+1)][col] -= 1
        col += 1
        
        # P3
        U[i_off * 4 + j_off][col] += 1
        V[j_off * 4 + (k_off+1)][col] += 1
        V[(j_off+1) * 4 + (k_off+1)][col] -= 1
        W[k_off * 3 + i_off][col] -= 1
        W[(k_off+1) * 3 + i_off][col] += 1
        col += 1
        
        # P4
        U[(i_off+1) * 4 + (j_off+1)][col] += 1
        V[(j_off+1) * 4 + k_off][col] += 1
        V[j_off * 4 + k_off][col] -= 1
        W[k_off * 3 + i_off][col] += 1
        W[k_off * 3 + (i_off+1)][col] += 1
        col += 1
        
        # P5
        U[i_off * 4 + j_off][col] += 1
        U[i_off * 4 + (j_off+1)][col] += 1
        V[(j_off+1) * 4 + (k_off+1)][col] += 1
        W[k_off * 3 + i_off][col] -= 1
        W[(k_off+1) * 3 + i_off][col] += 1
        col += 1
        
        # P6
        U[(i_off+1) * 4 + j_off][col] += 1
        U[i_off * 4 + j_off][col] -= 1
        V[j_off * 4 + k_off][col] += 1
        V[j_off * 4 + (k_off+1)][col] += 1
        W[(k_off+1) * 3 + (i_off+1)][col] += 1
        col += 1
        
        # P7
        U[i_off * 4 + (j_off+1)][col] += 1
        U[(i_off+1) * 4 + (j_off+1)][col] -= 1
        V[(j_off+1) * 4 + k_off][col] += 1
        V[(j_off+1) * 4 + (k_off+1)][col] += 1
        W[k_off * 3 + i_off][col] += 1
        W[(k_off+1) * 3 + (i_off+1)][col] -= 1
        col += 1

    # Apply Strassen to the 2x2x2 blocks for i in {0,1}
    # Block 1: i_off=0, j_off=0, k_off=0
    add_strassen_block(0, 0, 0)
    # Block 2: i_off=0, j_off=0, k_off=2
    add_strassen_block(0, 0, 2)
    # Block 3: i_off=0, j_off=2, k_off=0
    add_strassen_block(0, 2, 0)
    # Block 4: i_off=0, j_off=2, k_off=2
    add_strassen_block(0, 2, 2)
    
    # Wait, the blocks cover i in {0,1}.
    # My add_strassen_block uses i_off and i_off+1.
    # So for i_off=0, it covers i=0 and i=1.
    # This covers all combinations of j in {0,1,2,3} and k in {0,1,2,3} for i in {0,1}.
    # Specifically:
    # add_strassen_block(0, 0, 0) covers A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # add_strassen_block(0, 0, 2) covers A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # add_strassen_block(0, 2, 0) covers A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # add_strassen_block(0, 2, 2) covers A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    
    # Now handle i=2 (the third row of A)
    # Schoolbook for 1x4x4
    for j in range(4):
        for k in range(4):
            U[2 * 4 + j][col] = 1
            V[j * 4 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
