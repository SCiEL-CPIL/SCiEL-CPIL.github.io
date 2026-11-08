import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank calculated: 39
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # j_off: offset for j index (0 or 1)
    # k_off: offset for k index
    def add_strassen_222(j_off, k_off):
        nonlocal col
        # Term 0
        U[0, col] += 1; U[1, col] += 1
        V[j_off+0, col] += 1; V[j_off+1, col] += 1
        W[0*2+0, col] += 1; W[1*2+1, col] += 1
        col += 1
        
        # Term 1
        U[1, col] += 1; U[1, col] += 1 # Wait, U[1, j_off+0] and U[1, j_off+1]
        # Correction: U[1, j_off+0] += 1; U[1, j_off+1] += 1
        U[1, j_off+0] += 1; U[1, j_off+1] += 1
        V[j_off+0, col] += 1
        W[0*2+1, col] += 1
        col += 1
        
        # Term 2
        U[0, j_off+0] += 1
        V[j_off+0, col] += 1; V[j_off+1, col] -= 1
        W[1*2+0, col] += 1
        col += 1
        
        # Term 3
        U[1, j_off+1] += 1
        V[j_off+1, col] += 1; V[j_off+0, col] -= 1
        W[0*2+0, col] += 1
        col += 1
        
        # Term 4
        U[0, j_off+0] += 1; U[0, j_off+1] += 1
        V[j_off+1, col] += 1
        W[1*2+1, col] += 1
        col += 1
        
        # Term 5
        U[1, j_off+0] += 1; U[0, j_off+0] -= 1
        V[j_off+0, col] += 1; V[j_off+0, col] += 1 # V[0,0]+V[0,1]
        # Correction: V[j_off+0, col] += 1; V[j_off+0, col] += 1 is wrong.
        # V[j_off+0, col] += 1; V[j_off+0, col] += 1 -> 2? No.
        # Term 5 V: B_00 + B_01.
        # V[j_off+0, col] += 1; V[j_off+0, col] += 1? No, k indices are 0 and 1.
        # V[j_off+0, col] += 1; V[j_off+0, col] += 1 is same index.
        # B_00 is V[j_off+0, k_off+0]. B_01 is V[j_off+0, k_off+1].
        # In local V matrix, row j, col k.
        # So V[j_off+0, col] += 1 (for k=0) and V[j_off+0, col] += 1 (for k=1)?
        # No, V is (m*p) x rank.
        # V index is j*p + k.
        # So V[(j_off+0)*p + (k_off+0), col] += 1
        # V[(j_off+0)*p + (k_off+1), col] += 1
        
        # Let's rewrite the helper to be precise with global indices.
        pass

    # Re-implementing helper with global indices clearly
    def add_strassen_222(j_off, k_off):
        nonlocal col
        # Local j in {0,1}, k in {0,1}
        # Global J = j_off + j_loc
        # Global K = k_off + k_loc
        
        # Term 0: A_00+A_11, B_00+B_11, C_00+C_11
        U[0*m + j_off+0, col] += 1
        U[1*m + j_off+1, col] += 1
        V[(j_off+0)*p + (k_off+0), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[(k_off+0)*n + 0, col] += 1
        W[(k_off+1)*n + 1, col] += 1
        col += 1
        
        # Term 1: A_10+A_11, B_00, C_10
        U[1*m + j_off+0, col] += 1
        U[1*m + j_off+1, col] += 1
        V[(j_off+0)*p + (k_off+0), col] += 1
        W[(k_off+0)*n + 1, col] += 1
        col += 1
        
        # Term 2: A_00, B_01-B_11, C_01
        U[0*m + j_off+0, col] += 1
        V[(j_off+0)*p + (k_off+1), col] += 1
        V[(j_off+1)*p + (k_off+1), col] -= 1
        W[(k_off+1)*n + 0, col] += 1
        col += 1
        
        # Term 3: A_11, B_10-B_00, C_00
        U[1*m + j_off+1, col] += 1
        V[(j_off+1)*p + (k_off+0), col] += 1
        V[(j_off+0)*p + (k_off+0), col] -= 1
        W[(k_off+0)*n + 0, col] += 1
        col += 1
        
        # Term 4: A_00+A_01, B_11, C_10
        U[0*m + j_off+0, col] += 1
        U[0*m + j_off+1, col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[(k_off+1)*n + 1, col] += 1
        col += 1
        
        # Term 5: A_10-A_00, B_00+B_01, -C_00+C_01
        U[1*m + j_off+0, col] += 1
        U[0*m + j_off+0, col] -= 1
        V[(j_off+0)*p + (k_off+0), col] += 1
        V[(j_off+0)*p + (k_off+1), col] += 1
        W[(k_off+0)*n + 0, col] -= 1
        W[(k_off+1)*n + 0, col] += 1
        col += 1
        
        # Term 6: A_01-A_11, B_10+B_11, -C_00+C_11
        U[0*m + j_off+1, col] += 1
        U[1*m + j_off+1, col] -= 1
        V[(j_off+1)*p + (k_off+0), col] += 1
        V[(j_off+1)*p + (k_off+1), col] += 1
        W[(k_off+0)*n + 0, col] -= 1
        W[(k_off+1)*n + 1, col] += 1
        col += 1

    def add_schoolbook_221(j_off, k_off):
        nonlocal col
        # j in {0,1}, k=0
        for i in range(2):
            for j in range(2):
                U[i*m + j_off+j, col] += 1
                V[(j_off+j)*p + (k_off+0), col] += 1
                W[(k_off+0)*n + i, col] += 1
                col += 1

    def add_schoolbook_217(j_idx, k_start, k_end):
        nonlocal col
        # j fixed, k in range
        for i in range(2):
            for k in range(k_start, k_end):
                U[i*m + j_idx, col] += 1
                V[j_idx*p + k, col] += 1
                W[k*n + i, col] += 1
                col += 1

    # Part 1a: j in {0,1}, k in {0,1}
    add_strassen_222(0, 0)
    
    # Part 1b-i: j in {0,1}, k in {2,3}
    add_strassen_222(0, 2)
    
    # Part 1b-ii-a: j in {0,1}, k in {4,5}
    add_strassen_222(0, 4)
    
    # Part 1b-ii-b: j in {0,1}, k=6
    add_schoolbook_221(0, 6)
    
    # Part 2: j=2, k in {0..6}
    add_schoolbook_217(2, 0, 7)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
