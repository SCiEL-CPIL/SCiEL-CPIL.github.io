"""Matrix multiplication 2,4,7 with rank-minimal decomposition using modular block construction."""
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target Rank: 50
    # Strategy: Split m=4 into two blocks of size 2.
    # For each block (size 2), split p=7 into three blocks of size 2 and one of size 1.
    # Block (2x2) * (2x2) -> Rank 7 (Strassen)
    # Block (2x2) * (2x1) -> Rank 4 (Schoolbook)
    # Total Rank per m-block: 3 * 7 + 1 * 4 = 25.
    # Total Rank: 2 * 25 = 50.
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_block(u_idx, v_idx, w_idx, offset):
        """
        Adds a Strassen 2x2x2 decomposition.
        u_idx: list of 4 indices for U (A matrix)
        v_idx: list of 4 indices for V (B matrix)
        w_idx: list of 4 indices for W (C matrix)
        offset: column offset in the global matrices
        """
        # U indices: [A00, A01, A10, A11]
        # V indices: [B00, B01, B10, B11]
        # W indices: [C00, C01, C10, C11] corresponding to W[k*n+i]
        # C00 is W[k0*n+i0], C01 is W[k0*n+i1], C10 is W[k1*n+i0], C11 is W[k1*n+i1]
        
        # M1 = (A00+A11)(B00+B11)
        U[u_idx[0], offset] = 1
        U[u_idx[3], offset] = 1
        V[v_idx[0], offset] = 1
        V[v_idx[3], offset] = 1
        W[w_idx[0], offset] = 1  # C00
        W[w_idx[2], offset] = -1 # C10
        W[w_idx[1], offset] = 1  # C01
        W[w_idx[3], offset] = 1  # C11
        offset += 1
        
        # M2 = (A10+A11)(B00+B01)
        U[u_idx[2], offset] = 1
        U[u_idx[3], offset] = 1
        V[v_idx[0], offset] = 1
        V[v_idx[1], offset] = 1
        W[w_idx[0], offset] = -1 # C00
        W[w_idx[2], offset] = 1  # C10
        offset += 1
        
        # M3 = (A00-A11)(B10+B11)
        U[u_idx[0], offset] = 1
        U[u_idx[3], offset] = -1
        V[v_idx[2], offset] = 1
        V[v_idx[3], offset] = 1
        W[w_idx[3], offset] = 1  # C11
        offset += 1
        
        # M4 = A00(B10-B11)
        U[u_idx[0], offset] = 1
        V[v_idx[2], offset] = 1
        V[v_idx[3], offset] = -1
        W[w_idx[0], offset] = 1  # C00
        W[w_idx[3], offset] = -1 # C11
        offset += 1
        
        # M5 = A11(B01-B00)
        U[u_idx[3], offset] = 1
        V[v_idx[1], offset] = 1
        V[v_idx[0], offset] = -1
        W[w_idx[0], offset] = -1 # C00
        W[w_idx[2], offset] = 1  # C10
        offset += 1
        
        # M6 = (A00+A01)B11
        U[u_idx[0], offset] = 1
        U[u_idx[1], offset] = 1
        V[v_idx[3], offset] = 1
        W[w_idx[0], offset] = -1 # C00
        W[w_idx[1], offset] = 1  # C01
        W[w_idx[3], offset] = -1 # C11
        offset += 1
        
        # M7 = (A10-A00)B00
        U[u_idx[2], offset] = 1
        U[u_idx[0], offset] = -1
        V[v_idx[0], offset] = 1
        W[w_idx[0], offset] = 1  # C00
        W[w_idx[2], offset] = -1 # C10
        offset += 1
        
        return offset

    def add_schoolbook_2x2x1_block(u_idx, v_idx, w_idx, offset):
        """
        Adds a schoolbook 2x2 * 2x1 decomposition.
        u_idx: [A00, A01, A10, A11]
        v_idx: [B00, B10] (B is 2x1, so only col 0)
        w_idx: [C0, C1] (C is 2x1)
        """
        # C0 = A00*B00 + A01*B10
        # C1 = A10*B00 + A11*B10
        
        # P1: A00 * B00
        U[u_idx[0], offset] = 1
        V[v_idx[0], offset] = 1
        W[w_idx[0], offset] = 1
        offset += 1
        
        # P2: A01 * B10
        U[u_idx[1], offset] = 1
        V[v_idx[1], offset] = 1
        W[w_idx[0], offset] = 1
        offset += 1
        
        # P3: A10 * B00
        U[u_idx[2], offset] = 1
        V[v_idx[0], offset] = 1
        W[w_idx[1], offset] = 1
        offset += 1
        
        # P4: A11 * B10
        U[u_idx[3], offset] = 1
        V[v_idx[1], offset] = 1
        W[w_idx[1], offset] = 1
        offset += 1
        
        return offset

    # Iterate over m-blocks (size 2)
    for jb in range(2):
        j_base = jb * 2
        
        # Iterate over p-blocks
        # 3 blocks of size 2, 1 block of size 1
        for kb in range(4):
            if kb < 3:
                # Size 2 block
                k_base = kb * 2
                
                # U indices for A[i, j] where i in {0,1}, j in {j_base, j_base+1}
                # A[0, j_base] -> 0*4 + j_base
                # A[0, j_base+1] -> 0*4 + j_base + 1
                # A[1, j_base] -> 1*4 + j_base
                # A[1, j_base+1] -> 1*4 + j_base + 1
                u_idx = [
                    0 * m + j_base,
                    0 * m + j_base + 1,
                    1 * m + j_base,
                    1 * m + j_base + 1
                ]
                
                # V indices for B[j, k] where j in {j_base, j_base+1}, k in {k_base, k_base+1}
                # B[j_base, k_base] -> j_base * p + k_base
                # B[j_base, k_base+1] -> j_base * p + k_base + 1
                # B[j_base+1, k_base] -> (j_base+1) * p + k_base
                # B[j_base+1, k_base+1] -> (j_base+1) * p + k_base + 1
                v_idx = [
                    j_base * p + k_base,
                    j_base * p + k_base + 1,
                    (j_base + 1) * p + k_base,
                    (j_base + 1) * p + k_base + 1
                ]
                
                # W indices for C[k, i] where k in {k_base, k_base+1}, i in {0,1}
                # C[k_base, 0] -> k_base * n + 0
                # C[k_base, 1] -> k_base * n + 1
                # C[k_base+1, 0] -> (k_base+1) * n + 0
                # C[k_base+1, 1] -> (k_base+1) * n + 1
                w_idx = [
                    k_base * n + 0,
                    k_base * n + 1,
                    (k_base + 1) * n + 0,
                    (k_base + 1) * n + 1
                ]
                
                col = add_strassen_block(u_idx, v_idx, w_idx, col)
            else:
                # Size 1 block (last one)
                k_base = 6
                
                u_idx = [
                    0 * m + j_base,
                    0 * m + j_base + 1,
                    1 * m + j_base,
                    1 * m + j_base + 1
                ]
                
                # V indices for B[j, k_base]
                v_idx = [
                    j_base * p + k_base,
                    (j_base + 1) * p + k_base
                ]
                
                # W indices for C[k_base, i]
                w_idx = [
                    k_base * n + 0,
                    k_base * n + 1
                ]
                
                col = add_schoolbook_2x2x1_block(u_idx, v_idx, w_idx, col)

    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
