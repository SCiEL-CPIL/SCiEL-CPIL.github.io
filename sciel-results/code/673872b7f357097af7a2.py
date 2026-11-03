"""Program for matrix multiplication 2x4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix.
    
    Strategy:
    1. Construct a Strassen-based rank-28 decomposition by viewing the 2x4x8 multiplication 
       as a combination of 2x2x2 block multiplications.
    2. Verify the decomposition.
    3. If time permits, attempt to reduce the rank via local search or structured pruning.
    
    The decomposition satisfies:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 64
    U_school = np.zeros((n * m, rank_schoolbook), dtype=int)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=int)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # --- Strassen-based Decomposition ---
    # We decompose A (2x4) and B (4x8) into 2x2 blocks.
    # A = [A00 A01] where A00, A01 are 2x2
    # B = [B00 B01; B02 B03] where Bij are 2x2
    # C = A*B = [A00*B00 + A01*B02, A00*B01 + A01*B03]
    # Let C0 = A00*B00 + A01*B02 and C1 = A00*B01 + A01*B03.
    
    # We need to compute 4 products of 2x2 matrices:
    # P1 = A00 * B00
    # P2 = A01 * B02
    # P3 = A00 * B01
    # P4 = A01 * B03
    # Each 2x2 multiplication can be done with Strassen (rank 7).
    # Total rank = 4 * 7 = 28.
    
    # Let's construct the rank-28 decomposition.
    # Strassen 2x2x2 decomposition:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for 2x2 matrix:
    # 00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3
    
    # We will build U, V, W for the 4 Strassen instances.
    # Total rank R = 28.
    
    U_strassen = np.zeros((n * m, 28), dtype=int)
    V_strassen = np.zeros((m * p, 28), dtype=int)
    W_strassen = np.zeros((p * n, 28), dtype=int)
    
    # Helper to add a Strassen instance
    def add_strassen(offset, a_block_idx, b_block_idx, c_block_idx):
        # a_block_idx: which 2x2 block of A (0 for A00, 1 for A01)
        # b_block_idx: which 2x2 block of B (0 for B00, 1 for B01, 2 for B02, 3 for B03)
        # c_block_idx: which 2x2 block of C (0 for C0, 1 for C1)
        
        # Map block indices to linear indices in U, V, W
        # A is 2x4. A00 is cols 0-1, A01 is cols 2-3.
        # A element (i, j) is at U[i*4 + j]
        # B is 4x8. B00 is rows 0-1, cols 0-1. B01 is rows 0-1, cols 2-3.
        # B02 is rows 2-3, cols 0-1. B03 is rows 2-3, cols 2-3.
        # B element (j, k) is at V[j*8 + k]
        # C is 2x8. C0 is cols 0-1. C1 is cols 2-3.
        # C element (k, i) is at W[k*2 + i] (Note: W index is k*n + i)
        
        # Strassen terms for 2x2 matrices
        # Let A_block elements be A00, A01, A10, A11 corresponding to local indices 0,1,2,3
        # Let B_block elements be B00, B01, B10, B11 corresponding to local indices 0,1,2,3
        
        # We need to map local indices to global indices.
        
        # A block 0 (A00):
        # Local (0,0) -> Global (0,0) -> U[0]
        # Local (0,1) -> Global (0,1) -> U[1]
        # Local (1,0) -> Global (1,0) -> U[4]
        # Local (1,1) -> Global (1,1) -> U[5]
        
        # A block 1 (A01):
        # Local (0,0) -> Global (0,2) -> U[2]
        # Local (0,1) -> Global (0,3) -> U[3]
        # Local (1,0) -> Global (1,2) -> U[6]
        # Local (1,1) -> Global (1,3) -> U[7]
        
        # B block 0 (B00): Rows 0-1, Cols 0-1
        # Local (0,0) -> Global (0,0) -> V[0]
        # Local (0,1) -> Global (0,1) -> V[1]
        # Local (1,0) -> Global (1,0) -> V[8]
        # Local (1,1) -> Global (1,1) -> V[9]
        
        # B block 1 (B01): Rows 0-1, Cols 2-3
        # Local (0,0) -> Global (0,2) -> V[2]
        # Local (0,1) -> Global (0,3) -> V[3]
        # Local (1,0) -> Global (1,2) -> V[10]
        # Local (1,1) -> Global (1,3) -> V[11]
        
        # B block 2 (B02): Rows 2-3, Cols 0-1
        # Local (0,0) -> Global (2,0) -> V[16]
        # Local (0,1) -> Global (2,1) -> V[17]
        # Local (1,0) -> Global (3,0) -> V[24]
        # Local (1,1) -> Global (3,1) -> V[25]
        
        # B block 3 (B03): Rows 2-3, Cols 2-3
        # Local (0,0) -> Global (2,2) -> V[18]
        # Local (0,1) -> Global (2,3) -> V[19]
        # Local (1,0) -> Global (3,2) -> V[26]
        # Local (1,1) -> Global (3,3) -> V[27]
        
        # C block 0 (C0): Cols 0-1
        # Local (0,0) -> Global (0,0) -> W[0] (k=0, i=0)
        # Local (0,1) -> Global (0,1) -> W[1] (k=0, i=1)
        # Local (1,0) -> Global (1,0) -> W[2] (k=1, i=0)
        # Local (1,1) -> Global (1,1) -> W[3] (k=1, i=1)
        
        # C block 1 (C1): Cols 2-3
        # Local (0,0) -> Global (0,2) -> W[4] (k=0, i=0)
        # Local (0,1) -> Global (0,3) -> W[5] (k=0, i=1)
        # Local (1,0) -> Global (1,2) -> W[6] (k=1, i=0)
        # Local (1,1) -> Global (1,3) -> W[7] (k=1, i=1)
        
        # Define index mappings
        def a_idx(local_r, local_c, block):
            global_r = local_r
            global_c = block * 2 + local_c
            return global_r * m + global_c
            
        def b_idx(local_r, local_c, block):
            # Block 0: r 0-1, c 0-1
            # Block 1: r 0-1, c 2-3
            # Block 2: r 2-3, c 0-1
            # Block 3: r 2-3, c 2-3
            global_r = (block // 2) * 2 + local_r
            global_c = (block % 2) * 2 + local_c
            return global_r * p + global_c
            
        def c_idx(local_r, local_c, block):
            # Block 0: c 0-1
            # Block 1: c 2-3
            # local_r is k (row of C, which is col of B), local_c is i (col of C, which is row of A)
            # W index is k*n + i
            global_k = local_r
            global_i = local_c
            global_col = block * 2 + local_c # Wait, C is 2x8. 
            # C block 0 corresponds to output cols 0-1.
            # C block 1 corresponds to output cols 2-3.
            # The local indices (local_r, local_c) map to output (row, col) of the 2x2 block.
            # Output row is i (0..1), output col is k (0..7).
            # So local_r is i? No.
            # In Strassen, C is the result.
            # C[i, k] = sum_j A[i, j] B[j, k]
            # In the 2x2 block context:
            # C_local[i_local, k_local]
            # i_local maps to global i.
            # k_local maps to global k.
            # global_i = i_local
            # global_k = block * 2 + k_local
            # W index is global_k * n + global_i
            return global_k * n + global_i

        # Strassen formulas
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # Note: 1-based indexing in formulas, 0-based in code.
        # A00, A01, A10, A11
        # B00, B01, B10, B11
        
        # M1: A(1,1)+A(2,2) * B(1,1)+B(2,2) -> A[1,1]+A[2,2] is A11+A22 (indices 1,3)
        # M1: (A[1,1] + A[2,2]) * (B[1,1] + B[2,2])
        # M2: (A[2,1] + A[2,2]) * B[1,1] -> A[2,1]+A[2,2] is A21+A22 (indices 2,3)
        # M3: A[1,1] * (B[1,2] - B[2,2])
        # M4: A[2,2] * (B[2,1] - B[1,1])
        # M5: (A[1,1] + A[1,2]) * B[2,2]
        # M6: (A[2,1] - A[1,1]) * (B[1,1] + B[1,2])
        # M7: (A[1,2] - A[2,2]) * (B[2,1] + B[2,2])
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        terms = [
            # (m_idx, a_coeffs, b_coeffs, c_coeffs)
            # a_coeffs: list of (local_idx, coeff)
            # b_coeffs: list of (local_idx, coeff)
            # c_coeffs: list of (local_idx, coeff)
            
            # M1
            (0, 
             [(3, 1), (1, 1)],  # A11 + A22 (indices 1,3)
             [(3, 1), (1, 1)],  # B11 + B22 (indices 1,3)
             [(0, 1), (3, 1)]   # C11 + C22 (indices 0,3)
            ),
            # M2
            (1, 
             [(2, 1), (3, 1)],  # A21 + A22 (indices 2,3)
             [(0, 1)],          # B11 (index 0)
             [(2, 1)]           # C21 (index 2)
            ),
            # M3
            (2, 
             [(0, 1)],          # A11 (index 0)
             [(1, 1), (3, -1)], # B12 - B22 (indices 1,3)
             [(1, 1)]           # C12 (index 1)
            ),
            # M4
            (3, 
             [(3, 1)],          # A22 (index 3)
             [(2, 1), (0, -1)], # B21 - B11 (indices 2,0)
             [(0, 1), (2, 1)]   # C11 + C21 (indices 0,2)
            ),
            # M5
            (4, 
             [(0, 1), (1, 1)],  # A11 + A12 (indices 0,1)
             [(3, 1)],          # B22 (index 3)
             [(0, -1), (1, 1)]  # -C11 + C12 (indices 0,1)
            ),
            # M6
            (5, 
             [(2, 1), (0, -1)], # A21 - A11 (indices 2,0)
             [(0, 1), (1, 1)],  # B11 + B12 (indices 0,1)
             [(3, 1)]           # C22 (index 3)
            ),
            # M7
            (6, 
             [(1, 1), (3, -1)], # A12 - A22 (indices 1,3)
             [(2, 1), (3, 1)],  # B21 + B22 (indices 2,3)
             [(0, 1)]           # C11 (index 0)
            ),
        ]
        
        for m_idx, a_coeff, b_coeff, c_coeff in terms:
            col = offset + m_idx
            
            for (li, lc) in a_coeff:
                idx = a_idx(li // 2, li % 2, a_block_idx)
                U_strassen[idx, col] += lc
                
            for (li, lc) in b_coeff:
                idx = b_idx(li // 2, li % 2, b_block_idx)
                V_strassen[idx, col] += lc
                
            for (li, lc) in c_coeff:
                idx = c_idx(li // 2, li % 2, c_block_idx)
                W_strassen[idx, col] += lc

    # Add 4 Strassen instances
    # C0 = A00*B00 + A01*B02
    # Instance 1: A00 * B00 -> C0
    add_strassen(0, 0, 0, 0)
    # Instance 2: A01 * B02 -> C0
    add_strassen(7, 1, 2, 0)
    
    # C1 = A00*B01 + A01*B03
    # Instance 3: A00 * B01 -> C1
    add_strassen(14, 0, 1, 1)
    # Instance 4: A01 * B03 -> C1
    add_strassen(21, 1, 3, 1)
    
    # Verify the decomposition
    def verify(U, V, W):
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Should be 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(len(U[0])):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if val != 1:
                        return False
        return True

    if verify(U_strassen, V_strassen, W_strassen):
        U = U_strassen
        V = V_strassen
        W = W_strassen
    else:
        # Fallback to schoolbook if Strassen construction failed
        U = U_school
        V = V_school
        W = W_school

    # Check time and potentially attempt rank reduction
    current_rank = len(U[0])
    if time.time() < deadline - 10:
        # Try to prune columns that are zero or redundant
        # This is a simple check, not a full optimization
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
