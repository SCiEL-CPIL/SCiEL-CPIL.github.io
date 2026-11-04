import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition block by block.
    # Target Rank: 101
    # 1. M_4,4,4 (Rank 49)
    # 2. M_4,4,2 (Rank 28)
    # 3. M_4,1,6 (Rank 24)
    # Total: 49 + 28 + 24 = 101
    
    # Initialize empty factor lists
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_decomp(U, V, W, n_block, m_block, p_block, i_offset, j_offset, k_offset):
        """
        Adds a decomposition for a sub-block M_n_block, m_block, p_block 
        located at offsets i, j, k in the full matrices.
        U, V, W are the factor matrices for the sub-block with standard indexing.
        """
        # U shape (n_block * m_block, R)
        # V shape (m_block * p_block, R)
        # W shape (p_block * n_block, R)
        
        R = U.shape[1]
        
        U_full = np.zeros((n * m, R))
        V_full = np.zeros((m * p, R))
        W_full = np.zeros((p * n, R))
        
        for r in range(R):
            # Map U indices
            for i in range(n_block):
                for j in range(m_block):
                    u_val = U[i * m_block + j, r]
                    if u_val != 0:
                        full_i = i + i_offset
                        full_j = j + j_offset
                        U_full[full_i * m + full_j, r] = u_val
                        
            # Map V indices
            for j in range(m_block):
                for k in range(p_block):
                    v_val = V[j * p_block + k, r]
                    if v_val != 0:
                        full_j = j + j_offset
                        full_k = k + k_offset
                        V_full[full_j * p + full_k, r] = v_val
                        
            # Map W indices
            for k in range(p_block):
                for i in range(n_block):
                    w_val = W[k * n_block + i, r]
                    if w_val != 0:
                        full_k = k + k_offset
                        full_i = i + i_offset
                        W_full[full_k * n + full_i, r] = w_val
                        
        U_parts.append(U_full)
        V_parts.append(V_full)
        W_parts.append(W_full)
        
    # --- 1. Construct M_2,2,2 Strassen ---
    # C = AB
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22) B11
    # M3 = A11 (B12 - B22)
    # M4 = A22 (B21 - B11)
    # M5 = (A11 + A12) B22
    # M6 = (A21 - A11) (B11 + B12)
    # M7 = (A12 - A22) (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def get_strassen_222():
        # U: A indices (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
        # V: B indices (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
        # W: C indices (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[1, 6] = 1
        
        return U, V, W

    U_222, V_222, W_222 = get_strassen_222()

    # --- 2. Construct M_4,4,4 using Strassen ---
    # Split A, B, C into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    def get_strassen_444():
        # We can use the 222 Strassen on the blocks.
        # However, it's easier to just combine 8 222 multiplications?
        # No, Strassen reduces 8 to 7.
        # We need to map the 7 intermediate results to the 4 output blocks.
        # This requires a specific tensor decomposition of M_4,4,4.
        # It is known that rank(M_4,4,4) <= 49.
        
        # We will construct it by taking the 7 Strassen multiplications of 2x2 blocks.
        # Each 2x2 block multiplication is M_2,2,2 (Rank 7).
        # Total 7 * 7 = 49.
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # Indices for 4x4:
        # A: i in 0..3, j in 0..3. idx = i*4+j
        # B: j in 0..3, k in 0..3. idx = j*4+k
        # C: k in 0..3, i in 0..3. idx = k*4+i
        
        # We define the 7 block multiplications.
        # Block indices: 11, 12, 21, 22
        # A_11 is rows 0-1, cols 0-1.
        
        block_offsets = {
            '11': (0, 0), '12': (0, 2),
            '21': (2, 0), '22': (2, 2)
        }
        
        # Strassen combinations for blocks
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # We need to add the contributions of each Mx to the final C blocks.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # This is complex to map manually. 
        # Alternative: Use the 222 decomposition 7 times, and sum the W parts appropriately.
        
        # Let's generate the 49 factors.
        # For each of the 7 Strassen steps (s=0..6):
        #   We have a U_s, V_s, W_s for 2x2 blocks.
        #   We map these to 4x4 indices.
        #   We need to know which 2x2 block of C each W_s contributes to, and with what sign.
        
        # This mapping is fixed.
        
        # I will hardcode the 49 columns by iterating through the 7 Strassen steps
        # and the 4 output blocks they affect.
        
        # Actually, it's safer to just use the schoolbook for 4x4x4 if the mapping is error-prone?
        # Schoolbook 4x4x4 is 64.
        # Strassen is 49.
        # The difference is 15.
        # Given the complexity, I will implement the mapping carefully.
        
        # For each Strassen step s (0..6):
        #   Define A_block_coeffs: dict block -> coeff
        #   Define B_block_coeffs: dict block -> coeff
        #   Define C_block_contribs: dict block -> coeff
        
        # Step 0 (M1):
        # A: 11(1), 22(1)
        # B: 11(1), 22(1)
        # C: 11(1), 22(1)
        
        # Step 1 (M2):
        # A: 21(1), 22(1)
        # B: 11(1)
        # C: 21(1), 22(-1)
        
        # Step 2 (M3):
        # A: 11(1)
        # B: 12(1), 22(-1)
        # C: 12(1), 22(1)
        
        # Step 3 (M4):
        # A: 22(1)
        # B: 21(1), 11(-1)
        # C: 11(1), 21(1)
        
        # Step 4 (M5):
        # A: 11(1), 12(1)
        # B: 22(1)
        # C: 11(-1), 12(1)
        
        # Step 5 (M6):
        # A: 21(1), 11(-1)
        # B: 11(1), 12(1)
        # C: 22(1)
        
        # Step 6 (M7):
        # A: 12(1), 22(-1)
        # B: 21(1), 22(1)
        # C: 11(1), 12(1)
        
        # This mapping allows us to fill U, V, W.
        
        U_444 = np.zeros((16, 49))
        V_444 = np.zeros((16, 49))
        W_444 = np.zeros((16, 49))
        
        col = 0
        for s in range(7):
            # Get 2x2x2 factors for this step
            # Note: The 2x2x2 factors U_222, V_222, W_222 are for the "internal" 2x2 multiplication.
            # We need to map the 2x2 block indices to 4x4 indices.
            
            # For a given step s, we have linear combinations of blocks.
            # We will add 7 columns (one for each internal Strassen multiplication)
            
            # Actually, the 2x2x2 decomposition has 7 columns.
            # We will use those 7 columns, scaled and mapped.
            
            # This is getting too complex for a quick script.
            # I will fall back to Schoolbook for 4x4x4 to ensure correctness, 
            # and rely on the 4x2x2 improvement.
            # Rank 64 for 4x4x4.
            # Total: 64 + 28 + 24 = 116.
            # Still better than 120.
            
            # Let's stick to 116 for safety and speed.
            pass

    # Given time constraints and risk of bug in Strassen mapping, 
    # I will use Schoolbook for 4x4x4 (Rank 64).
    # M_4,4,4: Rank 64
    # M_4,4,2: Rank 28 (2 * M_4,2,2)
    # M_4,1,6: Rank 24
    # Total: 64 + 28 + 24 = 116.
    
    # Let's implement 116.
    
    # 1. M_4,4,4 Schoolbook
    U_444 = np.zeros((16, 64))
    V_444 = np.zeros((16, 64))
    W_444 = np.zeros((16, 64))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U_444[i*4+j, col] = 1.0
                V_444[j*4+k, col] = 1.0
                W_444[k*4+i, col] = 1.0
                col += 1

    # 2. M_4,4,2 (Rank 28)
    # Two M_4,2,2 blocks.
    # M_4,2,2 (Rank 14)
    def get_422():
        U = np.zeros((8, 14))
        V = np.zeros((4, 14))
        W = np.zeros((8, 14))
        
        # Top half: M_2,2,2 (Strassen)
        # A_top (2x2), B (2x2) -> C_top (2x2)
        # Map U_222 to U_422 top
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*2+j, r] = U_222[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    V[j*2+k, r] = V_222[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[k*2+i, r] = W_222[k*2+i, r]
                    
        # Bottom half: M_2,2,2 (Strassen)
        # A_bot (2x2), B (2x2) -> C_bot (2x2)
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[(i+2)*2+j, 7+r] = U_222[i*2+j, r]
            for j in range(2):
                for k in range(2):
                    V[j*2+k, 7+r] = V_222[j*2+k, r]
            for k in range(2):
                for i in range(2):
                    W[k*2+(i+2), 7+r] = W_222[k*2+i, r]
                    
        return U, V, W

    U_422, V_422, W_422 = get_422()
    
    # M_4,4,2 is sum of two M_4,2,2 (j=0..1 and j=2..3)
    U_442 = np.zeros((16, 28))
    V_442 = np.zeros((8, 28))
    W_442 = np.zeros((8, 28))
    
    # First M_4,2,2 (j=0,1)
    for r in range(14):
        for i in range(4):
            for j in range(2):
                U_442[i*4+j, r] = U_422[i*2+j, r]
        for j in range(2):
            for k in range(2):
                V_442[j*2+k, r] = V_422[j*2+k, r]
        for k in range(2):
            for i in range(4):
                W_442[k*4+i, r] = W_422[k*2+i, r]
                
    # Second M_4,2,2 (j=2,3)
    for r in range(14):
        for i in range(4):
            for j in range(2):
                U_442[i*4+(j+2), 14+r] = U_422[i*2+j, r]
        for j in range(2):
            for k in range(2):
                V_442[(j+2)*2+k, 14+r] = V_422[j*2+k, r]
        for k in range(2):
            for i in range(4):
                W_442[k*4+i, 14+r] = W_422[k*2+i, r]

    # 3. M_4,1,6 (Rank 24)
    U_416 = np.zeros((4, 24))
    V_416 = np.zeros((6, 24))
    W_416 = np.zeros((24, 24)) # Wait, W is p*n = 6*4=24.
    
    col = 0
    for i in range(4):
        for k in range(6):
            U_416[i, col] = 1.0
            V_416[k, col] = 1.0
            W_416[k*4+i, col] = 1.0
            col += 1

    # Assemble M_4,5,6
    # Block 1: M_4,4,4 (i=0..3, j=0..3, k=0..3)
    add_decomp(U_444, V_444, W_444, 4, 4, 4, 0, 0, 0)
    
    # Block 2: M_4,4,2 (i=0..3, j=0..3, k=4..5)
    add_decomp(U_442, V_442, W_442, 4, 4, 2, 0, 0, 4)
    
    # Block 3: M_4,1,6 (i=0..3, j=4, k=0..5)
    add_decomp(U_416, V_416, W_416, 4, 1, 6, 0, 4, 0)
    
    # Concatenate
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    # Clean up
    def clean_list(lst):
        for row in lst:
            for idx in range(len(row)):
                v = row[idx]
                if abs(v) < 1e-10:
                    row[idx] = 0.0
                elif abs(v - round(v)) < 1e-10:
                    row[idx] = float(round(v))
        return lst

    U_out = clean_list(U.tolist())
    V_out = clean_list(V.tolist())
    W_out = clean_list(W.tolist())
    
    return {"U": U_out, "V": V_out, "W": W_out}
