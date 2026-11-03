import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = 4, 5, 5
    
    # Rank 69 solution: 49 (4x4x4) + 16 (4x4x1) + 4 (4x1x5)
    # We will build this explicitly.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a term
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)

    # 1. 4x4x4 part (Rank 49)
    # Indices I (0..15), J (0..15), K (0..15)
    # I = i*4 + j
    # J = j*4 + k
    # K = k*4 + i
    
    # We use Strassen for 4x4.
    # This is equivalent to 7 multiplications of 2x2 matrices.
    # Each 2x2 multiplication is rank 7.
    # So 49 terms.
    
    # Strassen 2x2 terms (U2, V2, W2) for 2x2x2
    # U2, V2, W2 are 4x7
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: M0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 1: M1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1
    
    # Term 2: M2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # Term 3: M3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # Term 4: M4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1; W2[2, 4] = 1
    
    # Term 5: M5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 6: M6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1; W2[3, 6] = 1
    
    # Now compose for 4x4.
    # Block indices for A: 00, 01, 10, 11
    # Block indices for B: 00, 01, 10, 11
    # Block indices for C: 00, 01, 10, 11
    
    # Strassen 4x4 formulas (using blocks)
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # We need to map these to scalar indices.
    # For each Strassen term s (0..6), we have a 2x2 matrix product.
    # This product is formed by 7 scalar multiplications (using U2, V2, W2).
    # So for each s, we add 7 terms.
    
    # Mapping of blocks to indices
    # A block 00: I in {0,1}, J in {0,1}
    # A block 01: I in {0,1}, J in {2,3}
    # A block 10: I in {2,3}, J in {0,1}
    # A block 11: I in {2,3}, J in {2,3}
    
    # For a given Strassen term s, the A-part is a linear combination of blocks.
    # e.g. M0: A00 + A11.
    # The B-part is B00 + B11.
    # The C-part is added to C00 and C11.
    
    # Let's define the coefficients for A, B, C blocks for each s.
    # A_coeff[s, block]
    # B_coeff[s, block]
    # C_coeff[s, block]
    
    A_coeff = np.zeros((7, 4))
    B_coeff = np.zeros((7, 4))
    C_coeff = np.zeros((7, 4))
    
    # M0
    A_coeff[0, 0] = 1; A_coeff[0, 3] = 1
    B_coeff[0, 0] = 1; B_coeff[0, 3] = 1
    C_coeff[0, 0] = 1; C_coeff[0, 3] = 1
    
    # M1
    A_coeff[1, 2] = 1; A_coeff[1, 3] = 1
    B_coeff[1, 0] = 1
    C_coeff[1, 2] = 1; C_coeff[1, 3] = -1
    
    # M2
    A_coeff[2, 0] = 1
    B_coeff[2, 1] = 1; B_coeff[2, 3] = -1
    C_coeff[2, 1] = 1; C_coeff[2, 3] = 1
    
    # M3
    A_coeff[3, 3] = 1
    B_coeff[3, 2] = 1; B_coeff[3, 0] = -1
    C_coeff[3, 0] = 1; C_coeff[3, 2] = 1
    
    # M4
    A_coeff[4, 0] = 1; A_coeff[4, 1] = 1
    B_coeff[4, 3] = 1
    C_coeff[4, 0] = -1; C_coeff[4, 1] = 1
    
    # M5
    A_coeff[5, 2] = 1; A_coeff[5, 0] = -1
    B_coeff[5, 0] = 1; B_coeff[5, 1] = 1
    C_coeff[5, 3] = 1
    
    # M6
    A_coeff[6, 1] = 1; A_coeff[6, 3] = -1
    B_coeff[6, 2] = 1; B_coeff[6, 3] = 1
    C_coeff[6, 0] = 1; C_coeff[6, 3] = 1
    
    # Now generate 49 terms
    for s in range(7):
        for t in range(7):
            # U vector
            U = np.zeros(16)
            V = np.zeros(16)
            W = np.zeros(16)
            
            # U is formed by combining A blocks
            for ab in range(4):
                if A_coeff[s, ab] != 0:
                    # Add A block ab scaled by A_coeff[s, ab]
                    # A block ab has indices I in block_ab_I, J in block_ab_J?
                    # No, U is indexed by I.
                    # U[I, r] = sum_{ab} A_coeff[s, ab] * U2[I_local, t]
                    # I_local is index within the 2x2 block.
                    # I_global = I_block * 2 + I_local?
                    # Block 00: I in {0,1}. I_local = I.
                    # Block 01: I in {0,1}. I_local = I.
                    # Block 10: I in {2,3}. I_local = I - 2.
                    # Block 11: I in {2,3}. I_local = I - 2.
                    
                    # Wait, A is 4x4.
                    # Block 00: rows 0,1. cols 0,1.
                    # Block 01: rows 0,1. cols 2,3.
                    # Block 10: rows 2,3. cols 0,1.
                    # Block 11: rows 2,3. cols 2,3.
                    
                    # I is row index.
                    # If ab is 0 or 1 (top blocks), I in {0,1}.
                    # If ab is 2 or 3 (bottom blocks), I in {2,3}.
                    
                    I_block = ab // 2
                    I_local = I_block * 2 + U2[0, t] # No, U2 is vector
                    # U2[local_I, t]
                    
                    for I_local in range(2):
                        I_global = I_block * 2 + I_local
                        U[I_global] += A_coeff[s, ab] * U2[I_local, t]
                        
            # V is formed by combining B blocks
            for bb in range(4):
                if B_coeff[s, bb] != 0:
                    # B block bb
                    # J is col index of A, row index of B.
                    # B is 4x4.
                    # Block 00: cols 0,1 (of A), rows 0,1 (of B).
                    # J in A corresponds to col.
                    # If bb is 0 or 2 (left blocks), J in {0,1}.
                    # If bb is 1 or 3 (right blocks), J in {2,3}.
                    
                    J_block = bb % 2
                    J_local = J_block * 2 + V2[0, t]
                    
                    for J_local in range(2):
                        J_global = J_block * 2 + J_local
                        V[J_global] += B_coeff[s, bb] * V2[J_local, t]
                        
            # W is formed by combining C blocks
            for cb in range(4):
                if C_coeff[s, cb] != 0:
                    # C block cb
                    # K index.
                    # K = k*4 + i.
                    # Block 00: k in {0,1}, i in {0,1}.
                    # Block 01: k in {2,3}, i in {0,1}.
                    # Block 10: k in {0,1}, i in {2,3}.
                    # Block 11: k in {2,3}, i in {2,3}.
                    
                    # K_block = cb
                    # k_block = cb % 2
                    # i_block = cb // 2
                    
                    k_block = cb % 2
                    i_block = cb // 2
                    
                    for k_local in range(2):
                        for i_local in range(2):
                            k_global = k_block * 2 + k_local
                            i_global = i_block * 2 + i_local
                            K_global = k_global * 4 + i_global
                            
                            # W2 index?
                            # W2 is for 2x2 C.
                            # W2[K_local, t]
                            # K_local = k_local * 2 + i_local
                            K_local = k_local * 2 + i_local
                            W[K_global] += C_coeff[s, cb] * W2[K_local, t]
                            
            add_term(U, V, W)
            
    # 2. 4x4x1 part (Rank 16)
    # A (4x4) * B (4x1) -> C (4x1)
    # I in 0..15 (4x4 A)
    # J in 16..19 (4x1 B, cols 4..7? No, B is 5x5. We used 0..3 for B.
    # Wait, B is 5x5. J = j*5 + k.
    # We used J in 0..15 for the 4x4 part?
    # J = j*4 + k? No.
    # In 4x4x4, J index is for B (4x4).
    # J_global = j*4 + k.
    # But in 4x5x5, J = j*5 + k.
    # So the indices are different!
    
    # We must be careful with indices.
    # n=4, m=5, p=5.
    # I = i*5 + j (i in 0..3, j in 0..4)
    # J = j*5 + k (j in 0..4, k in 0..4)
    # K = k*4 + i (k in 0..4, i in 0..3)
    
    # My 4x4x4 construction assumed m=4, p=4.
    # I = i*4 + j.
    # This is WRONG for the full problem.
    
    # I need to map the 4x4x4 solution to the 4x5x5 indices.
    # The 4x4x4 part uses A (4x4) and B (4x4).
    # In 4x5x5:
    # A (4x5). We use cols 0..3.
    # B (5x5). We use rows 0..3, cols 0..3.
    # C (4x5). We get cols 0..3.
    
    # Indices:
    # I = i*5 + j. (j in 0..3)
    # J = j*5 + k. (j in 0..3, k in 0..3)
    # K = k*4 + i. (k in 0..3, i in 0..3)
    
    # So I need to adjust the U, V, W vectors.
    # U has length 20.
    # V has length 25.
    # W has length 20.
    
    # Let's restart the construction with correct indices.
    
    U_list = []
    V_list = []
    W_list = []
    
    # 1. 4x4x4 part
    for s in range(7):
        for t in range(7):
            U = np.zeros(20)
            V = np.zeros(25)
            W = np.zeros(20)
            
            for ab in range(4):
                if A_coeff[s, ab] != 0:
                    I_block = ab // 2
                    for I_local in range(2):
                        i = I_block * 2 + I_local
                        j = ab % 2 * 2 + I_local # Wait.
                        # A block ab:
                        # ab=0 (00): i in 0,1. j in 0,1.
                        # ab=1 (01): i in 0,1. j in 2,3.
                        # ab=2 (10): i in 2,3. j in 0,1.
                        # ab=3 (11): i in 2,3. j in 2,3.
                        
                        j_block = ab % 2
                        j = j_block * 2 + I_local # No, I_local is row.
                        # A is 4x4.
                        # Row i, Col j.
                        # I = i*5 + j.
                        
                        # For block ab:
                        # i in {2*I_block, 2*I_block+1}
                        # j in {2*j_block, 2*j_block+1}
                        
                        # U2[I_local, t] corresponds to row I_local in 2x2 block.
                        # So i = 2*I_block + I_local.
                        # But j?
                        # U is indexed by I (row of A).
                        # So only i matters for U.
                        # But A is 4x4.
                        # The term U[I] V[J] W[K].
                        # U[I] depends on i, j?
                        # No, U is vector of length nm.
                        # U[i*m + j].
                        # In 4x4x4, U[i*4 + j].
                        # Here U[i*5 + j].
                        
                        # The 2x2 Strassen term U2[I_local, t] is the coefficient for A[i, j]?
                        # No, U2 is for the 2x2 matrix.
                        # The 2x2 matrix A_block has entries A[2*I_block+u, 2*j_block+v].
                        # U2[u, t] is the coefficient for row u.
                        # But it multiplies B.
                        # So U[i*5 + j] should be A_coeff * U2[u, t] * (something for j)?
                        # No.
                        # The term is U[I] V[J] W[K].
                        # Sum over r.
                        # For a specific r, U[I] is fixed.
                        # U[I] = sum_{blocks} coeff * U2[local_I, t].
                        # But U[I] is for a specific I (i,j).
                        # In the 2x2 block, there are 4 entries.
                        # U2 has 4 entries (0,1,2,3).
                        # U2[0] corresponds to (0,0).
                        # U2[1] corresponds to (0,1).
                        # U2[2] corresponds to (1,0).
                        # U2[3] corresponds to (1,1).
                        
                        # So for block ab:
                        # u = I_local (row in 2x2)
                        # v = ?
                        # U2 index is u*2 + v?
                        # In my U2 definition:
                        # U2[0] -> (0,0)
                        # U2[1] -> (0,1)
                        # U2[2] -> (1,0)
                        # U2[3] -> (1,1)
                        
                        # So U2[idx] corresponds to (idx//2, idx%2).
                        
                        # For U:
                        # i = 2*I_block + u
                        # j = 2*j_block + v
                        # I = i*5 + j
                        
                        for u in range(2):
                            for v in range(2):
                                idx2 = u*2 + v
                                if U2[idx2, t] != 0:
                                    i = 2*I_block + u
                                    j = 2*j_block + v
                                    I = i*5 + j
                                    U[I] += A_coeff[s, ab] * U2[idx2, t]
                                    
            for bb in range(4):
                if B_coeff[s, bb] != 0:
                    j_block = bb % 2
                    k_block = bb // 2
                    for u in range(2):
                        for v in range(2):
                            idx2 = u*2 + v
                            if V2[idx2, t] != 0:
                                j = 2*j_block + u
                                k = 2*k_block + v
                                J = j*5 + k
                                V[J] += B_coeff[s, bb] * V2[idx2, t]
                                
            for cb in range(4):
                if C_coeff[s, cb] != 0:
                    i_block = cb // 2
                    k_block = cb % 2
                    for u in range(2):
                        for v in range(2):
                            idx2 = u*2 + v
                            if W2[idx2, t] != 0:
                                i = 2*i_block + u
                                k = 2*k_block + v
                                K = k*4 + i
                                W[K] += C_coeff[s, cb] * W2[idx2, t]
                                
            add_term(U, V, W)
            
    # 2. 4x4x1 part (A 4x4, B 4x1, C 4x1)
    # A cols 0..3. B rows 0..3, col 4. C rows 0..3, col 4.
    # I = i*5 + j (j in 0..3)
    # J = j*5 + 4
    # K = 4*4 + i = 16 + i
    
    for i in range(4):
        for j in range(4):
            U = np.zeros(20)
            V = np.zeros(25)
            W = np.zeros(20)
            
            I = i*5 + j
            J = j*5 + 4
            K = 16 + i
            
            U[I] = 1
            V[J] = 1
            W[K] = 1
            
            add_term(U, V, W)
            
    # 3. 4x1x5 part (A 4x1, B 1x5, C 4x5)
    # A col 4. B row 4, cols 0..4. C rows 0..3, cols 0..4.
    # I = i*5 + 4
    # J = 4*5 + k = 20 + k
    # K = k*4 + i
    
    for i in range(4):
        for k in range(5):
            U = np.zeros(20)
            V = np.zeros(25)
            W = np.zeros(20)
            
            I = i*5 + 4
            J = 20 + k
            K = k*4 + i
            
            U[I] = 1
            V[J] = 1
            W[K] = 1
            
            add_term(U, V, W)
            
    U = np.array(U_list)
    V = np.array(V_list)
    W = np.array(W_list)
    
    # Verify
    is_exact = True
    for i in range(4):
        for j in range(5):
            for k in range(5):
                I = i*5 + j
                J = j*5 + k
                K = k*4 + i
                val = np.dot(U[I, :], V[J, :] * W[K, :])
                if abs(val - 1.0) > 1e-9:
                    is_exact = False
                    break
            if not is_exact: break
        if not is_exact: break
        
    if is_exact:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback
    return {"U": [], "V": [], "W": []}
