import time
import numpy as np

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition, rank 7."""
    # Strassen: 7 products
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1+M4-M5+M7
    # C[0,1] = M3+M5
    # C[1,0] = M2+M4
    # C[1,1] = M1-M2+M3+M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0,0]=1; U[1,0]=1; V[0,0]=1; V[1,0]=1; W[0,0]=1
    # M2 = (c+d)e
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
    # M3 = a(f-h)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1
    # M4 = d(g-e)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[2,3]=1
    # M5 = (a+b)h
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[1,4]=1
    # M6 = (c-a)(e+f)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (b-d)(g+h)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[3,6]=1
    
    return U, V, W

def _naive_mm(n, m, p):
    """Naive matrix multiplication decomposition, rank n*m*p."""
    total_rank = n * m * p
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _combine_blocks(blocks):
    """
    Combine a list of (U, V, W) tuples into a single decomposition.
    Each block is a numpy array.
    """
    if not blocks:
        return None, None, None
        
    U_list = []
    V_list = []
    W_list = []
    
    for U, V, W in blocks:
        U_list.append(U)
        V_list.append(V)
        W_list.append(W)
        
    U_res = np.hstack(U_list)
    V_res = np.hstack(V_list)
    W_res = np.hstack(W_list)
    
    return U_res, V_res, W_res

def _construct():
    n, m, p = 4, 5, 5
    
    # Partition A (4x5) into Top (2x5) and Bottom (2x5)
    # Partition B (5x5) into Left (5x2) and Right (5x3)
    # Inner dimension 5 is split into 2 and 3 for the multiplication
    
    # Block 1: A_Top (2x5) * B_Left (5x2) -> C_TopLeft (2x2)
    # Inner dim 5 split into 2 (from A cols 0-1) and 3 (from A cols 2-4)
    # A_Top = [A_TL (2x2) | A_TR (2x3)]
    # B_Left = [B_TL (2x2); B_BL (3x2)]
    # Product = A_TL * B_TL + A_TR * B_BL
    
    # Sub-block 1a: A_TL (2x2) * B_TL (2x2) -> 2x2x2
    U1a, V1a, W1a = _strassen_2x2x2()
    
    # Sub-block 1b: A_TR (2x3) * B_BL (3x2) -> 2x3x2
    # Split inner 3 into 1 and 2
    # A_TR = [A_T1 (2x1) | A_T2 (2x2)]
    # B_BL = [B_B1 (1x2); B_B2 (2x2)]
    # Product = A_T1 * B_B1 + A_T2 * B_B2
    
    # Sub-sub 1b-i: A_T1 (2x1) * B_B1 (1x2) -> 2x1x2
    U1bi, V1bi, W1bi = _naive_mm(2, 1, 2)
    
    # Sub-sub 1b-ii: A_T2 (2x2) * B_B2 (2x2) -> 2x2x2
    U1bii, V1bii, W1bii = _strassen_2x2x2()
    
    # Combine 1b
    U1b, V1b, W1b = _combine_blocks([(U1bi, V1bi, W1bi), (U1bii, V1bii, W1bii)])
    
    # Combine 1
    U1, V1, W1 = _combine_blocks([(U1a, V1a, W1a), (U1b, V1b, W1b)])
    
    # Block 2: A_Top (2x5) * B_Right (5x3) -> C_TopRight (2x3)
    # A_Top = [A_TL (2x2) | A_TR (2x3)]
    # B_Right = [B_TR (2x3); B_BR (3x3)]
    # Product = A_TL * B_TR + A_TR * B_BR
    
    # Sub-block 2a: A_TL (2x2) * B_TR (2x3) -> 2x2x3
    # Split inner 2? No, inner is 2. 
    # Wait, 2x2x3 means A is 2x2, B is 2x3. Inner dim is 2.
    # We can view this as two 2x2x1 blocks?
    # Or use the fact that rank(2x2x3) = rank(2x2x2) + rank(2x2x1) = 7 + 4 = 11.
    # A_TL (2x2) * B_TR (2x3)
    # B_TR = [B_TR1 (2x1) | B_TR2 (2x1) | B_TR3 (2x1)]? No, B_TR is 2x3.
    # Let's split B_TR cols into 2 and 1.
    # B_TR = [B_Tra (2x2) | B_Trb (2x1)]
    # Product = A_TL * B_Tra + A_TL * B_Trb
    # A_TL * B_Tra is 2x2x2 (Rank 7)
    # A_TL * B_Trb is 2x2x1 (Rank 4)
    
    U2ai, V2ai, W2ai = _strassen_2x2x2() # A_TL * B_Tra
    U2aii, V2aii, W2aii = _naive_mm(2, 2, 1) # A_TL * B_Trb
    
    U2a, V2a, W2a = _combine_blocks([(U2ai, V2ai, W2ai), (U2aii, V2aii, W2aii)])
    
    # Sub-block 2b: A_TR (2x3) * B_BR (3x3) -> 2x3x3
    # Split inner 3 into 1 and 2
    # A_TR = [A_T1 (2x1) | A_T2 (2x2)]
    # B_BR = [B_B1 (1x3); B_B2 (2x3)]
    # Product = A_T1 * B_B1 + A_T2 * B_B2
    
    # A_T1 * B_B1 is 2x1x3 (Rank 6)
    U2bi, V2bi, W2bi = _naive_mm(2, 1, 3)
    
    # A_T2 * B_B2 is 2x2x3 (Rank 11)
    # Same logic as 2a: Split B_B2 (2x3) into 2x2 and 2x1
    U2bii_a, V2bii_a, W2bii_a = _strassen_2x2x2()
    U2bii_b, V2bii_b, W2bii_b = _naive_mm(2, 2, 1)
    U2bii, V2bii, W2bii = _combine_blocks([(U2bii_a, V2bii_a, W2bii_a), (U2bii_b, V2bii_b, W2bii_b)])
    
    U2b, V2b, W2b = _combine_blocks([(U2bi, V2bi, W2bi), (U2bii, V2bii, W2bii)])
    
    U2, V2, W2 = _combine_blocks([(U2a, V2a, W2a), (U2b, V2b, W2b)])
    
    # Block 3: A_Bottom (2x5) * B_Left (5x2) -> C_BottomLeft (2x2)
    # Symmetric to Block 1
    U3, V3, W3 = _combine_blocks([(U1a, V1a, W1a), (U1b, V1b, W1b)])
    # Note: We can reuse the same Strassen blocks if the indices map correctly, 
    # but since we are constructing the full tensor, we just concatenate the decompositions.
    # The U, V, W matrices are for the specific sub-problem.
    # When we combine, we just stack columns.
    # However, the indices in U, V, W are local to the block.
    # We need to map these local indices to the global indices.
    
    # CRITICAL: The helper functions return U, V, W with local indices.
    # We must shift these indices to the correct positions in the global 4x5x5 tensor.
    
    # Let's redefine the assembly to handle index shifting.
    
    def shift_indices(U, V, W, i_off, j_off, k_off, n, m, p):
        """
        Shift indices of a block decomposition.
        U is (n_b * m_b, rank). Local i in [0, n_b), local j in [0, m_b).
        Global i = i_off + local_i, global j = j_off + local_j.
        V is (m_b * p_b, rank). Local j in [0, m_b), local k in [0, p_b).
        Global j = j_off + local_j, global k = k_off + local_k.
        W is (p_b * n_b, rank). Local k in [0, p_b), local i in [0, n_b).
        Global k = k_off + local_k, global i = i_off + local_i.
        
        Note: The standard tensor decomposition interface uses:
        U[i*m + j, c], V[j*p + k, c], W[k*n + i, c].
        """
        n_b = U.shape[0] // m_b
        m_b = U.shape[0] // n_b # This is ambiguous if n_b==m_b.
        # We need to pass n_b, m_b, p_b explicitly.
        pass

    # This shifting logic is error-prone. 
    # Alternative: Construct the full U, V, W matrices directly by iterating over the blocks and placing the non-zeros.
    
    U_full = np.zeros((20, 92)) # 4*5 = 20
    V_full = np.zeros((25, 92)) # 5*5 = 25
    W_full = np.zeros((20, 92)) # 5*4 = 20
    
    col_offset = 0
    
    def add_block(U_b, V_b, W_b, i_off, j_off, k_off, n_b, m_b, p_b):
        nonlocal col_offset
        rank_b = U_b.shape[1]
        
        for c in range(rank_b):
            # U
            for u_idx in range(U_b.shape[0]):
                if U_b[u_idx, c] != 0:
                    # u_idx = i_local * m_b + j_local
                    i_loc = u_idx // m_b
                    j_loc = u_idx % m_b
                    i_glob = i_off + i_loc
                    j_glob = j_off + j_loc
                    U_full[i_glob * 5 + j_glob, col_offset + c] = U_b[u_idx, c]
                    
            # V
            for v_idx in range(V_b.shape[0]):
                if V_b[v_idx, c] != 0:
                    # v_idx = j_local * p_b + k_local
                    j_loc = v_idx // p_b
                    k_loc = v_idx % p_b
                    j_glob = j_off + j_loc
                    k_glob = k_off + k_loc
                    V_full[j_glob * 5 + k_glob, col_offset + c] = V_b[v_idx, c]
                    
            # W
            for w_idx in range(W_b.shape[0]):
                if W_b[w_idx, c] != 0:
                    # w_idx = k_local * n_b + i_local
                    k_loc = w_idx // n_b
                    i_loc = w_idx % n_b
                    k_glob = k_off + k_loc
                    i_glob = i_off + i_loc
                    W_full[k_glob * 4 + i_glob, col_offset + c] = W_b[w_idx, c]
                    
        col_offset += rank_b

    # Block 1: A_Top (2x5) * B_Left (5x2)
    # i_off=0, j_off=0, k_off=0
    # n_b=2, m_b=2, p_b=2 for Strassen part?
    # No, the blocks are:
    # 1a: 2x2x2. i_off=0, j_off=0, k_off=0.
    add_block(U1a, V1a, W1a, 0, 0, 0, 2, 2, 2)
    
    # 1b: 2x3x2.
    # This block is A_TR (2x3) * B_BL (3x2).
    # i_off=0, j_off=2, k_off=0.
    # But 1b is composed of 1bi (2x1x2) and 1bii (2x2x2).
    # 1bi: A_T1 (2x1) * B_B1 (1x2).
    # i_off=0, j_off=2, k_off=0. n_b=2, m_b=1, p_b=2.
    add_block(U1bi, V1bi, W1bi, 0, 2, 0, 2, 1, 2)
    
    # 1bii: A_T2 (2x2) * B_B2 (2x2).
    # i_off=0, j_off=3, k_off=0. n_b=2, m_b=2, p_b=2.
    add_block(U1bii, V1bii, W1bii, 0, 3, 0, 2, 2, 2)
    
    # Block 2: A_Top (2x5) * B_Right (5x3)
    # i_off=0, j_off=0, k_off=2.
    # 2a: 2x2x3. A_TL (2x2) * B_TR (2x3).
    # Composed of 2ai (2x2x2) and 2aii (2x2x1).
    # 2ai: A_TL (2x2) * B_Tra (2x2).
    # i_off=0, j_off=0, k_off=2. n_b=2, m_b=2, p_b=2.
    add_block(U2ai, V2ai, W2ai, 0, 0, 2, 2, 2, 2)
    
    # 2aii: A_TL (2x2) * B_Trb (2x1).
    # i_off=0, j_off=0, k_off=4. n_b=2, m_b=2, p_b=1.
    add_block(U2aii, V2aii, W2aii, 0, 0, 4, 2, 2, 1)
    
    # 2b: 2x3x3. A_TR (2x3) * B_BR (3x3).
    # Composed of 2bi (2x1x3) and 2bii (2x2x3).
    # 2bi: A_T1 (2x1) * B_B1 (1x3).
    # i_off=0, j_off=2, k_off=2. n_b=2, m_b=1, p_b=3.
    add_block(U2bi, V2bi, W2bi, 0, 2, 2, 2, 1, 3)
    
    # 2bii: A_T2 (2x2) * B_B2 (2x3).
    # Composed of 2bii_a (2x2x2) and 2bii_b (2x2x1).
    # 2bii_a: i_off=0, j_off=3, k_off=2. n_b=2, m_b=2, p_b=2.
    add_block(U2bii_a, V2bii_a, W2bii_a, 0, 3, 2, 2, 2, 2)
    
    # 2bii_b: i_off=0, j_off=3, k_off=4. n_b=2, m_b=2, p_b=1.
    add_block(U2bii_b, V2bii_b, W2bii_b, 0, 3, 4, 2, 2, 1)
    
    # Block 3: A_Bottom (2x5) * B_Left (5x2)
    # i_off=2, j_off=0, k_off=0.
    # 3a: 2x2x2. A_BL (2x2) * B_TL (2x2).
    # i_off=2, j_off=0, k_off=0.
    add_block(U1a, V1a, W1a, 2, 0, 0, 2, 2, 2)
    
    # 3b: 2x3x2. A_BR (2x3) * B_BL (3x2).
    # 3bi: A_B1 (2x1) * B_B1 (1x2).
    # i_off=2, j_off=2, k_off=0.
    add_block(U1bi, V1bi, W1bi, 2, 2, 0, 2, 1, 2)
    
    # 3bii: A_B2 (2x2) * B_B2 (2x2).
    # i_off=2, j_off=3, k_off=0.
    add_block(U1bii, V1bii, W1bii, 2, 3, 0, 2, 2, 2)
    
    # Block 4: A_Bottom (2x5) * B_Right (5x3)
    # i_off=2, j_off=0, k_off=2.
    # 4a: 2x2x3. A_BL (2x2) * B_TR (2x3).
    # 4ai: A_BL (2x2) * B_Tra (2x2).
    # i_off=2, j_off=0, k_off=2.
    add_block(U2ai, V2ai, W2ai, 2, 0, 2, 2, 2, 2)
    
    # 4aii: A_BL (2x2) * B_Trb (2x1).
    # i_off=2, j_off=0, k_off=4.
    add_block(U2aii, V2aii, W2aii, 2, 0, 4, 2, 2, 1)
    
    # 4b: 2x3x3. A_BR (2x3) * B_BR (3x3).
    # 4bi: A_B1 (2x1) * B_B1 (1x3).
    # i_off=2, j_off=2, k_off=2.
    add_block(U2bi, V2bi, W2bi, 2, 2, 2, 2, 1, 3)
    
    # 4bii: A_B2 (2x2) * B_B2 (2x3).
    # 4bii_a: i_off=2, j_off=3, k_off=2.
    add_block(U2bii_a, V2bii_a, W2bii_a, 2, 3, 2, 2, 2, 2)
    
    # 4bii_b: i_off=2, j_off=3, k_off=4.
    add_block(U2bii_b, V2bii_b, W2bii_b, 2, 3, 4, 2, 2, 1)
    
    # Verify rank
    final_rank = col_offset
    # Trim if necessary (should be exactly 92)
    U_out = U_full[:, :final_rank]
    V_out = V_full[:, :final_rank]
    W_out = W_full[:, :final_rank]
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

def construct():
    start_time = time.time()
    result = _construct()
    return result
