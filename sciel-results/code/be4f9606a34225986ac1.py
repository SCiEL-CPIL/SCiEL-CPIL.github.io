import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = delta_{col_A, row_B}."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] V[:,r] W[:,r]^T = T."""
    recon = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.allclose(recon, T, atol=tol)

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition of the 2x2x2 matrix multiplication tensor.
    """
    U = np.array([
        [1, 0, 1, 0, 1, 0, 1],   
        [0, 0, 0, 0, 1, -1, 0],  
        [0, 1, 0, 0, 0, 1, 0],   
        [1, 1, 0, 1, 0, 0, -1],  
    ], dtype=np.float64)

    V = np.array([
        [1, 1, 0, 0, 0, 0, 0],   
        [0, 0, 1, 0, 0, 1, 0],   
        [0, 0, 0, 1, 0, 0, 1],   
        [1, 0, -1, -1, 1, 0, 1], 
    ], dtype=np.float64)

    W = np.array([
        [1, 0, 0, 0, 1, 0, 1],   
        [0, 0, 1, 0, 1, 0, 0],   
        [0, 1, 0, 1, 0, 0, 0],   
        [1, -1, 1, 0, 0, 1, 0],  
    ], dtype=np.float64)

    return U, V, W

def _build_4x4x4_strassen():
    """
    Build rank-49 decomposition of 4x4x4 via Kronecker product of Strassen.
    """
    Us, Vs, Ws = _strassen_2x2x2()
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for i2 in range(2):
                    U4[i1 * 2 + i2, r] = Us[i1, r1] * Us[i2, r2]
            for j1 in range(2):
                for j2 in range(2):
                    V4[j1 * 2 + j2, r] = Vs[j1, r1] * Vs[j2, r2]
            for k1 in range(2):
                for k2 in range(2):
                    W4[k1 * 2 + k2, r] = Ws[k1, r1] * Ws[k2, r2]
    
    return U4, V4, W4

def _build_4x4x3_strassen_tail():
    """
    Build a low-rank decomposition for the 4x4x3 part (columns 4-6 of B).
    We split this into 4x4x2 + 4x4x1.
    4x4x2: Use Strassen-like structure on 2x2 blocks.
    4x4x1: Schoolbook (rank 16).
    
    For 4x4x2, we can view it as two 4x4x1 problems.
    However, a known result is that 4x4x2 can be done in rank 28? 
    Actually, let's use the "block Strassen" approach for 4x4x2.
    Or simpler: use the rank-28 construction from literature if available, 
    or just use schoolbook 32 for 4x4x2 and 16 for 4x4x1 = 48.
    
    Let's try a hybrid: 
    4x4x2: 
    A is 4x4, B is 4x2.
    Split A into 2x2 blocks A11, A12, A21, A22.
    B into 2x2 blocks B11, B12 (top), B21, B22 (bottom). Note B is 4x2, so blocks are 2x1.
    Wait, B is 4x2. 
    Rows 0-1: B_top (2x2). Rows 2-3: B_bot (2x2).
    C = A @ B (4x2).
    C_top (2x2) = A11@B_top + A12@B_bot.
    C_bot (2x2) = A21@B_top + A22@B_bot.
    
    This is 4 multiplications of 2x2 @ 2x2? No.
    A11 (2x2) @ B_top (2x2) -> 2x2 result.
    This is exactly the 2x2x2 problem, rank 7.
    So C_top requires 2 * 7 = 14 mults? 
    And C_bot requires 2 * 7 = 14 mults?
    Total 28 mults for 4x4x2.
    
    Let's construct it.
    """
    Us, Vs, Ws = _strassen_2x2x2()
    
    # 4x4x2 part
    # U: (16, 28), V: (8, 28), W: (8, 28)
    R_2 = 28
    U_2 = np.zeros((16, R_2))
    V_2 = np.zeros((8, R_2))
    W_2 = np.zeros((8, R_2))
    
    # C_top (rows 0,1 of C) = A11@B_top + A12@B_bot
    # A11 is A[0:2, 0:2] -> U indices 0,1,4,5
    # A12 is A[0:2, 2:4] -> U indices 2,3,6,7
    # B_top is B[0:2, :] -> V indices 0,1,2,3 (since B is 4x2, flat is 0,1,2,3 for first 2 rows? 
    # Wait, B is 4x2. Flat index j*2+k.
    # j=0: k=0,1 -> 0,1
    # j=1: k=0,1 -> 2,3
    # j=2: k=0,1 -> 4,5
    # j=3: k=0,1 -> 6,7
    # So B_top (rows 0,1) corresponds to V indices 0,1,2,3.
    # B_bot (rows 2,3) corresponds to V indices 4,5,6,7.
    
    # C_top output (rows 0,1 of C, cols 0,1).
    # C is 4x2. Flat index k*4+i.
    # k=0: i=0,1,2,3 -> 0,1,2,3
    # k=1: i=0,1,2,3 -> 4,5,6,7
    # C_top (rows 0,1) corresponds to W indices 0,1 (for k=0) and 4,5 (for k=1).
    
    # Let's map the 2x2x2 Strassen to the blocks.
    # Block 1: A11 @ B_top -> contributes to C_top
    # A11 indices in U: 0,1,4,5. Map to 0,1,2,3 in Strassen U.
    # B_top indices in V: 0,1,2,3. Map to 0,1,2,3 in Strassen V.
    # C_top indices in W: 0,1,4,5. Map to 0,1,2,3 in Strassen W.
    
    # We have 7 Strassen mults for this term.
    # We have 7 Strassen mults for A12 @ B_bot.
    # A12 indices in U: 2,3,6,7. Map to 0,1,2,3.
    # B_bot indices in V: 4,5,6,7. Map to 0,1,2,3.
    # C_top indices in W: 0,1,4,5. Map to 0,1,2,3.
    
    # So for C_top, we use 14 mults.
    # For C_bot (rows 2,3 of C), it is A21@B_top + A22@B_bot.
    # A21 indices in U: 8,9,12,13.
    # A22 indices in U: 10,11,14,15.
    # B_top, B_bot same as above.
    # C_bot indices in W: 2,3,6,7.
    
    # Total 28 mults.
    
    for r in range(7):
        # Term 1: A11 @ B_top
        # U part
        U_2[0, r] += Us[0, r]
        U_2[1, r] += Us[1, r]
        U_2[4, r] += Us[2, r]
        U_2[5, r] += Us[3, r]
        # V part
        V_2[0, r] += Vs[0, r]
        V_2[1, r] += Vs[1, r]
        V_2[2, r] += Vs[2, r]
        V_2[3, r] += Vs[3, r]
        # W part
        W_2[0, r] += Ws[0, r]
        W_2[1, r] += Ws[1, r]
        W_2[4, r] += Ws[2, r]
        W_2[5, r] += Ws[3, r]
        
        # Term 2: A12 @ B_bot
        r2 = r + 7
        U_2[2, r2] += Us[0, r]
        U_2[3, r2] += Us[1, r]
        U_2[6, r2] += Us[2, r]
        U_2[7, r2] += Us[3, r]
        V_2[4, r2] += Vs[0, r]
        V_2[5, r2] += Vs[1, r]
        V_2[6, r2] += Vs[2, r]
        V_2[7, r2] += Vs[3, r]
        W_2[0, r2] += Ws[0, r]
        W_2[1, r2] += Ws[1, r]
        W_2[4, r2] += Ws[2, r]
        W_2[5, r2] += Ws[3, r]
        
        # Term 3: A21 @ B_top
        r3 = r + 14
        U_2[8, r3] += Us[0, r]
        U_2[9, r3] += Us[1, r]
        U_2[12, r3] += Us[2, r]
        U_2[13, r3] += Us[3, r]
        V_2[0, r3] += Vs[0, r]
        V_2[1, r3] += Vs[1, r]
        V_2[2, r3] += Vs[2, r]
        V_2[3, r3] += Vs[3, r]
        W_2[2, r3] += Ws[0, r]
        W_2[3, r3] += Ws[1, r]
        W_2[6, r3] += Ws[2, r]
        W_2[7, r3] += Ws[3, r]
        
        # Term 4: A22 @ B_bot
        r4 = r + 21
        U_2[10, r4] += Us[0, r]
        U_2[11, r4] += Us[1, r]
        U_2[14, r4] += Us[2, r]
        U_2[15, r4] += Us[3, r]
        V_2[4, r4] += Vs[0, r]
        V_2[5, r4] += Vs[1, r]
        V_2[6, r4] += Vs[2, r]
        V_2[7, r4] += Vs[3, r]
        W_2[2, r4] += Ws[0, r]
        W_2[3, r4] += Ws[1, r]
        W_2[6, r4] += Ws[2, r]
        W_2[7, r4] += Ws[3, r]
        
    return U_2, V_2, W_2

def _build_4x4x1_schoolbook():
    """
    Build rank-16 decomposition for 4x4x1.
    U: (16, 16), V: (4, 16), W: (4, 16)
    """
    R = 16
    U = np.zeros((16, R))
    V = np.zeros((4, R))
    W = np.zeros((4, R))
    col = 0
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1.0
            V[j, col] = 1.0
            W[i, col] = 1.0
            col += 1
    return U, V, W

def _build_4x4x7_combined():
    """
    Combine 4x4x4 (rank 49) + 4x4x2 (rank 28) + 4x4x1 (rank 16) = 93.
    """
    U4, V4, W4 = _build_4x4x4_strassen()
    U2, V2, W2 = _build_4x4x3_strassen_tail() # Returns 4x4x2 part
    U1, V1, W1 = _build_4x4x1_schoolbook()
    
    R4 = 49
    R2 = 28
    R1 = 16
    R_total = R4 + R2 + R1
    
    U = np.zeros((16, R_total))
    V = np.zeros((28, R_total))
    W = np.zeros((28, R_total))
    
    # 4x4x4 part
    U[:, :R4] = U4
    V[:16, :R4] = V4
    W[:16, :R4] = W4
    
    # 4x4x2 part (cols 4,5 of B, rows 4,5 of C)
    U[:, R4:R4+R2] = U2
    V[16:24, R4:R4+R2] = V2 # B rows 2,3 of the 4x2 block? 
    # Wait, B is 4x7.
    # Cols 0-3 handled by 4x4x4.
    # Cols 4-5 handled by 4x4x2.
    # B is 4x7. Rows 0-3.
    # In 4x4x2, B is 4x2.
    # V2 is (8, 28). Indices 0-3 correspond to B rows 0-1? 
    # My _build_4x4x3_strassen_tail assumed B is 4x2 with rows 0-1 and 2-3.
    # So V2[0:4] corresponds to B rows 0-1.
    # V2[4:8] corresponds to B rows 2-3.
    # In the full 4x4x7, B rows 0-3 are always the same.
    # So for the 4x4x2 part (cols 4,5), V should be placed in rows 0-3 of the full V?
    # No, V is (m*p, rank) = (4*7, rank) = (28, rank).
    # V index = j*p + k.
    # For cols 4,5: k=4,5.
    # j=0: 0*7+4=4, 0*7+5=5
    # j=1: 1*7+4=11, 1*7+5=12
    # j=2: 2*7+4=18, 2*7+5=19
    # j=3: 3*7+4=25, 3*7+5=26
    
    # My V2 is (8, 28).
    # V2[0] -> B row 0, col 0 of the 2-col block?
    # In _build_4x4x3_strassen_tail, V2[0] was mapped to Vs[0] which is B row 0, col 0.
    # So V2[0] corresponds to B row 0, local col 0 (which is global col 4).
    # V2[1] corresponds to B row 0, local col 1 (which is global col 5).
    # V2[2] corresponds to B row 1, local col 0 (global col 4).
    # V2[3] corresponds to B row 1, local col 1 (global col 5).
    # V2[4] corresponds to B row 2, local col 0 (global col 4).
    # V2[5] corresponds to B row 2, local col 1 (global col 5).
    # V2[6] corresponds to B row 3, local col 0 (global col 4).
    # V2[7] corresponds to B row 3, local col 1 (global col 5).
    
    # So we need to scatter V2 into V at the correct global indices.
    # Global indices for k=4,5:
    # j=0: 4, 5
    # j=1: 11, 12
    # j=2: 18, 19
    # j=3: 25, 26
    
    v_indices = [4, 5, 11, 12, 18, 19, 25, 26]
    V[v_indices, R4:R4+R2] = V2
    
    # W for 4x4x2 part.
    # W is (p*n, rank) = (7*4, rank) = (28, rank).
    # W index = k*n + i.
    # For cols 4,5: k=4,5.
    # k=4: 4*4+0=16, 17, 18, 19
    # k=5: 5*4+0=20, 21, 22, 23
    
    # My W2 is (8, 28).
    # W2[0] -> C row 0, local col 0 (global col 4).
    # W2[1] -> C row 1, local col 0 (global col 4).
    # W2[2] -> C row 2, local col 0 (global col 4).
    # W2[3] -> C row 3, local col 0 (global col 4).
    # W2[4] -> C row 0, local col 1 (global col 5).
    # W2[5] -> C row 1, local col 1 (global col 5).
    # W2[6] -> C row 2, local col 1 (global col 5).
    # W2[7] -> C row 3, local col 1 (global col 5).
    
    # Global indices for k=4,5:
    # k=4: 16, 17, 18, 19
    # k=5: 20, 21, 22, 23
    
    w_indices = [16, 17, 18, 19, 20, 21, 22, 23]
    W[w_indices, R4:R4+R2] = W2
    
    # 4x4x1 part (col 6 of B, row 6 of C)
    U[:, R4+R2:] = U1
    # V for col 6:
    # j=0: 0*7+6=6
    # j=1: 1*7+6=13
    # j=2: 2*7+6=20
    # j=3: 3*7+6=27
    v_indices_1 = [6, 13, 20, 27]
    V[v_indices_1, R4+R2:] = V1
    
    # W for row 6:
    # k=6: 6*4+0=24, 25, 26, 27
    w_indices_1 = [24, 25, 26, 27]
    W[w_indices_1, R4+R2:] = W1
    
    return U, V, W

def _snap_rational(mat, tol=1e-10):
    """Snap values close to half-integers to exact half-integers."""
    mat = np.array(mat, dtype=np.float64)
    snapped = np.round(mat * 2) / 2
    if np.allclose(mat, snapped, atol=tol):
        return snapped
    return mat

def _greedy_sparse_code(U, V, W, T, max_iter=50, tol=1e-9):
    """
    Attempt to reduce the rank of the decomposition (U, V, W) by removing 
    linearly dependent columns or redundant terms.
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    R = U.shape[1]
    
    # We want to find a subset of columns of size R' < R such that the tensor is preserved.
    # This is equivalent to finding a low-rank approximation that is exact.
    # If the current decomposition has linear dependencies, we can remove them.
    
    # Check for linear dependencies in U, V, W columns.
    # If column r is a linear combination of others, we can potentially remove it if the 
    # corresponding tensor contribution can be absorbed.
    # However, for exact tensor decomposition, we can use the following:
    # If the vectors u_r, v_r, w_r are linearly dependent with others in a specific way, 
    # we might be able to eliminate r.
    
    # A simpler approach: Try to remove one column at a time and see if the residual 
    # can be represented by the remaining columns.
    # Residual T_res = T - sum_{r != r0} u_r v_r w_r^T.
    # If T_res can be written as u' v' w'^T (rank 1), then we can keep it.
    # But we want to REDUCE rank, so we want T_res = 0.
    # This happens if the term r was redundant.
    
    # Let's check if any term is exactly zero or linearly dependent.
    # This is hard to do generally.
    
    # Alternative: Use ALS to refine the decomposition and then snap.
    # But ALS might not reduce rank.
    
    # Let's try a specific check: 
    # Compute the tensor T_current from U, V, W.
    # T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
    # If T_current is close to T, we are good.
    
    # To reduce rank, we can try to find a basis for the column space of U, V, W 
    # and see if a smaller set of rank-1 tensors spans the same space.
    
    # This is complex. For now, let's just return the input if no obvious reduction is found.
    # The main improvement will come from the construction itself (93 rank).
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140
    
    T = _build_tensor(n, m, p)
    
    # Build the rank-93 decomposition
    U, V, W = _build_4x4x7_combined()
    
    # Verify
    if _verify(U, V, W, T, tol=1e-9):
        # Try to snap to rationals
        U = _snap_rational(U)
        V = _snap_rational(V)
        W = _snap_rational(W)
        
        # Verify again after snapping
        if _verify(U, V, W, T, tol=1e-9):
            pass
        else:
            # Revert if snapping broke it
            U, V, W = _build_4x4x7_combined()
    else:
        # Fallback to schoolbook if construction failed
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1

    # Convert to lists
    U_out = [[float(x) for x in U[r]] for r in range(U.shape[0])]
    V_out = [[float(x) for x in V[r]] for r in range(V.shape[0])]
    W_out = [[float(x) for x in W[r]] for r in range(W.shape[0])]
    
    return {'U': U_out, 'V': V_out, 'W': W_out}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
