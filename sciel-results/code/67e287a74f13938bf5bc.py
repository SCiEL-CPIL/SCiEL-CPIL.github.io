def construct():
    n, m, p = 2, 5, 5
    # Initial rank estimate based on block decomposition
    # 4 * Strassen(2x2x2) + 2 * Schoolbook(2x2x1) + 2 * Rank1(2x1x2) + 1 * Rank1(2x1x1)
    # 4 * 7 + 2 * 4 + 2 * 1 + 1 * 1 = 28 + 8 + 2 + 1 = 39
    rank = 39
    U = [[0]*rank for _ in range(n*m)]  # 10 x 39
    V = [[0]*rank for _ in range(m*p)]  # 25 x 39
    W = [[0]*rank for _ in range(p*n)]  # 10 x 39
    
    c = 0  # current column index
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal c
        # Strassen for 2x2x2
        # Adds 7 columns
        
        x00 = i_off * m + j_off
        x01 = i_off * m + j_off + 1
        x10 = (i_off+1) * m + j_off
        x11 = (i_off+1) * m + j_off + 1
        
        y00 = j_off * p + k_off
        y01 = j_off * p + k_off + 1
        y10 = (j_off+1) * p + k_off
        y11 = (j_off+1) * p + k_off + 1
        
        z00 = k_off * n + i_off
        z01 = (k_off+1) * n + i_off
        z10 = k_off * n + (i_off+1)
        z11 = (k_off+1) * n + (i_off+1)
        
        # M1 = (X00+X11)(Y00+Y11)
        U[x00][c] += 1; U[x11][c] += 1
        V[y00][c] += 1; V[y11][c] += 1
        W[z00][c] += 1; W[z11][c] += 1
        c += 1
        
        # M2 = (X10+X11)Y00
        U[x10][c] += 1; U[x11][c] += 1
        V[y00][c] += 1
        W[z10][c] += 1; W[z11][c] -= 1
        c += 1
        
        # M3 = X00(Y01-Y11)
        U[x00][c] += 1
        V[y01][c] += 1; V[y11][c] -= 1
        W[z01][c] += 1; W[z11][c] += 1
        c += 1
        
        # M4 = X11(Y10-Y00)
        U[x11][c] += 1
        V[y10][c] += 1; V[y00][c] -= 1
        W[z00][c] += 1; W[z10][c] += 1
        c += 1
        
        # M5 = (X00+X01)Y11
        U[x00][c] += 1; U[x01][c] += 1
        V[y11][c] += 1
        W[z00][c] -= 1; W[z01][c] += 1
        c += 1
        
        # M6 = (X10-X00)(Y00+Y01)
        U[x10][c] += 1; U[x00][c] -= 1
        V[y00][c] += 1; V[y01][c] += 1
        W[z11][c] += 1
        c += 1
        
        # M7 = (X01-X11)(Y10+Y11)
        U[x01][c] += 1; U[x11][c] -= 1
        V[y10][c] += 1; V[y11][c] += 1
        W[z00][c] += 1
        c += 1

    def add_schoolbook_2x2_x_2x1(i_off, j_off, k_off):
        nonlocal c
        # 2x2 A, 2x1 B, 2x1 C
        # Rank 4
        for ii in range(2):
            for jj in range(2):
                U[(i_off+ii)*m + (j_off+jj)][c] += 1
                V[(j_off+jj)*p + k_off][c] += 1
                W[k_off*n + (i_off+ii)][c] += 1
                c += 1

    def add_rank1_2x1_x_2x1(i_off, j_idx, k_off, k_len):
        # This is for A[:, j_idx] * B[j_idx, k_off:k_off+k_len]
        # C[:, k_off:k_off+k_len] = A[:, j_idx] * B[j_idx, :]
        # Rank 1
        # U: 1s at i_off, i_off+1
        # V: 1 at j_idx
        # W: 1s at all k in range, for both i
        nonlocal c
        # U
        U[i_off*m + j_idx][c] = 1
        U[(i_off+1)*m + j_idx][c] = 1
        # V
        V[j_idx*p + k_off][c] = 0 # Wait, V index is j*p + k? No, V is (m*p, rank).
        # V index for B[j, k] is j*p + k.
        # But here we are using a single column j_idx.
        # The term is A[i, j_idx] * B[j_idx, k].
        # So V should pick out row j_idx.
        # V[j_idx*p + k] is the element.
        # In the tensor decomposition, V is a vector of size m*p.
        # We need V[r] to be 1 at index j_idx*p + k?
        # No, the decomposition is sum u_r v_r w_r.
        # v_r is a vector in R^{m*p}.
        # We need v_r to be e_{j_idx, k}? No.
        # The term is A_{i, j} B_{j, k}.
        # For fixed j, we sum over k? No, k is an output index.
        # The tensor term is sum_{i,k} A_{i,j} B_{j,k} e_i \otimes e_j \otimes e_k.
        # This is (sum_i A_{i,j} e_i) \otimes e_j \otimes (sum_k B_{j,k} e_k).
        # This is not rank 1 in the tensor sense unless we fix the structure.
        # My previous analysis:
        # T_{i,j,k} = delta_{j, j_idx}.
        # Decomposition: u \otimes v \otimes w.
        # u_i v_j w_k = delta_{j, j_idx}.
        # v = e_{j_idx}.
        # u_i w_k = 1 for all i, k in range.
        # u = [1, 1]^T.
        # w = [1, 1, ..., 1]^T (length k_len? No, length p).
        # But w is in R^{p*n}.
        # w_{k*n + i} = 1 for all k in range, i in {0,1}.
        
        # So:
        U[i_off*m + j_idx][c] = 1
        U[(i_off+1)*m + j_idx][c] = 1
        
        V[j_idx*p + 0][c] = 0 # V is size m*p. Index is j*p + k?
        # No, V corresponds to B. B is m x p.
        # V vector has size m*p.
        # We need V to be e_{j_idx} in the "row" sense?
        # The term is A_{i, j_idx} B_{j_idx, k}.
        # This is a product of a column of A and a row of B.
        # In the tensor M, the middle index is j.
        # So V must be e_{j_idx} in the j-dimension.
        # But V is flattened.
        # V[r, col] contributes to B_{j, k} if r = j*p + k.
        # We need the sum over r to pick out j_idx.
        # If we set V[j_idx*p + k][c] = 1 for all k?
        # Then sum_k V[j_idx*p + k] B_{j_idx, k}? No.
        # The contraction is sum_{i,j,k} U[i*m+j] V[j*p+k] W[k*n+i] A[i*m+j] B[j*p+k].
        # We want this to equal A[i, j_idx] B[j_idx, k].
        # So we need U[i*m+j_idx] = 1, V[j_idx*p+k] = 1, W[k*n+i] = 1.
        # Then term is A[i, j_idx] * 1 * B[j_idx, k] * 1 = A[i, j_idx] B[j_idx, k].
        # Yes.
        
        for kk in range(k_len):
            V[j_idx*p + (k_off+kk)][c] = 1
            W[(k_off+kk)*n + i_off][c] = 1
            W[(k_off+kk)*n + i_off+1][c] = 1
            
        c += 1

    def add_rank1_2x1_x_1x1(i_off, j_idx, k_idx):
        nonlocal c
        # A[:, j_idx] * B[j_idx, k_idx] -> C[:, k_idx]
        # T_{i,j,k} = delta_{j, j_idx} delta_{k, k_idx}
        # u = [1, 1]^T
        # v = e_{j_idx} (in B space)
        # w = e_{k_idx} (in C space, for both i)
        
        U[i_off*m + j_idx][c] = 1
        U[(i_off+1)*m + j_idx][c] = 1
        
        V[j_idx*p + k_idx][c] = 1
        
        W[k_idx*n + i_off][c] = 1
        W[k_idx*n + i_off+1][c] = 1
        
        c += 1

    # Block 1: j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0)
    
    # Block 2: j in {0,1}, k in {2,3}
    add_strassen_block(0, 0, 2)
    
    # Block 3: j in {0,1}, k in {4}
    add_schoolbook_2x2_x_2x1(0, 0, 4)
    
    # Block 4: j in {2,3}, k in {0,1}
    add_strassen_block(0, 2, 0)
    
    # Block 5: j in {2,3}, k in {2,3}
    add_strassen_block(0, 2, 2)
    
    # Block 6: j in {2,3}, k in {4}
    add_schoolbook_2x2_x_2x1(0, 2, 4)
    
    # Block 7: j in {4}, k in {0,1}
    add_rank1_2x1_x_2x1(0, 4, 0, 2)
    
    # Block 8: j in {4}, k in {2,3}
    add_rank1_2x1_x_2x1(0, 4, 2, 2)
    
    # Block 9: j in {4}, k in {4}
    add_rank1_2x1_x_1x1(0, 4, 4)
    
    return {"U": U, "V": V, "W": W}
