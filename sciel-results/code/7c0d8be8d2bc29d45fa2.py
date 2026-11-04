import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 4, 5, 6
    
    # We decompose M_{n,m,p} = M_{n, m1, p} ⊕ M_{n, m2, p} where m1+m2 = m
    # Then further decompose each piece.
    #
    # Strategy:
    # M_{4,5,6} = M_{4,4,6} ⊕ M_{4,1,6}
    # M_{4,4,6} = M_{4,4,4} ⊕ M_{4,4,2}
    # M_{4,4,4} = (M_{2,2,2} ⊗ M_{2,2,2}) -- Kronecker product, rank 49
    # M_{4,4,2} = M_{2,4,2} ⊕ M_{2,4,2} where M_{2,4,2} = M_{2,2,2} ⊗ M_{2,2,1}, rank 7*1=7 each
    # M_{4,1,6} = rank 1 (it's sum_i sum_k e_i ⊗ e_0 ⊗ e_k = (sum e_i) ⊗ e_0 ⊗ (sum e_k))
    #
    # Total rank = 49 + 14 + 1 = 64
    
    def strassen_2x2():
        """Return (U, V, W) for M_{2,2,2} with rank 7.
        U shape (4,7), V shape (4,7), W shape (4,7)
        Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        # Standard Strassen decomposition
        # m1 = (A11+A22)(B11+B22)
        # m2 = (A21+A22)B11
        # m3 = A11(B12-B22)
        # m4 = A22(B21-B11)
        # m5 = (A11+A12)B22
        # m6 = (A21-A11)(B11+B12)
        # m7 = (A12-A22)(B21+B22)
        #
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        #
        # U rows: [A11, A12, A21, A22]
        # V rows: [B11, B12, B21, B22]
        # W rows: [C00, C01, C10, C11] where C00=C11, C01=C12, C10=C21, C11=C22
        
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # m1 = (A11+A22)(B11+B22)
        U[0, 0] = 1  # A11
        U[3, 0] = 1  # A22
        V[0, 0] = 1  # B11
        V[3, 0] = 1  # B22
        W[0, 0] = 1  # C11
        W[3, 0] = 1  # C22
        
        # m2 = (A21+A22)B11
        U[2, 1] = 1  # A21
        U[3, 1] = 1  # A22
        V[0, 1] = 1  # B11
        W[2, 1] = 1  # C21
        W[3, 1] = -1  # C22
        
        # m3 = A11(B12-B22)
        U[0, 2] = 1  # A11
        V[1, 2] = 1  # B12
        V[3, 2] = -1  # B22
        W[1, 2] = 1  # C12
        W[3, 2] = 1  # C22
        
        # m4 = A22(B21-B11)
        U[3, 3] = 1  # A22
        V[2, 3] = 1  # B21
        V[0, 3] = -1  # B11
        W[0, 3] = 1  # C11
        W[2, 3] = 1  # C21
        
        # m5 = (A11+A12)B22
        U[0, 4] = 1  # A11
        U[1, 4] = 1  # A12
        V[3, 4] = 1  # B22
        W[0, 4] = -1  # C11
        W[1, 4] = 1  # C12
        
        # m6 = (A21-A11)(B11+B12)
        U[2, 5] = 1  # A21
        U[0, 5] = -1  # A11
        V[0, 5] = 1  # B11
        V[1, 5] = 1  # B12
        W[3, 5] = 1  # C22
        
        # m7 = (A12-A22)(B21+B22)
        U[1, 6] = 1  # A12
        U[3, 6] = -1  # A22
        V[2, 6] = 1  # B21
        V[3, 6] = 1  # B22
        W[0, 6] = 1  # C11
        
        return U, V, W
    
    def kron_tensor(U1, V1, W1, U2, V2, W2):
        """Kronecker product of two tensor decompositions.
        If T1 = sum U1_c ⊗ V1_c ⊗ W1_c and T2 = sum U2_c ⊗ V2_c ⊗ W2_c,
        then T1⊗T2 = sum_{c1,c2} (U1_c1⊗U2_c2) ⊗ (V1_c1⊗V2_c2) ⊗ (W1_c1⊗W2_c2)
        """
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r = r1 * r2
        
        n1, m1, p1 = U1.shape[0], V1.shape[0], W1.shape[0]
        n2, m2, p2 = U2.shape[0], V2.shape[0], W2.shape[0]
        
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        
        U = np.zeros((n, r), dtype=int)
        V = np.zeros((m, r), dtype=int)
        W = np.zeros((p, r), dtype=int)
        
        for c1 in range(r1):
            for c2 in range(r2):
                c = c1 * r2 + c2
                U[:, c] = np.kron(U1[:, c1], U2[:, c2])
                V[:, c] = np.kron(V1[:, c1], V2[:, c2])
                W[:, c] = np.kron(W1[:, c1], W2[:, c2])
        
        return U, V, W
    
    def block_concat(U_list, V_list, W_list, row_splits):
        """Concatenate decompositions where each handles a block of rows.
        row_splits: list of (start, end) for U, V, W row indices for each block.
        """
        total_rank = sum(U.shape[1] for U in U_list)
        total_rows_u = max(row_splits[i][1] for i in range(len(U_list)))
        
        # Find max rows
        n = max(row_splits[i][1] for i in range(len(U_list)))
        m = max(row_splits[i][1] for i in range(len(V_list)))
        p = max(row_splits[i][1] for i in range(len(W_list)))
        
        U = np.zeros((n, total_rank), dtype=int)
        V = np.zeros((m, total_rank), dtype=int)
        W = np.zeros((p, total_rank), dtype=int)
        
        col = 0
        for idx, (U_b, V_b, W_b) in enumerate(zip(U_list, V_list, W_list)):
            u_start, u_end = row_splits[idx][0] if idx < len(row_splits) else (0, U_b.shape[0])
            # Actually we need separate splits for U, V, W
            r = U_b.shape[1]
            U[u_start:u_start+U_b.shape[0], col:col+r] = U_b
            V[u_start:u_start+V_b.shape[0], col:col+r] = V_b
            W[u_start:u_start+W_b.shape[0], col:col+r] = W_b
            col += r
        
        return U, V, W
    
    def identity_decomp(d):
        """M_{1,1,d} or similar rank-1 decomposition for a 'broadcast' tensor.
        Actually for M_{a,1,c}: T[i,0,k] = 1. This is rank 1.
        U is (a,1) all ones, V is (1,1) [1], W is (c,1) all ones.
        """
        pass
    
    def rank1_broadcast(n, m, p):
        """M_{n,m,p} where T[i,j,k] = 1 for all i,j,k? No, that's not right.
        For M_{n,1,p}: T[i,0,k] = 1. Rank 1.
        U[i,0]=1 for all i, V[0,0]=1, W[k,0]=1 for all k.
        """
        U = np.ones((n, 1), dtype=int)
        V = np.ones((1, 1), dtype=int)
        W = np.ones((p, 1), dtype=int)
        return U, V, W
    
    # Get Strassen decomposition for 2x2x2
    U_s, V_s, W_s = strassen_2x2()
    
    # M_{4,4,4} = M_{2,2,2} ⊗ M_{2,2,2}
    # But we need to be careful about the row ordering.
    # For M_{2,2,2}: U rows are [A00, A01, A10, A11] (i*2+j)
    # For M_{4,4,4}: U rows should be [A00,A01,A02,A03, A10,...,A33] (i*4+j)
    # Kronecker: U[i1*2+i2, j1*2+j2, c1*7+c2] = U_s[i1,j1,c1] * U_s[i2,j2,c2]
    # But i = i1*2+i2, j = j1*2+j2, so U[i,j,c] = U_s[i1,j1,c1]*U_s[i2,j2,c2]
    # This is exactly np.kron(U_s[:,c1], U_s[:,c2]) for the right c.
    U_444, V_444, W_444 = kron_tensor(U_s, V_s, W_s, U_s, V_s, W_s)
    # U_444 shape (16, 49), V_444 shape (16, 49), W_444 shape (16, 49)
    # Rows: U is i*4+j for i,j in 0..3. The kron gives rows (i1*2+i2, j1*2+j2) -> index (i1*2+i2)*4 + (j1*2+j2)? 
    # No! np.kron(U_s[:,c1], U_s[:,c2]) gives a vector of length 16 where element [a*4+b] = U_s[a,c1]*U_s[b,c2].
    # But we want U[i*4+j] where i = i1*2+i2, j = j1*2+j2.
    # So U[i*4+j, c] should = U_s[i1*2+j1, c1] * U_s[i2*2+j2, c2] where i=i1*2+i2, j=j1*2+j2.
    # np.kron gives [a*4+b] = U_s[a,c1]*U_s[b,c2]. So a=i, b=j. That's U[i,j,c] = U_s[i1,j1,c1]*U_s[i2,j2,c2] 
    # but a = i1*2+i2 and b = j1*2+j2. So yes, np.kron gives exactly what we want!
    
    if time.time() > deadline:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # M_{4,4,2}: We need T[i,j,k] = 1 for i in 0..3, j in 0..3, k in 0..1.
    # Split into k=0 and k=1: M_{4,4,1} ⊕ M_{4,4,1}
    # M_{4,4,1}: T[i,j,0] = 1. This is (sum_i e_i) ⊗ (sum_j e_j) ⊗ e_0. Rank 1!
    # U[i,0] = 1, V[j,0] = 1, W[0,0] = 1.
    # So M_{4,4,2} = M_{4,4,1} ⊕ M_{4,4,1}, rank 2.
    # Wait, but the W dimension is p=2. W[k*n+i, c]. For M_{4,4,1}, W has shape (1*4, 1) = (4,1), W[0,0]=1 (k=0,i=0).
    # For the second block (k=1), W[1*4+i, c] = W[4+i, c].
    
    # Actually, let me reconsider. The full tensor is M_{4,5,6}.
    # T[i,j,k] = 1 for i in 0..3, j in 0..4, k in 0..5.
    # U[i*m+j, c] = U[i*5+j, c]
    # V[j*p+k, c] = V[j*6+k, c]
    # W[k*n+i, c] = W[k*4+i, c]
    
    # Let me build it more carefully by direct construction.
    # 
    # I'll build the full decomposition by combining:
    # Block 1: i in 0..3, j in 0..3, k in 0..3 -> M_{4,4,4}, rank 49
    # Block 2: i in 0..3, j in 0..3, k in 4..5 -> M_{4,4,2}, rank 2
    # Block 3: i in 0..3, j in 4, k in 0..5 -> M_{4,1,6}, rank 1
    #
    # For Block 1: U rows [i*5+j for i in 0..3, j in 0..3], V rows [j*6+k for j in 0..3, k in 0..3], W rows [k*4+i for k in 0..3, i in 0..3]
    # For Block 2: U rows [i*5+j for i in 0..3, j in 0..3], V rows [j*6+k for j in 0..3, k in 4..5], W rows [k*4+i for k in 4..5, i in 0..3]
    # For Block 3: U rows [i*5+4 for i in 0..3], V rows [4*6+k for k in 0..5], W rows [k*4+i for k in 0..5, i in 0..3]
    
    total_rank = 49 + 2 + 1  # = 52
    
    U_full = np.zeros((n*m, total_rank), dtype=int)  # (20, 52)
    V_full = np.zeros((m*p, total_rank), dtype=int)  # (30, 52)
    W_full = np.zeros((p*n, total_rank), dtype=int)  # (24, 52)
    
    col = 0
    
    # Block 1: M_{4,4,4} using Strassen Kronecker
    # U_444 has shape (16, 49) with rows indexed by i*4+j for i,j in 0..3
    # We need to map to U_full[i*5+j, col:col+49]
    for i in range(4):
        for j in range(4):
            U_full[i*5+j, col:col+49] = U_444[i*4+j, :]
    
    # V_444 has shape (16, 49) with rows indexed by j*4+k for j,k in 0..3
    for j in range(4):
        for k in range(4):
            V_full[j*6+k, col:col+49] = V_444[j*4+k, :]
    
    # W_444 has shape (16, 49) with rows indexed by k*4+i for k,i in 0..3
    for k in range(4):
        for i in range(4):
            W_full[k*4+i, col:col+49] = W_444[k*4+i, :]
    
    col += 49
    
    # Block 2: M_{4,4,2}, rank 2
    # k=0: U[i*5+j]=1, V[j*6+0]=1, W[0*4+i]=1
    # k=1: U[i*5+j]=1, V[j*6+1]=1, W[1*4+i]=1
    # Actually this is two rank-1 tensors.
    for k in range(2):
        for i in range(4):
            for j in range(4):
                U_full[i*5+j, col] = 1
        for j in range(4):
            V_full[j*6+k, col] = 1
        for i in range(4):
            W_full[k*4+i, col] = 1
        col += 1
    
    # Block 3: M_{4,1,6}, rank 1
    # U[i*5+4]=1, V[4*6+k]=1, W[k*4+i]=1
    for i in range(4):
        U_full[i*5+4, col] = 1
    for k in range(6):
        V_full[4*6+k, col] = 1
    for k in range(6):
        for i in range(4):
            W_full[k*4+i, col] = 1
    col += 1
    
    # Verify the decomposition
    # C = A @ B where C[i,k] = sum_j A[i,j] * B[j,k]
    # Our decomposition: C[i,k] = sum_c (sum_j U[i*m+j,c] * W[k*n+i,c]) * (sum_j V[j*p+k,c] * ???)
    # Wait, I need to re-examine the convention.
    # 
    # The task says: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # The decomposition should satisfy: for each (i,j,k), T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] = 1 for all valid i,j,k.
    #
    # Let me verify this is correct.
    
    # Quick verification for a few entries
    # T[i,j,k] should = sum_c U[i*5+j,c] * V[j*6+k,c] * W[k*4+i,c]
    
    # Check T[0,0,0]
    val = 0
    for c in range(total_rank):
        val += U_full[0*5+0, c] * V_full[0*6+0, c] * W_full[0*4+0, c]
    if val != 1:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col_s] = 1
                    V[j*p+k, col_s] = 1
                    W[k*n+i, col_s] = 1
                    col_s += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # More thorough verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(total_rank):
                    val += U_full[i*m+j, c] * V_full[j*p+k, c] * W_full[k*n+i, c]
                if val != 1:
                    # Fallback
                    rank = n * m * p
                    U = np.zeros((n*m, rank), dtype=int)
                    V = np.zeros((m*p, rank), dtype=int)
                    W = np.zeros((p*n, rank), dtype=int)
                    col_s = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U[ii*m+jj, col_s] = 1
                                V[jj*p+kk, col_s] = 1
                                W[kk*n+ii, col_s] = 1
                                col_s += 1
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
