# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Build the target tensor T with shape (n*m, m*p, p*n)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-8):
        """Check if decomposition is exact."""
        r = U.shape[1]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_approx - T)) < tol
    
    def schoolbook():
        """Standard O(nmp) decomposition."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank

    def add_strassen_block(U, V, W, col, i_off, j_off, k_off):
        """
        Adds 7 Strassen terms for a 2x2x2 block starting at (i_off, j_off, k_off).
        Returns the next column index.
        """
        # Local indices i', j', k' in {0, 1}
        # Global i = i' + i_off, j = j' + j_off, k = k' + k_off
        
        # Helper to get global indices
        def u_idx(ip, jp): return (ip + i_off) * m + (jp + j_off)
        def v_idx(jp, kp): return (jp + j_off) * p + (kp + k_off)
        def w_idx(kp, ip): return (kp + k_off) * n + (ip + i_off)
        
        # Term 1: (A00+A11)(B00+B11) -> C00, C11
        U[u_idx(0,0), col] += 1.0
        U[u_idx(1,1), col] += 1.0
        V[v_idx(0,0), col] += 1.0
        V[v_idx(1,1), col] += 1.0
        W[w_idx(0,0), col] += 1.0
        W[w_idx(1,1), col] += 1.0
        col += 1
        
        # Term 2: (A10+A11)B00 -> C10
        U[u_idx(1,0), col] += 1.0
        U[u_idx(1,1), col] += 1.0
        V[v_idx(0,0), col] += 1.0
        W[w_idx(0,1), col] += 1.0
        col += 1
        
        # Term 3: A00(B01-B11) -> C01
        U[u_idx(0,0), col] += 1.0
        V[v_idx(0,1), col] += 1.0
        V[v_idx(1,1), col] += -1.0
        W[w_idx(1,0), col] += 1.0
        col += 1
        
        # Term 4: A11(B10-B00) -> C00, C10
        U[u_idx(1,1), col] += 1.0
        V[v_idx(1,0), col] += 1.0
        V[v_idx(0,0), col] += -1.0
        W[w_idx(0,0), col] += 1.0
        W[w_idx(0,1), col] += 1.0
        col += 1
        
        # Term 5: (A00+A01)B11 -> C01, -C00
        U[u_idx(0,0), col] += 1.0
        U[u_idx(0,1), col] += 1.0
        V[v_idx(1,1), col] += 1.0
        W[w_idx(1,0), col] += 1.0
        W[w_idx(0,0), col] += -1.0
        col += 1
        
        # Term 6: (A10-A00)(B00+B01) -> C11
        U[u_idx(1,0), col] += 1.0
        U[u_idx(0,0), col] += -1.0
        V[v_idx(0,0), col] += 1.0
        V[v_idx(0,1), col] += 1.0
        W[w_idx(1,1), col] += 1.0
        col += 1
        
        # Term 7: (A01-A11)(B10+B11) -> C00
        U[u_idx(0,1), col] += 1.0
        U[u_idx(1,1), col] += -1.0
        V[v_idx(1,0), col] += 1.0
        V[v_idx(1,1), col] += 1.0
        W[w_idx(0,0), col] += 1.0
        col += 1
        
        return col

    def add_dot_product_block(U, V, W, col, i_val, j_start, j_end, k_val):
        """
        Adds schoolbook terms for a 1 x (j_end-j_start) x 1 block.
        i is fixed, k is fixed.
        """
        for j in range(j_start, j_end + 1):
            U[i_val * m + j, col] += 1.0
            V[j * p + k_val, col] += 1.0
            W[k_val * n + i_val, col] += 1.0
            col += 1
        return col

    def strassen_decomposition_3x4x5():
        """
        Construct 3x4x5 MM using Strassen for 2x2x2 blocks and schoolbook for the rest.
        Rank 56.
        """
        rank = 56
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        col = 0
        
        # 4 Strassen blocks for i in {0,1}
        # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
        col = add_strassen_block(U, V, W, col, 0, 0, 0)
        # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
        col = add_strassen_block(U, V, W, col, 0, 2, 0)
        # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
        col = add_strassen_block(U, V, W, col, 0, 0, 2)
        # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
        col = add_strassen_block(U, V, W, col, 0, 2, 2)
        
        # Remaining for i in {0,1}: k=4
        # Block 5: i in {0,1}, j in {0,1}, k=4
        # This is 2x2x1. We can treat it as two 1x2x1? No, i varies.
        # C_{0,4} = A_{0,0}B_{0,4} + A_{0,1}B_{1,4}
        # C_{1,4} = A_{1,0}B_{0,4} + A_{1,1}B_{1,4}
        # Terms:
        # 1. A_{0,0} B_{0,4} -> C_{0,4}
        U[0*4+0, col] += 1.0; V[0*5+4, col] += 1.0; W[4*3+0, col] += 1.0; col += 1
        # 2. A_{0,1} B_{1,4} -> C_{0,4}
        U[0*4+1, col] += 1.0; V[1*5+4, col] += 1.0; W[4*3+0, col] += 1.0; col += 1
        # 3. A_{1,0} B_{0,4} -> C_{1,4}
        U[1*4+0, col] += 1.0; V[0*5+4, col] += 1.0; W[4*3+1, col] += 1.0; col += 1
        # 4. A_{1,1} B_{1,4} -> C_{1,4}
        U[1*4+1, col] += 1.0; V[1*5+4, col] += 1.0; W[4*3+1, col] += 1.0; col += 1
        
        # Block 6: i in {0,1}, j in {2,3}, k=4
        # 5. A_{0,2} B_{2,4} -> C_{0,4}
        U[0*4+2, col] += 1.0; V[2*5+4, col] += 1.0; W[4*3+0, col] += 1.0; col += 1
        # 6. A_{0,3} B_{3,4} -> C_{0,4}
        U[0*4+3, col] += 1.0; V[3*5+4, col] += 1.0; W[4*3+0, col] += 1.0; col += 1
        # 7. A_{1,2} B_{2,4} -> C_{1,4}
        U[1*4+2, col] += 1.0; V[2*5+4, col] += 1.0; W[4*3+1, col] += 1.0; col += 1
        # 8. A_{1,3} B_{3,4} -> C_{1,4}
        U[1*4+3, col] += 1.0; V[3*5+4, col] += 1.0; W[4*3+1, col] += 1.0; col += 1
        
        # Remaining for i=2: k in {0,1,2,3,4}, j in {0,1,2,3}
        # 5 independent 1x4x1 blocks
        for k in range(5):
            for j in range(4):
                U[2*4+j, col] += 1.0
                V[j*5+k, col] += 1.0
                W[k*3+2, col] += 1.0
                col += 1
        
        if col != rank:
            raise ValueError(f"Column count mismatch: {col} != {rank}")
            
        return U, V, W, rank

    # Try the Strassen-based approach
    try:
        U_str, V_str, W_str, rank_str = strassen_decomposition_3x4x5()
        if verify(U_str, V_str, W_str, tol=1e-8):
            best_U, best_V, best_W = U_str.copy(), V_str.copy(), W_str.copy()
            best_rank = rank_str
        else:
            raise ValueError("Strassen decomposition failed verification")
    except Exception:
        U_sb, V_sb, W_sb, rank_sb = schoolbook()
        best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
        best_rank = rank_sb
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
