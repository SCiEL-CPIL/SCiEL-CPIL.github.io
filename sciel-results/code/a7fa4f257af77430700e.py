import time
import numpy as np

def construct():
    start_time = time.time()
    timeout = 130  # safety margin before 150s limit
    
    n, m, p = 3, 5, 5
    
    # We'll build the decomposition incrementally.
    # Start with the known 2x2x2 Strassen decomposition (rank 7).
    # Then extend to 3x3x3, 3x5x3, 3x5x5 using tensor sum / block structure.
    
    # The 2x2x2 Strassen coefficients (rank 7):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # In tensor form, for 2x2x2:
    # U has 4 rows (A11,A12,A21,A22), V has 4 rows (B11,B12,B21,B22), W has 4 rows (C11,C12,C21,C22)
    # Each of the 7 terms contributes to U, V, W.
    
    def strassen_222():
        """Return U, V, W for 2x2x2 Strassen (rank 7)."""
        # Index convention: U[i*2+j], V[j*2+k], W[k*2+i]
        # U rows: A11=0, A12=1, A21=2, A22=3
        # V rows: B11=0, B12=1, B21=2, B22=3
        # W rows: C11=0, C12=1, C21=2, C22=3
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = -1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    # Now extend to 3x3x3.
    # We can write 3x3x3 as 2x2x2 (top-left 2x2 block) plus cross terms.
    # Actually, a cleaner approach: use the tensor product / Kronecker structure.
    
    # Let me use a different strategy: build the full 3x5x5 decomposition directly
    # by combining Strassen 2x2x2 with schoolbook for the remaining parts.
    
    # Split n=3 as 2+1, m=5 as 2+2+1, p=5 as 2+2+1
    # This gives 3x3x3 blocks of the tensor.
    
    # Actually, let me use a simpler and more reliable approach:
    # Use Strassen on 2x2x2 blocks and schoolbook for 1x1x1 and mixed blocks.
    
    # Let's define the block structure:
    # n = 2+1, m = 2+2+1, p = 2+2+1
    # So we have blocks:
    #   (2,2,2): Strassen rank 7
    #   (2,2,1): schoolbook rank 4
    #   (2,1,2): schoolbook rank 4
    #   (2,1,1): schoolbook rank 2
    #   (1,2,2): schoolbook rank 4
    #   (1,2,1): schoolbook rank 2
    #   (1,1,2): schoolbook rank 2
    #   (1,1,1): schoolbook rank 1
    
    # Total: 7 + 4+4+2 + 4+2+2 + 1 = 7 + 10 + 8 + 1 = 26
    # Wait, let me recalculate.
    
    # Blocks for (n_block, m_block, p_block):
    # (2,2,2): rank 7 (Strassen)
    # (2,2,1): 2*2*1 = 4
    # (2,2,1): 2*2*1 = 4 (second p block of size 1? No, p=2+2+1, so p blocks are 2,2,1)
    
    # Let me be more careful:
    # n blocks: 2, 1
    # m blocks: 2, 2, 1
    # p blocks: 2, 2, 1
    
    # All combinations:
    # (2,2,2): 7
    # (2,2,2): 7 (second 2 in m, second 2 in p)
    # (2,2,1): 4 (second 2 in m, 1 in p)
    # (2,1,2): 4 (1 in m, second 2 in p)
    # (2,1,2): 4 (1 in m, second 2 in p... wait I'm confusing myself)
    
    # Let me enumerate systematically:
    # n_size in [2,1], m_size in [2,2,1], p_size in [2,2,1]
    
    # (2,2,2): 7
    # (2,2,2): 7 (m=2nd block of 2, p=1st block of 2)
    # (2,2,1): 4 (m=1st block of 2, p=block of 1)
    # (2,2,1): 4 (m=2nd block of 2, p=block of 1)
    # (2,1,2): 4 (m=block of 1, p=1st block of 2)
    # (2,1,2): 4 (m=block of 1, p=2nd block of 2)
    # (2,1,1): 2 (m=block of 1, p=block of 1)
    # (1,2,2): 4
    # (1,2,2): 4
    # (1,2,1): 2
    # (1,2,1): 2
    # (1,1,2): 2
    # (1,1,2): 2
    # (1,1,1): 1
    
    # Sum: 7+7+4+4+4+4+2+4+4+2+2+2+2+1 = 49
    
    # That's worse than schoolbook 75? No, 49 < 75. But wait, I only used Strassen for two blocks.
    # Let me use Strassen for all (2,2,2) blocks.
    
    # Actually, this approach gives rank 49 for 3x5x5, which is better than 75.
    # But can we do better?
    
    # Let me try: n=2+1, m=3+2, p=3+2
    # (2,3,3): We need a decomposition. Use Strassen on 2x2x2 part + schoolbook for rest.
    # This is getting complex. Let me just implement the block approach above.
    
    # Let's implement it cleanly.
    
    def schoolbook_block(ns, ms, ps):
        """Return U, V, W for schoolbook ns x ms x ps."""
        U = np.zeros((ns * ms, ns * ms * ps))
        V = np.zeros((ms * ps, ns * ms * ps))
        W = np.zeros((ps * ns, ns * ms * ps))
        col = 0
        for i in range(ns):
            for j in range(ms):
                for k in range(ps):
                    U[i * ms + j, col] = 1
                    V[j * ps + k, col] = 1
                    W[k * ns + i, col] = 1
                    col += 1
        return U, V, W
    
    def embed_strassen_222(offset_u, offset_v, offset_w):
        """Embed Strassen 2x2x2 into larger arrays at given offsets."""
        U_s, V_s, W_s = strassen_222()
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[offset_u:offset_u+4, :] = U_s
        V[offset_v:offset_v+4, :] = V_s
        W[offset_w:offset_w+4, :] = W_s
        return U, V, W
    
    # Build the full decomposition
    n_sizes = [2, 1]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 1]
    
    # Calculate total rank
    total_rank = 0
    for ns in n_sizes:
        for ms in m_sizes:
            for ps in p_sizes:
                if ns == 2 and ms == 2 and ps == 2:
                    total_rank += 7
                else:
                    total_rank += ns * ms * ps
    
    # Build U, V, W
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col = 0
    u_offset = 0
    v_offset = 0
    w_offset = 0
    
    for ni, ns in enumerate(n_sizes):
        for mi, ms in enumerate(m_sizes):
            for pi, ps in enumerate(p_sizes):
                if ns == 2 and ms == 2 and ps == 2:
                    U_s, V_s, W_s = strassen_222()
                    U_full[u_offset:u_offset+4, col:col+7] = U_s
                    V_full[v_offset:v_offset+4, col:col+7] = V_s
                    W_full[w_offset:w_offset+4, col:col+7] = W_s
                    col += 7
                else:
                    U_sb, V_sb, W_sb = schoolbook_block(ns, ms, ps)
                    U_full[u_offset:u_offset+ns*ms, col:col+ns*ms*ps] = U_sb
                    V_full[v_offset:v_offset+ms*ps, col:col+ns*ms*ps] = V_sb
                    W_full[w_offset:w_offset+ps*ns, col:col+ns*ms*ps] = W_sb
                    col += ns * ms * ps
                
                w_offset += ps * ns
            v_offset += ms * p
        u_offset += ns * m
    
    # Verify correctness on random inputs
    if time.time() - start_time < timeout - 5:
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from factors
        C_computed = np.zeros((n, p))
        for r in range(total_rank):
            # Extract A part from U[:, r]
            A_r = U_full[:, r].reshape(n, m)
            # Extract B part from V[:, r]
            B_r = V_full[:, r].reshape(m, p)
            # Extract C part from W[:, r]
            C_r = W_full[:, r].reshape(p, n)
            # C[i,k] += sum_j A_r[i,j] * B_r[j,k] * C_r[k,i]
            # Wait, the decomposition is: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is a tensor contraction. Let me compute it properly.
            pass
        
        # Actually, let me verify differently. The tensor identity is:
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k])
        
        # Let me check a few entries
        error = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_expected = 1  # since C[i,k] includes A[i,j]*B[j,k] with coefficient 1
                    T_computed = 0
                    for r in range(total_rank):
                        T_computed += U_full[i*m+j, r] * V_full[j*p+k, r] * W_full[k*n+i, r]
                    error += abs(T_expected - T_computed)
        
        if error > 1e-6:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, column] = 1
                        V[j * p + k, column] = 1
                        W[k * n + i, column] = 1
                        column += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
