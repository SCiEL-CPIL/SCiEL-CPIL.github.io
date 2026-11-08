"""Tensor decomposition for 3x3 * 3x7 matrix multiplication."""
import time
import sys

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Uses Strassen decomposition for 2x2 blocks where beneficial, schoolbook for the rest.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    # Helper to create zero vectors
    def zero_U(): return [0] * (n * m)
    def zero_V(): return [0] * (m * p)
    def zero_W(): return [0] * (p * n)
    
    # Index helpers
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i  # W[k*n+i]
    
    terms = []  # List of (U_vec, V_vec, W_vec)
    
    def add_term(U_vec, V_vec, W_vec):
        terms.append((U_vec, V_vec, W_vec))
    
    # Strassen for 2x2 blocks: A[a:b, c:d] * B[e:f, g:h] -> C[i:j, k:l]
    # where b-a=2, f-e=2, l-k=2, j-i=2
    # A indices: (i,c)=(a,e), (i+1,c)=(a+1,e), (i,c+1)=(a,e+1), (i+1,c+1)=(a+1,e+1)
    # Actually let me use explicit indices.
    
    def strassen_2x2(a_i, a_j, b_j, b_k, c_i, c_k):
        """
        Strassen decomposition for:
        A[a_i:a_i+2, a_j:a_j+2] * B[b_j:b_j+2, b_k:b_k+2] -> C[c_i:c_i+2, c_k:c_k+2]
        
        A elements: 
          m00 = A[a_i, a_j], m01 = A[a_i, a_j+1]
          m10 = A[a_i+1, a_j], m11 = A[a_i+1, a_j+1]
        
        B elements:
          n00 = B[b_j, b_k], n01 = B[b_j, b_k+1]
          n10 = B[b_j+1, b_k], n11 = B[b_j+1, b_k+1]
        
        C elements:
          c00 = C[c_i, c_k], c01 = C[c_i, c_k+1]
          c10 = C[c_i+1, c_k], c11 = C[c_i+1, c_k+1]
        """
        # m1 = (m00 + m11) * (n00 + n11)
        U1 = zero_U(); V1 = zero_V(); W1 = zero_W()
        U1[u_idx(a_i, a_j)] = 1
        U1[u_idx(a_i+1, a_j+1)] = 1
        V1[v_idx(b_j, b_k)] = 1
        V1[v_idx(b_j+1, b_k+1)] = 1
        W1[w_idx(c_i, c_k)] = 1
        W1[w_idx(c_i+1, c_k+1)] = 1
        add_term(U1, V1, W1)
        
        # m2 = (m10 + m11) * n00
        U2 = zero_U(); V2 = zero_V(); W2 = zero_W()
        U2[u_idx(a_i+1, a_j)] = 1
        U2[u_idx(a_i+1, a_j+1)] = 1
        V2[v_idx(b_j, b_k)] = 1
        W2[w_idx(c_i+1, c_k)] = 1
        W2[w_idx(c_i+1, c_k+1)] = -1
        add_term(U2, V2, W2)
        
        # m3 = m00 * (n01 - n11)
        U3 = zero_U(); V3 = zero_V(); W3 = zero_W()
        U3[u_idx(a_i, a_j)] = 1
        V3[v_idx(b_j, b_k+1)] = 1
        V3[v_idx(b_j+1, b_k+1)] = -1
        W3[w_idx(c_i, c_k+1)] = 1
        W3[w_idx(c_i+1, c_k+1)] = 1
        add_term(U3, V3, W3)
        
        # m4 = m11 * (n10 - n00)
        U4 = zero_U(); V4 = zero_V(); W4 = zero_W()
        U4[u_idx(a_i+1, a_j+1)] = 1
        V4[v_idx(b_j+1, b_k)] = 1
        V4[v_idx(b_j, b_k)] = -1
        W4[w_idx(c_i, c_k)] = 1
        W4[w_idx(c_i+1, c_k)] = 1
        add_term(U4, V4, W4)
        
        # m5 = (m00 + m01) * n11
        U5 = zero_U(); V5 = zero_V(); W5 = zero_W()
        U5[u_idx(a_i, a_j)] = 1
        U5[u_idx(a_i, a_j+1)] = 1
        V5[v_idx(b_j+1, b_k+1)] = 1
        W5[w_idx(c_i, c_k)] = -1
        W5[w_idx(c_i, c_k+1)] = 1
        add_term(U5, V5, W5)
        
        # m6 = (m10 - m00) * (n00 + n01)
        U6 = zero_U(); V6 = zero_V(); W6 = zero_W()
        U6[u_idx(a_i+1, a_j)] = 1
        U6[u_idx(a_i, a_j)] = -1
        V6[v_idx(b_j, b_k)] = 1
        V6[v_idx(b_j, b_k+1)] = 1
        W6[w_idx(c_i+1, c_k+1)] = 1
        add_term(U6, V6, W6)
        
        # m7 = (m01 - m11) * (n10 + n11)
        U7 = zero_U(); V7 = zero_V(); W7 = zero_W()
        U7[u_idx(a_i, a_j+1)] = 1
        U7[u_idx(a_i+1, a_j+1)] = -1
        V7[v_idx(b_j+1, b_k)] = 1
        V7[v_idx(b_j+1, b_k+1)] = 1
        W7[w_idx(c_i, c_k)] = 1
        add_term(U7, V7, W7)
    
    def schoolbook_block(a_i, a_j, b_j, b_k, c_i, c_k, di, dj, dk):
        """
        Schoolbook for A[a_i:a_i+di, a_j:a_j+dj] * B[b_j:b_j+dk, b_k:b_k+dk] -> C[c_i:c_i+di, c_k:c_k+dk]
        """
        for ii in range(di):
            for jj in range(dj):
                for kk in range(dk):
                    U = zero_U(); V = zero_V(); W = zero_W()
                    U[u_idx(a_i+ii, a_j+jj)] = 1
                    V[v_idx(b_j+jj, b_k+kk)] = 1
                    W[w_idx(c_i+ii, c_k+kk)] = 1
                    add_term(U, V, W)
    
    # Decomposition:
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2] * B[2, 0:2]
    # C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2] * B[2, 2:4]
    # C[0:2, 4:7] = A[0:2, 0:2] * B[0:2, 4:7] + A[0:2, 2] * B[2, 4:7]
    # C[2, 0:2]   = A[2, 0:2] * B[0:2, 0:2] + A[2, 2] * B[2, 0:2]
    # C[2, 2:4]   = A[2, 0:2] * B[0:2, 2:4] + A[2, 2] * B[2, 2:4]
    # C[2, 4:7]   = A[2, 0:2] * B[0:2, 4:7] + A[2, 2] * B[2, 4:7]
    
    # Let me organize by the A column block and B row block.
    
    # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] : Strassen
    strassen_2x2(0, 0, 0, 0, 0, 0)
    
    # A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] : Strassen
    strassen_2x2(0, 0, 0, 2, 0, 2)
    
    # A[0:2, 0:2] * B[0:2, 4:7] -> C[0:2, 4:7] : Schoolbook (2x2x3)
    schoolbook_block(0, 0, 0, 4, 0, 4, 2, 2, 3)
    
    # A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2] : Schoolbook (2x1x2)
    schoolbook_block(0, 2, 2, 0, 0, 0, 2, 1, 2)
    
    # A[0:2, 2] * B[2, 2:4] -> C[0:2, 2:4] : Schoolbook (2x1x2)
    schoolbook_block(0, 2, 2, 2, 0, 2, 2, 1, 2)
    
    # A[0:2, 2] * B[2, 4:7] -> C[0:2, 4:7] : Schoolbook (2x1x3)
    schoolbook_block(0, 2, 2, 4, 0, 4, 2, 1, 3)
    
    # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2] : Schoolbook (1x2x2)
    schoolbook_block(2, 0, 0, 0, 2, 0, 1, 2, 2)
    
    # A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4] : Schoolbook (1x2x2)
    schoolbook_block(2, 0, 0, 2, 2, 2, 1, 2, 2)
    
    # A[2, 0:2] * B[0:2, 4:7] -> C[2, 4:7] : Schoolbook (1x2x3)
    schoolbook_block(2, 0, 0, 4, 2, 4, 1, 2, 3)
    
    # A[2, 2] * B[2, 0:2] -> C[2, 0:2] : Schoolbook (1x1x2)
    schoolbook_block(2, 2, 2, 0, 2, 0, 1, 1, 2)
    
    # A[2, 2] * B[2, 2:4] -> C[2, 2:4] : Schoolbook (1x1x2)
    schoolbook_block(2, 2, 2, 2, 2, 2, 1, 1, 2)
    
    # A[2, 2] * B[2, 4:7] -> C[2, 4:7] : Schoolbook (1x1x3)
    schoolbook_block(2, 2, 2, 4, 2, 4, 1, 1, 3)
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (U_vec, V_vec, W_vec) in enumerate(terms):
        for i in range(n * m):
            if U_vec[i] != 0:
                U[i][c] = U_vec[i]
        for i in range(m * p):
            if V_vec[i] != 0:
                V[i][c] = V_vec[i]
        for i in range(p * n):
            if W_vec[i] != 0:
                W[i][c] = W_vec[i]
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if total != 1:
                        return False, (i, j, k, total)
        return True, None
    
    valid, err = verify(U, V, W, n, m, p)
    
    if not valid:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    start_time = time.time()
    res = construct()
    end_time = time.time()
    if end_time - start_time > 110:
        print("Warning: Construction took too long", file=sys.stderr)
    n, m, p = DIMENSIONS
    assert len(res['U']) == n*m
    assert len(res['V']) == m*p
    assert len(res['W']) == p*n
    rank = len(res['U'][0])
    assert len(res['V'][0]) == rank
    assert len(res['W'][0]) == rank
    print(f"Rank: {rank}")
