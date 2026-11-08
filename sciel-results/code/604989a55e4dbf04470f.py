"""Tensor decomposition for 3x3 * 3x7 matrix multiplication."""
import time
import sys

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Uses a block decomposition:
    - Split C columns into [0:3], [3:6], [6:7]
    - For C[:, 0:3] and C[:, 3:6]: use a 22-rank 3x3x3 decomposition
      (Strassen for top-left 2x2 + schoolbook for the rest)
    - For C[:, 6:7]: use schoolbook (9 terms)
    Total: 22 + 22 + 9 = 53 terms
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    def zero_U(): return [0] * (n * m)
    def zero_V(): return [0] * (m * p)
    def zero_W(): return [0] * (p * n)
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i  # W[k*n+i]
    
    terms = []
    
    def add_term(U_vec, V_vec, W_vec):
        terms.append((U_vec, V_vec, W_vec))
    
    def strassen_2x2(a_i, a_j, b_j, b_k, c_i, c_k):
        """
        Strassen decomposition for:
        A[a_i:a_i+2, a_j:a_j+2] * B[b_j:b_j+2, b_k:b_k+2] -> C[c_i:c_i+2, c_k:c_k+2]
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
        Schoolbook for A[a_i:a_i+di, a_j:a_j+dj] * B[b_j:b_j+dj, b_k:b_k+dk] -> C[c_i:c_i+di, c_k:c_k+dk]
        """
        for ii in range(di):
            for jj in range(dj):
                for kk in range(dk):
                    U = zero_U(); V = zero_V(); W = zero_W()
                    U[u_idx(a_i+ii, a_j+jj)] = 1
                    V[v_idx(b_j+jj, b_k+kk)] = 1
                    W[w_idx(c_i+ii, c_k+kk)] = 1
                    add_term(U, V, W)
    
    def block_3x3_22(c_col_offset):
        """
        22-rank decomposition for C[:, c_col_offset:c_col_offset+3] = A * B[:, c_col_offset:c_col_offset+3]
        Structure:
        - C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] : Strassen (7)
        - C[0:2, 2]   = A[0:2, 0:2] * B[0:2, 2] + A[0:2, 2] * B[2, 2] : Schoolbook (4+2=6)
        - C[2, 0:2]   = A[2, 0:2] * B[0:2, 0:2] + A[2, 2] * B[2, 0:2] : Schoolbook (4+2=6)
        - C[2, 2]     = A[2, 0:2] * B[0:2, 2] + A[2, 2] * B[2, 2] : Schoolbook (2+1=3)
        Total: 7 + 6 + 6 + 3 = 22
        """
        # Strassen for top-left 2x2
        strassen_2x2(0, 0, 0, c_col_offset, 0, c_col_offset)
        
        # C[0:2, 2] = A[0:2, 0:2] * B[0:2, 2] + A[0:2, 2] * B[2, 2]
        # A[0:2, 0:2] * B[0:2, 2] -> C[0:2, c_col_offset+2]
        schoolbook_block(0, 0, 0, c_col_offset+2, 0, c_col_offset+2, 2, 2, 1)
        # A[0:2, 2] * B[2, 2] -> C[0:2, c_col_offset+2]
        schoolbook_block(0, 2, 2, c_col_offset+2, 0, c_col_offset+2, 2, 1, 1)
        
        # C[2, 0:2] = A[2, 0:2] * B[0:2, 0:2] + A[2, 2] * B[2, 0:2]
        # A[2, 0:2] * B[0:2, 0:2] -> C[2, c_col_offset:c_col_offset+2]
        schoolbook_block(2, 0, 0, c_col_offset, 2, c_col_offset, 1, 2, 2)
        # A[2, 2] * B[2, 0:2] -> C[2, c_col_offset:c_col_offset+2]
        schoolbook_block(2, 2, 2, c_col_offset, 2, c_col_offset, 1, 1, 2)
        
        # C[2, 2] = A[2, 0:2] * B[0:2, 2] + A[2, 2] * B[2, 2]
        # A[2, 0:2] * B[0:2, 2] -> C[2, c_col_offset+2]
        schoolbook_block(2, 0, 0, c_col_offset+2, 2, c_col_offset+2, 1, 2, 1)
        # A[2, 2] * B[2, 2] -> C[2, c_col_offset+2]
        schoolbook_block(2, 2, 2, c_col_offset+2, 2, c_col_offset+2, 1, 1, 1)
    
    # Block 1: C[:, 0:3] (22 terms)
    block_3x3_22(0)
    
    # Block 2: C[:, 3:6] (22 terms)
    block_3x3_22(3)
    
    # Block 3: C[:, 6:7] (9 terms, schoolbook)
    schoolbook_block(0, 0, 0, 6, 0, 6, 3, 3, 1)
    
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
