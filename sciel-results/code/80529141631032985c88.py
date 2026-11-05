"""Combined program: Strassen for 2x2 blocks + schoolbook for 1x4 strips."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy:
    - Partition A (3x4) into A1 (2x4) and A2 (1x4)
    - Partition B (4x8) into B1 (4x4) and B2 (4x4)
    - For A1*B1: use Strassen on 2x2 blocks (4 blocks, each rank 7 = 28 total)
    - For A1*B2: use Strassen on 2x2 blocks (4 blocks, each rank 7 = 28 total)
    - For A2*B1: schoolbook 1x4*4x4, rank 16
    - For A2*B2: schoolbook 1x4*4x4, rank 16
    Total rank: 28 + 28 + 16 + 16 = 88
    """
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # We'll build the decomposition by accumulating terms
    # U has shape (n*m, rank) = (12, rank)
    # V has shape (m*p, rank) = (32, rank)
    # W has shape (p*n, rank) = (24, rank)
    
    terms = []  # List of (u_vector, v_vector, w_vector)
    
    def add_strassen_block(i0, j0, k0):
        """
        Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        
        Strassen's algorithm for 2x2*2x2:
        Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        
        M1 = (a11 + a22) * (b11 + b22)
        M2 = (a21 + a22) * b11
        M3 = a11 * (b12 - b22)
        M4 = a22 * (b21 - b11)
        M5 = (a11 + a12) * b22
        M6 = (a21 - a11) * (b11 + b12)
        M7 = (a12 - a22) * (b21 + b22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # Map local indices to global indices
        # A indices: rows i0, i0+1; cols j0, j0+1
        # B indices: rows j0, j0+1; cols k0, k0+1
        # C indices: rows i0, i0+1; cols k0, k0+1
        
        # U index: (i, j) -> i*m + j
        # V index: (j, k) -> j*p + k
        # W index: (k, i) -> k*n + i
        
        a11_u = (i0) * m + j0
        a12_u = (i0) * m + (j0 + 1)
        a21_u = (i0 + 1) * m + j0
        a22_u = (i0 + 1) * m + (j0 + 1)
        
        b11_v = j0 * p + k0
        b12_v = j0 * p + (k0 + 1)
        b21_v = (j0 + 1) * p + k0
        b22_v = (j0 + 1) * p + (k0 + 1)
        
        c11_w = k0 * n + i0
        c12_w = (k0 + 1) * n + i0
        c21_w = k0 * n + (i0 + 1)
        c22_w = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a11 + a22) * (b11 + b22)
        u1 = [0.0] * (n * m); u1[a11_u] = 1.0; u1[a22_u] = 1.0
        v1 = [0.0] * (m * p); v1[b11_v] = 1.0; v1[b22_v] = 1.0
        w1 = [0.0] * (p * n); w1[c11_w] = 1.0; w1[c22_w] = 1.0
        terms.append((u1, v1, w1))
        
        # M2 = (a21 + a22) * b11
        u2 = [0.0] * (n * m); u2[a21_u] = 1.0; u2[a22_u] = 1.0
        v2 = [0.0] * (m * p); v2[b11_v] = 1.0
        w2 = [0.0] * (p * n); w2[c21_w] = 1.0; w2[c22_w] = -1.0
        terms.append((u2, v2, w2))
        
        # M3 = a11 * (b12 - b22)
        u3 = [0.0] * (n * m); u3[a11_u] = 1.0
        v3 = [0.0] * (m * p); v3[b12_v] = 1.0; v3[b22_v] = -1.0
        w3 = [0.0] * (p * n); w3[c12_w] = 1.0; w3[c22_w] = 1.0
        terms.append((u3, v3, w3))
        
        # M4 = a22 * (b21 - b11)
        u4 = [0.0] * (n * m); u4[a22_u] = 1.0
        v4 = [0.0] * (m * p); v4[b21_v] = 1.0; v4[b11_v] = -1.0
        w4 = [0.0] * (p * n); w4[c11_w] = 1.0; w4[c21_w] = 1.0
        terms.append((u4, v4, w4))
        
        # M5 = (a11 + a12) * b22
        u5 = [0.0] * (n * m); u5[a11_u] = 1.0; u5[a12_u] = 1.0
        v5 = [0.0] * (m * p); v5[b22_v] = 1.0
        w5 = [0.0] * (p * n); w5[c11_w] = -1.0; w5[c12_w] = 1.0
        terms.append((u5, v5, w5))
        
        # M6 = (a21 - a11) * (b11 + b12)
        u6 = [0.0] * (n * m); u6[a21_u] = 1.0; u6[a11_u] = -1.0
        v6 = [0.0] * (m * p); v6[b11_v] = 1.0; v6[b12_v] = 1.0
        w6 = [0.0] * (p * n); w6[c22_w] = 1.0
        terms.append((u6, v6, w6))
        
        # M7 = (a12 - a22) * (b21 + b22)
        u7 = [0.0] * (n * m); u7[a12_u] = 1.0; u7[a22_u] = -1.0
        v7 = [0.0] * (m * p); v7[b21_v] = 1.0; v7[b22_v] = 1.0
        w7 = [0.0] * (p * n); w7[c11_w] = 1.0
        terms.append((u7, v7, w7))
    
    def add_schoolbook_1x4_4x4(i_row, j0, k0):
        """
        Add schoolbook decomposition for A[i_row, j0:j0+4] * B[j0:j0+4, k0:k0+4]
        This is a 1x4 * 4x4 multiplication, rank 16.
        """
        for j in range(4):
            for k in range(4):
                u_idx = i_row * m + (j0 + j)
                v_idx = (j0 + j) * p + (k0 + k)
                w_idx = (k0 + k) * n + i_row
                u_vec = [0.0] * (n * m); u_vec[u_idx] = 1.0
                v_vec = [0.0] * (m * p); v_vec[v_idx] = 1.0
                w_vec = [0.0] * (p * n); w_vec[w_idx] = 1.0
                terms.append((u_vec, v_vec, w_vec))
    
    # Check time before starting
    if time.time() - start_time > deadline - 10:
        # Fallback to schoolbook
        return _schoolbook_fallback(n, m, p)
    
    # A1*B1: A1 is 2x4, B1 is 4x4
    # Split into 2x2 blocks:
    # Block (0,0): A1[0:2, 0:2] * B1[0:2, 0:2]
    add_strassen_block(i0=0, j0=0, k0=0)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # Block (0,1): A1[0:2, 2:4] * B1[2:4, 0:2]
    add_strassen_block(i0=0, j0=2, k0=0)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # Block (1,0): A1[0:2, 0:2] * B1[0:2, 2:4]
    # Wait, I need to be more careful. Let me reconsider the block structure.
    # A1 is 2x4, B1 is 4x4
    # A1 = [[A1_11, A1_12], [A1_21, A1_22]] where each is 2x2
    # B1 = [[B1_11, B1_12], [B1_21, B1_22]] where each is 2x2
    # C = A1*B1 = [[A1_11*B1_11 + A1_12*B1_21, A1_11*B1_12 + A1_12*B1_22],
    #              [A1_21*B1_11 + A1_22*B1_21, A1_21*B1_12 + A1_22*B1_22]]
    
    # Actually, the standard way to apply Strassen to a 2x4 * 4x4 product:
    # We can view it as two independent 2x2 * 2x2 products side by side
    # A1*B1 = [A1_left * B1_left | A1_left * B1_right] + [A1_right * B1_left | A1_right * B1_right]
    # But that's not quite right either.
    
    # Let me think again. A1 is 2x4, B1 is 4x4.
    # We can split A1 = [A1_L | A1_R] where A1_L is 2x2, A1_R is 2x2
    # And B1 = [B1_T; B1_B] where B1_T is 2x4, B1_B is 2x4
    # Then A1*B1 = A1_L*B1_T + A1_R*B1_B
    # Each of these is 2x2 * 2x4, which we can further split.
    
    # Actually, the simplest approach: split into 4 independent 2x2*2x2 blocks
    # A1[0:2, 0:2] * B1[0:2, 0:2] -> contributes to C[0:2, 0:2]
    # A1[0:2, 0:2] * B1[0:2, 2:4] -> contributes to C[0:2, 2:4]
    # A1[0:2, 2:4] * B1[2:4, 0:2] -> contributes to C[0:2, 0:2]
    # A1[0:2, 2:4] * B1[2:4, 2:4] -> contributes to C[0:2, 2:4]
    
    # Wait, that's not right. The matrix product is:
    # C[i,k] = sum_j A1[i,j] * B1[j,k]
    # For i in {0,1}, k in {0,1,2,3}, j in {0,1,2,3}
    
    # So C[0:2, 0:2] = A1[0:2, 0:2]*B1[0:2, 0:2] + A1[0:2, 2:4]*B1[2:4, 0:2]
    # C[0:2, 2:4] = A1[0:2, 0:2]*B1[0:2, 2:4] + A1[0:2, 2:4]*B1[2:4, 2:4]
    
    # Each of these is a sum of two 2x2*2x2 products. We can apply Strassen to each.
    
    # Let me redo this properly.
    # Clear terms and restart
    terms.clear()
    
    # A1*B1: 2x4 * 4x4
    # Split A1 into left (cols 0-1) and right (cols 2-3)
    # Split B1 into top (rows 0-1) and bottom (rows 2-3)
    
    # C[0:2, 0:2] = A1[:, 0:2] * B1[0:2, 0:2] + A1[:, 2:4] * B1[2:4, 0:2]
    # Apply Strassen to each 2x2*2x2 product
    
    # A1[:, 0:2] * B1[0:2, 0:2]: i0=0, j0=0, k0=0
    add_strassen_block(i0=0, j0=0, k0=0)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # A1[:, 2:4] * B1[2:4, 0:2]: i0=0, j0=2, k0=0
    add_strassen_block(i0=0, j0=2, k0=0)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # C[0:2, 2:4] = A1[:, 0:2] * B1[0:2, 2:4] + A1[:, 2:4] * B1[2:4, 2:4]
    
    # A1[:, 0:2] * B1[0:2, 2:4]: i0=0, j0=0, k0=2
    add_strassen_block(i0=0, j0=0, k0=2)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # A1[:, 2:4] * B1[2:4, 2:4]: i0=0, j0=2, k0=2
    add_strassen_block(i0=0, j0=2, k0=2)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # A1*B2: 2x4 * 4x4 (B2 is columns 4-7 of B)
    # C[0:2, 4:6] = A1[:, 0:2] * B2[0:2, 0:2] + A1[:, 2:4] * B2[2:4, 0:2]
    # B2[0:2, 0:2] corresponds to B[0:2, 4:6], so j0=0, k0=4
    add_strassen_block(i0=0, j0=0, k0=4)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # A1[:, 2:4] * B2[2:4, 0:2]: j0=2, k0=4
    add_strassen_block(i0=0, j0=2, k0=4)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # C[0:2, 6:8] = A1[:, 0:2] * B2[0:2, 2:4] + A1[:, 2:4] * B2[2:4, 2:4]
    # B2[0:2, 2:4] corresponds to B[0:2, 6:8], so j0=0, k0=6
    add_strassen_block(i0=0, j0=0, k0=6)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # A1[:, 2:4] * B2[2:4, 2:4]: j0=2, k0=6
    add_strassen_block(i0=0, j0=2, k0=6)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # A2*B1: 1x4 * 4x4, schoolbook, rank 16
    add_schoolbook_1x4_4x4(i_row=2, j0=0, k0=0)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # A2*B2: 1x4 * 4x4, schoolbook, rank 16
    add_schoolbook_1x4_4x4(i_row=2, j0=0, k0=4)
    
    if time.time() - start_time > deadline - 10:
        return _schoolbook_fallback(n, m, p)
    
    # Now build the U, V, W matrices from the terms
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            if u[i] != 0:
                U[i][r] = u[i]
        for j in range(m * p):
            if v[j] != 0:
                V[j][r] = v[j]
        for k in range(p * n):
            if w[k] != 0:
                W[k][r] = w[k]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_fallback(n, m, p):
    """Fallback to schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
