import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify tensor decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """Create the schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W

def _find_rank7_2x2x3():
    """
    Search for a rank-7 decomposition of 2x2 x 2x3 matrix multiplication.
    Returns (U, V, W) of shapes (4, 7), (6, 7), (6, 7) if found, else None.
    """
    n, m, p = 2, 2, 3
    target_rank = 7
    # Random search over {-1, 0, 1}
    np.random.seed(42)
    for _ in range(5000):
        U = np.random.choice([-1, 0, 1], size=(n*m, target_rank))
        V = np.random.choice([-1, 0, 1], size=(m*p, target_rank))
        W = np.random.choice([-1, 0, 1], size=(p*n, target_rank))
        
        # Check
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if valid:
            # Also check zero entries to ensure it's the exact tensor
            zero_valid = True
            for i in range(n):
                for j in range(m):
                    for j2 in range(m):
                        if j == j2: continue
                        for k in range(p):
                            s = np.dot(U[i*m+j], V[j2*p+k] * W[k*n+i])
                            if abs(s) > 1e-9:
                                zero_valid = False
                                break
                        if not zero_valid: break
                    if not zero_valid: break
                if not zero_valid: break
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            if k == k2: continue
                            s = np.dot(U[i*m+j], V[j*p+k2] * W[k*n+i])
                            if abs(s) > 1e-9:
                                zero_valid = False
                                break
                            for i2 in range(n):
                                if i == i2: continue
                                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i2])
                                if abs(s) > 1e-9:
                                    zero_valid = False
                                    break
                            if not zero_valid: break
                        if not zero_valid: break
                    if not zero_valid: break
                if not zero_valid: break
            if zero_valid:
                return U, V, W
    return None

def _strassen_4x4_terms():
    """Generate 49 Strassen terms for 4x4x4 via Kronecker product of 2x2 Strassen."""
    # 2x2 Strassen terms
    base_terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),
        ([0,0,0,1], [0,0,1,-1], [-1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [-1,0,0,-1])
    ]
    
    terms_4x4 = []
    for c1 in range(7):
        for c2 in range(7):
            U4 = [0] * 16
            V4 = [0] * 16
            W4 = [0] * 16
            for I in range(4):
                for J in range(4):
                    I1, I0 = I // 2, I % 2
                    J1, J0 = J // 2, J % 2
                    U4[I*4+J] = base_terms[c1][0][I1*2+J1] * base_terms[c2][0][I0*2+J0]
            for J in range(4):
                for K in range(4):
                    J1, J0 = J // 2, J % 2
                    K1, K0 = K // 2, K % 2
                    V4[J*4+K] = base_terms[c1][1][J1*2+K1] * base_terms[c2][1][J0*2+K0]
            for K in range(4):
                for I in range(4):
                    K1, K0 = K // 2, K % 2
                    I1, I0 = I // 2, I % 2
                    W4[K*4+I] = base_terms[c1][2][K1*2+I1] * base_terms[c2][2][K0*2+I0]
            terms_4x4.append((U4, V4, W4))
    return terms_4x4

def _make_optimized_decomposition():
    """
    Create a rank-77 decomposition:
    49 (Strassen 4x4x4) + 4 * 7 (Rank 7 for 2x2x3)
    """
    n, m, p = 4, 4, 7
    
    # Part 1: Strassen 4x4x4 (Rank 49)
    terms_4x4 = _strassen_4x4_terms()
    
    U_list = []
    V_list = []
    W_list = []
    
    for U4, V4, W4 in terms_4x4:
        # Extend V to 4x7 (first 4 cols of B)
        V_ext = [0] * 28
        for J in range(4):
            for K in range(4):
                V_ext[J*7+K] = V4[J*4+K]
        # W is 4x4 (K is 0..3, I is 0..3)
        W_ext = [0] * 28
        for K in range(4):
            for I in range(4):
                W_ext[K*4+I] = W4[K*4+I]
        
        U_list.append(list(U4))
        V_list.append(V_ext)
        W_list.append(W_ext)
        
    # Part 2: 4 blocks of 2x2x3 (Rank 28)
    # Find rank-7 decomposition for 2x2x3
    r7_decomp = _find_rank7_2x2x3()
    
    if r7_decomp:
        U7, V7, W7 = r7_decomp
        # We need to apply this to 4 blocks:
        # Block 1: A[0:2, 0:2] * B[0:2, 0:3] -> C[0:2, 0:3]
        # Block 2: A[0:2, 2:4] * B[2:4, 0:3] -> C[0:2, 0:3] (Add to C)
        # Block 3: A[2:4, 0:2] * B[0:2, 0:3] -> C[2:4, 0:3]
        # Block 4: A[2:4, 2:4] * B[2:4, 0:3] -> C[2:4, 0:3]
        
        # Note: The decomposition U7, V7, W7 is for local indices.
        # Local U: 2x2 (4 entries)
        # Local V: 2x3 (6 entries)
        # Local W: 3x2 (6 entries)
        
        blocks = [
            (0, 0, 0, 0), # (i_start, j_start, k_start, block_id) - actually just offsets
            (0, 2, 2, 1),
            (2, 0, 0, 2),
            (2, 2, 2, 3)
        ]
        
        for i_off, j_off, k_off, _ in blocks:
            # k_off is row offset for B and C.
            # i_off is row offset for A and C.
            # j_off is col offset for A and row offset for B.
            
            for r in range(7):
                U_row = [0] * 16
                V_row = [0] * 28
                W_row = [0] * 28
                
                # Fill U (A part)
                for li in range(2):
                    for lj in range(2):
                        local_idx = li * 2 + lj
                        global_idx = (i_off + li) * 4 + (j_off + lj)
                        U_row[global_idx] = U7[local_idx, r]
                        
                # Fill V (B part)
                for lj in range(2): # local row of B (corresponds to col of A)
                    for lk in range(3): # local col of B
                        local_idx = lj * 3 + lk
                        global_idx = (j_off + lj) * 7 + (k_off + lk)
                        V_row[global_idx] = V7[local_idx, r]
                        
                # Fill W (C part)
                for lk in range(3): # local col of C
                    for li in range(2): # local row of C
                        local_idx = lk * 2 + li
                        global_idx = (k_off + lk) * 4 + (i_off + li)
                        W_row[global_idx] = W7[local_idx, r]
                        
                U_list.append(U_row)
                V_list.append(V_row)
                W_list.append(W_row)
    else:
        # Fallback: Schoolbook for the 4x4x3 part
        # This adds rank 48
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U_row = [0] * 16
                    V_row = [0] * 28
                    W_row = [0] * 28
                    U_row[i*4+j] = 1
                    V_row[j*7+(k+4)] = 1
                    W_row[(k+4)*4+i] = 1
                    U_list.append(U_row)
                    V_list.append(V_row)
                    W_list.append(W_row)
                    
    rank = len(U_list)
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    for c in range(rank):
        U[:, c] = U_list[c]
        V[:, c] = V_list[c]
        W[:, c] = W_list[c]
        
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try optimized decomposition
    U, V, W = _make_optimized_decomposition()
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook_decomposition(n, m, p)
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
