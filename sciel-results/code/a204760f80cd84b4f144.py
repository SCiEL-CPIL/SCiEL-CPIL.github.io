import time
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 145.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Verify with random matrices
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            A_c = np.zeros((n, m))
            for idx in range(n * m):
                i, j = idx // m, idx % m
                A_c[i, j] = uc[idx]
            B_c = np.zeros((m, p))
            for idx in range(m * p):
                j, k = idx // p, idx % p
                B_c[j, k] = vc[idx]
            C_c = np.zeros((n, p))
            for idx in range(p * n):
                k, i = idx // n, idx % n
                C_c[i, k] = wc[idx]
            C_computed += C_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_strassen_444_kron(U, V, W):
    """
    Build rank-49 Strassen decomposition for 4x4 x 4x4 using Kronecker products.
    Fills columns 0..48 of U, V, W.
    """
    # 2x2 Strassen: 7 terms
    # Each term has: u (2x2), v (2x2), w (2x2) coefficients
    # U index: (i,j) -> i*2+j for A (2x2)
    # V index: (j,k) -> j*2+k for B (2x2)  
    # W index: (k,i) -> k*2+i for C (2x2) [trace convention W[k*n+i,r]]
    
    # 2x2 Strassen terms as (u_matrix, v_matrix, w_matrix)
    # u_matrix[i,j] = coefficient for A[i,j]
    # v_matrix[j,k] = coefficient for B[j,k]
    # w_matrix[k,i] = coefficient for C[i,k] (note: W index is k*2+i)
    
    S = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        (np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])),
        # M2 = (A10+A11)B00 -> C10 += M2
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,0],[1,0]])),
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        (np.array([[1,0],[0,0]]), np.array([[0,1],[0,-1]]), np.array([[0,1],[0,1]])),
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        (np.array([[0,0],[0,1]]), np.array([[0,-1],[1,0]]), np.array([[1,0],[1,0]])),
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        (np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([[-1,1],[0,0]])),
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        (np.array([[-1,0],[1,0]]), np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]])),
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        (np.array([[0,1],[0,-1]]), np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]])),
    ]
    
    # 4x4 Strassen: 7 outer terms P0..P6
    # Each P uses 2x2 block combinations of A and B, then applies 2x2 Strassen internally
    # This gives 7*7 = 49 rank-1 terms
    
    # For each outer term p_idx, define:
    # a_blocks: dict {(br,bc): coeff} - which 2x2 blocks of A to use
    # b_blocks: dict {(br,bc): coeff} - which 2x2 blocks of B to use
    # c_blocks: dict {(br,bc): coeff} - which 2x2 blocks of C to contribute to
    
    outer_terms = [
        # P0 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P1 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # P2 = A00(B01-B11)
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # P3 = A11(B10-B00)
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P4 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # P5 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P6 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    col_idx = 0
    for p_idx in range(7):
        a_blocks, b_blocks, c_blocks = outer_terms[p_idx]
        
        for t_idx in range(7):
            u2, v2, w2 = S[t_idx]
            
            # Build 4x4 U column
            for i in range(4):
                for j in range(4):
                    # i = br*2 + lr, j = bc*2 + lc
                    br, lr = i // 2, i % 2
                    bc, lc = j // 2, j % 2
                    coeff = 0
                    # Sum over all block combinations that map to (i,j)
                    # The coefficient is sum of a_block_coeff * u2[lr, lc] for blocks where br,bc matches
                    # Actually: for each (br,bc) in a_blocks, if br==i//2 and bc==j//2, 
                    # then add a_blocks[(br,bc)] * u2[lr,lc]
                    key_a = (br, bc)
                    if key_a in a_blocks:
                        coeff = a_blocks[key_a] * u2[lr, lc]
                    U[i*4+j, col_idx] = coeff
            
            # Build 4x4 V column
            for j in range(4):
                for k in range(4):
                    br, lr = j // 2, j % 2
                    bc, lc = k // 2, k % 2
                    coeff = 0
                    key_b = (br, bc)
                    if key_b in b_blocks:
                        coeff = b_blocks[key_b] * v2[lr, lc]
                    V[j*4+k, col_idx] = coeff
            
            # Build 4x4 W column (W index is k*4 + i)
            for k in range(4):
                for i in range(4):
                    br, lr = k // 2, k % 2  # k = br*2 + lr
                    bc, lc = i // 2, i % 2  # i = bc*2 + lc
                    # w2 is indexed as [k_local, i_local] where k_local = lr_k, i_local = lr_i
                    # Actually w2[k_local, i_local] gives coefficient for C[i_local, k_local]
                    # But we need C[i, k] where i = bc*2+lc, k = br*2+lr
                    # w2[lr_k, lr_i] where lr_k = k%2, lr_i = i%2
                    coeff = 0
                    key_c = (br, bc)  # block (br, bc) in C
                    if key_c in c_blocks:
                        # w2[k_local, i_local] = w2[lr, lc] where lr = k%2, lc = i%2
                        coeff = c_blocks[key_c] * w2[lr, lc]
                    W[k*4+i, col_idx] = coeff
            
            col_idx += 1

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Rank 93: 49 (Strassen 4x4x4) + 16 (schoolbook 4x4x1) + 28 (Strassen-like 4x4x2)
    # Actually let's just do 49 + 48 = 97 first, verify it works, then optimize
    # The key structural change is using the Kronecker product approach which is cleaner
    
    rank = 97
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Fill Strassen part for first 4 output columns
    _build_strassen_444_kron(U, V, W)
    
    # Fill Schoolbook part for k=4,5,6
    col_idx = 49
    for k in range(4, 7):
        for i in range(n):
            for j in range(m):
                U[i*m+j, col_idx] = 1
                V[j*p+k, col_idx] = 1
                W[k*n+i, col_idx] = 1
                col_idx += 1
                
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
        rank = n*m*p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
                    if time.time() - start_time > DEADLINE_SECONDS:
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
