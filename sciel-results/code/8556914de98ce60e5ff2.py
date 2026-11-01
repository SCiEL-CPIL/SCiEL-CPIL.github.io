import numpy as np

DIMENSIONS = [2, 4, 8]

def _strassen_2x2_decomp():
    """Return (U_block, V_block, W_block) for a single 2x2x2 Strassen block.
    U_block: 4 entries (indices for A elements)
    V_block: 4 entries (indices for B elements)
    W_block: 4 entries (indices for C elements)
    Each is a list of 7 column-vectors (each a dict mapping index->value).
    """
    # A indices: a00=0, a01=1, a10=2, a11=3
    # B indices: b00=0, b01=1, b10=2, b11=3
    # C indices: c00=0, c01=1, c10=2, c11=3
    
    # 7 multiplication patterns
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    patterns = [
        # (u_coeffs, v_coeffs, w_coeffs)
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),           # M1
        ({2:1, 3:1}, {0:1},     {2:1, 3:-1}),           # M2
        ({0:1},       {1:1, 3:-1}, {1:1, 3:1}),         # M3
        ({3:1},       {2:1, 0:-1}, {0:1, 2:1}),         # M4
        ({0:1, 1:1}, {3:1},     {0:-1, 1:1}),           # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),               # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),               # M7
    ]
    
    U = [{} for _ in range(7)]
    V = [{} for _ in range(7)]
    W = [{} for _ in range(7)]
    
    for s, (u_c, v_c, w_c) in enumerate(patterns):
        for idx, val in u_c.items():
            U[s][idx] = val
        for idx, val in v_c.items():
            V[s][idx] = val
        for idx, val in w_c.items():
            W[s][idx] = val
    
    return U, V, W

def _build_rank_56():
    """Construct rank-56 decomposition for 2x4 @ 4x8."""
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    total_rank = 56  # 8 blocks * 7 per block
    
    # U: shape (n*m, rank) = (8, 56)
    # V: shape (m*p, rank) = (32, 56)
    # W: shape (p*n, rank) = (16, 56)
    # W index convention: W[k*n + i, r] corresponds to C[i, k]
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Get the Strassen block decomposition
    strassen_U, strassen_V, strassen_W = _strassen_2x2_decomp()
    
    # 8 blocks: u in {0,1} for the 2x2 partition of the middle dimension (m=4 -> 2 blocks of 2)
    #            v in {0,1,2,3} for the 4x4 partition of the last dimension (p=8 -> 4 blocks of 2)
    for u in range(2):
        for v in range(4):
            base_col = 7 * (4 * u + v)
            
            # A block: A[i, 2u] and A[i, 2u+1] for i in {0,1}
            # A index: i * m + j = i * 4 + j
            # Within the 2x2 block: a00=A[0,2u], a01=A[0,2u+1], a10=A[1,2u], a11=A[1,2u+1]
            
            # B block: B[2u, 2v] and B[2u+1, 2v] etc.
            # B index: j * p + k = j * 8 + k
            # Within the 2x2 block: b00=B[2u,2v], b01=B[2u,2v+1], b10=B[2u+1,2v], b11=B[2u+1,2v+1]
            
            # C block: C[i, 2v] and C[i, 2v+1] for i in {0,1}
            # C index: k * n + i = k * 2 + i
            # Within the 2x2 block: c00=C[0,2v]=W[2v*2+0], c01=C[0,2v+1]=W[(2v+1)*2+0]
            #                   c10=C[1,2v]=W[2v*2+1],   c11=C[1,2v+1]=W[(2v+1)*2+1]
            
            # Map local indices to global indices
            a_map = {
                0: 0 * m + 2 * u,       # a00
                1: 0 * m + 2 * u + 1,   # a01
                2: 1 * m + 2 * u,       # a10
                3: 1 * m + 2 * u + 1,   # a11
            }
            b_map = {
                0: 2 * u * p + 2 * v,        # b00
                1: 2 * u * p + 2 * v + 1,    # b01
                2: (2 * u + 1) * p + 2 * v,  # b10
                3: (2 * u + 1) * p + 2 * v + 1,  # b11
            }
            c_map = {
                0: 2 * v * n + 0,        # c00 -> W[2v*2 + 0]
                1: (2 * v + 1) * n + 0,  # c01 -> W[(2v+1)*2 + 0]
                2: 2 * v * n + 1,        # c10 -> W[2v*2 + 1]
                3: (2 * v + 1) * n + 1,  # c11 -> W[(2v+1)*2 + 1]
            }
            
            for s in range(7):
                col = base_col + s
                for local_idx, val in strassen_U[s].items():
                    U[a_map[local_idx]][col] += val
                for local_idx, val in strassen_V[s].items():
                    V[b_map[local_idx]][col] += val
                for local_idx, val in strassen_W[s].items():
                    W[c_map[local_idx]][col] += val
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    U = np.array(result["U"], dtype=np.int64)
    V = np.array(result["V"], dtype=np.int64)
    W = np.array(result["W"], dtype=np.int64)
    
    rank = U.shape[1]
    
    # Check shapes
    if U.shape[0] != n * m:
        print(f"U shape mismatch: {U.shape[0]} != {n*m}")
        return False
    if V.shape[0] != m * p:
        print(f"V shape mismatch: {V.shape[0]} != {m*p}")
        return False
    if W.shape[0] != p * n:
        print(f"W shape mismatch: {W.shape[0]} != {p*n}")
        return False
    
    # For each rank term r, check that U[:,r] * V[:,r] matches the outer-product contribution
    # The decomposition says: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j
    # Actually, the tensor T_{ijk} = U_{i,j} * V_{j,k} * W_{k,i} ... no.
    # The standard form: T_{i,j,k} = sum_r U_{(i*m+j), r} * V_{(j*p+k), r} * W_{(k*n+i), r}
    # where T_{i,j,k} = 1 if... no, T is the matrix multiplication tensor.
    # Actually: (AB)_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The tensor T[i,j,k] such that T[i,j,k] = B_{j,k} (with A as a separate vector)
    # Or more precisely, the rank-1 term is: U_vec ⊗ V_vec ⊗ W_vec
    # where the multiplication is: for each r, contribution to C[i,k] = W[k*n+i, r] * sum_j U[i*m+j, r] * V[j*p+k, r] * A[i,j] * B[j,k]
    # Hmm, let me think again.
    
    # Standard tensor decomposition for matrix multiplication:
    # M = sum_r (sum_i A_i * u_i_r) ⊗ (sum_j B_j * v_j_r) ⊗ ... 
    # Actually for A(B): result C[i,k] = sum_j A[i,j] B[j,k]
    # Rank-r decomposition: C[i,k] = sum_s W[s, k*n+i] * (sum_j U[s, i*m+j] * A[i,j]) * (sum_l V[s, j*p+k] * B[j,k])
    # Wait, that's not quite right either. Let me just verify numerically.
    
    # The contract says: entries in U, V, W such that the tensor decomposition holds.
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # A rank-r decomposition means: C[i,k] = sum_s (linear in A) * (linear in B) * W_entry
    # Specifically: C[i,k] = sum_s W[k*n+i, s] * (sum_j U[s, i*m+j] * A[i,j]) * (sum_l V[s, l*p+k] * B[l,k])
    
    # Let's test with random A and B
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m))
    B = np.random.randint(-5, 6, size=(m, p))
    C_true = A @ B
    
    C_decomp = np.zeros((n, p), dtype=np.int64)
    for s in range(rank):
        # linear form in A for rank s
        LA = np.zeros((n, m), dtype=np.int64)  # actually it's per i
        # For each i: L_A[i] = sum_j U[s, i*m+j] * A[i,j]
        L_A = np.zeros(n, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                L_A[i] += U[s, i*m + j] * A[i, j]
        
        # linear form in B for rank s
        # For each k: L_B[k] = sum_l V[s, l*p+k] * B[l,k]
        L_B = np.zeros(p, dtype=np.int64)
        for k in range(p):
            for l in range(m):
                L_B[k] += V[s, l*p + k] * B[l, k]
        
        w_s = W[:, s]  # shape (p*n,)
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += w_s[k*n + i] * L_A[i] * L_B[k]
    
    if not np.allclose(C_decomp, C_true):
        print("Decomposition verification FAILED")
        print(f"C_true:\n{C_true}")
        print(f"C_decomp:\n{C_decomp}")
        return False
    
    # Also test with a few more random matrices
    for seed in range(10):
        np.random.seed(seed)
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_true = A @ B
        C_decomp = np.zeros((n, p), dtype=np.int64)
        for s in range(rank):
            L_A = np.zeros(n, dtype=np.int64)
            for i in range(n):
                for j in range(m):
                    L_A[i] += U[s, i*m + j] * A[i, j]
            L_B = np.zeros(p, dtype=np.int64)
            for k in range(p):
                for l in range(m):
                    L_B[k] += V[s, l*p + k] * B[l, k]
            w_s = W[:, s]
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += w_s[k*n + i] * L_A[i] * L_B[k]
        if not np.allclose(C_decomp, C_true):
            print(f"Decomposition verification FAILED for seed {seed}")
            return False
    
    print(f"Decomposition verified with rank {rank}")
    return True

def construct():
    n, m, p = DIMENSIONS
    result = _build_rank_56()
    
    # Verify in debug mode (comment out for final if too slow)
    # _verify_decomposition(result)
    
    return result
