import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """Build the 2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # P1: (x11+x22)(y11+y22) -> c11+=1, c22+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # P2: (x21+x22)(y11) -> c21+=1, c22+=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # P3: (x11)(y12-y22) -> c12+=1, c22+=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # P4: (x22)(y21-y11) -> c11+=1, c21+=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # P5: (x11+x12)(y22) -> c11+=-1, c12+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # P6: (x21-x11)(y11+y12) -> c22+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # P7: (x12-x22)(y21+y22) -> c11+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, seed=42, atol=1e-9):
    np.random.seed(seed)
    A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
    C_expected = A_test @ B_test
    rank = U.shape[1]
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_r = U[:, r].reshape(n, m)
        v_r = V[:, r].reshape(m, p)
        w_r = W[:, r].reshape(p, n)
        M1 = A_test * u_r
        M2 = v_r * B_test
        partial = M1 @ M2
        C_computed += partial * w_r.T
    return np.allclose(C_computed, C_expected, atol=atol)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS
    
    # Build Strassen 2x2
    U22, V22, W22 = _build_strassen_2x2()
    
    # Kronecker product of Strassen 2x2 with itself gives 8x8 decomposition
    # Rank = 7 * 7 = 49
    # Shapes: U22 is (4, 7). U88 will be (64, 49).
    # Index mapping for Kronecker product of vectors a (len 4) and b (len 7)?
    # No, Kronecker product of matrices.
    # U88[i8*8+j8, r8] = U22[i2*2+j2, r2] * U22[i2'*2+j2', r2']
    # where r8 = r2 * 7 + r2'
    # and i8 = i2 * 2 + i2', j8 = j2 * 2 + j2'
    
    rank_88 = 7 * 7
    U88 = np.zeros((64, rank_88), dtype=float)
    V88 = np.zeros((64, rank_88), dtype=float)
    W88 = np.zeros((64, rank_88), dtype=float)
    
    for r2 in range(7):
        for r2_prime in range(7):
            r8 = r2 * 7 + r2_prime
            u2_col = U22[:, r2]
            v2_col = V22[:, r2]
            w2_col = W22[:, r2]
            u2p_col = U22[:, r2_prime]
            v2p_col = V22[:, r2_prime]
            w2p_col = W22[:, r2_prime]
            
            # U88[i8*8+j8, r8]
            for i2 in range(2):
                for j2 in range(2):
                    for i2p in range(2):
                        for j2p in range(2):
                            i8 = i2 * 2 + i2p
                            j8 = j2 * 2 + j2p
                            U88[i8 * 8 + j8, r8] = u2_col[i2 * 2 + j2] * u2p_col[i2p * 2 + j2p]
                            
            # V88[j8*8+k8, r8]
            for j2 in range(2):
                for k2 in range(2):
                    for j2p in range(2):
                        for k2p in range(2):
                            j8 = j2 * 2 + j2p
                            k8 = k2 * 2 + k2p
                            V88[j8 * 8 + k8, r8] = v2_col[j2 * 2 + k2] * v2p_col[j2p * 2 + k2p]
                            
            # W88[k8*8+i8, r8]
            for k2 in range(2):
                for i2 in range(2):
                    for k2p in range(2):
                        for i2p in range(2):
                            k8 = k2 * 2 + k2p
                            i8 = i2 * 2 + i2p
                            W88[k8 * 8 + i8, r8] = w2_col[k2 * 2 + i2] * w2p_col[k2p * 2 + i2p]
                            
    # Restrict to 6x6
    # Keep rows where index < 36 (0..35)
    # For U: row index i8*8+j8. i8<6, j8<6.
    # Since i8, j8 are 0..7, we keep i8 in 0..5, j8 in 0..5.
    # This corresponds to the first 36 rows if we iterate i8=0..5, j8=0..5?
    # Row index = i8*8+j8.
    # i8=0, j8=0..5 -> 0..5
    # i8=1, j8=0..5 -> 8..13
    # ...
    # i8=5, j8=0..5 -> 40..45
    # So we cannot just take the first 36 rows. We must filter.
    
    U66_rows = []
    V66_rows = []
    W66_rows = []
    
    for idx in range(64):
        i8 = idx // 8
        j8 = idx % 8
        if i8 < 6 and j8 < 6:
            U66_rows.append(idx)
            
    for idx in range(64):
        j8 = idx // 8
        k8 = idx % 8
        if j8 < 6 and k8 < 6:
            V66_rows.append(idx)
            
    for idx in range(64):
        k8 = idx // 8
        i8 = idx % 8
        if k8 < 6 and i8 < 6:
            W66_rows.append(idx)
            
    U66 = U88[U66_rows, :]
    V66 = V88[V66_rows, :]
    W66 = W88[W66_rows, :]
    
    # Remove zero columns
    # A column is zero if U66[:, c] is all 0, OR V66[:, c] is all 0, OR W66[:, c] is all 0?
    # Actually, if U66[:, c] is all 0, the term contributes 0.
    # We can remove any column where ANY of the three vectors is all zero.
    # Or if the product is zero.
    # Simplest: check if U66[:, c] has any non-zero, V66[:, c] has any non-zero, W66[:, c] has any non-zero.
    # If any is all zero, the term is 0.
    
    valid_cols = []
    for c in range(rank_88):
        if np.any(U66[:, c] != 0) and np.any(V66[:, c] != 0) and np.any(W66[:, c] != 0):
            valid_cols.append(c)
            
    U66 = U66[:, valid_cols]
    V66 = V66[:, valid_cols]
    W66 = W66[:, valid_cols]
    
    rank_final = len(valid_cols)
    
    # Verify
    if time.time() < deadline - 5:
        if not _verify_decomposition(U66, V66, W66, n, m, p, seed=42):
            # Fallback to schoolbook 6x6 (rank 216)
            U_fb = np.zeros((36, 216), dtype=float)
            V_fb = np.zeros((36, 216), dtype=float)
            W_fb = np.zeros((36, 216), dtype=float)
            for i in range(6):
                for j in range(6):
                    for k in range(6):
                        col = i * 36 + j * 6 + k
                        U_fb[i * 6 + j, col] = 1
                        V_fb[j * 6 + k, col] = 1
                        W_fb[k * 6 + i, col] = 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
            
    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}

# EVOLVE-BLOCK-END
