"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy: 
    1. Build rank-77 via Kronecker: Strassen(2x2x2) ⊗ hybrid(2x2x3)
    2. Attempt rank reduction using linear algebra on the coefficient tensor
    3. Return best valid decomposition found within time budget
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 110.0
    _test_A = None
    _test_B = None
    _test_C = None
    
    # ---- Strassen decomposition for 2x2x2 ----
    U1 = np.zeros((4, 7), dtype=int)
    V1 = np.zeros((4, 7), dtype=int)
    W1 = np.zeros((4, 7), dtype=int)
    
    # M1 = (a00+a11)(b00+b11)
    U1[0,0] = 1; U1[3,0] = 1
    V1[0,0] = 1; V1[3,0] = 1
    W1[0,0] = 1; W1[3,0] = 1
    
    # M2 = (a10+a11)b00
    U1[2,1] = 1; U1[3,1] = 1
    V1[0,1] = 1
    W1[1,1] = 1; W1[3,1] = -1
    
    # M3 = a00(b01-b11)
    U1[0,2] = 1
    V1[1,2] = 1; V1[3,2] = -1
    W1[2,2] = 1; W1[3,2] = 1
    
    # M4 = a11(b10-b00)
    U1[3,3] = 1
    V1[2,3] = 1; V1[0,3] = -1
    W1[0,3] = 1; W1[1,3] = 1
    
    # M5 = (a00+a01)b11
    U1[0,4] = 1; U1[1,4] = 1
    V1[3,4] = 1
    W1[0,4] = -1; W1[2,4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U1[2,5] = 1; U1[0,5] = -1
    V1[0,5] = 1; V1[1,5] = 1
    W1[3,5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U1[1,6] = 1; U1[3,6] = -1
    V1[2,6] = 1; V1[3,6] = 1
    W1[0,6] = 1
    
    # ---- 2x2x3 decomposition (rank 11) ----
    U2 = np.zeros((4, 11), dtype=int)
    V2 = np.zeros((6, 11), dtype=int)
    W2 = np.zeros((6, 11), dtype=int)
    
    # Strassen part for columns 0-1 (terms 0-6)
    for c in range(7):
        U2[0,c] = U1[0,c]
        U2[1,c] = U1[1,c]
        U2[2,c] = U1[2,c]
        U2[3,c] = U1[3,c]
        
        V2[0,c] = V1[0,c]
        V2[1,c] = V1[1,c]
        V2[3,c] = V1[2,c]
        V2[4,c] = V1[3,c]
        V2[2,c] = 0
        V2[5,c] = 0
        
        W2[0,c] = W1[0,c]
        W2[1,c] = W1[1,c]
        W2[2,c] = W1[2,c]
        W2[3,c] = W1[3,c]
        W2[4,c] = 0
        W2[5,c] = 0
    
    # Trivial part for column 2 (terms 7-10)
    U2[0,7] = 1; V2[2,7] = 1; W2[4,7] = 1
    U2[1,8] = 1; V2[5,8] = 1; W2[4,8] = 1
    U2[2,9] = 1; V2[2,9] = 1; W2[5,9] = 1
    U2[3,10] = 1; V2[5,10] = 1; W2[5,10] = 1
    
    # ---- Kronecker product to get full 4x4x6 decomposition ----
    rank_total = 7 * 11
    U = np.zeros((16, rank_total), dtype=int)
    V = np.zeros((24, rank_total), dtype=int)
    W = np.zeros((24, rank_total), dtype=int)
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1, c1]
                            u2 = U2[i2*2+j2, c2]
                            if u1 != 0 and u2 != 0:
                                U[i1*8 + i2*4 + j1*2 + j2, c] = u1 * u2
            
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1, c1]
                            v2 = V2[j2*3+k2, c2]
                            if v1 != 0 and v2 != 0:
                                V[j1*12 + j2*6 + k1*3 + k2, c] = v1 * v2
            
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1, c1]
                            w2 = W2[k2*2+i2, c2]
                            if w1 != 0 and w2 != 0:
                                W[k1*12 + k2*4 + i1*2 + i2, c] = w1 * w2
    
    def _init_test_matrices():
        nonlocal _test_A, _test_B, _test_C
        if _test_A is None:
            np.random.seed(42)
            _test_A = np.random.randn(n, m)
            _test_B = np.random.randn(m, p)
            _test_C = _test_A @ _test_B

    def _verify_fast(U_mat, V_mat, W_mat):
        _init_test_matrices()
        C_computed = np.zeros((n, p))
        for r in range(U_mat.shape[1]):
            alpha_r = np.dot(_test_A.ravel(), U_mat[:, r])
            beta_r = np.dot(_test_B.ravel(), V_mat[:, r])
            C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
        return np.allclose(C_computed, _test_C, atol=1e-10)

    def _verify_multi(U_mat, V_mat, W_mat, n_seeds=3):
        for seed in range(n_seeds):
            np.random.seed(seed * 1000 + 7)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            C_computed = np.zeros((n, p))
            for r in range(U_mat.shape[1]):
                alpha_r = np.dot(A_test.ravel(), U_mat[:, r])
                beta_r = np.dot(B_test.ravel(), V_mat[:, r])
                C_computed += alpha_r * beta_r * W_mat[:, r].reshape(p, n).T
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    if not _verify_multi(U, V, W, n_seeds=2):
        rank_school = n * m * p
        U_s = np.zeros((n * m, rank_school), dtype=int)
        V_s = np.zeros((m * p, rank_school), dtype=int)
        W_s = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # ---- Attempt rank reduction via linear algebra ----
    if time.time() < deadline:
        U_float = U.astype(float)
        V_float = V.astype(float)
        W_float = W.astype(float)
        
        _init_test_matrices()
        test_A_flat = _test_A.ravel()
        test_B_flat = _test_B.ravel()
        
        # Precompute per-column contributions for fast verification
        def _precompute_contribs(Uf, Vf, Wf):
            """Precompute alpha_r, beta_r, and W contributions."""
            n_cols = Uf.shape[1]
            alphas = np.zeros(n_cols)
            betas = np.zeros(n_cols)
            W_contribs = np.zeros((n_cols, p, n))
            for r in range(n_cols):
                alphas[r] = np.dot(test_A_flat, Uf[:, r])
                betas[r] = np.dot(test_B_flat, Vf[:, r])
                W_contribs[r] = Wf[:, r].reshape(p, n).T
            return alphas, betas, W_contribs
        
        def _verify_from_contribs(alphas, betas, W_contribs, active_mask):
            """Verify using precomputed contributions with active mask."""
            C_computed = np.zeros((n, p))
            for r in range(len(active_mask)):
                if active_mask[r]:
                    C_computed += alphas[r] * betas[r] * W_contribs[r]
            return np.allclose(C_computed, _test_C, atol=1e-10)
        
        # Greedy column removal with multiple passes
        num_cols = U_float.shape[1]
        keep_mask = np.ones(num_cols, dtype=bool)
        best_rank = num_cols
        
        for pass_num in range(8):
            if time.time() > deadline - 20:
                break
            
            removed_this_pass = 0
            alphas, betas, W_contribs = _precompute_contribs(U_float, V_float, W_float)
            
            for r in range(num_cols):
                if time.time() > deadline - 15:
                    break
                if not keep_mask[r]:
                    continue
                
                keep_mask[r] = False
                if _verify_from_contribs(alphas, betas, W_contribs, keep_mask):
                    # Commit removal
                    mask = keep_mask.copy()
                    U_float = U_float[:, mask]
                    V_float = V_float[:, mask]
                    W_float = W_float[:, mask]
                    num_cols = U_float.shape[1]
                    best_rank = num_cols
                    removed_this_pass += 1
                    keep_mask = np.ones(num_cols, dtype=bool)
                    alphas, betas, W_contribs = _precompute_contribs(U_float, V_float, W_float)
                else:
                    keep_mask[r] = True
            
            if removed_this_pass == 0:
                break
        
        # Second strategy: try removing pairs of columns
        if time.time() < deadline - 20 and best_rank > 1:
            num_cols = U_float.shape[1]
            keep_mask = np.ones(num_cols, dtype=bool)
            alphas, betas, W_contribs = _precompute_contribs(U_float, V_float, W_float)
            
            for r1 in range(num_cols):
                if time.time() > deadline - 15:
                    break
                if not keep_mask[r1]:
                    continue
                for r2 in range(r1 + 1, num_cols):
                    if time.time() > deadline - 15:
                        break
                    if not keep_mask[r2]:
                        continue
                    
                    keep_mask[r1] = False
                    keep_mask[r2] = False
                    
                    if _verify_from_contribs(alphas, betas, W_contribs, keep_mask):
                        mask = keep_mask.copy()
                        U_float = U_float[:, mask]
                        V_float = V_float[:, mask]
                        W_float = W_float[:, mask]
                        num_cols = U_float.shape[1]
                        best_rank = num_cols
                        keep_mask = np.ones(num_cols, dtype=bool)
                        alphas, betas, W_contribs = _precompute_contribs(U_float, V_float, W_float)
                        break
                    else:
                        keep_mask[r1] = True
                        keep_mask[r2] = True
        
        # Third strategy: try removing triples
        if time.time() < deadline - 25 and best_rank > 2:
            num_cols = U_float.shape[1]
            keep_mask = np.ones(num_cols, dtype=bool)
            alphas, betas, W_contribs = _precompute_contribs(U_float, V_float, W_float)
            
            for r1 in range(num_cols):
                if time.time() > deadline - 20:
                    break
                if not keep_mask[r1]:
                    continue
                for r2 in range(r1 + 1, num_cols):
                    if time.time() > deadline - 20:
                        break
                    if not keep_mask[r2]:
                        continue
                    for r3 in range(r2 + 1, num_cols):
                        if time.time() > deadline - 20:
                            break
                        if not keep_mask[r3]:
                            continue
                        
                        keep_mask[r1] = False
                        keep_mask[r2] = False
                        keep_mask[r3] = False
                        
                        if _verify_from_contribs(alphas, betas, W_contribs, keep_mask):
                            mask = keep_mask.copy()
                            U_float = U_float[:, mask]
                            V_float = V_float[:, mask]
                            W_float = W_float[:, mask]
                            num_cols = U_float.shape[1]
                            best_rank = num_cols
                            keep_mask = np.ones(num_cols, dtype=bool)
                            alphas, betas, W_contribs = _precompute_contribs(U_float, V_float, W_float)
                            break
                        else:
                            keep_mask[r1] = True
                            keep_mask[r2] = True
                            keep_mask[r3] = True
                            if time.time() > deadline - 20:
                                break
        
        # Final verification with multiple seeds
        if _verify_multi(U_float, V_float, W_float, n_seeds=3):
            U = U_float
            V = V_float
            W = W_float
        
        # Convert back to int if possible
        if np.allclose(U, np.round(U), atol=1e-10):
            U = np.round(U).astype(int)
        if np.allclose(V, np.round(V), atol=1e-10):
            V = np.round(V).astype(int)
        if np.allclose(W, np.round(W), atol=1e-10):
            W = np.round(W).astype(int)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
