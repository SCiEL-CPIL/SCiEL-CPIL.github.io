"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_decomposition(n, m, p):
    """
    Builds a rank-77 decomposition for 4x4 x 4x6 matrix multiplication.
    Uses Kronecker product of Strassen 2x2x2 (rank 7) and a 2x2x3 decomposition (rank 11).
    The 2x2x3 decomposition uses Strassen for the first 2 columns (rank 7) and
    trivial terms for the 3rd column (rank 4), total rank 11.
    Total rank: 7 * 11 = 77.
    """
    # 1. Define Strassen 2x2x2 terms as sparse dicts
    # Each term: (U_dict, V_dict, W_dict) where keys are (row, col) pairs
    strassen_terms_2x2 = [
        # M1 = (A00+A11)(B00+B11)
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        # M2 = (A10+A11)B00
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        # M3 = A00(B01-B11)
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        # M4 = A11(B10-A00)
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        # M5 = (A00+A01)B11
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    
    # 2. Define 2x2x3 decomposition (rank 11)
    # For 2x2 * 2x3:
    # - First 2 columns: use Strassen (7 terms)
    # - 3rd column: use trivial terms (4 terms)
    
    # The 2x2x3 terms are (U_dict, V_dict, W_dict)
    # U: (2x2), V: (2x3), W: (3x2)
    
    # Strassen terms for the 2x2 part (columns 0,1), with W having only entries in positions (0,0),(0,1),(1,0),(1,1)
    # but we need W to be 3x2, so we pad with zeros for column 2
    strassen_2x2_3 = []
    for u_dict, v_dict, w_dict in strassen_terms_2x2:
        # Pad V: add zeros for column 2
        v_padded = dict(v_dict)  # keys are (row, col) where col is 0 or 1
        # No need to add explicit zeros; missing keys mean 0
        
        # Pad W: add zeros for row 2
        w_padded = dict(w_dict)  # keys are (row, col) where row is 0 or 1
        # No need to add explicit zeros
        
        strassen_2x2_3.append((u_dict, v_padded, w_padded))
    
    # Trivial terms for the 3rd column (column index 2)
    trivial_2x2_3 = []
    for i in range(2):
        for j in range(2):
            u_dict = {(i, j): 1}
            v_dict = {(j, 2): 1}
            w_dict = {(2, i): 1}
            trivial_2x2_3.append((u_dict, v_dict, w_dict))
    
    # Combine: 7 + 4 = 11 terms for 2x2x3
    terms_2x2_3 = strassen_2x2_3 + trivial_2x2_3
    
    rank_terms = []
    
    # 3. Generate terms via tensor product of 2x2x2 (7 terms) and 2x2x3 (11 terms)
    # 2x2x2 handles the "outer" 2x2 blocks
    # 2x2x3 handles the "inner" 2x2x3 blocks
    
    # The 4x4 matrix A is partitioned as:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # The 4x6 matrix B is partitioned as:
    # B = [[B00, B01, B02], [B10, B11, B12]] where each block is 2x2
    
    # For the tensor product:
    # U_term = U_outer (2x2) ⊗ U_inner (2x2) -> 4x4
    # V_term = V_outer (2x2) ⊗ V_inner (2x3) -> 4x6
    # W_term = W_outer (2x2) ⊗ W_inner (3x2) -> 6x4
    
    for s in range(7):
        u_s, v_s, w_s = strassen_terms_2x2[s]
        for t in range(11):
            u_t, v_t, w_t = terms_2x2_3[t]
            
            U_global = {}
            V_global = {}
            W_global = {}
            
            # U: (i1, j1) from outer, (i2, j2) from inner -> (2*i1+i2, 2*j1+j2)
            for (i1, j1), cu in u_s.items():
                for (i2, j2), cu2 in u_t.items():
                    gi = 2 * i1 + i2
                    gj = 2 * j1 + j2
                    coeff = cu * cu2
                    key = (gi, gj)
                    U_global[key] = U_global.get(key, 0) + coeff
                    
            # V: (j1, k1) from outer, (j2, k2) from inner -> (2*j1+j2, 2*k1+k2)
            # Note: V_inner has k2 in {0,1,2}, so gk = 2*k1 + k2 gives 0..5
            for (j1, k1), cv in v_s.items():
                for (j2, k2), cv2 in v_t.items():
                    gj = 2 * j1 + j2
                    gk = 2 * k1 + k2
                    coeff = cv * cv2
                    key = (gj, gk)
                    V_global[key] = V_global.get(key, 0) + coeff
                    
            # W: (k1, i1) from outer, (k2, i2) from inner -> (2*k1+k2, 2*i1+i2)
            # Note: W_outer has k1 in {0,1}, W_inner has k2 in {0,1,2}
            # So gk = 2*k1 + k2 gives 0..5
            for (k1, i1), cw in w_s.items():
                for (k2, i2), cw2 in w_t.items():
                    gk = 2 * k1 + k2
                    gi = 2 * i1 + i2
                    coeff = cw * cw2
                    key = (gk, gi)
                    W_global[key] = W_global.get(key, 0) + coeff
                    
            rank_terms.append((U_global, V_global, W_global))
                
    rank = len(rank_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k, r] = coeff
        for (k, i), coeff in w_dict.items():
            W[k * n + i, r] = coeff
            
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W[:, r].reshape(p, n).T
        
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _remove_redundant(U, V, W, n, m, p, tol=1e-10):
    """
    Try to remove redundant rank terms by checking linear dependencies.
    Returns (U_reduced, V_reduced, W_reduced) with potentially lower rank.
    """
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Check for linearly dependent columns
    # Use SVD to find rank
    U_svd, _, _ = np.linalg.svd(U, full_matrices=False)
    V_svd, _, _ = np.linalg.svd(V, full_matrices=False)
    W_svd, _, _ = np.linalg.svd(W, full_matrices=False)
    
    # Find tolerance for numerical rank
    U_rank = np.sum(U_svd > tol * U_svd[0])
    V_rank = np.sum(V_svd > tol * V_svd[0])
    W_rank = np.sum(W_svd > tol * W_svd[0])
    
    # If any matrix has lower numerical rank, we might be able to reduce
    # But this is complex; for now, just return as is if no clear redundancy
    # A more sophisticated approach would identify specific redundant terms
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 110.0 # Safety margin
    
    # Build decomposition
    U, V, W = _build_decomposition(n, m, p)
    
    # Try to remove redundant terms
    U, V, W = _remove_redundant(U, V, W, n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
