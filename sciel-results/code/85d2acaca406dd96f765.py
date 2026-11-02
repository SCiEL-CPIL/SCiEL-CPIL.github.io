import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_strassen_rank28():
    """Build the known rank-28 Strassen decomposition for 2x4 * 4x4."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2*2x2 rank-1 components
    strassen_components = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # 4 block terms: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    terms = [
        (0, 0, 0, 0),  # A0 * B00 -> C0
        (2, 2, 0, 0),  # A1 * B10 -> C0
        (0, 0, 2, 2),  # A0 * B01 -> C1
        (2, 2, 2, 2),  # A1 * B11 -> C1
    ]
    
    col = 0
    for (a_co, b_ro, b_co, c_co) in terms:
        for (u_pat, v_pat, w_pat) in strassen_components:
            for (pi, pj), val in u_pat.items():
                U[pi * m + (a_co + pj)][col] += val
            for (qi, qj), val in v_pat.items():
                V[(b_ro + qi) * p + (b_co + qj)][col] += val
            for (ri, rj), val in w_pat.items():
                W[(c_co + rj) * n + ri][col] += val
            col += 1
    
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)


def _verify_decomposition(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    residual = np.max(np.abs(recon - T))
    return residual < tol, residual


def _try_rank27_from_strassen(U28, V28, W28, max_attempts=50, seed_base=0):
    """Try to find a rank-27 decomposition by modifying the Strassen rank-28 solution.
    
    Strategy: The rank-28 Strassen solution has 4 independent 2x2*2x2 blocks.
    We try to "merge" two of the rank-7 components across blocks by finding
    a linear dependency that allows us to eliminate one rank term.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    best_rank = 28
    best_U, best_V, best_W = U28.copy(), V28.copy(), W28.copy()
    
    for attempt in range(max_attempts):
        rng = np.random.RandomState(seed_base + attempt)
        
        # Try different strategies:
        # 1. Random column perturbation
        # 2. Structured merging of similar components
        
        U = U28.copy()
        V = V28.copy()
        W = W28.copy()
        
        # Strategy: Pick two columns that have similar structure and try to merge them
        # by making one a linear combination of the other plus a small perturbation
        col1 = rng.randint(0, 28)
        col2 = rng.randint(0, 28)
        if col1 == col2:
            continue
        
        # Try to express col2 as a linear combination involving col1
        # with a small random perturbation to break symmetry
        alpha = rng.uniform(-2, 2)
        
        # Set col2 = alpha * col1 + small random noise
        noise_scale = 0.01
        U[:, col2] = alpha * U[:, col1] + rng.randn(n * m) * noise_scale
        V[:, col2] = alpha * V[:, col1] + rng.randn(m * p) * noise_scale
        W[:, col2] = alpha * W[:, col1] + rng.randn(p * n) * noise_scale
        
        # Now try to re-optimize to see if we can drop one column
        # Use a simple gradient-free approach: try zeroing each column and checking
        for test_col in range(28):
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, test_col] = 0
            V_test[:, test_col] = 0
            W_test[:, test_col] = 0
            
            U3t = U_test.reshape(n, m, -1)
            V3t = V_test.reshape(m, p, -1)
            W3t = W_test.reshape(p, n, -1)
            recon_test = np.einsum('ijr,jkr,kir->ijk', U3t, V3t, W3t)
            residual_test = np.max(np.abs(recon_test - T))
            
            if residual_test < 1e-8:
                # Check if the remaining 27 columns form a valid decomposition
                mask = np.ones(28, dtype=bool)
                mask[test_col] = False
                U_new = U_test[:, mask]
                V_new = V_test[:, mask]
                W_new = W_test[:, mask]
                
                # Verify
                U3n = U_new.reshape(n, m, -1)
                V3n = V_new.reshape(m, p, -1)
                W3n = W_new.reshape(p, n, -1)
                recon_new = np.einsum('ijr,jkr,kir->ijk', U3n, V3n, W3n)
                residual_new = np.max(np.abs(recon_new - T))
                
                if residual_new < 1e-6:
                    best_rank = 27
                    best_U = U_new
                    best_V = V_new
                    best_W = W_new
                    return best_U, best_V, best_W, best_rank
    
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin before 150s limit
    
    # Define target tensor T[i,j,k] = delta_{j,k}
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    # Start with known rank-28 Strassen solution
    U28, V28, W28 = _build_strassen_rank28()
    
    def to_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    best_result = {"U": to_lists(U28), "V": to_lists(V28), "W": to_lists(W28)}
    best_rank = 28
    
    # Try to reduce to rank 27 with a time budget
    if time.time() < deadline:
        try:
            U_r, V_r, W_r, rank_r = _try_rank27_from_strassen(U28, V28, W28, max_attempts=30, seed_base=42)
            if rank_r < best_rank:
                valid, res = _verify_decomposition(U_r, V_r, W_r, tol=1e-6)
                if valid:
                    best_rank = rank_r
                    best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
        except Exception:
            pass
    
    # If still at rank 28 and we have time, try a few more structured approaches
    if time.time() < deadline and best_rank == 28:
        # Try exploiting the fact that the tensor has special structure
        # The matmul tensor T[i,j,k] = delta_{j,k} has a lot of zeros
        # This might allow for a more efficient decomposition
        
        # For 2x4*4x4, the tensor is 2*4*4 = 32 entries, with only 8 non-zero
        # (T[i,j,j] = 1 for i in {0,1}, j in {0,1,2,3})
        
        # Try a direct construction approach for rank 27
        # This is based on the idea that we might be able to share some
        # computation between the two rows of the output matrix
        
        for seed in range(20):
            if time.time() >= deadline:
                break
            
            rng = np.random.RandomState(seed)
            
            # Initialize with the Strassen solution but with a small perturbation
            # on one pair of columns that might be mergeable
            U = U28.copy()
            V = V28.copy()
            W = W28.copy()
            
            # Perturb columns 0 and 14 (first component of first and third blocks)
            # These both have similar structure in the U and V matrices
            U[:, 14] = U[:, 0] + rng.randn(n * m) * 0.001
            V[:, 14] = V[:, 0] + rng.randn(m * p) * 0.001
            W[:, 14] = W[:, 0] + rng.randn(p * n) * 0.001
            
            # Check if we can now drop column 14
            U_test = U.copy()
            V_test = V.copy()
            W_test = W.copy()
            U_test[:, 14] = 0
            V_test[:, 14] = 0
            W_test[:, 14] = 0
            
            U3t = U_test.reshape(n, m, -1)
            V3t = V_test.reshape(m, p, -1)
            W3t = W_test.reshape(p, n, -1)
            recon_test = np.einsum('ijr,jkr,kir->ijk', U3t, V3t, W3t)
            residual_test = np.max(np.abs(recon_test - T))
            
            if residual_test < 1e-6:
                mask = np.ones(28, dtype=bool)
                mask[14] = False
                U_new = U_test[:, mask]
                V_new = V_test[:, mask]
                W_new = W_test[:, mask]
                
                valid, res = _verify_decomposition(U_new, V_new, W_new, tol=1e-6)
                if valid:
                    best_rank = 27
                    best_result = {"U": to_lists(U_new), "V": to_lists(V_new), "W": to_lists(W_new)}
                    break
    
    return best_result
# EVOLVE-BLOCK-END
