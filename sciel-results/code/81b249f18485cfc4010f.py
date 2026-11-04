"""Model-and-optimize approach for 2x4x7 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.
    
    Strategy: Build rank-50 Strassen decomposition, then use vectorized
    verification and structured column elimination to reduce rank.
    """
    n, m, p = 2, 4, 7
    deadline = time.time() + 110  # safety margin before 150s limit
    
    # Build the rank-50 decomposition as starting point
    rank_init = 50
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks for the 2x4x6 part (k=0..5)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    assert col == rank_init, f"Expected rank {rank_init}, got {col}"
    
    # Precompute index arrays for fast vectorized verification
    # For each output (i,k), the contributing (j,r) pairs
    # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    def compute_C(U_m, V_m, W_m, A, B):
        """Vectorized computation of C from decomposition."""
        n_local, m_local, p_local = n, m, p
        rank_m = U_m.shape[1]
        C = np.zeros((n_local, p_local))
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U_m.reshape(n_local, m_local, rank_m)
        V_r = V_m.reshape(m_local, p_local, rank_m)
        W_r = W_m.reshape(p_local, n_local, rank_m)
        # C[i,k] = sum_j sum_r A[i,j] * B[j,k] * U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
        for i in range(n_local):
            for k in range(p_local):
                # sum over j and r
                # A[i,:] * U_r[i,:,:] has shape (m, rank)
                # B[:,k] * V_r[:,k,:] has shape (m, rank)
                X = A[i, :, None] * U_r[i, :, :]
                Y = B[:, k, None] * V_r[:, k, :] * W_r[k, i, None]
                term = np.sum(X * Y)
                C[i, k] = term
        return C
    
    def verify_decomposition(U_m, V_m, W_m, num_tests=10, seed=42):
        """Fast vectorized verification using random integer inputs."""
        rng = np.random.RandomState(seed)
        for _ in range(num_tests):
            A = rng.randint(-3, 4, size=(n, m))
            B = rng.randint(-3, 4, size=(m, p))
            C_expected = A @ B
            C_comp = compute_C(U_m, V_m, W_m, A, B)
            if not np.allclose(C_comp, C_expected, atol=1e-9):
                return False
        return True
    
    def verify_exact(U_m, V_m, W_m):
        """Exact verification with multiple seeds."""
        for seed in range(20):
            rng = np.random.RandomState(seed)
            A = rng.randint(-2, 3, size=(n, m))
            B = rng.randint(-2, 3, size=(m, p))
            C_expected = A @ B
            C_comp = compute_C(U_m, V_m, W_m, A, B)
            if not np.allclose(C_comp, C_expected, atol=1e-6):
                return False
        return True
    
    # Verify initial decomposition
    if not verify_decomposition(U, V, W, num_tests=10):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init
    
    def try_eliminate_column(U_arr, V_arr, W_arr, col_idx):
        """
        Try to eliminate column col_idx by checking if it's redundant.
        """
        r = U_arr.shape[1]
        if r <= 1:
            return None
        
        # First: try simple removal
        keep = [i for i in range(r) if i != col_idx]
        U_new = U_arr[:, keep]
        V_new = V_arr[:, keep]
        W_new = W_arr[:, keep]
        
        if verify_decomposition(U_new, V_new, W_new, num_tests=3, seed=col_idx):
            return U_new, V_new, W_new
        
        # Try pairwise: can column c be absorbed into column d?
        # Check if U_c, V_c, W_c are proportional to U_d, V_d, W_d
        Uc, Vc, Wc = U_arr[:, col_idx], V_arr[:, col_idx], W_arr[:, col_idx]
        
        for d in keep:
            Ud, Vd, Wd = U_arr[:, d], V_arr[:, d], W_arr[:, d]
            
            # Check proportionality
            if np.allclose(Uc, 0) or np.allclose(Ud, 0):
                continue
            mask = Ud != 0
            if not np.any(mask):
                continue
            ratios = Uc[mask] / Ud[mask]
            if len(ratios) < 2:
                continue
            if np.allclose(ratios, ratios[0], atol=1e-10):
                alpha = ratios[0]
                if np.allclose(Vc, alpha * Vd, atol=1e-10) and np.allclose(Wc, alpha * Wd, atol=1e-10):
                    keep2 = [i for i in range(r) if i != col_idx]
                    U_new2 = U_arr[:, keep2]
                    V_new2 = V_arr[:, keep2]
                    W_new2 = W_arr[:, keep2]
                    if verify_decomposition(U_new2, V_new2, W_new2, num_tests=3, seed=col_idx*100+d):
                        return U_new2, V_new2, W_new2
        
        return None
    
    def try_merge_columns(U_arr, V_arr, W_arr, c1, c2):
        """
        Try to merge two columns by finding coefficients alpha, beta such that
        the combined column replaces both. This is a more powerful operation.
        We look for: U_new = alpha*U_c1 + beta*U_c2, etc.
        such that the total tensor contribution is preserved.
        """
        r = U_arr.shape[1]
        if r <= 2:
            return None
        
        Uc1, Vc1, Wc1 = U_arr[:, c1], V_arr[:, c1], W_arr[:, c1]
        Uc2, Vc2, Wc2 = U_arr[:, c2], V_arr[:, c2], W_arr[:, c2]
        
        # Try: replace both with a single column that is a linear combination
        # The tensor T = T_c1 + T_c2 should equal T_new
        # T_new = U_new ⊗ V_new ⊗ W_new
        # This is generally not possible with a single rank-1 term unless
        # T_c1 and T_c2 are proportional.
        
        # Alternative: try to find if there's a column d (other than c1,c2)
        # such that we can adjust d to absorb both c1 and c2.
        keep = [i for i in range(r) if i not in (c1, c2)]
        if not keep:
            return None
        
        # Try removing both and checking
        U_new = U_arr[:, keep]
        V_new = V_arr[:, keep]
        W_new = W_arr[:, keep]
        if verify_decomposition(U_new, V_new, W_new, num_tests=2, seed=c1*100+c2):
            return U_new, V_new, W_new
        
        return None
    
    def greedy_reduce(U_arr, V_arr, W_arr, deadline):
        """
        Greedy column elimination with multiple strategies.
        """
        U_arr = U_arr.copy()
        V_arr = V_arr.copy()
        W_arr = W_arr.copy()
        r = U_arr.shape[1]
        
        max_iter = 200
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            if r <= 1:
                break
            
            found = False
            
            # Try single column elimination
            for c in range(r):
                if time.time() > deadline - 10:
                    break
                result = try_eliminate_column(U_arr, V_arr, W_arr, c)
                if result is not None:
                    U_arr, V_arr, W_arr = result
                    r = U_arr.shape[1]
                    found = True
                    break
            
            if found:
                continue
            
            # Try pairwise column elimination
            for c1 in range(r):
                if time.time() > deadline - 10:
                    break
                for c2 in range(c1 + 1, r):
                    if time.time() > deadline - 10:
                        break
                    result = try_merge_columns(U_arr, V_arr, W_arr, c1, c2)
                    if result is not None:
                        U_arr, V_arr, W_arr = result
                        r = U_arr.shape[1]
                        found = True
                        break
                if found:
                    break
            
            if not found:
                break
        
        return U_arr, V_arr, W_arr, r
    
    # Attempt structured reduction
    if time.time() < deadline - 30:
        U_red, V_red, W_red, new_rank = greedy_reduce(U, V, W, deadline)
        if new_rank < best_rank and verify_decomposition(U_red, V_red, W_red, num_tests=10):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank
    
    # Try multiple passes with different column orderings
    np.random.seed(123)
    for attempt in range(10):
        if time.time() > deadline - 15:
            break
        r = best_U.shape[1]
        perm = np.random.permutation(r)
        U_shuf = best_U[:, perm]
        V_shuf = best_V[:, perm]
        W_shuf = best_W[:, perm]
        
        U_red, V_red, W_red, new_rank = greedy_reduce(U_shuf, V_shuf, W_shuf, deadline)
        if new_rank < best_rank and verify_decomposition(U_red, V_red, W_red, num_tests=10):
            best_U, best_V, best_W = U_red, V_red, W_red
            best_rank = new_rank
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W, num_tests=20):
        best_U, best_V, best_W = U, V, W
        best_rank = rank_init
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
