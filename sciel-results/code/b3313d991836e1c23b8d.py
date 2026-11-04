import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen (rank 7)."""
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[0,2]=-1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[1,3]=-1; W[3,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=1; W[1,4]=-1; W[3,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=-1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[2,6]=-1
    return U, V, W


def _build_4x4x4_49():
    """4x4x4 via 49-term Strassen recursion."""
    U2, V2, W2 = _strassen_2x2x2()
    U4 = np.zeros((16, 49), dtype=np.int64)
    V4 = np.zeros((16, 49), dtype=np.int64)
    W4 = np.zeros((16, 49), dtype=np.int64)
    
    A_TL = [0, 1, 4, 5]
    A_TR = [2, 3, 6, 7]
    A_BL = [8, 9, 12, 13]
    A_BR = [10, 11, 14, 15]
    
    B_TL = [0, 1, 4, 5]
    B_TR = [2, 3, 6, 7]
    B_BL = [8, 9, 12, 13]
    B_BR = [10, 11, 14, 15]
    
    C_TL = [0, 1, 4, 5]
    C_TR = [2, 3, 6, 7]
    C_BL = [8, 9, 12, 13]
    C_BR = [10, 11, 14, 15]
    
    U_blocks = {
        0: (A_TL + A_BR, [1]*8),
        1: (A_BL + A_BR, [1]*8),
        2: (A_TL, [1]*4),
        3: (A_BR, [1]*4),
        4: (A_TL + A_TR, [1]*8),
        5: (A_BR + A_BL, [1]*8),
        6: (A_TR + A_BR, [1]*4 + [-1]*4),
    }
    
    V_blocks = {
        0: (B_TL + B_BR, [1]*8),
        1: (B_TL, [1]*4),
        2: (B_TR + B_BR, [1]*4 + [-1]*4),
        3: (B_BL + B_TL, [1]*4 + [-1]*4),
        4: (B_BR, [1]*4),
        5: (B_TR + B_BR, [1]*8),
        6: (B_BL + B_TL, [1]*8),
    }
    
    C_blocks_for_M = {
        0: [(C_TL, 1), (C_BR, 1)],
        1: [(C_TR, 1), (C_BR, -1)],
        2: [(C_BL, 1), (C_BR, 1)],
        3: [(C_TL, 1)],
        4: [(C_TR, 1), (C_BR, 1)],
        5: [(C_TL, -1), (C_BL, 1)],
        6: [(C_TL, 1)],
    }
    
    for outer_t in range(7):
        u_idx_list, u_coefs = U_blocks[outer_t]
        v_idx_list, v_coefs = V_blocks[outer_t]
        
        for inner_s in range(7):
            col = outer_t * 7 + inner_s
            
            for local_i in range(4):
                val = U2[local_i, inner_s]
                if val != 0:
                    for bi, coef in zip(u_idx_list, u_coefs):
                        U4[bi, col] += val * coef
            
            for local_j in range(4):
                val = V2[local_j, inner_s]
                if val != 0:
                    for bj, coef in zip(v_idx_list, v_coefs):
                        V4[bj, col] += val * coef
            
            for local_k in range(4):
                val = W2[local_k, inner_s]
                if val != 0:
                    for c_list, c_coef in C_blocks_for_M[outer_t]:
                        for bk in c_list:
                            W4[bk, col] += val * c_coef
    
    return U4, V4, W4


def _build_4x4x8_98():
    """4x4x8 via two 49-term 4x4x4 blocks (rank 98)."""
    U4, V4, W4 = _build_4x4x4_49()
    U = np.zeros((16, 98), dtype=np.int64)
    V = np.zeros((32, 98), dtype=np.int64)
    W = np.zeros((32, 98), dtype=np.int64)
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    U[:, 49:98] = U4
    for j in range(4):
        for kl in range(4):
            V[j*8+kl+4, 49:98] = V4[j*4+kl, 0:49]
    for kl in range(4):
        for i in range(4):
            W[(kl+4)*4+i, 49:98] = W4[kl*4+i, 0:49]
    return U, V, W


def _verify_decomposition(U, V, W, T, atol=1e-9):
    """Verify that the decomposition exactly reconstructs T."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=atol)


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T


def _kronecker_block_decompose(T, n, m, p, target_rank, deadline):
    """
    Structured Kronecker-product search for the 4x4x8 tensor.
    
    Model: Treat the 4x4x8 problem as a 2x2 block structure.
    A (4x4) = [[A00, A01], [A10, A11]] where each block is 2x2.
    B (4x8) = [[B00, B01, B02, B03], [B10, B11, B12, B13]] where each block is 2x4.
    C (4x8) = [[C00, C01, C02, C03], [C10, C11, C12, C13]] where each block is 2x4.
    
    We model each 2x2 factor matrix in the Kronecker decomposition as a product of 2x2 matrices.
    Specifically, for each rank term r, we have:
    U_r = kron(Ua_r, Ub_r) where Ua_r, Ub_r are 2x2.
    V_r = kron(Va_r, Vb_r) where Va_r, Vb_r are 2x2.
    W_r = kron(Wa_r, Wb_r) where Wa_r, Wb_r are 2x2.
    
    This gives 12 parameters per rank term (6 for U, 6 for V, 6 for W... wait, 4+4+4=12).
    Total parameters = 12 * target_rank.
    
    We minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||_F^2 using Gauss-Newton.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    # Initialize with random 2x2 factors
    np.random.seed(42)
    
    # Parameters: for each rank term r, we have:
    # Ua_r (2x2), Ub_r (2x2), Va_r (2x2), Vb_r (2x2), Wa_r (2x2), Wb_r (2x2)
    # Total 12 params per term.
    
    n_params = 12 * target_rank
    
    # Initial guess: try to match the Strassen structure
    # For simplicity, start with random initialization
    x = np.random.randn(n_params) * 0.1
    
    def _compute_residual(x):
        """Compute the residual T - sum_r U_r ⊗ V_r ⊗ W_r."""
        recon = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
        for r in range(target_rank):
            base = r * 12
            Ua = x[base:base+4].reshape(2, 2)
            Ub = x[base+4:base+8].reshape(2, 2)
            Va = x[base+8:base+12].reshape(2, 2)
            # Wait, I need 6 matrices: Ua, Ub, Va, Vb, Wa, Wb
            # That's 6*4 = 24 params per term, not 12.
            pass
        return recon
    
    # Actually, let's use a simpler approach: 6 matrices of 2x2 = 24 params per term.
    # But that's a lot. Let's use a different structure.
    
    # Alternative: Use the fact that for 4x4x8, we can use a 2x2 block structure
    # where each block multiplication is a 2x2x2 (or 2x2x4) problem.
    
    # Let's try a different approach: direct optimization with a custom Gauss-Newton.
    # We'll use the schoolbook as initial guess and try to reduce rank.
    
    # For now, return None to indicate this method is not ready.
    return None


def _custom_gauss_newton(T, n, m, p, target_rank, deadline):
    """
    Custom Gauss-Newton optimization for tensor decomposition.
    T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    
    # Initialize with schoolbook-like structure
    np.random.seed(42)
    
    # Start with random initialization
    U = np.random.randn(dim_a, target_rank) * 0.1
    V = np.random.randn(dim_b, target_rank) * 0.1
    W = np.random.randn(dim_c, target_rank) * 0.1
    
    # Normalize
    for r in range(target_rank):
        U[:, r] /= (np.linalg.norm(U[:, r]) + 1e-10)
        V[:, r] /= (np.linalg.norm(V[:, r]) + 1e-10)
        W[:, r] /= (np.linalg.norm(W[:, r]) + 1e-10)
    
    best_residual = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(50):
        if deadline and time.time() > deadline - 10:
            break
        
        # Compute residual
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        residual = T - T_recon
        res_norm = np.linalg.norm(residual)
        
        if res_norm < best_residual:
            best_residual = res_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if res_norm < 1e-8:
            break
        
        # Gauss-Newton step: linearize around current (U, V, W)
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # d(T_recon)/dU[a,r] = V[b,r] W[c,r]
        # d(T_recon)/dV[b,r] = U[a,r] W[c,r]
        # d(T_recon)/dW[c,r] = U[a,r] V[b,r]
        
        # Build the Jacobian (this is large, so we'll use a simplified approach)
        # For efficiency, we'll do coordinate descent instead of full Gauss-Newton.
        
        # Update U: for each a, U[a,:] should minimize
        # sum_{b,c} (T[a,b,c] - sum_r U[a,r] V[b,r] W[c,r])^2
        # This is a linear least squares problem.
        
        # Build the system: for each a,
        # T[a,:,:] = sum_r U[a,r] * outer(V[:,r], W[:,r])
        
        M = np.zeros((dim_b * dim_c, target_rank), dtype=np.float64)
        for r in range(target_rank):
            M[:, r] = np.outer(V[:, r], W[:, r]).flatten()
        
        T_flat = T.reshape(dim_a, dim_b * dim_c)
        
        try:
            U_new, residuals, _, _ = np.linalg.lstsq(M, T_flat, rcond=1e-12)
        except:
            break
        
        # Check if the update improves the residual
        T_recon_new = np.einsum('ar,br,cr->abc', U_new, V, W)
        res_new = np.linalg.norm(T - T_recon_new)
        
        if res_new < res_norm:
            U = U_new
        
        # Similarly for V and W
        M2 = np.zeros((dim_a * dim_c, target_rank), dtype=np.float64)
        for r in range(target_rank):
            M2[:, r] = np.outer(U[:, r], W[:, r]).flatten()
        
        T_flat2 = T.reshape(dim_b, dim_a * dim_c)
        
        try:
            V_new, _, _, _ = np.linalg.lstsq(M2, T_flat2, rcond=1e-12)
        except:
            break
        
        T_recon_new = np.einsum('ar,br,cr->abc', U, V_new, W)
        res_new = np.linalg.norm(T - T_recon_new)
        
        if res_new < res_norm:
            V = V_new
        
        M3 = np.zeros((dim_a * dim_b, target_rank), dtype=np.float64)
        for r in range(target_rank):
            M3[:, r] = np.outer(U[:, r], V[:, r]).flatten()
        
        T_flat3 = T.reshape(dim_c, dim_a * dim_b)
        
        try:
            W_new, _, _, _ = np.linalg.lstsq(M3, T_flat3, rcond=1e-12)
        except:
            break
        
        T_recon_new = np.einsum('ar,br,cr->abc', U, V, W_new)
        res_new = np.linalg.norm(T - T_recon_new)
        
        if res_new < res_norm:
            W = W_new
    
    return best_U, best_V, best_W, best_residual


def _try_column_pruning(U, V, W, T, deadline):
    """
    Try to prune columns that are linearly dependent.
    Use a greedy approach: for each column, check if removing it still leaves
    a valid decomposition (approximately).
    """
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    rank = U.shape[1]
    
    keep = [True] * rank
    changed = True
    iter_count = 0
    max_iter = 20
    
    while changed and iter_count < max_iter:
        changed = False
        iter_count += 1
        if deadline and time.time() > deadline - 15:
            break
        
        for r in range(rank):
            if not keep[r]:
                continue
            
            # Try removing column r
            others = [s for s in range(rank) if s != r and keep[s]]
            if len(others) == 0:
                continue
            
            U_test = U[:, others]
            V_test = V[:, others]
            W_test = W[:, others]
            
            if _verify_decomposition(U_test, V_test, W_test, T, atol=1e-6):
                keep[r] = False
                changed = True
    
    idx = [i for i in range(rank) if keep[i]]
    return U[:, idx], V[:, idx], W[:, idx]


def _to_half_integer_list(mat):
    """Convert a matrix to a list of lists with half-integer or integer entries."""
    mat = np.asarray(mat, dtype=np.float64)
    # Try half-integers first
    mat2 = mat * 2
    rounded2 = np.round(mat2)
    if np.allclose(mat2, rounded2, atol=1e-6):
        return (rounded2 / 2).tolist()
    # Try integers
    rounded_int = np.round(mat)
    if np.allclose(mat, rounded_int, atol=1e-6):
        return rounded_int.astype(int).tolist()
    # Fall back to float list
    return mat.tolist()


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 8
    dim_a, dim_b, dim_c = n*m, m*p, p*n

    # Build the full 4x4x8 multiplication tensor
    T = _build_tensor(n, m, p)

    # Fallback: Schoolbook (rank 128)
    rank_schoolbook = n * m * p
    U_school = np.zeros((dim_a, rank_schoolbook), dtype=np.int64)
    V_school = np.zeros((dim_b, rank_schoolbook), dtype=np.int64)
    W_school = np.zeros((dim_c, rank_schoolbook), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook

    # Try 98-rank constructive solution
    if time.time() < deadline - 30:
        try:
            U98, V98, W98 = _build_4x4x8_98()
            if _verify_decomposition(U98, V98, W98, T, atol=1e-9):
                best_U, best_V, best_W = U98, V98, W98
                best_rank = 98
        except Exception:
            pass

    # Try column pruning on the 98-term solution
    if best_rank == 98 and time.time() < deadline - 20:
        try:
            U_pr, V_pr, W_pr = _try_column_pruning(
                best_U.astype(np.float64),
                best_V.astype(np.float64),
                best_W.astype(np.float64),
                T,
                deadline
            )
            if U_pr.shape[1] < best_rank:
                if _verify_decomposition(U_pr, V_pr, W_pr, T, atol=1e-6):
                    best_U, best_V, best_W = U_pr, V_pr, W_pr
                    best_rank = U_pr.shape[1]
        except Exception:
            pass

    # Try custom Gauss-Newton optimization to reduce rank
    if best_rank > 80 and time.time() < deadline - 15:
        try:
            for target in [90, 85, 80]:
                if time.time() > deadline - 10:
                    break
                U_gn, V_gn, W_gn, res = _custom_gauss_newton(T, n, m, p, target, deadline)
                if res < 1e-6:
                    # Verify
                    T_recon = np.einsum('ar,br,cr->abc', U_gn, V_gn, W_gn)
                    if np.allclose(T_recon, T, atol=1e-6):
                        best_U, best_V, best_W = U_gn, V_gn, W_gn
                        best_rank = target
                        break
        except Exception:
            pass

    # Final verification
    T_recon = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
    if not np.allclose(T_recon, T, atol=1e-6):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_schoolbook

    U_out = _to_half_integer_list(best_U)
    V_out = _to_half_integer_list(best_V)
    W_out = _to_half_integer_list(best_W)

    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END
