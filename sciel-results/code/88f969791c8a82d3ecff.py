"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time as _time
import numpy as _np
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _build_strassen_36():
    """Build the known rank-36 block-Strassen decomposition."""
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0

    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices

        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1

        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1

        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1

        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1

        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1

        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1

        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)

    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)

    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)

    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)

    U[u_idx(0,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
    U[u_idx(0,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
    U[u_idx(1,0)][col] = 1; V[v_idx(0,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
    U[u_idx(1,1)][col] = 1; V[v_idx(1,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1

    U[u_idx(0,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
    U[u_idx(0,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,0)][col] = 1; col += 1
    U[u_idx(1,2)][col] = 1; V[v_idx(2,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1
    U[u_idx(1,3)][col] = 1; V[v_idx(3,4)][col] = 1; W[w_idx(4,1)][col] = 1; col += 1

    return {"U": U, "V": V, "W": W}


def _tensor_error(U, V, W):
    """Compute the Frobenius norm of the difference between the tensor
    represented by (U,V,W) and the true matrix-multiplication tensor."""
    n, m, p = 2, 4, 5
    # True tensor T[i,j,k] = 1 if k == j (in the sense: C[i,k] = sum_j A[i,j]*B[j,k])
    # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Build the tensor from U,V,W
    Ua = _np.array(U, dtype=float)  # (n*m, rank)
    Va = _np.array(V, dtype=float)  # (m*p, rank)
    Wa = _np.array(W, dtype=float)  # (p*n, rank)
    rank = Ua.shape[1]
    # Reshape to 3D
    U3 = Ua.reshape(n, m, rank)   # (n, m, r)
    V3 = Va.reshape(m, p, rank)   # (m, p, r)
    W3 = Wa.reshape(p, n, rank)   # (p, n, r)
    # Compute tensor: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    T = _np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # True tensor: T_true[i,j,k] = 1 if j == k else 0
    T_true = _np.zeros((n, m, p))
    for j in range(m):
        T_true[:, j, j] = 1.0
    return _np.linalg.norm(T - T_true)


def _verify_decomposition(U, V, W, tol=1e-9):
    """Check if (U,V,W) exactly decomposes the matrix multiplication tensor."""
    return _tensor_error(U, V, W) < tol


def _try_rank_reduction(U, V, W, target_rank, max_iters=200, deadline=None, seed=42):
    """Try to reduce rank from current to target_rank using ALS with border-rank perturbation.
    
    Returns (U_new, V_new, W_new) if successful, else None.
    """
    n, m, p = 2, 4, 5
    nm, mp, pn = n*m, m*p, p*n
    cur_rank = len(U[0])
    
    rng = _np.random.RandomState(seed)
    
    # Start with the current decomposition, add small perturbation
    Ua = _np.array(U, dtype=float).copy()
    Va = _np.array(V, dtype=float).copy()
    Wa = _np.array(W, dtype=float).copy()
    
    # Add small random perturbation to encourage exploration
    eps = 1e-4
    Ua += eps * rng.randn(nm, cur_rank)
    Va += eps * rng.randn(mp, cur_rank)
    Wa += eps * rng.randn(pn, cur_rank)
    
    # Re-normalize: scale each column so that the decomposition is preserved
    # For each column r, we need U[:,r], V[:,r], W[:,r] to multiply correctly
    # Actually, let's just run ALS from the perturbed starting point
    
    # ALS: alternate fixing two factors and solving for the third
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Fix V,W -> solve for U: for each (i,j), U[i,j,:] = argmin sum_k (T[i,j,k] - sum_r U[i,j,r] V[j,k,r] W[k,i,r])^2
    # This is a linear system.
    
    # Precompute the true tensor
    T_true = _np.zeros((n, m, p))
    for j in range(m):
        T_true[:, j, j] = 1.0
    
    best_err = _tensor_error(Ua.tolist(), Va.tolist(), Wa.tolist())
    
    for it in range(max_iters):
        if deadline is not None and _time.time() > deadline:
            break
        
        # Update U: fix V, W
        # For each (i,j): U[i,j,r] should minimize sum_k (T[i,j,k] - sum_r' U[i,j,r'] * (V[j,:,r'] * W[:,i,r']))^2
        # Let B[i,j,k] = sum_r' V[j,k,r'] * W[k,i,r'] (this is the "contraction" of V and W)
        # Then we need U[i,j,:] such that sum_r U[i,j,r] * B[i,j,k] ≈ T[i,j,k]
        # Wait, that's not right. Let me think again.
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Fix V and W. For a fixed (i,j), define c_r = V[j,:,r] * W[:,i,r] element-wise? No.
        # For fixed (i,j): T[i,j,k] = sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r])
        # So define A[i,j,k,r] = V[j,k,r] * W[k,i,r]
        # Then T[i,j,:] = U[i,j,:] @ A[i,j,:,:].T ... no
        # T[i,j,k] = sum_r U[i,j,r] * A[i,j,k,r]
        # So for each (i,j), we have a linear system: A[i,j,:,:] @ U[i,j,:] = T[i,j,:]
        # where A[i,j,k,r] = V[j,k,r] * W[k,i,r]
        
        for i in range(n):
            for j in range(m):
                # Build A matrix: (p, rank) where A[k,r] = V[j,k,r] * W[k,i,r]
                # V[j,k,r] is Va[j*p+k, r]
                # W[k,i,r] is Wa[k*n+i, r]
                A_mat = Va[j*p:(j+1)*p, :] * Wa[_np.arange(p)*n + i][:, None, :]  # (p, rank)
                # Solve A_mat.T @ x = T_true[i,j,:]
                # i.e., A_mat.T @ x = T_true[i,j,:]
                # This is an underdetermined system (rank x >= p typically)
                # Use least squares
                x, *_ = _np.linalg.lstsq(A_mat.T, T_true[i, j, :], rcond=None)
                Ua[i*m+j, :] = x
        
        # Update V: fix U, W
        # T[i,j,k] = sum_r V[j,k,r] * (U[i,j,r] * W[k,i,r])
        # For fixed (j,k): T[:,j,k] = V[j,k,:] @ B[j,k,:,:] where B[j,k,i,r] = U[i,j,r]*W[k,i,r]
        for j in range(m):
            for k in range(p):
                # B[i,r] = U[i,j,r] * W[k,i,r] = Ua[i*m+j, r] * Wa[k*n+i, r]
                B_mat = Ua[:n*m, :] * Wa[_np.arange(n)*n + k][:, None, :]  # (n, rank)
                # B_mat[i,r] = Ua[i*m+j, r] * Wa[k*n+i, r]
                B_mat = (Ua[:n*m, :].T * Wa[_np.arange(n)*n + k][:, None, :]).T  # wrong
                # Let me be more careful
                # For row i: B[i,r] = Ua[i*m + j, r] * Wa[k*n + i, r]
                B_mat = _np.zeros((n, cur_rank))
                for ii in range(n):
                    B_mat[ii, :] = Ua[ii*m + j, :] * Wa[k*n + ii, :]
                # Solve B_mat.T @ x = T_true[:,j,k]
                x, *_ = _np.linalg.lstsq(B_mat.T, T_true[:, j, k], rcond=None)
                Va[j*p+k, :] = x
        
        # Update W: fix U, V
        # T[i,j,k] = sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r])
        # For fixed (k,i): T[:,j,k]... wait
        # T[i,j,k] = sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r])
        # For fixed (k,i): for all j, T[i,j,k] = sum_r W[k,i,r] * (U[i,j,r] * V[j,k,r])
        # Define C[j,r] = U[i,j,r] * V[j,k,r]
        # Then T[i,:,k] = C @ W[k,i,:] ... i.e., sum_r C[j,r] * W[k,i,r] = T[i,j,k]
        for k in range(p):
            for i in range(n):
                C_mat = _np.zeros((m, cur_rank))
                for jj in range(m):
                    C_mat[jj, :] = Ua[i*m + jj, :] * Va[jj*p + k, :]
                # Solve C_mat.T @ x = T_true[i,:,k]
                x, *_ = _np.linalg.lstsq(C_mat.T, T_true[i, :, k], rcond=None)
                Wa[k*n+i, :] = x
        
        # Check error
        err = _tensor_error(Ua.tolist(), Va.tolist(), Wa.tolist())
        if err < best_err:
            best_err = err
        
        # Check if we can drop a column (border rank check)
        if err < 1e-7:
            # Try to find a column that is nearly zero or linearly dependent
            # Check if any column has very small norm
            for r in range(cur_rank):
                if (cur_rank - 1) >= target_rank and cur_rank > target_rank:
                    # Try removing column r
                    U_test = _np.delete(Ua, r, axis=1)
                    V_test = _np.delete(Va, r, axis=1)
                    W_test = _np.delete(Wa, r, axis=1)
                    err_test = _tensor_error(U_test.tolist(), V_test.tolist(), W_test.tolist())
                    if err_test < 1e-7:
                        return U_test.tolist(), V_test.tolist(), W_test.tolist()
    
    return None


def construct():
    _start_time = _time.time()
    _deadline = _start_time + 120  # safety margin before 150s kill
    
    # First, get the known rank-36 solution
    base = _build_strassen_36()
    
    # Verify it's correct
    if not _verify_decomposition(base["U"], base["V"], base["W"]):
        # Should not happen, but return it anyway
        return base
    
    # Try to reduce to rank 35 using border-rank / ALS
    # Try multiple seeds and initializations
    for seed in range(5):
        if _time.time() > _deadline:
            break
        result = _try_rank_reduction(
            base["U"], base["V"], base["W"],
            target_rank=35,
            max_iters=150,
            deadline=_deadline,
            seed=seed
        )
        if result is not None:
            U_new, V_new, W_new = result
            # Verify
            if _verify_decomposition(U_new, V_new, W_new):
                return {"U": U_new, "V": V_new, "W": W_new}
    
    # Try rank 34 as well (less likely but worth a shot)
    if _time.time() < _deadline - 10:
        for seed in range(3):
            if _time.time() > _deadline:
                break
            result = _try_rank_reduction(
                base["U"], base["V"], base["W"],
                target_rank=34,
                max_iters=100,
                deadline=_deadline,
                seed=100 + seed
            )
            if result is not None:
                U_new, V_new, W_new = result
                if _verify_decomposition(U_new, V_new, W_new):
                    return {"U": U_new, "V": V_new, "W": W_new}
    
    # Fallback: return the known rank-36 solution
    return base
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    # We use a rank 30 decomposition:
    # - 4 blocks of 2x2x2 Strassen (7 each) = 28
    # - 2 blocks of 2x2x1 (rank 1 each) = 2
    # Total: 30
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1: (y00+y11)(z00+z11) -> X00, X11
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M2: (y10+y11)z00 -> X10, -X11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] -= 1
        col += 1
        
        # M3: y00(z01-z11) -> X01, X11
        U[u_idx(y00[0], y00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M4: y11(z10-z00) -> X00, X10
        U[u_idx(y11[0], y11[1])][col] = 1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z00[0], z00[1])][col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1])][col] += 1
        col += 1
        
        # M5: (y00+y01)z11 -> -X00, X01
        U[u_idx(y00[0], y00[1])][col] = 1
        U[u_idx(y01[0], y01[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1])][col] += 1
        col += 1
        
        # M6: (y10-y00)(z00+z01) -> X11
        U[u_idx(y10[0], y10[1])][col] = 1
        U[u_idx(y00[0], y00[1])][col] = -1
        V[v_idx(z00[0], z00[1])][col] = 1
        V[v_idx(z01[0], z01[1])][col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1])][col] += 1
        col += 1
        
        # M7: (y01-y11)(z10+z11) -> X00
        U[u_idx(y01[0], y01[1])][col] = 1
        U[u_idx(y11[0], y11[1])][col] = -1
        V[v_idx(z10[0], z10[1])][col] = 1
        V[v_idx(z11[0], z11[1])][col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1])][col] += 1
        col += 1

    # Block 1: A_L (cols 0,1) * B_11 (rows 0,1; cols 0,1)
    # Result: C rows 0,1; cols 0,1 -> W indices (k=0,i=0), (k=1,i=0), (k=0,i=1), (k=1,i=1)
    y1 = [(0,0), (0,1), (1,0), (1,1)]
    z1 = [(0,0), (0,1), (1,0), (1,1)]
    w1 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y1, z1, w1)
    
    # Block 2: A_R (cols 2,3) * B_21 (rows 2,3; cols 0,1)
    # Result: C rows 0,1; cols 0,1
    y2 = [(0,2), (0,3), (1,2), (1,3)]
    z2 = [(2,0), (2,1), (3,0), (3,1)]
    w2 = [(0,0), (1,0), (0,1), (1,1)]
    strassen_block(y2, z2, w2)
    
    # Block 3: A_L (cols 0,1) * B_12 (rows 0,1; cols 2,3)
    # Result: C rows 0,1; cols 2,3 -> W indices (k=2,i=0), (k=3,i=0), (k=2,i=1), (k=3,i=1)
    y3 = [(0,0), (0,1), (1,0), (1,1)]
    z3 = [(0,2), (0,3), (1,2), (1,3)]
    w3 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y3, z3, w3)
    
    # Block 4: A_R (cols 2,3) * B_22 (rows 2,3; cols 2,3)
    # Result: C rows 0,1; cols 2,3
    y4 = [(0,2), (0,3), (1,2), (1,3)]
    z4 = [(2,2), (2,3), (3,2), (3,3)]
    w4 = [(2,0), (3,0), (2,1), (3,1)]
    strassen_block(y4, z4, w4)
    
    # Block 5: A_L (cols 0,1) * B_15 (rows 0,1; col 4)
    # 2x2x1: rank 4
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    
    # Term 1: A_00 * B_0 -> C_0
    U[u_idx(0,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    # Term 2: A_01 * B_1 -> C_0
    U[u_idx(0,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    # Term 3: A_10 * B_0 -> C_1
    U[u_idx(1,0)][col] = 1
    V[v_idx(0,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Term 4: A_11 * B_1 -> C_1
    U[u_idx(1,1)][col] = 1
    V[v_idx(1,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Block 6: A_R (cols 2,3) * B_25 (rows 2,3; col 4)
    # 2x2x1: rank 4
    
    # Term 1: A_02 * B_24 -> C_0
    U[u_idx(0,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    # Term 2: A_03 * B_34 -> C_0
    U[u_idx(0,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,0)][col] = 1
    col += 1
    
    # Term 3: A_12 * B_24 -> C_1
    U[u_idx(1,2)][col] = 1
    V[v_idx(2,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
    # Term 4: A_13 * B_34 -> C_1
    U[u_idx(1,3)][col] = 1
    V[v_idx(3,4)][col] = 1
    W[w_idx(4,1)][col] = 1
    col += 1
    
# EVOLVE-BLOCK-END
