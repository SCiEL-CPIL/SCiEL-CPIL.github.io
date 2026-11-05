"""Matrix multiplication 2x4x8 via Strassen with algebraic rank reduction."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = 2, 4, 8
    
    # Build rank-28 decomposition: 4 independent 2x4x2 Strassen blocks
    # Each block handles columns (2*c, 2*c+1) of B -> column c of C
    # For each block, use Strassen 2x2x2 (rank 7) with 4x2 B submatrix
    
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 4 blocks, each with 7 Strassen operations
    # Block b handles B columns (2b, 2b+1) and C column b
    # A is 2x4, B_block is 4x2, C_block is 2x1 (one column)
    
    # For 2x4x2: A(2x4) * B(4x2) = C(2x2)
    # Split A into A_left(2x2) cols 0,1 and A_right(2x2) cols 2,3
    # Split B into B_top(2x2) rows 0,1 and B_bottom(2x2) rows 2,3
    # C = A_left @ B_top + A_right @ B_bottom
    # Use Strassen on each 2x2x2 product (rank 7 each), total rank 14 per 2x4x2
    # But we can be smarter...
    
    # Actually for 2x4x2: we have A(2x4) and B(4x2)
    # C[i,k] = sum_j A[i,j]*B[j,k] for j=0..3, k=0,1
    # This is a 2x4x2 tensor. Known rank: let's use Strassen-like.
    # Split A into A1(2x2) [cols 0,1] and A2(2x2) [cols 2,3]
    # C = A1 @ B1 + A2 @ B2 where B1=B[0:2, :], B2=B[2:4, :]
    # Each 2x2x2 has rank 7, so total rank 14 for 2x4x2.
    # For 4 such blocks: 4*14 = 56. Same as before!
    
    # Alternative: don't split A. Use the full 2x4x2 structure.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # We can group: (A[i,0]+A[i,2])*(B[0,k]+B[2,k]) + (A[i,1]+A[i,3])*(B[1,k]+B[3,k])
    #   - A[i,2]*B[2,k] - A[i,0]*B[0,k] - A[i,3]*B[3,k] - A[i,1]*B[1,k]
    #   + cross terms...
    # This is getting complex. Let me just use the rank-56 and try to reduce.
    
    # Let me try a different approach: use the known result that
    # 2x4x8 has bilinear complexity <= 28 using specific decompositions.
    
    # Actually, let me just build the rank-56 and try aggressive reduction.
    
    U = np.zeros((n * m, 56))
    V = np.zeros((m * p, 56))
    W = np.zeros((p * n, 56))
    
    # 8 Strassen blocks: (a_col_off, b_row_off, b_col_off, c_col_off)
    blocks = [
        (0, 0, 0, 0), (0, 0, 2, 2), (0, 0, 4, 4), (0, 0, 6, 6),
        (2, 2, 0, 0), (2, 2, 2, 2), (2, 2, 4, 4), (2, 2, 6, 6),
    ]
    
    col = 0
    for (a_col_off, b_row_off, b_col_off, c_col_off) in blocks:
        a00 = 0 * 4 + a_col_off
        a01 = 0 * 4 + a_col_off + 1
        a10 = 1 * 4 + a_col_off
        a11 = 1 * 4 + a_col_off + 1
        
        b00 = b_row_off * 8 + b_col_off
        b01 = b_row_off * 8 + b_col_off + 1
        b10 = (b_row_off + 1) * 8 + b_col_off
        b11 = (b_row_off + 1) * 8 + b_col_off + 1
        
        c00 = c_col_off * 2 + 0
        c01 = (c_col_off + 1) * 2 + 0
        c10 = c_col_off * 2 + 1
        c11 = (c_col_off + 1) * 2 + 1
        
        # P1 = (a00+a11)(b00+b11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # P2 = (a10+a11)*b00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] += -1
        col += 1
        
        # P3 = a00*(b01-b11)
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] += -1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # P4 = a11*(b10-b00)
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] += -1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # P5 = (a00+a01)*b11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] += -1; W[c01, col] += 1
        col += 1
        
        # P6 = (a10-a00)*(b00+b01)
        U[a10, col] += 1; U[a00, col] += -1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # P7 = (a01-a11)*(b10+b11)
        U[a01, col] += 1; U[a11, col] += -1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
    
    assert col == 56
    
    def verify(U_mat, V_mat, W_mat, n_tests=3):
        rng_v = np.random.RandomState(12345)
        for _ in range(n_tests):
            A_t = rng_v.randn(n, m)
            B_t = rng_v.randn(m, p)
            C_exp = A_t @ B_t
            C_comp = np.zeros((n, p))
            for c in range(U_mat.shape[1]):
                for i in range(n):
                    for j in range(m):
                        u_val = U_mat[i * m + j, c]
                        if abs(u_val) < 1e-14:
                            continue
                        for k in range(p):
                            v_val = V_mat[j * p + k, c]
                            if abs(v_val) < 1e-14:
                                continue
                            w_val = W_mat[k * n + i, c]
                            C_comp[i, k] += u_val * v_val * w_val
            if not np.allclose(C_comp, C_exp, atol=1e-8):
                return False
        return True
    
    # Verify base decomposition
    if not verify(U, V, W, 2):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the evaluation matrix M where M[s,c] = U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    all_triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    n_triples = len(all_triples)  # 64
    
    M = np.zeros((n_triples, 56))
    for s, (i, j, k) in enumerate(all_triples):
        for c in range(56):
            M[s, c] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
    
    # SVD to find numerical rank
    U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=True)
    tol = 1e-10
    num_rank = int(np.sum(s_vals > tol))
    
    best_rank = 56
    best_factors = (U.copy(), V.copy(), W.copy())
    
    # Aggressive greedy elimination: try to remove columns one by one
    # and solve for adjusted values if possible
    keep = list(range(56))
    
    # Sort by column norm (remove smallest first)
    col_norms = [(np.linalg.norm(M[:, c]), c) for c in range(56)]
    col_norms.sort()
    
    for _, c in col_norms:
        if time.time() > deadline - 30:
            break
        if c not in keep:
            continue
        other = [d for d in keep if d != c]
        
        # Check if column c is in span of others
        if np.linalg.norm(M[:, c]) < 1e-14:
            keep.remove(c)
            continue
        
        try:
            A_check = M[:, other]
            b_check = M[:, c]
            alpha, residuals, rank_c, _ = np.linalg.lstsq(A_check, b_check, rcond=1e-10)
            residual = np.linalg.norm(A_check @ alpha - b_check)
            if residual / max(np.linalg.norm(b_check), 1e-14) < 1e-8:
                keep.remove(c)
        except:
            pass
    
    r_final = len(keep)
    if r_final < 56:
        U_red = U[:, keep]
        V_red = V[:, keep]
        W_red = W[:, keep]
        if verify(U_red, V_red, W_red, 2):
            best_rank = r_final
            best_factors = (U_red, V_red, W_red)
    
    # Try to further reduce by solving for new coefficients
    # For each candidate rank r < best_rank, try to find a decomposition
    # This is NP-hard in general, but we can try random initialization
    
    if time.time() < deadline - 40:
        # Try random search for lower rank
        for target_rank in range(max(best_rank - 4, 16), best_rank):
            if time.time() > deadline - 35:
                break
            
            # Random initialization
            for trial in range(20):
                if time.time() > deadline - 35:
                    break
                
                rng = np.random.RandomState(42 + trial)
                U_rand = rng.randn(n * m, target_rank)
                V_rand = rng.randn(m * p, target_rank)
                W_rand = rng.randn(p * n, target_rank)
                
                # Build M_rand and check if it can match the target
                M_rand = np.zeros((n_triples, target_rank))
                for s, (i, j, k) in enumerate(all_triples):
                    for c in range(target_rank):
                        M_rand[s, c] = U_rand[i * m + j, c] * V_rand[j * p + k, c] * W_rand[k * n + i, c]
                
                # The target tensor values are all 1s for valid (i,j,k)
                # Actually, T[i,j,k] = 1 for all i,j,k (it's the identity)
                # So we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
                
                # This is a nonlinear system. Skip for now.
                pass
    
    # Return best found
    U_best, V_best, W_best = best_factors
    if verify(U_best, V_best, W_best, 2):
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
