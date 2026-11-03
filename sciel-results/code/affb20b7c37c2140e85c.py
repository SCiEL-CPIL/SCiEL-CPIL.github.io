import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 4, 8
    dim_u = n * m  # 8
    dim_v = m * p  # 32
    dim_w = p * n  # 16

    # 1. Build the target multiplication tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # 2. Define a robust schoolbook fallback (Rank 56)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W

    # 3. Randomized rank-deflation search
    # We try to find a decomposition of rank R by:
    # a) Randomly generating sparse integer/half-integer factors for the first term.
    # b) Subtracting that rank-1 tensor from the residual.
    # c) Repeating until the residual is zero (or too small).
    
    best_U, best_V, best_W = schoolbook()
    best_rank = best_U.shape[1]

    def check_and_finalize(U_list, V_list, W_list, rank):
        U = np.array(U_list, dtype=np.float64)
        V = np.array(V_list, dtype=np.float64)
        W = np.array(W_list, dtype=np.float64)
        # Verify exactness
        T_recon = np.zeros((dim_u, dim_v, dim_w))
        for c in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        if np.allclose(T_recon, T, atol=1e-9):
            return True
        return False

    # Search for ranks lower than schoolbook
    # Theoretical lower bound is max(dim_u, dim_v, dim_w) = 32.
    # Let's try down to 32.
    
    for target_rank in range(32, best_rank):
        if time.time() > deadline - 15.0:
            break
            
        found = False
        # Limit iterations per rank to avoid hanging
        max_iters = 10000
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            U_cur = np.zeros((dim_u, target_rank))
            V_cur = np.zeros((dim_v, target_rank))
            W_cur = np.zeros((dim_w, target_rank))
            Residual = T.copy()
            
            success = True
            for c in range(target_rank):
                # Generate random sparse factors
                # Sparsity: ~20% non-zeros
                # Values: 0, 0.5, 1, -0.5, -1
                
                def gen_sparse_vec(dim, sparsity=0.2):
                    vec = np.zeros(dim)
                    nz_count = max(1, int(dim * sparsity))
                    idx = np.random.choice(dim, size=nz_count, replace=False)
                    vals = np.random.choice([-1.0, -0.5, 0.5, 1.0], size=nz_count)
                    vec[idx] = vals
                    return vec

                u_vec = gen_sparse_vec(dim_u)
                v_vec = gen_sparse_vec(dim_v)
                w_vec = gen_sparse_vec(dim_w)
                
                # Ensure they are not trivially zero
                if np.all(u_vec == 0) or np.all(v_vec == 0) or np.all(w_vec == 0):
                    continue

                term = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
                
                # Check if this term is "useful" (reduces residual norm significantly)
                # Actually, for exact decomposition, we need to solve for the next term.
                # A better approach: 
                # 1. Pick random u, v.
                # 2. Solve for w in Residual ≈ u ⊗ v ⊗ w.
                #    Residual[i, j, k] = sum_c u_c[i] v_c[j] w_c[k]
                #    For the current step c, we want Residual_new = Residual_old - u ⊗ v ⊗ w.
                #    This is an underdetermined system for w if we don't fix u, v.
                #    Instead, let's use a "greedy" approach:
                #    1. Randomly pick u, v.
                #    2. Compute the "best" w that minimizes ||Residual - u⊗v⊗w||.
                #       This is a least squares problem:
                #       Residual[i, j, k] ≈ u[i] v[j] w[k]
                #       Let M[i, j] = u[i] v[j].
                #       We want to approximate Residual (flattened over i,j for each k) by M ⊗ w.
                #       Actually, it's easier to view it as:
                #       Residual[i, j, k] = M[i, j] * w[k]
                #       This is a rank-1 approximation of the tensor Residual by a tensor with fixed u, v.
                #       We can solve for w by minimizing sum_{i,j,k} (Residual[i,j,k] - M[i,j]w[k])^2.
                #       This is separable in k? No, w[k] is shared for all i,j.
                #       Let's flatten Residual to (dim_u*dim_v, dim_w).
                #       Let R_flat be (dim_u*dim_v, dim_w).
                #       Let M_flat be (dim_u*dim_v,).
                #       We want to minimize ||R_flat - M_flat * w^T||_F^2.
                #       This is a simple linear least squares problem for w.
                
                R_flat = Residual.reshape(dim_u * dim_v, dim_w)
                M_flat = (u_vec * v_vec.reshape(-1, 1)).reshape(-1) # Wait, v_vec is 1D.
                # u_vec is (dim_u,), v_vec is (dim_v,).
                # M[i, j] = u_vec[i] * v_vec[j].
                # M_flat index is i*dim_v + j.
                M_flat = np.outer(u_vec, v_vec).reshape(-1)
                
                # Solve M_flat * w^T ≈ R_flat
                # w^T is (1, dim_w).
                # R_flat is (dim_u*dim_v, dim_w).
                # For each column k of R_flat, we have R_flat[:, k] ≈ M_flat * w[k].
                # So w[k] = dot(R_flat[:, k], M_flat) / dot(M_flat, M_flat).
                
                denom = np.dot(M_flat, M_flat)
                if denom < 1e-12:
                    continue
                
                # Compute w
                # w[k] = sum_{i,j} R_flat[i*dim_v+j, k] * M_flat[i*dim_v+j] / denom
                # This is equivalent to:
                # w = (R_flat.T @ M_flat) / denom
                w_sol = (R_flat.T @ M_flat) / denom
                
                # Snap w_sol to nearest half-integer or integer?
                # The problem allows integer/half-integer.
                # If we snap, we might lose exactness.
                # However, if the true solution has integer/half-integer entries,
                # and our random u, v are "aligned" with the structure, w_sol might be close.
                # Let's try snapping to nearest 0.5.
                w_snap = np.round(w_sol * 2.0) / 2.0
                
                # Check if snapping is close
                if not np.allclose(w_snap, w_sol, atol=1e-6):
                    # If snapping is bad, maybe skip this candidate or try without snapping?
                    # Let's try with snapping first, as we need exact integer/half-integer output.
                    pass
                
                # Update Residual
                term = np.einsum('a,b,c->abc', u_vec, v_vec, w_snap)
                Residual -= term
                
                U_cur[:, c] = u_vec
                V_cur[:, c] = v_vec
                W_cur[:, c] = w_snap
                
                # Early termination if residual is zero
                if np.allclose(Residual, 0, atol=1e-9):
                    if c < target_rank - 1:
                        # We finished early, which means the rank is actually c+1.
                        # But we are searching for a specific rank. 
                        # If we finish early, we have found a decomposition of rank c+1.
                        # This is a success!
                        if check_and_finalize(U_cur[:, :c+1], V_cur[:, :c+1], W_cur[:, :c+1], c+1):
                            best_U = U_cur[:, :c+1]
                            best_V = V_cur[:, :c+1]
                            best_W = W_cur[:, :c+1]
                            best_rank = c + 1
                            found = True
                            break
                    else:
                        # Finished exactly at target_rank
                        if check_and_finalize(U_cur, V_cur, W_cur, target_rank):
                            best_U = U_cur
                            best_V = V_cur
                            best_W = W_cur
                            best_rank = target_rank
                            found = True
                            break

            if found:
                break

    # Final verification and return
    if time.time() < deadline - 5.0:
        if not check_and_finalize(best_U, best_V, best_W, best_rank):
            # Fallback to schoolbook if something went wrong
            U, V, W = schoolbook()
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
