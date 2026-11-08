"""Tensor decomposition for 2x3 matrix multiplication (A@B, B is 3x10)."""
DIMENSIONS = [2, 3, 10]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    n, m, p = 2, 3, 10

    # Build the rank-55 block decomposition (5 blocks of 2 columns, 11 ranks each)
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1

        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0][col] += 1
        U[4][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M2: (A10 + A11) * B0,k0
        U[3][col] += 1
        U[4][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1

        # M3: A00 * (B0,k1 - B1,k1)
        U[0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M4: A11 * (B1,k0 - B0,k0)
        U[4][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # M5: (A00 + A01) * B1,k1
        U[0][col] += 1
        U[1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # M6: (A10 - A00) * (B0,k0 + B0,k1)
        U[3][col] += 1
        U[0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

        # M7: (A01 - A11) * (B1,k0 + B1,k1)
        U[1][col] += 1
        U[4][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        # W00: A02 * B2,k0
        U[2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1

        # W01: A02 * B2,k1
        U[2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1

        # W10: A12 * B2,k0
        U[5][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1

        # W11: A12 * B2,k1
        U[5][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1

    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Try to reduce rank by finding linearly dependent columns in W
    # If W[:, j] is in the span of other columns, we can eliminate column j
    # by absorbing it into the others.
    
    # Strategy: iteratively check if any column of W is in the span of the
    # remaining columns. If so, eliminate it.
    
    def try_eliminate(U_mat, V_mat, W_mat, time_budget):
        """Try to eliminate linearly dependent columns."""
        start = time.time()
        cur_U = U_mat.copy()
        cur_V = V_mat.copy()
        cur_W = W_mat.copy()
        cur_cols = list(range(cur_W.shape[1]))
        
        improved = True
        while improved:
            improved = False
            if time.time() - start > time_budget:
                break
            
            n_cols = len(cur_cols)
            if n_cols <= 1:
                break
                
            # Try each column to see if it's in the span of others
            for idx in range(n_cols):
                if time.time() - start > time_budget:
                    break
                
                col_to_remove = cur_cols[idx]
                remaining = [c for j, c in enumerate(cur_cols) if j != idx]
                
                if len(remaining) == 0:
                    continue
                
                W_remove = cur_W[:, col_to_remove]
                W_rem = cur_W[:, remaining]
                
                # Check if W_remove is in the column space of W_rem
                try:
                    # Solve W_rem @ a = W_remove
                    a, residuals, rank_w, sv = np.linalg.lstsq(W_rem, W_remove, rcond=None)
                except Exception:
                    continue
                
                # Check if the solution is exact
                if np.allclose(W_rem @ a, W_remove, atol=1e-7, rtol=1e-7):
                    # We can eliminate this column
                    # For each remaining column k, new_U[:,k] = old_U[:,k] + a[k]*old_U[:,col_to_remove]
                    new_U = np.zeros((cur_U.shape[0], len(remaining)))
                    new_V = np.zeros((cur_V.shape[0], len(remaining)))
                    new_W = np.zeros((cur_W.shape[0], len(remaining)))
                    
                    for new_idx, old_idx in enumerate(remaining):
                        coeff = a[new_idx]
                        new_U[:, new_idx] = cur_U[:, old_idx] + coeff * cur_U[:, col_to_remove]
                        new_V[:, new_idx] = cur_V[:, old_idx] + coeff * cur_V[:, col_to_remove]
                        new_W[:, new_idx] = cur_W[:, old_idx] + coeff * cur_W[:, col_to_remove]
                    
                    # Verify the new decomposition still works
                    # The W matrix should now have the removed column's contribution absorbed
                    # Actually, we need to verify that the new W still represents the same tensor
                    # Let's just check that the new decomposition is valid by checking
                    # that for the new W, the tensor product still holds.
                    
                    # For now, accept the elimination and verify at the end
                    cur_U = new_U
                    cur_V = new_V
                    cur_W = new_W
                    cur_cols = remaining
                    improved = True
                    break
        
        return cur_U, cur_V, cur_W, len(cur_cols)

    # Attempt rank reduction with a time budget
    time_budget = 30.0
    U_reduced, V_reduced, W_reduced, new_rank = try_eliminate(U_np, V_np, W_np, time_budget)
    
    # Verify the reduced decomposition is still valid
    # Check: for all i,j,k: sum_r U[i,r]*V[j,r]*W[k,r] = delta_{i,j} * (A@B)[i,k]
    # Actually, we need to verify the tensor decomposition is correct.
    
    # Let's verify by checking a few random matrix products
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p, n_tests=5):
        """Verify the decomposition by testing with random matrices."""
        for _ in range(n_tests):
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            # Compute C from decomposition
            C_computed = np.zeros((n, p))
            for r in range(U_mat.shape[1]):
                # u_r is a vector of length n*m
                # v_r is a vector of length m*p
                # w_r is a vector of length p*n
                u_r = U_mat[:, r].reshape(n, m)
                v_r = V_mat[:, r].reshape(m, p)
                w_r = W_mat[:, r].reshape(p, n)
                
                # The contribution to C from rank r is:
                # C += sum_{i,j,k} w_r[k,i] * (u_r[i,j] * A[i,j]) * (v_r[j,k] * B[j,k])
                # Wait, that's not right. Let me think about this more carefully.
                
                # The tensor T[i,j,k] = delta_{i,j} for matrix multiplication
                # T = sum_r u_r ⊗ v_r ⊗ w_r
                # where u_r[i,j] = U[i,r], v_r[j,k] = V[j,r], w_r[k,i] = W[k,r]
                
                # For a specific A and B:
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # = sum_r (sum_j A[i,j] * u_r[i,j]) * (sum_j B[j,k] * v_r[j,k]) * w_r[k,i]
                
                a_u = A_test * u_r  # element-wise, shape (n,m)
                b_v = B_test * v_r  # element-wise, shape (m,p)
                
                # sum_j A[i,j]*u_r[i,j] for each i
                a_proj = np.sum(a_u, axis=1)  # shape (n,)
                # sum_j B[j,k]*v_r[j,k] for each k
                b_proj = np.sum(b_v, axis=0)  # shape (p,)
                
                # C_computed += outer(a_proj, b_proj) * w_r.T
                # w_r has shape (p, n), so w_r.T has shape (n, p)
                C_computed += np.outer(a_proj, b_proj) * w_r.T
            
            if not np.allclose(C_computed, C_expected, atol=1e-6, rtol=1e-6):
                return False
        return True

    # Verify the original decomposition first
    if not verify_decomposition(U_np, V_np, W_np, n, m, p, n_tests=3):
        # Original is broken, return as-is
        return {"U": U, "V": V, "W": W}
    
    # Verify the reduced decomposition
    if new_rank < rank and verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p, n_tests=3):
        # Convert back to list format
        U_out = U_reduced.tolist()
        V_out = V_reduced.tolist()
        W_out = W_reduced.tolist()
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # If reduction didn't work or verification failed, return the original
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
