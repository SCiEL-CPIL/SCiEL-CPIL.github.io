"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build initial rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 1
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] -= 1

        col = offset + 2
        U[(i_start+0)*m + (j_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 3
        U[(i_start+1)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+0)][col] -= 1
        W[(k_start+0)*n + (i_start+0)][col] += 1
        W[(k_start+0)*n + (i_start+1)][col] += 1

        col = offset + 4
        U[(i_start+0)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+1)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] -= 1
        W[(k_start+1)*n + (i_start+0)][col] += 1

        col = offset + 5
        U[(i_start+1)*m + (j_start+0)][col] += 1
        U[(i_start+0)*m + (j_start+0)][col] -= 1
        V[(j_start+0)*p + (k_start+0)][col] += 1
        V[(j_start+0)*p + (k_start+1)][col] += 1
        W[(k_start+1)*n + (i_start+1)][col] += 1

        col = offset + 6
        U[(i_start+0)*m + (j_start+1)][col] += 1
        U[(i_start+1)*m + (j_start+1)][col] -= 1
        V[(j_start+1)*p + (k_start+0)][col] += 1
        V[(j_start+1)*p + (k_start+1)][col] += 1
        W[(k_start+0)*n + (i_start+0)][col] += 1

    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)

    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Verify the decomposition is correct
    def verify_decomp(U_arr, V_arr, W_arr):
        """Check that sum_k U[:,k] * V[:,k] * W[:,k] equals the matmul tensor."""
        dim_in = n * m
        dim_mid = m * p
        dim_out = p * n
        rank_test = U_arr.shape[1]
        
        T_decomp = np.zeros((n, m, p))
        for r in range(rank_test):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i,j,k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        
        T_expected = np.ones((n, m, p))
        return np.allclose(T_decomp, T_expected, atol=1e-10)

    if not verify_decomp(U_np, V_np, W_np):
        return {"U": U, "V": W, "W": W}

    best_rank = rank
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()

    # --- Support Pruning Approach ---
    # We want to find a rank-40 decomposition.
    # Strategy: Remove 2 columns from the rank-42 Strassen pattern.
    # Then, solve for the coefficients of the remaining 40 columns such that
    # the tensor equation is satisfied.
    # Since the original rank-42 decomposition works, if we remove 2 columns,
    # the remaining 40 columns must span the tensor. 
    # Wait, the tensor is fixed (all 1s). 
    # If we remove 2 columns, we have a system of 48 equations (n*m*p) and 40 unknowns (coefficients).
    # The system is overdetermined. A solution exists only if the target tensor lies in the span
    # of the remaining 40 rank-1 tensors.
    # However, the "support" is fixed to the entries of the Strassen pattern.
    # Actually, the "support pruning" strategy suggests:
    # 1. Pick a subset S of size 40 from the 42 columns.
    # 2. Keep the *structure* (non-zero positions) of these 40 columns from the Strassen pattern.
    # 3. Solve for the *values* (coefficients) of these non-zero entries such that the decomposition holds.
    
    # Let's formalize:
    # Let the support of column r be defined by the non-zero entries in U[:,r], V[:,r], W[:,r].
    # In the Strassen pattern, each column has a specific sparse structure.
    # If we fix the support (which entries are non-zero) for a subset of 40 columns,
    # we have a linear system for the values.
    
    # Total number of unknowns: sum of non-zeros in the 40 selected columns.
    # Total number of equations: n*m*p = 48.
    
    # For a generic Strassen column, there are 3 non-zeros in U, 2 in V, 2 in W (approx).
    # Let's count exactly for one block.
    # M1: U(2), V(2), W(2) -> 6 non-zeros
    # M2: U(2), V(1), W(2) -> 5 non-zeros
    # M3: U(1), V(2), W(2) -> 5 non-zeros
    # M4: U(1), V(2), W(2) -> 5 non-zeros
    # M5: U(2), V(1), W(2) -> 5 non-zeros
    # M6: U(2), V(2), W(1) -> 5 non-zeros
    # M7: U(2), V(2), W(1) -> 5 non-zeros
    # Total per block: 6+5+5+5+5+5+5 = 36 non-zeros.
    # For 6 blocks (42 cols), total non-zeros = 6 * 36 = 216.
    # If we pick 40 columns, we have roughly 40/42 * 216 ~ 205 non-zeros.
    # Equations: 48.
    # This is a very underdetermined system (205 unknowns, 48 equations).
    # A solution almost certainly exists.
    # The constraint is that the entries must be integer or half-integer.
    # The linear system will yield rational solutions. We need to check if they are half-integers.
    
    # Let's try all pairs of columns to remove.
    # There are C(42, 2) = 861 pairs.
    # For each pair, we build the linear system.
    # We can vectorize this or do it efficiently.
    
    # Precompute the tensor basis vectors for each column?
    # No, the coefficients are unknowns.
    # Let x be the vector of non-zero values in the selected 40 columns.
    # We want sum_{r in S} x_r * (u_r * v_r * w_r) = 1 (tensor of ones)
    # Where u_r, v_r, w_r are the *unit* vectors corresponding to the support positions?
    # No, the support positions are fixed, but the values are unknown.
    # Let the support of column r be defined by indices I_r (for U), J_r (for V), K_r (for W).
    # Let the values be alpha_{r, i}, beta_{r, j}, gamma_{r, k}.
    # The term in the tensor is product of these.
    # This is NON-LINEAR in the coefficients!
    # T[i,j,k] = sum_{r} U[i,r] V[j,r] W[k,r]
    # If U[i,r], V[j,r], W[k,r] are variables, this is a polynomial system of degree 3.
    # This is hard.
    
    # Alternative interpretation of "support pruning":
    # Maybe we fix the *values* to be +/- 1 (or similar) and only prune the support?
    # Or maybe we fix the values from the Strassen pattern and just check if removing 2 columns
    # and rescaling the rest works? (This is the peeling idea).
    
    # Let's stick to the peeling idea but make it more robust.
    # The peeling idea: Remove column r. Check if the tensor T_r is in the span of the others.
    # If yes, remove it.
    # The current code does this. It found rank 42 (no peeling).
    # Why? Because the Strassen decomposition is likely minimal for that structure,
    # or the numerical tolerance failed.
    
    # Let's try a different structural change:
    # Use a different base decomposition.
    # Strassen is 7 for 2x2x2.
    # What if we use a 2x4x2 decomposition?
    # Rank of 2x4x2 matmul?
    # Schoolbook: 2*4*2 = 16.
    # Strassen-like?
    # We can view 2x4 as two 2x2 blocks.
    # (A1 A2) * (B1) = (A1 B1 + A2 B2)
    #             (B2)
    # This is not a standard tensor decomposition.
    
    # Let's try to optimize the coefficients of the Strassen pattern.
    # Maybe the optimal rank is 41 or 40, and the coefficients are not integers.
    # The task allows half-integers.
    
    # Let's use the "Support Pruning" strategy as suggested:
    # Enumerate sparse rank-40 decompositions with support sets derived from Strassen.
    # This implies we fix the support (which entries are non-zero) to a subset of the Strassen supports.
    # And we solve for the values.
    # Since the system is non-linear (tensor rank decomposition is non-linear),
    # we can linearize it if we fix two of the three matrices.
    
    # Strategy:
    # 1. Fix U and V to their Strassen values (for a subset of columns).
    # 2. Solve for W.
    # This is a linear system for W!
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
    # For fixed U, V, this is linear in W.
    # We have 48 equations (i,j,k) and 40*12 unknowns? No.
    # W is (p*n) x rank = 12 x 40.
    # Unknowns: 480.
    # Equations: 48.
    # Very underdetermined.
    # We can find a solution for W.
    # Then we check if W has half-integer entries.
    # If not, we can try to adjust U and V slightly?
    
    # Let's try this:
    # 1. Select 40 columns from the 42 Strassen columns.
    # 2. Fix U and V to their Strassen values for these 40 columns.
    # 3. Solve the linear system for W:
    #    For each (i,j,k): sum_{r in S} U[i,r] V[j,r] W[k*n+i, r] = 1
    #    This is a system of 48 equations with 40*12 = 480 unknowns.
    #    We want a sparse or simple solution?
    #    Actually, we just want *any* solution that has half-integer entries.
    #    Since it's underdetermined, we can impose constraints.
    #    Or, we can minimize the norm of W, or something.
    #    But we need EXACT half-integers.
    
    # Better:
    # Solve for W using least squares, then round to half-integers and verify.
    # If verification fails, try a different subset of 40 columns.
    
    # There are 861 subsets. This is feasible.
    
    # Let's implement this.
    
    # Precompute U_strassen, V_strassen, W_strassen (original)
    U_orig = U_np.copy()
    V_orig = V_np.copy()
    W_orig = W_np.copy()
    
    # List of all column indices
    all_cols = list(range(42))
    
    # Iterate over all pairs to remove
    for i_remove in range(42):
        if time.time() > deadline - 5:
            break
        for j_remove in range(i_remove + 1, 42):
            if time.time() > deadline - 5:
                break
            
            # Keep all columns except i_remove and j_remove
            keep_cols = [c for c in all_cols if c != i_remove and c != j_remove]
            
            # Extract U, V for kept columns
            U_keep = U_orig[:, keep_cols]  # 8 x 40
            V_keep = V_orig[:, keep_cols]  # 24 x 40
            
            # Build linear system for W
            # Unknowns: W_keep is 12 x 40. Flatten to 480 vector.
            # Equations: 48 (i, j, k)
            
            # A * W_flat = b
            # A is 48 x 480
            
            A = np.zeros((48, 480))
            b = np.ones(48)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Equation: sum_{r in keep_cols} U[i, r] * V[j, r] * W[k*n+i, r] = 1
                        # W index in flat vector: (k*n+i) * 40 + col_index_in_keep
                        for col_idx, r in enumerate(keep_cols):
                            u_val = U_orig[i, r]
                            v_val = V_orig[j, r]
                            if u_val != 0 and v_val != 0:
                                w_row = k * n + i
                                w_flat_idx = w_row * 40 + col_idx
                                A[idx, w_flat_idx] = u_val * v_val
                        idx += 1
            
            # Solve A * W_flat = b
            # Use least squares
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-12)
            except Exception:
                continue
            
            # Reshape W_flat to 12 x 40
            W_sol = W_flat.reshape((12, 40))
            
            # Check if W_sol is close to half-integers
            # Round to half-integers
            W_rounded = np.round(W_sol * 2) / 2
            
            # Verify the decomposition with rounded W
            # Check residual
            W_check = W_rounded
            # Reconstruct tensor
            T_check = np.zeros((n, m, p))
            for col_idx, r in enumerate(keep_cols):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_check[i, j, k] += U_orig[i, r] * V_orig[j, r] * W_check[k * n + i, col_idx]
            
            if np.allclose(T_check, np.ones((n, m, p)), atol=1e-10):
                # We found a valid rank-40 decomposition!
                # Construct the full U, V, W matrices
                U_new = np.zeros((n * m, 40))
                V_new = np.zeros((m * p, 40))
                W_new = np.zeros((p * n, 40))
                
                for col_idx, r in enumerate(keep_cols):
                    U_new[:, col_idx] = U_orig[:, r]
                    V_new[:, col_idx] = V_orig[:, r]
                    W_new[:, col_idx] = W_rounded[:, col_idx]
                
                # Check if U and V are also half-integers (they are from Strassen, so yes)
                # But let's ensure the returned format is correct.
                
                best_rank = 40
                best_U = U_new
                best_V = V_new
                best_W = W_new
                break
        
        if best_rank == 40:
            break
    
    # If we didn't find rank 40, try rank 41?
    # Or just return the best we have.
    
    if best_rank < rank:
        # Convert to lists
        def to_list(arr):
            return arr.tolist()
        
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    else:
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
