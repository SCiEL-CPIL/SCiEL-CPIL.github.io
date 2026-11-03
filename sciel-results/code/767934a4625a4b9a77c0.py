import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    deadline = time.time() + 145
    
    # Build the rank-36 Strassen solution
    U36, V36, W36 = _build_strassen_36(n, m, p)
    U36 = np.array(U36, dtype=np.float64)
    V36 = np.array(V36, dtype=np.float64)
    W36 = np.array(W36, dtype=np.float64)
    R36 = 36
    
    best_rank = R36
    best_U, best_V, best_W = U36.copy(), V36.copy(), W36.copy()
    
    # Iterative reduction strategy
    # Current set of indices to keep
    current_indices = list(range(R36))
    current_U = U36.copy()
    current_V = V36.copy()
    current_W = W36.copy()
    current_rank = R36
    
    # We will try to drop columns one by one
    # To do this efficiently, we maintain the tensor matrix M
    # M has shape (n*m*p, current_rank)
    
    def build_M(U, V, W, rank):
        M = np.zeros((n * m * p, rank), dtype=np.float64)
        for r in range(rank):
            u_r = U[:, r]
            v_r = V[:, r]
            w_r = W[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * (m * p) + j * p + k
                        M[idx, r] = u_r[i * m + j] * v_r[j * p + k] * w_r[k * n + i]
        return M

    M = build_M(current_U, current_V, current_W, current_rank)
    
    # Loop to find dependencies
    # We try to drop a column if it is in the span of the others
    # We repeat until no more drops are possible or time is up
    
    dropped_any = True
    while dropped_any and time.time() < deadline:
        dropped_any = False
        # Try to drop each column
        # We should probably try to drop the column with the largest norm first? 
        # Or just iterate through all.
        # Since the matrix is small, iterating through all is fine.
        
        for col_to_drop in range(current_rank):
            if time.time() > deadline:
                break
                
            # Skip if this column is already "dropped" (not in current_indices)
            # Actually, current_U/V/W are compacted. So col_to_drop is valid index.
            
            if current_rank == 1:
                break
                
            M_sub = np.delete(M, col_to_drop, axis=1)
            b = M[:, col_to_drop]
            
            try:
                c, residuals, rank_sol, sv = np.linalg.lstsq(M_sub, b, rcond=None)
            except:
                continue
            
            if residuals is None or len(residuals) == 0:
                continue
                
            if residuals[0] < 1e-8:
                # We can drop this column
                # Update the remaining columns
                # T_new = sum_{i != col} (1 + c[i]) T_i
                
                new_rank = current_rank - 1
                new_U = np.zeros((n * m, new_rank))
                new_V = np.zeros((m * p, new_rank))
                new_W = np.zeros((p * n, new_rank))
                
                idx = 0
                for i in range(current_rank):
                    if i == col_to_drop:
                        continue
                    scale = 1.0 + c[idx]
                    new_U[:, idx] = current_U[:, i] * scale
                    new_V[:, idx] = current_V[:, i]
                    new_W[:, idx] = current_W[:, i]
                    idx += 1
                
                # Verify the solution
                valid = True
                # Reconstruct tensor from new_U, new_V, new_W
                # and compare with original M (which is the sum of original terms)
                # Actually, we just need to check if the new decomposition equals the original tensor.
                # The original tensor is M @ ones? No.
                # The tensor is T = sum_r T_r.
                # M is the matrix of T_r columns.
                # The sum is M @ np.ones(current_rank).
                # We should check if new_M @ np.ones(new_rank) == M @ np.ones(current_rank).
                
                M_new = build_M(new_U, new_V, new_W, new_rank)
                target_sum = M @ np.ones(current_rank)
                new_sum = M_new @ np.ones(new_rank)
                
                if np.allclose(target_sum, new_sum, atol=1e-8):
                    # Update state
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    current_rank = new_rank
                    M = M_new
                    dropped_any = True
                    # Since we modified the set, we need to restart the loop over columns
                    # because the indices have shifted.
                    break 
                else:
                    # Numerical issue, try rounding?
                    # For now, skip this drop
                    pass

    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = current_U, current_V, current_W

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _build_strassen_36(n, m, p):
    """Build Strassen rank-36 decomposition for 2x4 by 4x5 matrix multiplication."""
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        nonlocal col
        c = col_start
        U[i0 * m + j0, c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[j0 * p + k0, c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col += 1

        c = col_start + 1
        U[(i0 + 1) * m + j0, c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[j0 * p + k0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] -= 1
        col += 1

        c = col_start + 2
        U[i0 * m + j0, c] += 1
        V[j0 * p + (k0 + 1), c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] -= 1
        W[(k0 + 1) * n + i0, c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col += 1

        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1), c] += 1
        V[(j0 + 1) * p + k0, c] += 1
        V[j0 * p + k0, c] -= 1
        W[k0 * n + i0, c] += 1
        W[k0 * n + (i0 + 1), c] += 1
        col += 1

        c = col_start + 4
        U[i0 * m + j0, c] += 1
        U[i0 * m + (j0 + 1), c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] -= 1
        W[(k0 + 1) * n + i0, c] += 1
        col += 1

        c = col_start + 5
        U[(i0 + 1) * m + j0, c] += 1
        U[i0 * m + j0, c] -= 1
        V[j0 * p + k0, c] += 1
        V[j0 * p + (k0 + 1), c] += 1
        W[(k0 + 1) * n + (i0 + 1), c] += 1
        col += 1

        c = col_start + 6
        U[i0 * m + (j0 + 1), c] += 1
        U[(i0 + 1) * m + (j0 + 1), c] -= 1
        V[(j0 + 1) * p + k0, c] += 1
        V[(j0 + 1) * p + (k0 + 1), c] += 1
        W[k0 * n + i0, c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj), c] = 1
                V[(j0 + dj) * p + k0, c] = 1
                W[k0 * n + (i0 + di), c] = 1
                c += 1
        return c

    col = add_strassen_222(0, 0, 0, 0)
    col = add_strassen_222(0, 2, 0, col)
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W
