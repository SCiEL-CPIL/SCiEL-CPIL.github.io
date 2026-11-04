"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy:
  1. Build a rank-77 decomposition using top-level Strassen (7 terms) where each
     term is a (2x2) x (2x3) product. Each such product uses inner Strassen for
     the 2x2x2 part (7 terms) plus schoolbook for the 2x2x1 part (4 terms) = 11.
     Total rank: 7 * 11 = 77.
  2. Try to reduce rank via structured column-merging and alternating minimization.
  3. Fallback to the rank-77 construction if no improvement is found.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# Strassen 2x2*2x2 coefficients
alpha = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: A00 + A11
    {0: 0, 1: 0, 2: 1, 3: 1},   # M2: A10 + A11
    {0: 1, 1: 0, 2: 0, 3: 0},   # M3: A00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M4: A11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M5: A00 + A01
    {0: -1, 1: 0, 2: 1, 3: 0},  # M6: A10 - A00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M7: A01 - A11
]
beta = [
    {0: 1, 1: 0, 2: 0, 3: 1},   # M1: B00 + B11
    {0: 1, 1: 0, 2: 0, 3: 0},   # M2: B00
    {0: 0, 1: 1, 2: 0, 3: -1},  # M3: B01 - B11
    {0: -1, 1: 0, 2: 1, 3: 0},  # M4: B10 - B00
    {0: 0, 1: 0, 2: 0, 3: 1},   # M5: B11
    {0: 1, 1: 1, 2: 0, 3: 0},   # M6: B00 + B01
    {0: 0, 1: 0, 2: 1, 3: 1},   # M7: B10 + B11
]
recon = [
    [(0, 0, 1), (1, 1, 1)],       # M1 -> C00(+1), C11(+1)
    [(1, 0, 1), (1, 1, -1)],      # M2 -> C10(+1), C11(-1)
    [(0, 1, 1), (1, 1, 1)],       # M3 -> C01(+1), C11(+1)
    [(0, 0, 1), (1, 0, 1)],       # M4 -> C00(+1), C10(+1)
    [(0, 0, -1), (0, 1, 1)],      # M5 -> C00(-1), C01(+1)
    [(1, 1, 1)],                  # M6 -> C11(+1)
    [(0, 0, 1)],                  # M7 -> C00(+1)
]


def _build_rank77():
    """Build the rank-77 decomposition using top-level Strassen with 2x2x2+2x2x1 inner."""
    n, m, p = DIMENSIONS  # 4, 4, 6
    R = 77
    U = np.zeros((n * m, R))  # 16 x 77
    V = np.zeros((m * p, R))  # 24 x 77
    W = np.zeros((p * n, R))  # 24 x 77
    
    col = 0
    for s in range(7):
        # For each top-level Strassen term s:
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            for a in range(4):
                if alpha[si].get(a, 0) != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s].get(r, 0) != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * m + (br * 2 + x_col)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            for b in range(4):
                if beta[si].get(b, 0) != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s].get(c, 0) != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * p + (bc * 3 + y_inner_col)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in recon[si]:
                    w_idx = (b * 3 + zb) * n + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s].get(r, 0) != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * m + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                for c in range(4):
                    if beta[s].get(c, 0) != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * p + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * n + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    assert col == R, f"Column mismatch: {col} != {R}"
    return U, V, W


def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _err_of(U, V, W, T):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.sum((T - recon) ** 2)


def _check(U, V, W, T, atol=1e-9):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=atol)


def _try_reduce_rank(U, V, W, T, target_rank, deadline):
    """Try to find a valid decomposition with target_rank columns using structured merging."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for restart in range(8):
        if time.time() > deadline - 15:
            break
        
        # Strategy: keep target_rank columns, zero out others, then optimize
        # Use a greedy approach: start from full rank, try removing columns
        # one at a time and see if we can still represent the tensor
        
        # Initialize with random subset of columns
        np.random.seed(restart * 42)
        cols = np.random.choice(R, size=target_rank, replace=False)
        cols = np.sort(cols)
        
        U_r = np.zeros((n * m, target_rank))
        V_r = np.zeros((m * p, target_rank))
        W_r = np.zeros((p * n, target_rank))
        
        for i, c in enumerate(cols):
            U_r[:, i] = U[:, c]
            V_r[:, i] = V[:, c]
            W_r[:, i] = W[:, c]
        
        # Add small perturbation to escape local minima
        U_r += np.random.randn(n * m, target_rank) * 0.1
        V_r += np.random.randn(m * p, target_rank) * 0.1
        W_r += np.random.randn(p * n, target_rank) * 0.1
        
        err = _err_of(U_r, V_r, W_r, T)
        
        # Alternating minimization with adaptive learning rate
        for it in range(500):
            if time.time() > deadline - 10:
                break
            if err < 1e-12:
                break
            
            recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
            diff = T - recon
            
            # Compute gradients
            grad_U = -2.0 * np.einsum('abc,br,cr->ar', diff, V_r, W_r)
            grad_V = -2.0 * np.einsum('abc,ar,cr->br', diff, U_r, W_r)
            grad_W = -2.0 * np.einsum('abc,ar,br->cr', diff, U_r, V_r)
            
            # Adaptive learning rate based on gradient magnitude
            lr = 0.05 / (1.0 + np.linalg.norm(grad_U, ord='fro') / (n * m))
            
            U_new = U_r - lr * grad_U
            V_new = V_r - lr * grad_V
            W_new = W_r - lr * grad_W
            
            new_err = _err_of(U_new, V_new, W_new, T)
            if new_err < err:
                U_r, V_r, W_r = U_new, V_new, W_new
                err = new_err
            else:
                # Try smaller step
                lr *= 0.5
                U_new = U_r - lr * grad_U
                V_new = V_r - lr * grad_V
                W_new = W_r - lr * grad_W
                new_err = _err_of(U_new, V_new, W_new, T)
                if new_err < err:
                    U_r, V_r, W_r = U_new, V_new, W_new
                    err = new_err
                else:
                    break
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-10:
        return best_U, best_V, best_W, True
    return None, None, None, False


def _try_structured_merge(U, V, W, T, deadline):
    """Try to reduce rank by merging near-dependent columns."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Identify pairs of columns that are nearly linearly dependent
    # For each pair, try to express one as a combination of the other and adjust
    for merge_round in range(5):
        if time.time() > deadline - 20:
            break
        
        current_rank = best_U.shape[1]
        if current_rank <= 50:
            break
        
        # Compute column norms and find near-dependent pairs
        U_norms = np.linalg.norm(best_U, axis=0)
        V_norms = np.linalg.norm(best_V, axis=0)
        W_norms = np.linalg.norm(best_W, axis=0)
        
        # Find pairs with small cosine similarity
        U_normed = best_U / (U_norms[np.newaxis, :] + 1e-10)
        sim_matrix = U_normed.T @ U_normed
        
        # Find the most similar pair (excluding diagonal)
        np.fill_diagonal(sim_matrix, -2)
        max_sim_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
        max_sim = sim_matrix[max_sim_idx]
        
        if max_sim < 0.95:
            break  # No near-dependent pairs
        
        i, j = max_sim_idx
        
        # Try to merge column j into column i
        # Express col_j = alpha * col_i + residual
        # Then adjust other columns to compensate
        
        U_i = best_U[:, i]
        U_j = best_U[:, j]
        
        # Least squares: U_j ≈ alpha * U_i
        alpha = np.dot(U_j, U_i) / (np.dot(U_i, U_i) + 1e-10)
        
        V_i = best_V[:, i]
        V_j = best_V[:, j]
        W_i = best_W[:, i]
        W_j = best_W[:, j]
        
        # After merging: new_col_i = U_i, and we need to adjust
        # The contribution of col_j is: U_j ⊗ V_j ⊗ W_j
        # We want to replace this with adjustments to other columns
        
        # For simplicity, try removing column j and see if error is acceptable
        new_rank = current_rank - 1
        keep_cols = [c for c in range(current_rank) if c != j]
        
        U_new = best_U[:, keep_cols].copy()
        V_new = best_V[:, keep_cols].copy()
        W_new = best_W[:, keep_cols].copy()
        
        # Add the merged contribution to column i
        U_new[:, i] += alpha * U_i  # This is approximate
        
        err = _err_of(U_new, V_new, W_new, T)
        
        if err < 1e-6:
            # Success! Try to polish
            for it in range(100):
                if time.time() > deadline - 15:
                    break
                recon = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                diff = T - recon
                grad_U = -2.0 * np.einsum('abc,br,cr->ar', diff, V_new, W_new)
                grad_V = -2.0 * np.einsum('abc,ar,cr->br', diff, U_new, W_new)
                grad_W = -2.0 * np.einsum('abc,ar,br->cr', diff, U_new, V_new)
                
                lr = 0.01
                U_new -= lr * grad_U
                V_new -= lr * grad_V
                W_new -= lr * grad_W
                
                new_err = _err_of(U_new, V_new, W_new, T)
                if new_err < err:
                    err = new_err
                else:
                    break
            
            if err < 1e-10:
                best_rank = new_rank
                best_U, best_V, best_W = U_new, V_new, W_new
            else:
                # Roll back
                pass
        else:
            break
    
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    deadline = time.time() + 115  # safety margin
    
    T = _build_target_tensor()
    
    # Build rank-77 decomposition
    U77, V77, W77 = _build_rank77()
    
    # Verify
    assert _check(U77, V77, W77, T, atol=1e-6), "Rank-77 verification failed"
    
    best_rank = 77
    best_U, best_V, best_W = U77.copy(), V77.copy(), W77.copy()
    
    # Try structured merging first
    if time.time() < deadline - 30:
        U_merged, V_merged, W_merged, merged_rank = _try_structured_merge(U77, V77, W77, T, deadline)
        if merged_rank < best_rank:
            best_rank = merged_rank
            best_U, best_V, best_W = U_merged, V_merged, W_merged
    
    # Try to reduce rank via alternating minimization
    for target_rank in range(min(best_rank - 1, 76), 50, -1):
        if time.time() > deadline - 20:
            break
        
        U_new, V_new, W_new, success = _try_reduce_rank(best_U, best_V, best_W, T, target_rank, deadline)
        
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U_new, V_new, W_new
            break
    
    # Final polish: ensure the solution is as accurate as possible
    if best_rank < 77:
        for it in range(200):
            if time.time() > deadline - 5:
                break
            recon = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
            diff = T - recon
            err = np.sum(diff ** 2)
            if err < 1e-14:
                break
            
            grad_U = -2.0 * np.einsum('abc,br,cr->ar', diff, best_V, best_W)
            grad_V = -2.0 * np.einsum('abc,ar,cr->br', diff, best_U, best_W)
            grad_W = -2.0 * np.einsum('abc,ar,br->cr', diff, best_U, best_V)
            
            lr = 0.005
            best_U -= lr * grad_U
            best_V -= lr * grad_V
            best_W -= lr * grad_W
        
        # Verify final result
        if not _check(best_U, best_V, best_W, T, atol=1e-8):
            # Fall back to rank-77
            best_rank = 77
            best_U, best_V, best_W = U77.copy(), V77.copy(), W77.copy()
    
    # Return the best result
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
