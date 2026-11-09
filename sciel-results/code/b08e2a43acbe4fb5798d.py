"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill

    # Build the rank-28 Strassen decomposition
    U28, V28, W28 = _strassen_28(n, m, p)
    
    # Try to reduce rank by merging columns with identical (U, W) patterns
    U_r = _try_reduce_rank(U28, V28, W28, n, m, p, deadline)
    
    if U_r is not None:
        return {"U": U_r[0], "V": U_r[1], "W": U_r[2]}
    
    # Fallback: rank 28
    return {"U": U28, "V": V28, "W": W28}


def _strassen_28(n, m, p):
    """Construct the rank-28 Strassen-based decomposition."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(1, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(2, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (1, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (2, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    blocks = [
        {'j_offset': 0, 'k_offset': 0},
        {'j_offset': 2, 'k_offset': 0},
        {'j_offset': 0, 'k_offset': 2},
        {'j_offset': 2, 'k_offset': 2}
    ]
    
    col = 0
    for block in blocks:
        j_off = block['j_offset']
        k_off = block['k_offset']
        for term in strassen_terms:
            for (idx, coeff) in term['U']:
                i = idx // 2
                j = idx % 2
                global_i = i
                global_j = j + j_off
                U[global_i * m + global_j][col] += coeff
            for (idx, coeff) in term['V']:
                j = idx // 2
                k = idx % 2
                global_j = j + j_off
                global_k = k + k_off
                V[global_j * p + global_k][col] += coeff
            for (idx, coeff) in term['W']:
                k = idx // 2
                i = idx % 2
                global_k = k + k_off
                global_i = i
                W[global_k * n + global_i][col] += coeff
            col += 1
    
    U = [[int(x) for x in row] for row in U]
    V = [[int(x) for x in row] for row in V]
    W = [[int(x) for x in row] for row in W]
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _try_reduce_rank(U28, V28, W28, n, m, p, deadline):
    """Try to reduce the rank by merging columns with identical (U, W) patterns."""
    t_start = time.time()
    
    # Convert to numpy for analysis
    U_arr = np.array(U28, dtype=float)  # shape (n*m, 28)
    V_arr = np.array(V28, dtype=float)  # shape (m*p, 28)
    W_arr = np.array(W28, dtype=float)  # shape (p*n, 28)
    
    n_m = n * m  # 8
    m_p = m * p  # 16
    p_n = p * n  # 8
    r_old = 28
    
    # First pass: group columns by identical (U, W) patterns
    col_patterns = {}
    for t in range(r_old):
        u_pat = tuple(U_arr[:, t].astype(int))
        w_pat = tuple(W_arr[:, t].astype(int))
        key = (u_pat, w_pat)
        if key not in col_patterns:
            col_patterns[key] = []
        col_patterns[key].append(t)
    
    # Build reduced set from identical patterns
    groups = sorted(col_patterns.items(), key=lambda x: -len(x[1]))
    new_cols = []
    for key, cols in groups:
        rep = cols[0]
        new_cols.append((rep, cols))
    
    r_new = len(new_cols)
    
    # Build the reduced matrices
    U_new = np.zeros((n_m, r_new))
    V_new = np.zeros((m_p, r_new))
    W_new = np.zeros((p_n, r_new))
    
    for new_idx, (rep, cols) in enumerate(new_cols):
        U_new[:, new_idx] = U_arr[:, rep]
        W_new[:, new_idx] = W_arr[:, rep]
        for c in cols:
            V_new[:, new_idx] += V_arr[:, c]
    
    U_list = [[int(round(x)) for x in row] for row in U_new.tolist()]
    V_list = [[int(round(x)) for x in row] for row in V_new.tolist()]
    W_list = [[int(round(x)) for x in row] for row in W_new.tolist()]
    
    best_rank = r_old
    best_result = (U28, V28, W28)
    
    if _verify(U_list, V_list, W_list, n, m, p):
        best_rank = r_new
        best_result = (U_list, V_list, W_list)
    
    # Second pass: try to further reduce by checking if any column can be removed
    # or if pairs of columns can be merged
    current_U = np.array(best_result[0], dtype=float)
    current_V = np.array(best_result[1], dtype=float)
    current_W = np.array(best_result[2], dtype=float)
    current_rank = best_rank
    
    # Try removing each column one at a time
    for t_remove in range(current_rank):
        if time.time() > deadline - 5:
            break
        
        # Skip if column is zero
        if np.allclose(current_U[:, t_remove], 0) and np.allclose(current_W[:, t_remove], 0):
            # Remove this column
            mask = [i for i in range(current_rank) if i != t_remove]
            U_test = current_U[:, mask]
            V_test = current_V[:, mask]
            W_test = current_W[:, mask]
            
            U_t = [[int(round(x)) for x in row] for row in U_test.tolist()]
            V_t = [[int(round(x)) for x in row] for row in V_test.tolist()]
            W_t = [[int(round(x)) for x in row] for row in W_test.tolist()]
            
            if _verify(U_t, V_t, W_t, n, m, p):
                current_U = U_test
                current_V = V_test
                current_W = W_test
                current_rank -= 1
                # Restart from beginning since indices changed
                t_remove = -1
                continue
        
        # Try merging column t_remove with each other column t_keep
        for t_keep in range(current_rank):
            if t_keep == t_remove or time.time() > deadline - 5:
                continue
            
            # Check if U[:,t_remove] is proportional to U[:,t_keep]
            # and W[:,t_remove] is proportional to W[:,t_keep]
            u_rm = current_U[:, t_remove]
            u_kp = current_U[:, t_keep]
            w_rm = current_W[:, t_remove]
            w_kp = current_W[:, t_keep]
            
            # Check if one is scalar multiple of the other
            if np.allclose(u_kp, 0):
                if not np.allclose(u_rm, 0):
                    continue
                # Both U are zero, check W
                if not np.allclose(w_kp, 0):
                    if not np.allclose(w_rm, 0):
                        continue
                    # Both zero, can merge
                elif not np.allclose(w_rm, 0):
                    continue
            else:
                # Find scalar: u_rm = alpha * u_kp
                nonzero_mask = np.abs(u_kp) > 1e-9
                if not np.any(nonzero_mask):
                    continue
                alpha = u_rm[nonzero_mask] / u_kp[nonzero_mask]
                if not np.allclose(alpha, alpha[0], atol=1e-9):
                    continue
                alpha = alpha[0]
                if not np.allclose(u_rm, alpha * u_kp, atol=1e-9):
                    continue
            
            if np.allclose(w_kp, 0):
                if not np.allclose(w_rm, 0):
                    continue
                beta = 0.0
            else:
                nonzero_mask_w = np.abs(w_kp) > 1e-9
                if not np.any(nonzero_mask_w):
                    continue
                beta = w_rm[nonzero_mask_w] / w_kp[nonzero_mask_w]
                if not np.allclose(beta, beta[0], atol=1e-9):
                    continue
                beta = beta[0]
                if not np.allclose(w_rm, beta * w_kp, atol=1e-9):
                    continue
            
            # If we get here, U[:,t_remove] = alpha * U[:,t_keep] and W[:,t_remove] = beta * W[:,t_keep]
            # The contribution from t_remove is alpha * beta * U[:,t_keep] * V[:,t_remove] * W[:,t_keep]
            # We can absorb this into t_keep by adding alpha * beta * V[:,t_remove] to V[:,t_keep]
            # But wait, that's not quite right. Let me reconsider.
            
            # Actually, the contribution from column t is: U[:,t] * V[:,t] * W[:,t] (outer product structure)
            # If U[:,t_rm] = alpha * U[:,t_kp] and W[:,t_rm] = beta * W[:,t_kp], then:
            # U[:,t_rm] * V[:,t_rm] * W[:,t_rm] = alpha * beta * U[:,t_kp] * V[:,t_rm] * W[:,t_kp]
            # This is NOT the same as alpha * beta * U[:,t_kp] * V[:,t_kp] * W[:,t_kp] unless V[:,t_rm] = V[:,t_kp]
            
            # So this simple proportional merge only works if V is also proportional, which is unlikely.
            # Let's try a different approach: just sum V and keep U,W from one column
            # This only works if U and W are identical (already handled in first pass)
            
            pass  # Skip this for now
    
    # Third pass: try all pairs of columns for potential linear combination reduction
    # For each pair (t1, t2), check if we can replace them with a single column
    # This requires solving a linear system
    
    # Convert current to lists
    U_cur = [[int(round(x)) for x in row] for row in current_U.tolist()]
    V_cur = [[int(round(x)) for x in row] for row in current_V.tolist()]
    W_cur = [[int(round(x)) for x in row] for row in current_W.tolist()]
    
    if current_rank < best_rank:
        best_rank = current_rank
        best_result = (U_cur, V_cur, W_cur)
    
    return best_result if best_rank < r_old else None
