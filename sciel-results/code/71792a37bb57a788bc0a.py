import time
import numpy as np

START = time.time()
DEADLINE = START + 130.0


def _build_strassen_2x2x2():
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_rank98():
    U2, V2, W2 = _build_strassen_2x2x2()

    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)

    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                for j in range(4):
                    i0, i1 = i // 2, i % 2
                    j0, j1 = j // 2, j % 2
                    U4[i * 4 + j, c] = U2[i0 * 2 + j0, c0] * U2[i1 * 2 + j1, c1]

            for j in range(4):
                for k in range(4):
                    j0, j1 = j // 2, j % 2
                    k0, k1 = k // 2, k % 2
                    V4[j * 4 + k, c] = V2[j0 * 2 + k0, c0] * V2[j1 * 2 + k1, c1]

            for k in range(4):
                for i in range(4):
                    k0, k1 = k // 2, k % 2
                    i0, i1 = i // 2, i % 2
                    W4[k * 4 + i, c] = W2[k0 * 2 + i0, c0] * W2[k1 * 2 + i1, c1]

    n, m, p = 4, 4, 8
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)

    for block in range(2):
        col_start = block * 49
        col_end = (block + 1) * 49

        U[:, col_start:col_end] = U4

        for j in range(4):
            for k_local in range(4):
                k_global = block * 4 + k_local
                V[j * 8 + k_global, col_start:col_end] = V4[j * 4 + k_local, :]

        for k_local in range(4):
            k_global = block * 4 + k_local
            for i in range(4):
                W[k_global * 4 + i, col_start:col_end] = W4[k_local * 4 + i, :]

    # Verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(total_rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-9:
                    return None
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


def _build_rank112():
    n, m, p = 4, 4, 8
    rank = 112
    U = np.zeros((n * m, rank), dtype=np.int16)
    V = np.zeros((m * p, rank), dtype=np.int16)
    W = np.zeros((p * n, rank), dtype=np.int16)

    col = 0

    m_coeffs_list = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]

    n_coeffs_list = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]

    w_coeffs_list = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1)],
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)]
    ]

    for i_b in range(2):
        for k_b in range(4):
            for s in range(7):
                for (i_l, j_l, coeff) in m_coeffs_list[s]:
                    u_idx = (i_b * 2 + i_l) * m + (0 * 2 + j_l)
                    U[u_idx, col] += coeff
                for (j_l, k_l, coeff) in n_coeffs_list[s]:
                    v_idx = (0 * 2 + j_l) * p + (k_b * 2 + k_l)
                    V[v_idx, col] += coeff
                for (i_l, k_l, coeff) in w_coeffs_list[s]:
                    w_idx = (k_b * 2 + k_l) * n + (i_b * 2 + i_l)
                    W[w_idx, col] += coeff
                col += 1

            for s in range(7):
                for (i_l, j_l, coeff) in m_coeffs_list[s]:
                    u_idx = (i_b * 2 + i_l) * m + (1 * 2 + j_l)
                    U[u_idx, col] += coeff
                for (j_l, k_l, coeff) in n_coeffs_list[s]:
                    v_idx = (1 * 2 + j_l) * p + (k_b * 2 + k_l)
                    V[v_idx, col] += coeff
                for (i_l, k_l, coeff) in w_coeffs_list[s]:
                    w_idx = (k_b * 2 + k_l) * n + (i_b * 2 + i_l)
                    W[w_idx, col] += coeff
                col += 1

    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-9:
                    return None
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


def _schoolbook(n=4, m=4, p=8):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_optimize(U0, V0, W0, n, m, p, time_budget):
    """Try to reduce rank using iterative coordinate descent with sparsification."""
    try:
        import scipy.optimize as opt

        U = U0.astype(np.float64).copy()
        V = V0.astype(np.float64).copy()
        W = W0.astype(np.float64).copy()
        rank = U.shape[1]

        def compute_residual():
            # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Residual is T - target (where target has 1s at diagonal positions)
            # But for optimization, we just want the Frobenius norm of T
            # Actually, the constraint is that T[i,j,k] = 1 if (i,j,k) is a "valid" index
            # In matrix mult, the "target" tensor has T[i,j,k] = 1 for all i,j,k
            # Wait, no. Let me re-examine.
            
            # The matrix multiplication tensor: C = A * B where A is n x m, B is m x p, C is n x p
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor T[i,j,k] (with i in [0,n), j in [0,m), k in [0,p)) has T[i,j,k] = 1 for all i,j,k
            # because each triple (i,j,k) contributes to C[i,k]
            
            # Actually, the standard formulation: the matrix multiplication tensor has
            # T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
            # Wait, that's not right either. Let me think again.
            
            # In the rank decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
            # For matrix multiplication C = AB: C[i,k] = sum_j A[i,j] B[j,k]
            # So T[i,j,k] = 1 for all i,j,k.
            # The verification in _build_rank98 checks that sum_r ... = 1.0 for all i,j,k.
            # So the target tensor is all-ones.
            
            T = np.zeros((n, m, p))
            for r in range(rank):
                T += np.einsum('ij,jk,ki->ijk', U[:, r], V[:, r], W[:, r])
            return T

        def objective(params):
            U_flat = params[0:U.shape[0] * rank].reshape(U.shape)
            V_flat = params[U.shape[0] * rank:U.shape[0] * rank + V.shape[0] * rank].reshape(V.shape)
            W_flat = params[U.shape[0] * rank + V.shape[0] * rank:].reshape(W.shape)
            T = np.einsum('ir,js,kt->ijk', U_flat, V_flat, W_flat)
            target = np.ones((n, m, p))
            return np.sum((T - target) ** 2)

        # Check current residual
        T = np.einsum('ir,js,kt->ijk', U, V, W)
        target = np.ones((n, m, p))
        init_err = np.sum((T - target) ** 2)
        if init_err > 1e-6:
            return None

        # Try iterative sparsification: zero out smallest entries
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank

        for iteration in range(50):
            if time.time() > START + time_budget:
                break
            
            # Find smallest magnitude entries across all three matrices
            all_entries = []
            for i in range(U.shape[0]):
                for r in range(rank):
                    if abs(U[i, r]) > 1e-10:
                        all_entries.append(('U', i, r, abs(U[i, r])))
            for j in range(V.shape[0]):
                for r in range(rank):
                    if abs(V[j, r]) > 1e-10:
                        all_entries.append(('V', j, r, abs(V[j, r])))
            for k in range(W.shape[0]):
                for r in range(rank):
                    if abs(W[k, r]) > 1e-10:
                        all_entries.append(('W', k, r, abs(W[k, r])))
            
            if not all_entries:
                break
            
            all_entries.sort(key=lambda x: x[3])
            
            # Try zeroing out the smallest entry
            trial_matrix, trial_i, trial_r, _ = all_entries[0]
            
            if trial_matrix == 'U':
                U[trial_i, trial_r] = 0
            elif trial_matrix == 'V':
                V[trial_i, trial_r] = 0
            else:
                W[trial_i, trial_r] = 0
            
            # Check if still valid
            T = np.einsum('ir,js,kt->ijk', U, V, W)
            err = np.sum((T - target) ** 2)
            
            if err > 1e-6:
                # Undo
                if trial_matrix == 'U':
                    U[trial_i, trial_r] = best_U[trial_i, trial_r]
                elif trial_matrix == 'V':
                    V[trial_i, trial_r] = best_V[trial_i, trial_r]
                else:
                    W[trial_i, trial_r] = best_W[trial_i, trial_r]
                break
            else:
                best_U[trial_i, trial_r] = 0
                best_V[trial_i, trial_r] = V[trial_i, trial_r]
                best_W[trial_i, trial_r] = W[trial_i, trial_r]
        
        # Now try to reduce rank by identifying zero columns
        # Check which columns are all zero
        zero_cols = []
        for r in range(rank):
            if np.allclose(U[:, r], 0) and np.allclose(V[:, r], 0) and np.allclose(W[:, r], 0):
                zero_cols.append(r)
        
        # Also check if a column has only one matrix non-zero (can't remove)
        # Actually, for the tensor decomposition, a column r contributes U[:,r] * V[:,r] * W[:,r]
        # If any of these is all zero, the column contributes nothing
        removable_cols = []
        for r in range(rank):
            if np.allclose(U[:, r], 0) or np.allclose(V[:, r], 0) or np.allclose(W[:, r], 0):
                removable_cols.append(r)
        
        if removable_cols:
            keep_cols = [r for r in range(rank) if r not in removable_cols]
            new_rank = len(keep_cols)
            if new_rank < best_rank:
                best_rank = new_rank
                U = U[:, keep_cols]
                V = V[:, keep_cols]
                W = W[:, keep_cols]
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
        
        # Verify final result
        T = np.einsum('ir,js,kt->ijk', best_U, best_V, best_W)
        err = np.sum((T - target) ** 2)
        if err > 1e-6:
            return None
        
        # Round to integers if possible
        U_rounded = np.round(best_U)
        V_rounded = np.round(best_V)
        W_rounded = np.round(best_W)
        
        T_check = np.einsum('ir,js,kt->ijk', U_rounded, V_rounded, W_rounded)
        err_check = np.sum((T_check - target) ** 2)
        
        if err_check < 1e-6:
            return {
                "U": U_rounded.astype(np.int16).tolist(),
                "V": V_rounded.astype(np.int16).tolist(),
                "W": W_rounded.astype(np.int16).tolist()
            }
        
        # If rounding fails, use original
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    except Exception:
        return None


def construct():
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                # Try to improve
                if time.time() < START + 120:
                    U98 = np.array(res['U'], dtype=np.float64)
                    V98 = np.array(res['V'], dtype=np.float64)
                    W98 = np.array(res['W'], dtype=np.float64)
                    improved = _try_optimize(U98, V98, W98, 4, 4, 8, 115)
                    if improved is not None:
                        return improved
                return res
    except Exception:
        pass

    try:
        if time.time() < DEADLINE:
            res = _build_rank112()
            if res is not None:
                return res
    except Exception:
        pass

    return _schoolbook()


if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    U, V, W = res['U'], res['V'], res['W']
    n, m, p = 4, 4, 8
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(len(U[0])):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
