import time
import numpy as np
from scipy.optimize import minimize

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
    if not _check_valid_fast(U, V, W, n, m, p):
        return None
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _compute_tensor_fast(U, V, W, n, m, p):
    T = np.zeros((n, m, p))
    for r in range(U.shape[1]):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', u, v, w)
    return T

def _check_valid_fast(U, V, W, n, m, p, tol=1e-9):
    T = _compute_tensor_fast(U, V, W, n, m, p)
    return np.allclose(T, 1.0, atol=tol)

def _try_optimize(U0, V0, W0, n, m, p, time_budget):
    try:
        U = U0.astype(np.float64).copy()
        V = V0.astype(np.float64).copy()
        W = W0.astype(np.float64).copy()
        rank = U.shape[1]
        
        if not _check_valid_fast(U, V, W, n, m, p):
            return None

        best_rank = rank
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()

        # Phase 1: Greedy sparsification
        max_iterations = 500 # Increased due to faster check
        for iteration in range(max_iterations):
            if time.time() > START + time_budget:
                break

            # Find smallest non-zero entry
            min_val = float('inf')
            min_loc = None

            # Vectorized search for min non-zero? 
            # U, V, W are small enough for loops, but let's try to be efficient.
            # Actually, just flattening and checking might be faster than nested loops in Python.
            # But we need the location.
            
            # Let's stick to loops for clarity and low overhead, but break early if found small enough?
            # No, we need the global minimum.
            
            for mat, name in [(U, 'U'), (V, 'V'), (W, 'W')]:
                # Get non-zero values
                nz = mat[mat != 0]
                if len(nz) > 0:
                    local_min = np.min(np.abs(nz))
                    if local_min < min_val:
                        min_val = local_min
                        # Find location
                        idx = np.argwhere(mat == nz[np.argmin(np.abs(nz))])
                        # There might be multiple, take first
                        min_loc = (name, idx[0, 0], idx[0, 1])

            if min_loc is None or min_val > 1e-10:
                break

            mat_name, idx, col_idx = min_loc
            saved_val = 0.0
            if mat_name == 'U':
                saved_val = U[idx, col_idx]
                U[idx, col_idx] = 0
            elif mat_name == 'V':
                saved_val = V[idx, col_idx]
                V[idx, col_idx] = 0
            else:
                saved_val = W[idx, col_idx]
                W[idx, col_idx] = 0

            if _check_valid_fast(U, V, W, n, m, p):
                pass
            else:
                if mat_name == 'U':
                    U[idx, col_idx] = saved_val
                elif mat_name == 'V':
                    V[idx, col_idx] = saved_val
                else:
                    W[idx, col_idx] = saved_val
                break

        # Phase 2: Remove zero columns
        removable = []
        for r in range(rank):
            if np.allclose(U[:, r], 0, atol=1e-10) or \
               np.allclose(V[:, r], 0, atol=1e-10) or \
               np.allclose(W[:, r], 0, atol=1e-10):
                removable.append(r)

        if removable:
            keep = [r for r in range(rank) if r not in removable]
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            rank = len(keep)

        if rank < best_rank:
            best_rank = rank
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()

        # Phase 3: L1 Optimization (Optional, might be slow)
        if time.time() < START + time_budget - 10:
            try:
                # Only attempt if we have time
                def l1_objective(params):
                    U_f = params[:U.shape[0] * rank].reshape(U.shape)
                    V_f = params[U.shape[0] * rank:U.shape[0] * rank + V.shape[0] * rank].reshape(V.shape)
                    W_f = params[U.shape[0] * rank + V.shape[0] * rank:].reshape(W.shape)
                    l1 = np.sum(np.abs(U_f)) + np.sum(np.abs(V_f)) + np.sum(np.abs(W_f))
                    T = _compute_tensor_fast(U_f, V_f, W_f, n, m, p)
                    penalty = np.sum((T - 1.0) ** 2)
                    return l1 + 100.0 * penalty

                x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                res = minimize(l1_objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-10})
                
                if res.success:
                    U_new = res.x[:U.shape[0] * rank].reshape(U.shape)
                    V_new = res.x[U.shape[0] * rank:U.shape[0] * rank + V.shape[0] * rank].reshape(V.shape)
                    W_new = res.x[U.shape[0] * rank + V.shape[0] * rank:].reshape(W.shape)

                    if _check_valid_fast(U_new, V_new, W_new, n, m, p):
                        # Sparsify again
                        for _ in range(50):
                            if time.time() > START + time_budget - 2:
                                break
                            min_val = float('inf')
                            min_loc = None
                            for mat, name in [(U_new, 'U'), (V_new, 'V'), (W_new, 'W')]:
                                nz = mat[mat != 0]
                                if len(nz) > 0:
                                    local_min = np.min(np.abs(nz))
                                    if local_min < min_val:
                                        min_val = local_min
                                        idx = np.argwhere(mat == nz[np.argmin(np.abs(nz))])
                                        min_loc = (name, idx[0, 0], idx[0, 1])
                            if min_loc is None or min_val > 1e-8:
                                break
                            mat_name, idx, col_idx = min_loc
                            saved_val = 0.0
                            if mat_name == 'U':
                                saved_val = U_new[idx, col_idx]
                                U_new[idx, col_idx] = 0
                            elif mat_name == 'V':
                                saved_val = V_new[idx, col_idx]
                                V_new[idx, col_idx] = 0
                            else:
                                saved_val = W_new[idx, col_idx]
                                W_new[idx, col_idx] = 0
                            if _check_valid_fast(U_new, V_new, W_new, n, m, p):
                                pass
                            else:
                                if mat_name == 'U':
                                    U_new[idx, col_idx] = saved_val
                                elif mat_name == 'V':
                                    V_new[idx, col_idx] = saved_val
                                else:
                                    W_new[idx, col_idx] = saved_val
                                break

                        removable = []
                        for r in range(rank):
                            if np.allclose(U_new[:, r], 0, atol=1e-8) or \
                               np.allclose(V_new[:, r], 0, atol=1e-8) or \
                               np.allclose(W_new[:, r], 0, atol=1e-8):
                                removable.append(r)
                        if removable:
                            keep = [r for r in range(rank) if r not in removable]
                            U_new = U_new[:, keep]
                            V_new = V_new[:, keep]
                            W_new = W_new[:, keep]
                            new_rank = len(keep)
                            if new_rank < best_rank and _check_valid_fast(U_new, V_new, W_new, n, m, p):
                                best_rank = new_rank
                                best_U = U_new.copy()
                                best_V = V_new.copy()
                                best_W = W_new.copy()
            except Exception:
                pass

        if not _check_valid_fast(best_U, best_V, best_W, n, m, p):
            return None

        U_rounded = np.round(best_U)
        V_rounded = np.round(best_V)
        W_rounded = np.round(best_W)

        if _check_valid_fast(U_rounded, V_rounded, W_rounded, n, m, p):
            return {
                "U": U_rounded.astype(np.int16).tolist(),
                "V": V_rounded.astype(np.int16).tolist(),
                "W": W_rounded.astype(np.int16).tolist()
            }

        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    except Exception:
        return None

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

def construct():
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                if time.time() < START + 118:
                    U98 = np.array(res['U'], dtype=np.float64)
                    V98 = np.array(res['V'], dtype=np.float64)
                    W98 = np.array(res['W'], dtype=np.float64)
                    improved = _try_optimize(U98, V98, W98, 4, 4, 8, 115)
                    if improved is not None:
                        return improved
                return res
    except Exception:
        pass

    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
