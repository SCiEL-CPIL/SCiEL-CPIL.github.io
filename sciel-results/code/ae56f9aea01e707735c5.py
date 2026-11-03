import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    deadline = time.time() + 140  # 140s safety margin

    strassen = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]
    
    def emit_block(a_cs, b_rs, b_cs, c_cs, off, U, V, W, rank):
        for s in range(7):
            col = off + s
            if col >= rank:
                continue
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r, cc = li // 2, li % 2
                U[r * m + a_cs + cc][col] += c
            for li, c in vc.items():
                r, cc = li // 2, li % 2
                V[(b_rs + r) * p + b_cs + cc][col] += c
            for li, c in wc.items():
                r, cc = li // 2, li % 2
                W[(c_cs + cc) * n + r][col] += c

    def build_strassen_28():
        rank = 28
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        emit_block(0, 0, 0, 0, 0, U, V, W, rank)
        emit_block(0, 0, 2, 2, 7, U, V, W, rank)
        emit_block(2, 2, 0, 0, 14, U, V, W, rank)
        emit_block(2, 2, 2, 2, 21, U, V, W, rank)
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if s != 1:
                        return False
        return True

    def is_rank1_tensor(T, nm, mp, pn):
        T000 = None
        i0, j0, k0 = None, None, None
        for i in range(nm):
            if T000 is not None:
                break
            for j in range(mp):
                if T000 is not None:
                    break
                for k in range(pn):
                    if T[i, j, k] != 0:
                        T000 = T[i, j, k]
                        i0, j0, k0 = i, j, k
                        break
        if T000 is None:
            return True
        
        # Use numpy for fast verification
        # T[i,j,k] = u[i]v[j]w[k]
        # u[i] = T[i,j0,k0]/T000
        # v[j] = T[i0,j,k0]/T000
        # w[k] = T[i0,j0,k]/T000
        
        u_vec = T[:, j0, k0].astype(float) / T000
        v_vec = T[i0, :, k0].astype(float) / T000
        w_vec = T[i0, j0, :].astype(float) / T000
        
        T_recon = np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
        
        # Check exact match (allowing for float precision if needed, but inputs are int)
        # Since T is integer, T_recon should be very close to integer if rank 1.
        if not np.allclose(T, T_recon, atol=1e-9):
            return False
            
        # Additional check: ensure the "vectors" derived are consistent with the rest of the tensor
        # Actually, if T_recon matches T, it is rank 1.
        return True

    def try_merge_two_terms(U, V, W, n, m, p):
        rank = len(U[0])
        nm, mp, pn = n * m, m * p, p * n
        
        # Pre-convert to numpy arrays for speed
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)

        for a in range(rank):
            if time.time() > deadline - 15:
                break
            for b in range(a + 1, rank):
                ua = U_np[:, a]
                ub = U_np[:, b]
                va = V_np[:, a]
                vb = V_np[:, b]
                wa = W_np[:, a]
                wb = W_np[:, b]

                # Compute sum of two rank-1 tensors
                T = np.einsum('i,j,k->ijk', ua, va, wa) + np.einsum('i,j,k->ijk', ub, vb, wb)

                if is_rank1_tensor(T, nm, mp, pn):
                    # Find pivot
                    i0, j0, k0 = None, None, None
                    T000 = None
                    # Find first non-zero
                    for i in range(nm):
                        if T000 is not None: break
                        for j in range(mp):
                            if T000 is not None: break
                            for k in range(pn):
                                if T[i, j, k] != 0:
                                    T000 = T[i, j, k]
                                    i0, j0, k0 = i, j, k
                                    break
                    if T000 is None:
                        continue

                    # Extract vectors
                    u_prime = T[:, j0, k0].astype(int)
                    v_prime = T[i0, :, k0].astype(int)
                    w_prime = T[i0, j0, :].astype(int)

                    new_rank = rank - 1
                    U_new = [[0] * new_rank for _ in range(nm)]
                    V_new = [[0] * new_rank for _ in range(mp)]
                    W_new = [[0] * new_rank for _ in range(pn)]

                    col_map = [r for r in range(rank) if r != a and r != b]
                    for new_c, old_c in enumerate(col_map):
                        for i in range(nm):
                            U_new[i][new_c] = U[i][old_c]
                        for i in range(mp):
                            V_new[i][new_c] = V[i][old_c]
                        for i in range(pn):
                            W_new[i][new_c] = W[i][old_c]

                    last_c = new_rank - 1
                    for i in range(nm):
                        U_new[i][last_c] = int(u_prime[i])
                    for i in range(mp):
                        V_new[i][last_c] = int(v_prime[i])
                    for i in range(pn):
                        W_new[i][last_c] = int(w_prime[i])

                    if verify_decomposition(U_new, V_new, W_new, n, m, p):
                        return U_new, V_new, W_new

        return None

    U28, V28, W28 = build_strassen_28()

    if not verify_decomposition(U28, V28, W28, n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    best_U, best_V, best_W = U28, V28, W28
    for attempt in range(5):
        if time.time() > deadline - 20:
            break
        merged = try_merge_two_terms(best_U, best_V, best_W, n, m, p)
        if merged is None:
            break
        best_U, best_V, best_W = merged

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
