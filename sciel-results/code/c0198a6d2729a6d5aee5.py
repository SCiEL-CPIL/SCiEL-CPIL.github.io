"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6

    # Build the rank-42 Strassen decomposition
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42

    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        """
        a_idx: [a00, a01, a10, a11] indices into U
        b_idx: [b00, b01, b10, b11] indices into V
        w_idx: [w00, w01, w10, w11] indices into W for the 2x2 output block
        """
        c = start_col
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][c] += 1; U[a_idx[3]][c] += 1
        V[b_idx[0]][c] += 1; V[b_idx[3]][c] += 1
        W[w_idx[0]][c] += 1; W[w_idx[3]][c] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][c+1] += 1; U[a_idx[3]][c+1] += 1
        V[b_idx[0]][c+1] += 1
        W[w_idx[2]][c+1] += 1; W[w_idx[3]][c+1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][c+2] += 1
        V[b_idx[1]][c+2] += 1; V[b_idx[3]][c+2] -= 1
        W[w_idx[1]][c+2] += 1; W[w_idx[3]][c+2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][c+3] += 1
        V[b_idx[2]][c+3] += 1; V[b_idx[0]][c+3] -= 1
        W[w_idx[0]][c+3] += 1; W[w_idx[2]][c+3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][c+4] += 1; U[a_idx[1]][c+4] += 1
        V[b_idx[3]][c+4] += 1
        W[w_idx[0]][c+4] -= 1; W[w_idx[1]][c+4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][c+5] += 1; U[a_idx[0]][c+5] -= 1
        V[b_idx[0]][c+5] += 1; V[b_idx[1]][c+5] += 1
        W[w_idx[3]][c+5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][c+6] += 1; U[a_idx[3]][c+6] -= 1
        V[b_idx[2]][c+6] += 1; V[b_idx[3]][c+6] += 1
        W[w_idx[0]][c+6] += 1

    # A is 2x4: indices i*m+j
    a_L = [0, 1, 4, 5]  # a00, a01, a10, a11
    a_R = [2, 3, 6, 7]

    # B is 4x6: indices j*p+k
    # B col block 0: cols 0,1
    b_1_T = [0, 1, 6, 7]
    b_1_B = [12, 13, 18, 19]
    w_1 = [0, 2, 1, 3]

    add_strassen_block(a_L, b_1_T, w_1, 0)
    add_strassen_block(a_R, b_1_B, w_1, 7)

    # B col block 1: cols 2,3
    b_2_T = [2, 3, 8, 9]
    b_2_B = [14, 15, 20, 21]
    w_2 = [4, 6, 5, 7]

    add_strassen_block(a_L, b_2_T, w_2, 14)
    add_strassen_block(a_R, b_2_B, w_2, 21)

    # B col block 2: cols 4,5
    b_3_T = [4, 5, 10, 11]
    b_3_B = [16, 17, 22, 23]
    w_3 = [8, 10, 9, 11]

    add_strassen_block(a_L, b_3_T, w_3, 28)
    add_strassen_block(a_R, b_3_B, w_3, 35)

    def verify_decomp(U, V, W, rank):
        """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all (i,j,k)"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if total != 1:
                        return False
        return True

    if not verify_decomp(U, V, W, rank):
        # Fallback: return schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m+j][r] = 1
                    V[j*p+k][r] = 1
                    W[k*n+i][r] = 1
        return {"U": U, "V": V, "W": W}

    # Try to reduce rank by finding linear dependencies among the 42 rank-1 terms
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Build matrix where column r is the flattened u_r ⊗ v_r ⊗ w_r
    flat_terms = []
    for r in range(rank):
        term = np.outer(np.outer(U_np[:, r], V_np[:, r]), W_np[:, r])
        flat_terms.append(term.flatten())

    M = np.array(flat_terms).T  # (2304, 42)

    # Check rank of M - if < 42, there are linear dependencies
    try:
        U_svd, s_vals, Vt_svd = np.linalg.svd(M, full_matrices=False)
        tol = 1e-10
        effective_rank = np.sum(s_vals > tol)

        if effective_rank < rank:
            null_space = Vt_svd[effective_rank:, :]
            num_deps = null_space.shape[0]

            new_U = U_np.copy()
            new_V = V_np.copy()
            new_W = W_np.copy()
            eliminated = set()

            for _ in range(num_deps):
                if time.time() > deadline - 15:
                    break

                remaining = [c for c in range(rank) if c not in eliminated]
                if len(remaining) <= 1:
                    break

                M_sub = M[:, remaining]

                try:
                    U_sub, s_sub, Vt_sub = np.linalg.svd(M_sub, full_matrices=False)
                except np.linalg.LinAlgError:
                    break

                tol_sub = 1e-10
                eff_rank_sub = np.sum(s_sub > tol_sub)

                if eff_rank_sub >= len(remaining):
                    break

                null_sub = Vt_sub[eff_rank_sub:, :]

                best_col = None
                best_score = -1
                best_null_idx = 0

                for nv_idx in range(null_sub.shape[0]):
                    nv = null_sub[nv_idx]
                    for local_c, global_c in enumerate(remaining):
                        score = abs(nv[local_c])
                        if score > best_score and score > 1e-8:
                            best_score = score
                            best_col = global_c
                            best_null_idx = nv_idx

                if best_col is None:
                    break

                nv = null_sub[best_null_idx]
                local_best = remaining.index(best_col)
                pivot_val = nv[local_best]

                if abs(pivot_val) < 1e-10:
                    continue

                for local_c, global_c in enumerate(remaining):
                    if global_c == best_col:
                        continue
                    coeff = -nv[local_c] / pivot_val
                    if abs(coeff) > 1e-10:
                        new_U[:, global_c] += coeff * new_U[:, best_col]
                        new_V[:, global_c] += coeff * new_V[:, best_col]
                        new_W[:, global_c] += coeff * new_W[:, best_col]

                eliminated.add(best_col)

            if eliminated:
                keep = [c for c in range(rank) if c not in eliminated]
                new_rank = len(keep)
                final_U = new_U[:, keep]
                final_V = new_V[:, keep]
                final_W = new_W[:, keep]

                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for r in range(new_rank):
                                total += final_U[i*m+j, r] * final_V[j*p+k, r] * final_W[k*n+i, r]
                            if abs(total - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break

                if valid:
                    def to_fraction_list(arr):
                        result = []
                        for row in arr:
                            result.append([])
                            for val in row:
                                frac = round(val * 2) / 2
                                if abs(frac - round(frac)) < 1e-10:
                                    frac = int(round(frac))
                                result[-1].append(frac)
                        return result

                    return {
                        "U": to_fraction_list(final_U),
                        "V": to_fraction_list(final_V),
                        "W": to_fraction_list(final_W)
                    }
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
