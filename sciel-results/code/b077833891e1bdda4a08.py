"""Matrix multiplication 2,4,6 - rank-minimal tensor decomposition using Strassen blocks with rank reduction."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    n, m, p = 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build the 42 Strassen terms
    rank_init = 42
    U = np.zeros((nm, rank_init), dtype=float)
    V = np.zeros((mp, rank_init), dtype=float)
    W = np.zeros((pn, rank_init), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset):
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)
        
        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)
        
        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)
        
        # M1: (A00+A11)(B00+B11)
        U[a00, offset] += 1.0
        U[a11, offset] += 1.0
        V[b00, offset] += 1.0
        V[b11, offset] += 1.0
        W[w00, offset] += 1.0
        W[w11, offset] += 1.0
        
        # M2: (A10+A11)B00
        U[a10, offset + 1] += 1.0
        U[a11, offset + 1] += 1.0
        V[b00, offset + 1] += 1.0
        W[w10, offset + 1] += 1.0
        W[w11, offset + 1] -= 1.0
        
        # M3: A00(B01-B11)
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0
        V[b11, offset + 2] -= 1.0
        W[w01, offset + 2] += 1.0
        W[w11, offset + 2] += 1.0
        
        # M4: A11(B10-B00)
        U[a11, offset + 3] += 1.0
        V[b10, offset + 3] += 1.0
        V[b00, offset + 3] -= 1.0
        W[w00, offset + 3] += 1.0
        W[w10, offset + 3] += 1.0
        
        # M5: (A00+A01)B11
        U[a00, offset + 4] += 1.0
        U[a01, offset + 4] += 1.0
        V[b11, offset + 4] += 1.0
        W[w00, offset + 4] -= 1.0
        W[w01, offset + 4] += 1.0
        
        # M6: (A10-A00)(B00+B01)
        U[a10, offset + 5] += 1.0
        U[a00, offset + 5] -= 1.0
        V[b00, offset + 5] += 1.0
        V[b01, offset + 5] += 1.0
        W[w11, offset + 5] += 1.0
        
        # M7: (A01-A11)(B10+B11)
        U[a01, offset + 6] += 1.0
        U[a11, offset + 6] -= 1.0
        V[b10, offset + 6] += 1.0
        V[b11, offset + 6] += 1.0
        W[w00, offset + 6] += 1.0

    # Apply 6 Strassen blocks
    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    def verify_decomp(U_mat, V_mat, W_mat, rank, tol=1e-8):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify_decomp(U, V, W, rank_init):
        # Fallback to schoolbook
        rank_trivial = n * m * p
        U_t = np.zeros((nm, rank_trivial), dtype=float)
        V_t = np.zeros((mp, rank_trivial), dtype=float)
        W_t = np.zeros((pn, rank_trivial), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_t[i * m + j, r] = 1.0
                    V_t[j * p + k, r] = 1.0
                    W_t[k * n + i, r] = 1.0
        return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}

    # Try to reduce rank by removing individual factors
    best_rank = rank_init
    best_U, best_V, best_W = U, V, W

    def try_remove_factors(U_mat, V_mat, W_mat, indices_to_remove):
        """Remove specified factor indices and verify."""
        rank = U_mat.shape[1]
        keep = [i for i in range(rank) if i not in indices_to_remove]
        if not keep:
            return None
        U_new = U_mat[:, keep]
        V_new = V_mat[:, keep]
        W_new = W_mat[:, keep]
        new_rank = len(keep)
        if verify_decomp(U_new, V_new, W_new, new_rank):
            return U_new, V_new, W_new, new_rank
        return None

    # Try removing single factors
    for r in range(rank_init):
        if time.time() > deadline - 30:
            break
        result = try_remove_factors(U, V, W, {r})
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            if new_rank < best_rank:
                best_rank = new_rank
                best_U, best_V, best_W = U_new, V_new, W_new
            break  # Found one, move to pairs

    # Try removing pairs of factors if we have time
    if time.time() < deadline - 20 and best_rank == rank_init:
        for r1 in range(rank_init):
            for r2 in range(r1 + 1, rank_init):
                if time.time() > deadline - 10:
                    break
                result = try_remove_factors(U, V, W, {r1, r2})
                if result is not None:
                    U_new, V_new, W_new, new_rank = result
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_U, best_V, best_W = U_new, V_new, W_new
                    break
            if time.time() > deadline - 10:
                break
            if best_rank < rank_init:
                break

    # Convert to lists with integer entries
    U_list = [[int(round(best_U[i, r])) for r in range(best_rank)] for i in range(nm)]
    V_list = [[int(round(best_V[i, r])) for r in range(best_rank)] for i in range(mp)]
    W_list = [[int(round(best_W[i, r])) for r in range(best_rank)] for i in range(pn)]

    return {"U": U_list, "V": V_list, "W": W_list}
