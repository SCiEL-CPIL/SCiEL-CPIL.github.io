import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    def verify(U_mat, V_mat, W_mat, rank, tol=1e-8):
        """Verify that the tensor decomposition exactly decomposes matrix multiplication."""
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
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    j_b = b // p
                    k = b % p
                    i_w = c % n
                    k_w = c // n
                    if i == i_w and j == j_b and k == k_w:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > tol:
                        return False
        return True

    def build_nz(n, m, p):
        nz = []
        nz_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    nz.append((a, b, c))
                    nz_set.add((a, b, c))
        return nz, nz_set

    def compute_error(U_r, V_r, W_r, nz_a, nz_b, nz_c, zero_a, zero_b, zero_c):
        prod_nz = np.einsum('ar,br,cr->r', U_r[nz_a], V_r[nz_b], W_r[nz_c])
        sum_nz = prod_nz.sum(axis=1)
        err_nz = np.sum((sum_nz - 1.0) ** 2)
        prod_z = np.einsum('ar,br,cr->r', U_r[zero_a], V_r[zero_b], W_r[zero_c])
        sum_z = prod_z.sum(axis=1)
        err_z = np.sum(sum_z ** 2)
        return err_nz + err_z

    def update_fixed_point(U_r, V_r, W_r, nz_a, nz_b, nz_c, n_a, n_b, n_c):
        U_new = U_r.copy()
        V_new = V_r.copy()
        W_new = W_r.copy()
        for a_idx in range(n_a):
            mask = nz_a == a_idx
            if not np.any(mask):
                continue
            V_sub = V_r[nz_b[mask]]
            W_sub = W_r[nz_c[mask]]
            prod = V_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            U_new[a_idx, valid] = num[valid] / den[valid]
        for b_idx in range(n_b):
            mask = nz_b == b_idx
            if not np.any(mask):
                continue
            U_sub = U_r[nz_a[mask]]
            W_sub = W_r[nz_c[mask]]
            prod = U_sub * W_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            V_new[b_idx, valid] = num[valid] / den[valid]
        for c_idx in range(n_c):
            mask = nz_c == c_idx
            if not np.any(mask):
                continue
            U_sub = U_r[nz_a[mask]]
            V_sub = V_r[nz_b[mask]]
            prod = U_sub * V_sub
            num = prod.sum(axis=0)
            den = (prod ** 2).sum(axis=0)
            valid = den > 1e-12
            W_new[c_idx, valid] = num[valid] / den[valid]
        return U_new, V_new, W_new

    def search_decomposition(n, m, p, target_ranks, max_attempts, max_iters):
        nz, nz_set = build_nz(n, m, p)
        n_a, n_b, n_c = n * m, m * p, p * n
        nz_a = np.array([x[0] for x in nz])
        nz_b = np.array([x[1] for x in nz])
        nz_c = np.array([x[2] for x in nz])
        
        zero_entries = []
        for a in range(n_a):
            for b in range(n_b):
                for c in range(n_c):
                    if (a, b, c) not in nz_set:
                        zero_entries.append((a, b, c))
        zero_a = np.array([x[0] for x in zero_entries])
        zero_b = np.array([x[1] for x in zero_entries])
        zero_c = np.array([x[2] for x in zero_entries])

        for target_rank in target_ranks:
            for attempt in range(max_attempts):
                if time.time() > deadline - 20:
                    return None
                np.random.seed(1000 + attempt * 7)
                U_r = np.random.randn(n_a, target_rank) * 0.5
                V_r = np.random.randn(n_b, target_rank) * 0.5
                W_r = np.random.randn(n_c, target_rank) * 0.5
                for it in range(max_iters):
                    U_r, V_r, W_r = update_fixed_point(U_r, V_r, W_r, nz_a, nz_b, nz_c, n_a, n_b, n_c)
                    if it % 5 == 0:
                        err = compute_error(U_r, V_r, W_r, nz_a, nz_b, nz_c, zero_a, zero_b, zero_c)
                        if err < 1e-10:
                            return U_r, V_r, W_r
                        if err < 1e-4:
                            for polish_it in range(100):
                                U_r, V_r, W_r = update_fixed_point(U_r, V_r, W_r, nz_a, nz_b, nz_c, n_a, n_b, n_c)
                                err = compute_error(U_r, V_r, W_r, nz_a, nz_b, nz_c, zero_a, zero_b, zero_c)
                                if err < 1e-12:
                                    break
                            if err < 1e-10:
                                return U_r, V_r, W_r
        return None

    # Strategy 1: Try to decompose M_{2,2,6} (rank < 21)
    # M_{2,4,6} = 2 blocks of M_{2,2,6} (j=0,1 and j=2,3)
    # If we find rank R for M_{2,2,6}, total rank is 2*R.
    # M_{2,2,6} has 3 M_{2,2,2} blocks. Strassen rank is 21.
    # Try ranks 18, 19, 20.
    
    best_rank_226 = None
    best_U_226 = None
    best_V_226 = None
    best_W_226 = None
    
    if time.time() < deadline - 40:
        res = search_decomposition(2, 2, 6, [18, 19, 20], 50, 200)
        if res:
            best_U_226, best_V_226, best_W_226 = res
            best_rank_226 = res[0].shape[1]

    if best_rank_226 is not None:
        rank_total = 2 * best_rank_226
        U_full = np.zeros((nm, rank_total), dtype=float)
        V_full = np.zeros((mp, rank_total), dtype=float)
        W_full = np.zeros((pn, rank_total), dtype=float)
        
        for j_start in [0, 2]:
            offset = (j_start // 2) * best_rank_226
            for i in range(2):
                for j_local in range(2):
                    j_global = j_start + j_local
                    a_global = i * m + j_global
                    a_local = i * 2 + j_local
                    for k in range(6):
                        b_global = j_global * p + k
                        b_local = j_local * 6 + k
                        c_global = k * n + i
                        c_local = k * 2 + i
                        for r in range(best_rank_226):
                            U_full[a_global, offset + r] += best_U_226[a_local, r]
                            V_full[b_global, offset + r] += best_V_226[b_local, r]
                            W_full[c_global, offset + r] += best_W_226[c_local, r]
        
        if verify(U_full, V_full, W_full, rank_total):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Strategy 2: Try to decompose M_{2,2,4} (rank < 14)
    # M_{2,4,6} = 2 * (M_{2,2,4} + M_{2,2,2})
    # M_{2,2,4} for k=0..3, M_{2,2,2} for k=4..5.
    # M_{2,2,2} rank is 7.
    # If we find rank R for M_{2,2,4}, total rank is 2 * (R + 7).
    # Strassen rank for M_{2,2,4} is 14 (2 * 7).
    # Try ranks 12, 13.

    best_rank_224 = None
    best_U_224 = None
    best_V_224 = None
    best_W_224 = None

    if time.time() < deadline - 20:
        res = search_decomposition(2, 2, 4, [12, 13], 50, 200)
        if res:
            best_U_224, best_V_224, best_W_224 = res
            best_rank_224 = res[0].shape[1]

    if best_rank_224 is not None:
        rank_222 = 7
        rank_total = 2 * (best_rank_224 + rank_222)
        U_full = np.zeros((nm, rank_total), dtype=float)
        V_full = np.zeros((mp, rank_total), dtype=float)
        W_full = np.zeros((pn, rank_total), dtype=float)
        
        # We need a rank 7 decomposition for M_{2,2,2}
        # Use standard Strassen
        U_222 = np.zeros((4, 7))
        V_222 = np.zeros((6, 7))
        W_222 = np.zeros((4, 7))
        
        # Strassen for 2x2x2
        # a00, a01, a10, a11
        # b00, b01, b10, b11
        # c00, c01, c10, c11
        # Indices: a=2i+j, b=2j+k, c=2k+i
        
        def add_strassen_222(offset):
            U_222[0, offset] += 1; U_222[3, offset] += 1
            V_222[0, offset] += 1; V_222[5, offset] += 1
            W_222[0, offset] += 1; W_222[3, offset] += 1
            
            U_222[2, offset+1] += 1; U_222[3, offset+1] += 1
            V_222[0, offset+1] += 1
            W_222[2, offset+1] += 1; W_222[3, offset+1] -= 1
            
            U_222[0, offset+2] += 1
            V_222[1, offset+2] += 1; V_222[5, offset+2] -= 1
            W_222[1, offset+2] += 1; W_222[3, offset+2] += 1
            
            U_222[3, offset+3] += 1
            V_222[2, offset+3] += 1; V_222[0, offset+3] -= 1
            W_222[0, offset+3] += 1; W_222[2, offset+3] += 1
            
            U_222[0, offset+4] += 1; U_222[1, offset+4] += 1
            V_222[5, offset+4] += 1
            W_222[0, offset+4] -= 1; W_222[1, offset+4] += 1
            
            U_222[2, offset+5] += 1; U_222[0, offset+5] -= 1
            V_222[0, offset+5] += 1; V_222[1, offset+5] += 1
            W_222[3, offset+5] += 1
            
            U_222[1, offset+6] += 1; U_222[3, offset+6] -= 1
            V_222[2, offset+6] += 1; V_222[3, offset+6] += 1
            W_222[0, offset+6] += 1

        add_strassen_222(0)

        for j_start in [0, 2]:
            offset_224 = (j_start // 2) * best_rank_224
            offset_222 = (j_start // 2) * best_rank_224 + best_rank_224
            
            for i in range(2):
                for j_local in range(2):
                    j_global = j_start + j_local
                    a_global = i * m + j_global
                    a_local = i * 2 + j_local
                    
                    # k=0..3 part (M_{2,2,4})
                    for k in range(4):
                        b_global = j_global * p + k
                        b_local = j_local * 4 + k
                        c_global = k * n + i
                        c_local = k * 2 + i
                        for r in range(best_rank_224):
                            U_full[a_global, offset_224 + r] += best_U_224[a_local, r]
                            V_full[b_global, offset_224 + r] += best_V_224[b_local, r]
                            W_full[c_global, offset_224 + r] += best_W_224[c_local, r]
                            
                    # k=4..5 part (M_{2,2,2})
                    for k in range(4, 6):
                        b_global = j_global * p + k
                        b_local = (j_local * 2 + (k - 4)) * 2 + (k - 4) # b_local for 2x2x2?
                        # Wait, M_{2,2,2} local indices:
                        # i_local, j_local, k_local (0,1)
                        # b_local = j_local * 2 + k_local
                        k_local = k - 4
                        b_local_222 = j_local * 2 + k_local
                        c_local_222 = k_local * 2 + i
                        for r in range(7):
                            U_full[a_global, offset_222 + r] += U_222[a_local, r]
                            V_full[b_global, offset_222 + r] += V_222[b_local_222, r]
                            W_full[c_global, offset_222 + r] += W_222[c_local_222, r]
        
        if verify(U_full, V_full, W_full, rank_total):
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback: rank 42 Strassen
    rank = 42
    U = np.zeros((nm, rank), dtype=float)
    V = np.zeros((mp, rank), dtype=float)
    W = np.zeros((pn, rank), dtype=float)

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
        
        U[a00, offset] += 1.0
        U[a11, offset] += 1.0
        V[b00, offset] += 1.0
        V[b11, offset] += 1.0
        W[w00, offset] += 1.0
        W[w11, offset] += 1.0
        
        U[a10, offset + 1] += 1.0
        U[a11, offset + 1] += 1.0
        V[b00, offset + 1] += 1.0
        W[w10, offset + 1] += 1.0
        W[w11, offset + 1] -= 1.0
        
        U[a00, offset + 2] += 1.0
        V[b01, offset + 2] += 1.0
        V[b11, offset + 2] -= 1.0
        W[w01, offset + 2] += 1.0
        W[w11, offset + 2] += 1.0
        
        U[a11, offset + 3] += 1.0
        V[b10, offset + 3] += 1.0
        V[b00, offset + 3] -= 1.0
        W[w00, offset + 3] += 1.0
        W[w10, offset + 3] += 1.0
        
        U[a00, offset + 4] += 1.0
        U[a01, offset + 4] += 1.0
        V[b11, offset + 4] += 1.0
        W[w00, offset + 4] -= 1.0
        W[w01, offset + 4] += 1.0
        
        U[a10, offset + 5] += 1.0
        U[a00, offset + 5] -= 1.0
        V[b00, offset + 5] += 1.0
        V[b01, offset + 5] += 1.0
        W[w11, offset + 5] += 1.0
        
        U[a01, offset + 6] += 1.0
        U[a11, offset + 6] -= 1.0
        V[b10, offset + 6] += 1.0
        V[b11, offset + 6] += 1.0
        W[w00, offset + 6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    if verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W_full.tolist()}

    # Fallback: schoolbook rank 48
    rank_school = 48
    U_s = np.zeros((nm, rank_school), dtype=float)
    V_s = np.zeros((mp, rank_school), dtype=float)
    W_s = np.zeros((pn, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j, col] = 1.0
                V_s[j * p + k, col] = 1.0
                W_s[k * n + i, col] = 1.0
                col += 1
    
    if verify(U_s, V_s, W_s, rank_school):
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
