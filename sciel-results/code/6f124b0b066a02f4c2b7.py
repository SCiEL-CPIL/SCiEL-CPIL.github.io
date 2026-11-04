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

    # Strategy: decompose 2x4x6 into two 2x2x3 sub-tensors
    # Split m=4 into two blocks of 2 (j_start=0 and j_start=2)
    # Each sub-tensor is 2x2x3, attempt rank-6 decomposition for each
    # Total rank: 2 * 6 = 12

    # First, let's try to find a rank-6 decomposition for the 2x2x3 tensor
    # The 2x2x3 tensor has dimensions: a in [0,3], b in [0,5], c in [0,5]
    # Nonzero entries as computed above

    def build_2x2x3_tensor():
        """Build the nonzero entries of the 2x2x3 multiplication tensor."""
        nz = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a = i * 2 + j
                    b = j * 3 + k
                    c = k * 2 + i
                    nz.append((a, b, c))
        return nz

    nz_223 = build_2x2x3_tensor()
    nz_set = set(nz_223)
    
    # Precompute index arrays for vectorized operations
    nz_a = np.array([x[0] for x in nz_223])
    nz_b = np.array([x[1] for x in nz_223])
    nz_c = np.array([x[2] for x in nz_223])
    
    zero_entries = []
    for a in range(4):
        for b in range(6):
            for c in range(6):
                if (a, b, c) not in nz_set:
                    zero_entries.append((a, b, c))
    zero_a = np.array([x[0] for x in zero_entries])
    zero_b = np.array([x[1] for x in zero_entries])
    zero_c = np.array([x[2] for x in zero_entries])

    def compute_error(U_r, V_r, W_r):
        prod_nz = np.einsum('ar,br,cr->r', U_r[nz_a], V_r[nz_b], W_r[nz_c])
        sum_nz = prod_nz.sum(axis=1)
        err_nz = np.sum((sum_nz - 1.0) ** 2)
        prod_z = np.einsum('ar,br,cr->r', U_r[zero_a], V_r[zero_b], W_r[zero_c])
        sum_z = prod_z.sum(axis=1)
        err_z = np.sum(sum_z ** 2)
        return err_nz + err_z

    def update_fixed_point(U_r, V_r, W_r):
        U_new = U_r.copy()
        V_new = V_r.copy()
        W_new = W_r.copy()
        for a_idx in range(4):
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
        for b_idx in range(6):
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
        for c_idx in range(6):
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

    best_rank_223 = None
    best_U_223 = None
    best_V_223 = None
    best_W_223 = None

    for target_rank in [6, 7, 8]:
        if time.time() > deadline - 60:
            break
        found = False
        for attempt in range(100):
            if time.time() > deadline - 60:
                break
            np.random.seed(42 + attempt)
            U_r = np.random.randn(4, target_rank) * 0.5
            V_r = np.random.randn(6, target_rank) * 0.5
            W_r = np.random.randn(6, target_rank) * 0.5
            for it in range(500):
                U_r, V_r, W_r = update_fixed_point(U_r, V_r, W_r)
                if it % 10 == 0:
                    err = compute_error(U_r, V_r, W_r)
                    if err < 1e-10:
                        found = True
                        best_rank_223 = target_rank
                        best_U_223 = U_r.copy()
                        best_V_223 = V_r.copy()
                        best_W_223 = W_r.copy()
                        break
                    if err < 1e-4:
                        for polish_it in range(200):
                            U_r, V_r, W_r = update_fixed_point(U_r, V_r, W_r)
                            err = compute_error(U_r, V_r, W_r)
                            if err < 1e-12:
                                break
                        if err < 1e-10:
                            found = True
                            best_rank_223 = target_rank
                            best_U_223 = U_r.copy()
                            best_V_223 = V_r.copy()
                            best_W_223 = W_r.copy()
                            break
            if found:
                break
        if found:
            break
    
    if best_rank_223 is not None and best_U_223 is not None:
        rank_total = 2 * best_rank_223
        
        U_full = np.zeros((nm, rank_total), dtype=float)
        V_full = np.zeros((mp, rank_total), dtype=float)
        W_full = np.zeros((pn, rank_total), dtype=float)
        
        # Block 1: j_start=0 (j=0,1), uses columns 0..best_rank_223-1
        # Block 2: j_start=2 (j=2,3), uses columns best_rank_223..2*best_rank_223-1
        
        for j_start in [0, 2]:
            offset = j_start // 2 * best_rank_223
            
            # Map local indices to global
            for i in range(2):
                for j_local in range(2):
                    j_global = j_start + j_local
                    a_global = i * m + j_global
                    a_local = i * 2 + j_local
                    
                    for k in range(3):
                        b_global = j_global * p + k
                        b_local = j_local * 3 + k
                        c_global = k * n + i
                        c_local = k * 2 + i
                        
                        for r in range(best_rank_223):
                            U_full[a_global, offset + r] += best_U_223[a_local, r]
                            V_full[b_global, offset + r] += best_V_223[b_local, r]
                            W_full[c_global, offset + r] += best_W_223[c_local, r]
        
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

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
