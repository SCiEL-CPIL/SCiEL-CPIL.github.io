import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        # Check all 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(s - 1.0) > tol:
                        return False
        
        # Check all 0s
        valid_set = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid_set.add((i*m+j, j*p+k, k*n+i))
                    
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a,b,c) in valid_set:
                        continue
                    s = 0.0
                    for r_idx in range(r):
                        s += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(s) > tol:
                        return False
        return True

    # Construct Rank 42 (Strassen blocks)
    rank42 = 42
    U42 = np.zeros((nm, rank42), dtype=float)
    V42 = np.zeros((mp, rank42), dtype=float)
    W42 = np.zeros((pn, rank42), dtype=float)

    def add_strassen_block(i_start, j_start, k_start, offset):
        # A indices
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        
        # B indices
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        
        # W indices
        w00 = k_start * n + i_start
        w01 = (k_start + 1) * n + i_start
        w10 = k_start * n + i_start + 1
        w11 = (k_start + 1) * n + i_start + 1
        
        # Term 0: M1 = (A00+A11)(B00+B11)
        U42[a00, offset] += 1.0; U42[a11, offset] += 1.0
        V42[b00, offset] += 1.0; V42[b11, offset] += 1.0
        W42[w00, offset] += 1.0; W42[w11, offset] += 1.0

        # Term 1: M2 = (A10+A11)B00
        U42[a10, offset+1] += 1.0; U42[a11, offset+1] += 1.0
        V42[b00, offset+1] += 1.0
        W42[w10, offset+1] += 1.0; W42[w11, offset+1] -= 1.0

        # Term 2: M3 = A00(B01-B11)
        U42[a00, offset+2] += 1.0
        V42[b01, offset+2] += 1.0; V42[b11, offset+2] -= 1.0
        W42[w01, offset+2] += 1.0; W42[w11, offset+2] += 1.0

        # Term 3: M4 = A11(B10-B00)
        U42[a11, offset+3] += 1.0
        V42[b10, offset+3] += 1.0; V42[b00, offset+3] -= 1.0
        W42[w00, offset+3] += 1.0; W42[w10, offset+3] += 1.0

        # Term 4: M5 = (A00+A01)B11
        U42[a00, offset+4] += 1.0; U42[a01, offset+4] += 1.0
        V42[b11, offset+4] += 1.0
        W42[w00, offset+4] -= 1.0; W42[w01, offset+4] += 1.0

        # Term 5: M6 = (A10-A00)(B00+B01)
        U42[a10, offset+5] += 1.0; U42[a00, offset+5] -= 1.0
        V42[b00, offset+5] += 1.0; V42[b01, offset+5] += 1.0
        W42[w11, offset+5] += 1.0

        # Term 6: M7 = (A01-A11)(B10+B11)
        U42[a01, offset+6] += 1.0; U42[a11, offset+6] -= 1.0
        V42[b10, offset+6] += 1.0; V42[b11, offset+6] += 1.0
        W42[w00, offset+6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    if verify(U42, V42, W42):
        # Attempt to reduce rank by finding linear dependencies
        best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()
        best_rank = 42
        
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            current_rank = best_U.shape[1]
            if current_rank <= 24:
                break
            
            # Try to find a linear dependency: find c such that sum_r c_r U_r ⊗ V_r ⊗ W_r = 0
            found_dep = False
            for b_val in range(mp):
                if time.time() > deadline - 15:
                    break
                    
                # Build system: for all (a,c), sum_r c_r U[a,r]V[b_val,r]W[c,r] = 0
                A_sys = np.zeros((nm * pn, current_rank), dtype=float)
                idx = 0
                for a_val in range(nm):
                    for c_val in range(pn):
                        for r_idx in range(current_rank):
                            A_sys[idx, r_idx] = best_U[a_val, r_idx] * best_V[b_val, r_idx] * best_W[c_val, r_idx]
                        idx += 1
                
                try:
                    U_svd, s_vals, Vt = np.linalg.svd(A_sys, full_matrices=True)
                    rank_A = np.sum(s_vals > 1e-8)
                    null_dim = current_rank - rank_A
                    
                    if null_dim > 0:
                        null_vecs = Vt[rank_A:]
                        
                        for nv_idx in range(null_dim):
                            c_vec = null_vecs[nv_idx]
                            max_idx = np.argmax(np.abs(c_vec))
                            if abs(c_vec[max_idx]) < 1e-10:
                                continue
                            
                            # Verify dependency holds for all (a,b,c)
                            valid_dep = True
                            for a_val in range(nm):
                                for b_val2 in range(mp):
                                    for c_val in range(pn):
                                        s = 0.0
                                        for r_idx in range(current_rank):
                                            s += c_vec[r_idx] * best_U[a_val, r_idx] * best_V[b_val2, r_idx] * best_W[c_val, r_idx]
                                        if abs(s) > 1e-6:
                                            valid_dep = False
                                            break
                                    if not valid_dep:
                                        break
                                if not valid_dep:
                                    break
                            
                            if valid_dep:
                                keep_indices = [r for r in range(current_rank) if r != max_idx]
                                scale_factors = np.ones(len(keep_indices))
                                for new_idx, old_idx in enumerate(keep_indices):
                                    scale_factors[new_idx] = 1.0 - c_vec[old_idx] / c_vec[max_idx]
                                
                                new_U = best_U[:, keep_indices].copy()
                                new_V = best_V[:, keep_indices].copy()
                                new_W = best_W[:, keep_indices].copy()
                                
                                for idx2 in range(len(keep_indices)):
                                    sf = scale_factors[idx2]
                                    if abs(sf) < 1e-15:
                                        continue
                                    if sf > 0:
                                        sqrt_sf = np.sqrt(sf)
                                        new_U[:, idx2] *= sqrt_sf
                                        new_V[:, idx2] *= sqrt_sf
                                    else:
                                        sqrt_sf = np.sqrt(-sf)
                                        new_U[:, idx2] *= sqrt_sf
                                        new_V[:, idx2] *= sqrt_sf
                                        new_W[:, idx2] *= -1.0
                                
                                if verify(new_U, new_V, new_W, tol=1e-6):
                                    best_U, best_V, best_W = new_U, new_V, new_W
                                    best_rank = len(keep_indices)
                                    changed = True
                                    found_dep = True
                                    break
                        
                        if found_dep:
                            break
                except np.linalg.LinAlgError:
                    pass
            
            if not found_dep:
                # Fallback: proportional column merge
                cols_to_keep = []
                used = [False] * current_rank
                
                for i in range(current_rank):
                    if used[i]:
                        continue
                    
                    merge_found = False
                    for j in range(i + 1, current_rank):
                        if used[j]:
                            continue
                        
                        nz_u_i = np.nonzero(best_U[:, i])[0]
                        nz_u_j = np.nonzero(best_U[:, j])[0]
                        if len(nz_u_i) == 0 or len(nz_u_j) == 0:
                            if len(nz_u_i) != len(nz_u_j):
                                continue
                            if len(nz_u_i) > 0:
                                continue
                        else:
                            if not np.array_equal(nz_u_i, nz_u_j):
                                continue
                            ratios_u = best_U[nz_u_i, i] / best_U[nz_u_i, j]
                            if np.std(ratios_u) > 1e-10:
                                continue
                            s = ratios_u[0]
                        
                        nz_v_i = np.nonzero(best_V[:, i])[0]
                        nz_v_j = np.nonzero(best_V[:, j])[0]
                        if len(nz_v_i) == 0 or len(nz_v_j) == 0:
                            if len(nz_v_i) != len(nz_v_j):
                                continue
                            if len(nz_v_i) > 0:
                                continue
                        else:
                            if not np.array_equal(nz_v_i, nz_v_j):
                                continue
                            ratios_v = best_V[nz_v_i, i] / best_V[nz_v_i, j]
                            if np.std(ratios_v) > 1e-10:
                                continue
                            t = ratios_v[0]
                        
                        scale = 1.0 / (s * t)
                        best_W[:, i] += scale * best_W[:, j]
                        used[j] = True
                        merge_found = True
                        changed = True
                        break
                    
                    if not used[i]:
                        cols_to_keep.append(i)
                
                if merge_found:
                    new_rank = len(cols_to_keep)
                    best_U = best_U[:, cols_to_keep]
                    best_V = best_V[:, cols_to_keep]
                    best_W = best_W[:, cols_to_keep]
                    
                    if not verify(best_U, best_V, best_W):
                        best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()
                        break
                    else:
                        best_rank = new_rank
                else:
                    break
        
        if best_rank < 42 and verify(best_U, best_V, best_W):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook rank 48
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=float)
        V = np.zeros((mp, rank0), dtype=float)
        W = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Constructed", file=sys.stderr)
