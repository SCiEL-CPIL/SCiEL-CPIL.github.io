"""Optimized block-Strassen decomposition for 6x6 matrix multiplication.
Uses 3x3 blocks of size 2x2 with Strassen (rank 7 per block), giving rank 189.
Optimized for speed: uses integer arithmetic, fast verification, and early exit.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    nb = 3  # number of blocks per dimension
    bs = 2  # block size
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Each entry is a list of (row, col, coeff)
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0: A00+A11
        [(1,0,1), (1,1,1)],      # M1: A10+A11
        [(0,0,1)],               # M2: A00
        [(1,1,1)],               # M3: A11
        [(0,0,1), (0,1,1)],      # M4: A00+A01
        [(1,0,1), (0,0,-1)],     # M5: A10-A00
        [(0,1,1), (1,1,-1)],     # M6: A01-A11
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0: B00+B11
        [(0,0,1)],               # M1: B00
        [(0,1,1), (1,1,-1)],     # M2: B01-B11
        [(1,0,1), (0,0,-1)],     # M3: B10-B00
        [(1,1,1)],               # M4: B11
        [(0,0,1), (0,1,1)],      # M5: B00+B01
        [(1,0,1), (1,1,1)],      # M6: B10+B11
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
    
    rank = nb * nb * nb * 7  # 9 * 3 * 7 = 189
    
    # Use integer dtype for speed and exactness
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    for ib in range(nb):
        for jb in range(nb):
            for kb in range(nb):
                for s in range(7):
                    # A block: rows ib*bs..(ib+1)*bs, cols jb*bs..(jb+1)*bs
                    # B block: rows jb*bs..(jb+1)*bs, cols kb*bs..(kb+1)*bs
                    # C block: rows ib*bs..(ib+1)*bs, cols kb*bs..(kb+1)*bs
                    
                    for (lr, lc, coeff) in strassen_x[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U[gi * m + gj, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_y[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V[gj * p + gk, col] += coeff
                    
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W[gk * n + gi, col] += coeff
                    
                    col += 1
    
    # Fast verification using integer arithmetic
    # T[i,j,k] should equal 1 for all i,j,k
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can check a subset of (i,j,k) to verify quickly
    
    def fast_verify(U_mat, V_mat, W_mat, n=n, m=m, p=p):
        """Verify decomposition by checking T[i,j,k] = 1 for sampled (i,j,k)."""
        rank_val = U_mat.shape[1]
        # Check all 216 positions (fast enough with integer arithmetic)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(rank_val):
                        u_val = U_mat[u_idx, r]
                        if u_val != 0:
                            v_val = V_mat[v_idx, r]
                            if v_val != 0:
                                w_val = W_mat[w_idx, r]
                                total += u_val * v_val * w_val
                    if total != 1:
                        return False
        return True
    
    if not fast_verify(U, V, W):
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = np.zeros((n*m, rank_school), dtype=np.int64)
        V_s = np.zeros((m*p, rank_school), dtype=np.int64)
        W_s = np.zeros((p*n, rank_school), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j, col] = 1
                    V_s[j*p + k, col] = 1
                    W_s[k*n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    # Try to reduce rank by finding proportional column pairs
    # Two columns c1, c2 can be merged if U[:,c1] = a*U[:,c2] and V[:,c1] = b*V[:,c2]
    # In that case, W[:,c1] += a*b*W[:,c2] and we remove column c2
    
    used = [True] * rank
    merged = 0
    
    def find_ratio_int(vec_a, vec_b):
        """Find integer r such that vec_b = r * vec_a, or None."""
        r = None
        for idx in range(len(vec_a)):
            a_val = vec_a[idx]
            b_val = vec_b[idx]
            if a_val != 0:
                if b_val == 0:
                    return None
                if b_val % a_val != 0:
                    return None
                candidate = b_val // a_val
                if r is None:
                    r = candidate
                elif candidate != r:
                    return None
            else:
                if b_val != 0:
                    return None
        if r is None:
            # All zeros in vec_a
            if np.any(vec_b != 0):
                return None
            r = 0
        return r
    
    # Check for time
    if time.time() < deadline - 20:
        for i in range(rank):
            if not used[i]:
                continue
            for j in range(i+1, rank):
                if not used[j]:
                    continue
                if time.time() > deadline - 15:
                    break
                
                ratio_u = find_ratio_int(U[:, i], U[:, j])
                if ratio_u is None:
                    continue
                ratio_v = find_ratio_int(V[:, i], V[:, j])
                if ratio_v is None:
                    continue
                
                # Merge: W[:,i] += ratio_u * ratio_v * W[:,j]
                W[:, i] += ratio_u * ratio_v * W[:, j]
                used[j] = False
                merged += 1
    
    if merged > 0:
        new_rank = sum(used)
        final_U = U[:, [i for i in range(rank) if used[i]]]
        final_V = V[:, [i for i in range(rank) if used[i]]]
        final_W = W[:, [i for i in range(rank) if used[i]]]
        
        # Verify the reduced decomposition
        if fast_verify(final_U, final_V, final_W):
            return {"U": final_U.tolist(), "V": final_V.tolist(), "W": final_W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
