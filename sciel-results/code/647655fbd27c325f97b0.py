import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 3, 4, 6
    
    def build_decomposition(a_split, b_split, c_split):
        """Build decomposition given splits.
        a_split: how to split rows of A (n rows), e.g., [1,2]
        b_split: how to split rows of B (m rows), e.g., [2,2]
        c_split: how to split cols of C (p cols), e.g., [2,2,2]
        For each output block (i_block, k_block):
          if i_block size == 1: schoolbook for each j_block pair
          if i_block size == 2: Strassen for each j_block pair
        """
        total_rank = 0
        # Count rank
        for ib in range(len(a_split)):
            for kb in range(len(c_split)):
                for jb in range(len(b_split)):
                    if a_split[ib] == 1:
                        total_rank += 4  # schoolbook 1x2 * 2x2
                    else:
                        total_rank += 7  # Strassen 2x2 * 2x2
        
        R = total_rank
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        
        # Compute offsets
        a_offsets = [0]
        for s in a_split:
            a_offsets.append(a_offsets[-1] + s)
        b_offsets = [0]
        for s in b_split:
            b_offsets.append(b_offsets[-1] + s)
        c_offsets = [0]
        for s in c_split:
            c_offsets.append(c_offsets[-1] + s)
        
        def add_factor(u_entries, v_entries, w_entries):
            nonlocal col
            for u_r, val in u_entries:
                U[u_r][col] += val
            for v_r, val in v_entries:
                V[v_r][col] += val
            for w_r, val in w_entries:
                W[w_r][col] += val
            col += 1
        
        for ib in range(len(a_split)):
            i0 = a_offsets[ib]
            isz = a_split[ib]
            for jb in range(len(b_split)):
                j0 = b_offsets[jb]
                jsz = b_split[jb]
                for kb in range(len(c_split)):
                    k0 = c_offsets[kb]
                    ksz = c_split[kb]
                    
                    if isz == 1 and jsz == 2 and ksz == 2:
                        # Schoolbook 1x2 * 2x2 -> 1x2
                        # A[i0, j0], A[i0, j0+1]
                        # B[j0, k0], B[j0, k0+1], B[j0+1, k0], B[j0+1, k0+1]
                        # C[i0, k0], C[i0, k0+1]
                        u0 = i0 * m + j0
                        u1 = i0 * m + j0 + 1
                        v00 = j0 * p + k0
                        v01 = j0 * p + k0 + 1
                        v10 = (j0 + 1) * p + k0
                        v11 = (j0 + 1) * p + k0 + 1
                        w0 = k0 * n + i0
                        w1 = (k0 + 1) * n + i0
                        
                        add_factor([(u0, 1)], [(v00, 1)], [(w0, 1)])
                        add_factor([(u0, 1)], [(v01, 1)], [(w1, 1)])
                        add_factor([(u1, 1)], [(v10, 1)], [(w0, 1)])
                        add_factor([(u1, 1)], [(v11, 1)], [(w1, 1)])
                    
                    elif isz == 2 and jsz == 2 and ksz == 2:
                        # Strassen 2x2 * 2x2 -> 2x2
                        i1 = i0 + 1
                        j1 = j0 + 1
                        k1 = k0 + 1
                        
                        u00 = i0 * m + j0
                        u01 = i0 * m + j1
                        u10 = i1 * m + j0
                        u11 = i1 * m + j1
                        
                        v00 = j0 * p + k0
                        v01 = j0 * p + k1
                        v10 = j1 * p + k0
                        v11 = j1 * p + k1
                        
                        w00 = k0 * n + i0
                        w01 = k1 * n + i0
                        w10 = k0 * n + i1
                        w11 = k1 * n + i1
                        
                        # M1: (a00+a11)(b00+b11) -> c00+c11
                        add_factor([(u00, 1), (u11, 1)], [(v00, 1), (v11, 1)], [(w00, 1), (w11, 1)])
                        # M2: (a10+a11)b00 -> c10+c11(-1)
                        add_factor([(u10, 1), (u11, 1)], [(v00, 1)], [(w10, 1), (w11, -1)])
                        # M3: a00(b01-b11) -> c01+c11
                        add_factor([(u00, 1)], [(v01, 1), (v11, -1)], [(w01, 1), (w11, 1)])
                        # M4: a11(b10-b00) -> c00+c10
                        add_factor([(u11, 1)], [(v10, 1), (v00, -1)], [(w00, 1), (w10, 1)])
                        # M5: (a00+a01)b11 -> c00(-1)+c01
                        add_factor([(u00, 1), (u01, 1)], [(v11, 1)], [(w00, -1), (w01, 1)])
                        # M6: (a10-a00)(b00+b01) -> c11
                        add_factor([(u10, 1), (u00, -1)], [(v00, 1), (v01, 1)], [(w11, 1)])
                        # M7: (a01-a11)(b10+b11) -> c00
                        add_factor([(u01, 1), (u11, -1)], [(v10, 1), (v11, 1)], [(w00, 1)])
        
        return U, V, W
    
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    expected = 1.0 if j == k else 0.0
                    max_err = max(max_err, abs(s - expected))
        return max_err
    
    def try_reduce_rank(U, V, W, max_rank):
        """Try to reduce rank by removing columns that are linearly dependent."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        
        if r <= max_rank:
            return U, V, W, r
        
        # Precompute tensor
        tensor = np.zeros((n, m, p, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor[i,j,k,:] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
        
        best_r = r
        best_U, best_V, best_W = U, V, W
        
        # Try removing each column
        for c in range(r):
            if time.time() > deadline - 5:
                break
            keep = [l for l in range(r) if l != c]
            U_new = U[:, keep]
            V_new = V[:, keep]
            W_new = W[:, keep]
            
            # Check if the removed column's contribution is in the span
            # Build system: tensor[i,j,k,c] = sum_{l in keep} alpha_l * tensor[i,j,k,l]
            eqs = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if abs(tensor[i,j,k,c]) > 1e-12:
                            eqs.append((i, j, k))
            
            if not eqs:
                # Column c contributes nothing - just remove it
                if verify(U_new.tolist(), V_new.tolist(), W_new.tolist()) < 1e-10:
                    best_r = r - 1
                    best_U, best_V, best_W = U_new, V_new, W_new
                    break
                continue
            
            A = np.zeros((len(eqs), len(keep)))
            b = np.zeros(len(eqs))
            for idx, (i, j, k) in enumerate(eqs):
                b[idx] = tensor[i,j,k,c]
                for ci, l in enumerate(keep):
                    A[idx, ci] = tensor[i,j,k,l]
            
            try:
                alpha, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                residual = np.max(np.abs(A @ alpha - b))
                if residual < 1e-10:
                    if verify(U_new.tolist(), V_new.tolist(), W_new.tolist()) < 1e-10:
                        best_r = r - 1
                        best_U, best_V, best_W = U_new, V_new, W_new
                        break
            except:
                pass
        
        return best_U, best_V, best_W, best_r
    
    # Try multiple partition strategies
    partitions = [
        ([1, 2], [2, 2], [2, 2, 2]),   # Original: A rows 1+2, B rows 2+2, C cols 2+2+2
        ([2, 1], [2, 2], [2, 2, 2]),   # A rows 2+1
        ([1, 2], [1, 1, 2], [2, 2, 2]),  # B rows 1+1+2
        ([1, 2], [2, 2], [1, 1, 2, 1, 1]),  # C cols finer
        ([1, 1, 1], [2, 2], [2, 2, 2]),  # A rows 1+1+1 (all schoolbook)
        ([1, 2], [1, 2, 1], [2, 2, 2]),  # B rows 1+2+1
        ([2, 1], [1, 2, 1], [2, 2, 2]),
    ]
    
    best_result = None
    best_rank = 999
    
    for a_split, b_split, c_split in partitions:
        if time.time() > deadline - 20:
            break
        
        U, V, W = build_decomposition(a_split, b_split, c_split)
        
        # Verify
        err = verify(U, V, W)
        if err > 1e-10:
            continue
        
        # Try rank reduction
        U_r, V_r, W_r, r = try_reduce_rank(U, V, W, 72)
        
        if r < best_rank:
            best_rank = r
            best_result = (U_r, V_r, W_r)
        
        if r <= 65:
            break
    
    if best_result is None:
        # Fallback to original construction
        U, V, W = build_decomposition([1, 2], [2, 2], [2, 2, 2])
        best_result = (np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float))
    
    U_best, V_best, W_best = best_result
    
    # Final verification
    final_err = verify(U_best.tolist(), V_best.tolist(), W_best.tolist())
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
