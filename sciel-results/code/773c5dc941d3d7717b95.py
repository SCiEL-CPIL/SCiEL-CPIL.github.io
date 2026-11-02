import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Use block-Strassen: 3x3 blocks of 2x2 matrices
    # This gives rank 9 * 3 * 7 = 189
    nb = 3  # number of blocks in each dimension
    bs = 2  # block size
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # 7 multiplications M0..M6
    # We need to express the 7 rank-1 terms for 2x2 matrix mult
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M5
    # C11 = M0 - M1 + M2 + M5
    #
    # M0 = (A00 + A11)(B00 + B11)
    # M1 = (A10 + A11)B00
    # M2 = A00(B01 - B11)
    # M3 = A11(B10 - B00)
    # M4 = (A00 + A01)B11
    # M5 = (A10 - A00)(B00 + B01)
    # M6 = (A01 - A11)(B10 + B11)
    
    # For each of the 7 Strassen multiplications, we need:
    # - U factor: which elements of A block (2x2) are multiplied
    # - V factor: which elements of B block (2x2) are multiplied
    # - W factor: which elements of C block (2x2) receive the product
    
    # Let's define the Strassen decomposition for a single 2x2 block
    # A is 2x2, B is 2x2, C = A*B is 2x2
    
    # M0 = (A00+A11)(B00+B11)
    #   U: A00 + A11, V: B00 + B11
    #   C00 += M0, C11 += M0
    
    # M1 = (A10+A11)B00
    #   U: A10 + A11, V: B00
    #   C10 += M1, C11 -= M1
    
    # M2 = A00(B01-B11)
    #   U: A00, V: B01 - B11
    #   C01 += M2, C11 += M2
    
    # M3 = A11(B10-B00)
    #   U: A11, V: B10 - B00
    #   C00 += M3, C10 += M3
    
    # M4 = (A00+A01)B11
    #   U: A00 + A01, V: B11
    #   C00 -= M4, C01 += M4
    
    # M5 = (A10-A00)(B00+B01)
    #   U: A10 - A00, V: B00 + B01
    #   C11 += M5
    
    # M6 = (A01-A11)(B10+B11)
    #   U: A01 - A11, V: B10 + B11
    #   C00 += M6
    
    # For each Strassen term s (0-6), we define:
    # a_coeffs: list of (row, col, coeff) for the A block
    # b_coeffs: list of (row, col, coeff) for the B block
    # c_coeffs: list of (row, col, coeff) for the C block
    
    strassen_a = [
        [(0,0,1), (1,1,1)],      # M0: A00 + A11
        [(1,0,1), (1,1,1)],      # M1: A10 + A11
        [(0,0,1)],               # M2: A00
        [(1,1,1)],               # M3: A11
        [(0,0,1), (0,1,1)],      # M4: A00 + A01
        [(1,0,1), (0,0,-1)],     # M5: A10 - A00
        [(0,1,1), (1,1,-1)],     # M6: A01 - A11
    ]
    strassen_b = [
        [(0,0,1), (1,1,1)],      # M0: B00 + B11
        [(0,0,1)],               # M1: B00
        [(0,1,1), (1,1,-1)],     # M2: B01 - B11
        [(1,0,1), (0,0,-1)],     # M3: B10 - B00
        [(1,1,1)],               # M4: B11
        [(0,0,1), (0,1,1)],      # M5: B00 + B01
        [(1,0,1), (1,1,1)],      # M6: B10 + B11
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # M0: C00 += 1, C11 += 1
        [(1,0,1), (1,1,-1)],     # M1: C10 += 1, C11 -= 1
        [(0,1,1), (1,1,1)],      # M2: C01 += 1, C11 += 1
        [(0,0,1), (1,0,1)],      # M3: C00 += 1, C10 += 1
        [(0,0,-1), (0,1,1)],     # M4: C00 -= 1, C01 += 1
        [(1,1,1)],               # M5: C11 += 1
        [(0,0,1)],               # M6: C00 += 1
    ]
    
    total_rank = nb * nb * nb * 7  # 27 * 7 = 189
    
    U = np.zeros((n*m, total_rank), dtype=np.float64)
    V = np.zeros((m*p, total_rank), dtype=np.float64)
    W = np.zeros((p*n, total_rank), dtype=np.float64)
    
    col = 0
    for ib in range(nb):  # block row for A and C
        for jb in range(nb):  # block col for A and block row for B
            for kb in range(nb):  # block col for B and C
                for s in range(7):  # Strassen term
                    # A block: rows ib*bs to (ib+1)*bs, cols jb*bs to (jb+1)*bs
                    # B block: rows jb*bs to (jb+1)*bs, cols kb*bs to (kb+1)*bs
                    # C block: rows ib*bs to (ib+1)*bs, cols kb*bs to (kb+1)*bs
                    
                    # Fill U: for each (lr, lc, coeff) in strassen_a[s]
                    for (lr, lc, coeff) in strassen_a[s]:
                        gi = ib * bs + lr
                        gj = jb * bs + lc
                        U[gi * m + gj, col] += coeff
                    
                    # Fill V: for each (lr, lc, coeff) in strassen_b[s]
                    for (lr, lc, coeff) in strassen_b[s]:
                        gj = jb * bs + lr
                        gk = kb * bs + lc
                        V[gj * p + gk, col] += coeff
                    
                    # Fill W: for each (lr, lc, coeff) in strassen_c[s]
                    for (lr, lc, coeff) in strassen_c[s]:
                        gi = ib * bs + lr
                        gk = kb * bs + lc
                        W[gk * n + gi, col] += coeff
                    
                    col += 1
    
    # Now attempt rank reduction by finding linear dependencies
    # among columns of U, V, W
    # Two columns c1, c2 can be merged if:
    #   U[:,c2] = alpha * U[:,c1] and V[:,c2] = beta * V[:,c1]
    #   and W[:,c2] = alpha * beta * W[:,c1]
    # In that case, we can remove column c2 and add its W contribution to c1.
    
    if time.time() > deadline - 15:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find mergeable pairs
    # This is O(rank^2 * dim) which could be expensive for rank 189
    # 189^2 * 36 ≈ 1.2M operations, should be fast enough
    
    used = [True] * total_rank
    merged_count = 0
    
    for i in range(total_rank):
        if not used[i]:
            continue
        for j in range(i + 1, total_rank):
            if not used[j]:
                continue
            if time.time() > deadline - 10:
                break
            
            # Check if U[:,j] is proportional to U[:,i]
            ui = U[:, i]
            uj = U[:, j]
            
            # Find ratio from first non-zero element of ui
            ratio_u = None
            for idx in range(n * m):
                if abs(ui[idx]) > 1e-12:
                    if abs(uj[idx]) < 1e-12:
                        ratio_u = None
                    else:
                        ratio_u = uj[idx] / ui[idx]
                    break
                elif abs(uj[idx]) > 1e-12:
                    ratio_u = None
                    break
            
            if ratio_u is None:
                continue
            
            # Verify the ratio
            valid = True
            for idx in range(n * m):
                expected = ratio_u * ui[idx]
                if abs(uj[idx] - expected) > 1e-9 * max(1, abs(expected)):
                    valid = False
                    break
            if not valid:
                continue
            
            # Check if V[:,j] is proportional to V[:,i]
            vi = V[:, i]
            vj = V[:, j]
            
            ratio_v = None
            for idx in range(m * p):
                if abs(vi[idx]) > 1e-12:
                    if abs(vj[idx]) < 1e-12:
                        ratio_v = None
                    else:
                        ratio_v = vj[idx] / vi[idx]
                    break
                elif abs(vj[idx]) > 1e-12:
                    ratio_v = None
                    break
            
            if ratio_v is None:
                continue
            
            # Verify the ratio
            valid = True
            for idx in range(m * p):
                expected = ratio_v * vi[idx]
                if abs(vj[idx] - expected) > 1e-9 * max(1, abs(expected)):
                    valid = False
                    break
            if not valid:
                continue
            
            # Check if W[:,j] = ratio_u * ratio_v * W[:,i]
            wi = W[:, i]
            wj = W[:, j]
            scale = ratio_u * ratio_v
            
            valid = True
            for idx in range(p * n):
                expected = scale * wi[idx]
                if abs(wj[idx] - expected) > 1e-9 * max(1, abs(expected)):
                    valid = False
                    break
            if valid:
                W[:, i] = wi + wj
                used[j] = False
                merged_count += 1
                if time.time() > deadline - 10:
                    break
    
    if merged_count > 0:
        new_rank = sum(used)
        final_U = U[:, [c for c in range(total_rank) if used[c]]]
        final_V = V[:, [c for c in range(total_rank) if used[c]]]
        final_W = W[:, [c for c in range(total_rank) if used[c]]]
        return {"U": final_U.tolist(), "V": final_V.tolist(), "W": final_W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
