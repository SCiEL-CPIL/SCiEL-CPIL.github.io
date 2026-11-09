import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    n, m, p = DIMENSIONS  # 4, 4, 9

    # ---------- Schoolbook fallback (rank 144), guaranteed correct ----------
    def schoolbook():
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

    # ---------- Strassen 4x4x4 rank-49 decomposition generator ----------
    # We build the 4x4x4 matrix multiplication tensor decomposition using
    # Strassen's algorithm applied hierarchically.
    #
    # The 4x4 matrices are treated as 2x2 block matrices of 2x2 blocks.
    # The 7 Strassen products S1..S7 are computed, each of which is a
    # 2x2 matrix of 2x2 block products. Each block product is expanded
    # via Strassen 2x2 into 7 rank-1 terms.
    #
    # Total: 7 outer * 7 inner = 49 rank-1 terms for 4x4x4.

    def zero2():
        return np.zeros((2, 2), dtype=int)

    def blk(r, c, sign=1):
        """2x2 block coefficient matrix for block (r,c) with uniform sign."""
        return np.full((2, 2), sign, dtype=int)

    def blockmat(b00, b01, b10, b11):
        """Assemble a 4x4 coefficient matrix from four 2x2 blocks."""
        M = np.zeros((4, 4), dtype=int)
        M[0:2, 0:2] = b00
        M[0:2, 2:4] = b01
        M[2:4, 0:2] = b10
        M[2:4, 2:4] = b11
        return M

    def outer_combos():
        Z = zero2()
        # S1: X=[[A00,0],[0,A11]], Y=[[B00,0],[0,B11]]
        S1 = ([blk(0,0), Z], [Z, blk(1,1)]), ([blk(0,0), Z], [Z, blk(1,1)])
        # S2: X=[[0,0],[A10,A11]], Y=[[B00,0],[0,0]]
        S2 = ([Z, Z], [blk(1,0), blk(1,1)]), ([blk(0,0), Z], [Z, Z])
        # S3: X=[[A00,0],[0,0]], Y=[[0,B01],[0,-B11]]
        S3 = ([blk(0,0), Z], [Z, Z]), ([Z, blk(0,1)], [Z, blk(1,1, -1)])
        # S4: X=[[0,0],[0,A11]], Y=[[0,0],[B10,-B00]]
        S4 = ([Z, Z], [Z, blk(1,1)]), ([Z, Z], [blk(1,0), blk(0,0, -1)])
        # S5: X=[[A00,A01],[0,0]], Y=[[0,0],[0,B11]]
        S5 = ([blk(0,0), blk(0,1)], [Z, Z]), ([Z, Z], [Z, blk(1,1)])
        # S6: X=[[-A00,0],[A10,0]], Y=[[B00,B01],[0,0]]
        S6 = ([blk(0,0, -1), Z], [blk(1,0), Z]), ([blk(0,0), blk(0,1)], [Z, Z])
        # S7: X=[[0,A01],[0,-A11]], Y=[[0,0],[B10,B11]]
        S7 = ([Z, blk(0,1)], [Z, blk(1,1, -1)]), ([Z, Z], [blk(1,0), blk(1,1)])
        return [S1, S2, S3, S4, S5, S6, S7]

    def expand_outer(X, Y):
        """Return list of 7 (u, v, w_with_C_block_placement) for one outer S."""
        X00, X01 = X[0][0], X[0][1]
        X10, X11 = X[1][0], X[1][1]
        Y00, Y01 = Y[0][0], Y[0][1]
        Y10, Y11 = Y[1][0], Y[1][1]

        def add2(a, b): return a + b
        def sub2(a, b): return a - b

        # 7 inner M's: (Xj, Yj)
        M = [
            (add2(X00, X11), add2(Y00, Y11)),  # M1
            (add2(X10, X11), Y00),              # M2
            (X00, sub2(Y01, Y11)),              # M3
            (X11, sub2(Y10, Y00)),              # M4
            (add2(X00, X01), Y11),              # M5
            (sub2(X10, X00), add2(Y00, Y01)),   # M6
            (sub2(X01, X11), add2(Y10, Y11)),   # M7
        ]
        # C-block placement signs for M1..M7:
        placement = {
            0: [(0, 0, 1), (1, 1, 1)],   # M1: + in C00, + in C11
            1: [(1, 0, 1), (1, 1, 1)],   # M2: + in C10, + in C11
            2: [(0, 1, 1), (1, 1, -1)],  # M3: + in C01, - in C11
            3: [(0, 0, -1), (1, 0, 1)],  # M4: - in C00, + in C10
            4: [(0, 1, 1), (1, 1, -1)],  # M5: + in C01, - in C11
            5: [(0, 0, 1)],              # M6: + in C00
            6: [(0, 0, -1)],             # M7: - in C00
        }
        terms = []
        for j in range(7):
            Xj, Yj = M[j]
            u = blockmat(Xj[0][0], Xj[0][1], Xj[1][0], Xj[1][1])
            v = blockmat(Yj[0][0], Yj[0][1], Yj[1][0], Yj[1][1])
            w = u @ v  # 4x4
            terms.append((u, v, w, placement[j]))
        return terms

    def build_strassen_4x4x4():
        """Return list of 49 (u16, v16, w16) integer vectors."""
        terms = []
        for (X, Y) in outer_combos():
            for (u, v, w, placement) in expand_outer(X, Y):
                u16 = u.flatten().tolist()
                v16 = v.flatten().tolist()
                w16 = [0] * 16
                for (cbr, cbc, sign) in placement:
                    wblock = w[2*cbr:2*cbr+2, 2*cbc:2*cbc+2]
                    for a in range(2):
                        for b in range(2):
                            idx = (2*cbr + a) * 4 + (2*cbc + b)
                            w16[idx] += sign * int(wblock[a, b])
                terms.append((u16, v16, w16))
        return terms

    # Build the rank-114 decomposition:
    # 2 Strassen 4x4x4 (rank 49 each) for B-columns 0-3 and 4-7,
    # + schoolbook for B-column 8 (4*4*1 = 16 terms).
    # Total = 49 + 49 + 16 = 114.
    strassen_terms = build_strassen_4x4x4()
    assert len(strassen_terms) == 49, f"Expected 49 Strassen terms, got {len(strassen_terms)}"

    total_rank = 49 + 49 + 16
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    def add_strassen_block(offset, b_col_offset):
        """Add 49 Strassen terms for B-columns [b_col_offset, b_col_offset+4)."""
        for t, (u16, v16, w16) in enumerate(strassen_terms):
            col = offset + t
            for idx, val in enumerate(u16):
                if val != 0:
                    i, j = idx // 4, idx % 4
                    U[i * m + j][col] += val
            for idx, val in enumerate(v16):
                if val != 0:
                    j, k = idx // 4, idx % 4
                    V[j * p + (b_col_offset + k)][col] += val
            for idx, val in enumerate(w16):
                if val != 0:
                    k, i = idx // 4, idx % 4
                    W[k * n + i][col] += val

    add_strassen_block(0, 0)       # B columns 0-3
    add_strassen_block(49, 4)      # B columns 4-7

    # Schoolbook for B column 8: 4 (i) * 4 (j) * 1 (k) = 16 terms
    col = 98
    for i in range(n):
        for j in range(m):
            for k in [8]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # ---------- Fast vectorized verification ----------
    def verify_fast(U_mat, V_mat, W_mat, rank):
        """Verify tensor decomposition using vectorized numpy operations.
        
        The tensor decomposition requires:
        T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        to equal the matrix multiplication tensor (which is 1 when the
        index pattern matches, i.e., T[i,j,k] contributes A[i,j]*B[j,k] to C[i,k]).
        
        Actually, for exact matrix multiplication:
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        The rank-r decomposition gives:
        C[i,k] = sum_t sum_j U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] * A[i,j] * B[j,k]
        
        For this to equal sum_j A[i,j]*B[j,k] for all A,B, we need:
        sum_t sum_j U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = delta_{i,k pattern}
        
        More precisely, the tensor T[i,j,k] = sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
        must satisfy: for all i,k: sum_j T[i,j,k] * A[i,j] * B[j,k] = sum_j A[i,j]*B[j,k]
        
        This means T[i,j,k] must be such that the "effective" contribution works out.
        The standard condition is that the rank-r decomposition represents the
        matrix multiplication tensor exactly.
        
        A simpler verification: compute C from random A,B using the decomposition
        and compare to A@B.
        """
        rng = np.random.default_rng(42)
        A = rng.integers(-3, 4, size=(n, m)).astype(float)
        B = rng.integers(-3, 4, size=(m, p)).astype(float)
        C_expected = A @ B
        
        # C[i,k] = sum_j sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] * A[i,j] * B[j,k]
        # = sum_t W[k*n+i,t] * (sum_j U[i*m+j,t] * A[i,j]) * (sum_j V[j*p+k,t] * B[j,k])
        # Let's compute: for each t, define:
        #   at[i,t] = sum_j U[i*m+j, t] * A[i,j]  -> but this mixes i in the sum over j
        # Actually: sum_j U[i*m+j,t] * A[i,j] is not separable cleanly.
        # 
        # Let me just do it directly but vectorized over t:
        # C[i,k] = sum_t W[k*n+i,t] * sum_j (U[i*m+j,t] * A[i,j]) * (V[j*p+k,t] * B[j,k])
        # 
        # This is tricky to fully vectorize. Let me use a different approach:
        # Build the tensor T[i,j,k] = sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t]
        # then check that C = sum_j T[i,j,k]*A[i,j]*B[j,k] = A@B
        
        # Alternative: directly verify by computing the implied C
        # C[i,k] = sum_j sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] * A[i,j] * B[j,k]
        
        # Vectorized: loop over j (only 4 values)
        C_computed = np.zeros((n, p))
        U_arr = U_mat
        V_arr = V_mat
        W_arr = W_mat
        for j in range(m):
            # U_slice[i, t] = U[i*m+j, t]
            U_slice = U_arr.reshape(n, m, rank)[:, j, :]  # (n, rank)
            # V_slice[k, t] = V[j*p+k, t]
            V_slice = V_arr.reshape(m, p, rank)[j, :, :]  # (p, rank)
            # W_slice[k, i, t] = W[k*n+i, t]
            W_slice = W_arr.reshape(p, n, rank)  # (p, n, rank)
            
            # For this j: contribution to C[i,k] = sum_t U_slice[i,t] * V_slice[k,t] * W_slice[k,i,t] * A[i,j] * B[j,k]
            # = A[i,j] * B[j,k] * sum_t U_slice[i,t] * V_slice[k,t] * W_slice[k,i,t]
            # Let S_j[i,k] = sum_t U_slice[i,t] * V_slice[k,t] * W_slice[k,i,t]
            # Then C[i,k] += A[i,j] * B[j,k] * S_j[i,k]
            
            # S_j[i,k] = sum_t U_slice[i,t] * (V_slice[k,t] * W_slice[k,i,t])
            # = sum_t U_slice[i,t] * D_j[i,k,t] where D_j[i,k,t] = V_slice[k,t]*W_slice[k,i,t]
            # This is a batched dot product: S_j = U_slice @ D_j where D_j[i,k,t]... 
            # Actually S_j[i,k] = sum_t U_slice[i,t] * V_slice[k,t] * W_slice[k,i,t]
            
            # Compute V_slice[k,t] * W_slice[k,i,t] -> shape (p, n, rank)
            # W_slice is (p, n, rank), V_slice[k,t] needs to be broadcast to (p, n, rank)
            VW = V_slice[:, np.newaxis, :] * W_slice  # (p, n, rank)
            # S_j[i,k] = sum_t U_slice[i,t] * VW[k,i,t]
            # = sum_t U_slice[i,t] * VW[k,i,t] -> this is like a 3-way contraction
            # Reshape: S_j = (U_slice @ VW_permuted) ... let me just do it with einsum
            S_j = np.einsum('it,kit->ki', U_slice, VW)  # (n, p)
            
            C_computed += A[:, j:j+1] * B[j:j+1, :] * S_j
        
        return np.allclose(C_computed, C_expected, atol=1e-9)

    # Verify the base rank-114 decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    base_verified = verify_fast(U_arr, V_arr, W_arr, total_rank)
    
    if not base_verified:
        # Something is wrong; fall back to schoolbook
        return schoolbook()

    # ---------- Try to reduce rank by pruning zero or redundant columns ----------
    # Strategy: check if any column can be removed while preserving correctness.
    # A column t can be removed if removing it still gives correct results.
    # This is expensive to check for all columns, so we try a greedy approach.
    
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    best_rank = total_rank
    best_verified = True

    # Greedy column pruning: try removing each column one at a time
    # and check if the decomposition is still valid.
    # Since rank is 114, checking all 114 columns individually is feasible
    # if each check is fast.
    
    current_U = U_arr.copy()
    current_V = V_arr.copy()
    current_W = W_arr.copy()
    current_rank = total_rank
    active = list(range(total_rank))
    
    # Try to prune columns greedily
    for attempt in range(50):  # limit iterations for time safety
        if time.time() > deadline - 10:
            break
        
        pruned = False
        for t_idx in range(len(active)):
            if time.time() > deadline - 10:
                break
            
            # Try removing column at position t_idx in active list
            col = active[t_idx]
            # Remove column col
            keep = [c for c in active if c != col]
            if not keep:
                continue
            
            new_U = current_U[:, keep]
            new_V = current_V[:, keep]
            new_W = current_W[:, keep]
            
            # Quick check: if the column is all zeros, it's definitely removable
            if np.allclose(new_U, current_U[:, active]) and \
               np.allclose(new_V, current_V[:, active]) and \
               np.allclose(new_W, current_W[:, active]):
                # Column was all zeros, safe to remove
                active.remove(col)
                current_U = new_U
                current_V = new_V
                current_W = new_W
                current_rank = len(active)
                pruned = True
                break
            else:
                # Need to verify
                if len(active) - 1 < best_rank:
                    if verify_fast(new_U, new_V, new_W, len(active) - 1):
                        active.remove(col)
                        current_U = new_U
                        current_V = new_V
                        current_W = new_W
                        current_rank = len(active)
                        best_rank = current_rank
                        best_U = current_U.copy()
                        best_V = current_V.copy()
                        best_W = current_W.copy()
                        pruned = True
                        break
    
        if not pruned:
            break
    
    # If we pruned some columns, update
    if best_rank < total_rank:
        U_final = best_U
        V_final = best_V
        W_final = best_W
        final_rank = best_rank
    else:
        U_final = U_arr
        V_final = V_arr
        W_final = W_arr
        final_rank = total_rank

    # Convert back to lists
    U_out = U_final.tolist()
    V_out = V_final.tolist()
    W_out = W_final.tolist()

    # Final verification
    if verify_fast(U_final, V_final, W_final, final_rank):
        return {"U": U_out, "V": V_out, "W": W_out}
    else:
        # Fallback to the base rank-114
        if base_verified:
            return {"U": U_arr.tolist(), "V": V_arr.tolist(), "W": W_arr.tolist()}
        else:
            return schoolbook()

