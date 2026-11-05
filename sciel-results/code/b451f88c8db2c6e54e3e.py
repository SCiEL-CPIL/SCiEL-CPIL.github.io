"""Matrix multiplication 4,4,9 tensor decomposition - rank reduction via linear algebra."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Build target tensor (sparse)
    target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target[a_idx, b_idx, c_idx] = 1.0
    
    target_flat = target.flatten()
    
    def verify(U, V, W, rank):
        T = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T.flatten(), target_flat, atol=1e-9)
    
    # Build rank-114 solution: 2x Strassen 4x4x4 (49 each) + schoolbook 4x4x1 (16)
    def build_rank_114():
        U_cols, V_cols, W_cols = [], [], []
        
        def add_term(u, v, w):
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
        
        # Strassen 2x2x2: 7 terms
        # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        s2x2 = [
            ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
            ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
            ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
            ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
            ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
            ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
            ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
        ]
        
        def add_strassen_4x4(B_col_off):
            def add_inner(A_blocks, B_blocks, C_blocks):
                for uc, vc, wc in s2x2:
                    u_vec = [0.0] * 16
                    v_vec = [0.0] * 36
                    w_vec = [0.0] * 36
                    
                    for loc, coeff in uc.items():
                        li, lj = loc // 2, loc % 2
                        for br, bc, bc_c in A_blocks:
                            gi, gj = br*2 + li, bc*2 + lj
                            u_vec[gi*m + gj] += coeff * bc_c
                    
                    for loc, coeff in vc.items():
                        lj, lk = loc // 2, loc % 2
                        for br, bc, bc_c in B_blocks:
                            gj, gk = br*2 + lj, B_col_off + bc*2 + lk
                            v_vec[gj*p + gk] += coeff * bc_c
                    
                    for loc, coeff in wc.items():
                        li, lk = loc // 2, loc % 2
                        for br, bc, bc_c in C_blocks:
                            gi, gk = br*2 + li, B_col_off + bc*2 + lk
                            w_vec[gk*n + gi] += coeff * bc_c
                    
                    add_term(u_vec, v_vec, w_vec)
            
            # 7 outer Strassen combinations
            add_inner([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
            add_inner([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)])
            add_inner([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)])
            add_inner([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])
            add_inner([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)])
            add_inner([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
            add_inner([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])
        
        add_strassen_4x4(0)
        add_strassen_4x4(4)
        
        # Schoolbook for last column (k=8)
        for i in range(4):
            for j in range(4):
                u_vec = [0.0] * 16
                v_vec = [0.0] * 36
                w_vec = [0.0] * 36
                u_vec[i*m + j] = 1.0
                v_vec[j*p + 8] = 1.0
                w_vec[8*n + i] = 1.0
                add_term(u_vec, v_vec, w_vec)
        
        rank = len(U_cols)
        U = np.array(U_cols).T  # (16, rank)
        V = np.array(V_cols).T  # (36, rank)
        W = np.array(W_cols).T  # (36, rank)
        return U, V, W, rank
    
    U, V, W, rank = build_rank_114()
    
    # Verify the base solution
    if not verify(U, V, W, rank):
        # Fallback: pure schoolbook rank 144
        U_fb = np.zeros((16, 144))
        V_fb = np.zeros((36, 144))
        W_fb = np.zeros((36, 144))
        idx = 0
        for i in range(4):
            for j in range(4):
                for k in range(9):
                    U_fb[i*4+j, idx] = 1
                    V_fb[j*9+k, idx] = 1
                    W_fb[k*4+i, idx] = 1
                    idx += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Strategy 1: Try to eliminate linearly dependent columns
    # The tensor equation is: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T
    # For each (a,b) pair: sum_r U[a,r]*V[b,r]*W[:,r] = T[a,b,:]
    # So: D @ W = T_mat where D[a*36+b, r] = U[a,r] * V[b,r]
    
    D = np.zeros((16*36, rank))
    for r in range(rank):
        D[:, r] = np.kron(U[:, r], V[:, r])
    
    T_mat = target.reshape(16*36, 36)
    
    # Check rank of D using SVD for numerical stability
    try:
        svals = np.linalg.svd(D, compute_uv=False)
        D_rank = int(np.sum(svals > 1e-10 * svals[0]))
    except:
        D_rank = rank
    
    if D_rank < best_rank:
        # Find a numerically stable basis using QR with column pivoting
        Q, R, pivots = np.linalg.qr(D, mode='complete', pivoting=True)
        independent = sorted(pivots[:D_rank])
        
        U_new = U[:, independent]
        V_new = V[:, independent]
        D_new = np.zeros((16*36, D_rank))
        for r in range(D_rank):
            D_new[:, r] = np.kron(U_new[:, r], V_new[:, r])
        
        # Solve D_new @ W_new = T_mat (both are (16*36, D_rank) and (16*36, 36))
        W_new, residuals, _, _ = np.linalg.lstsq(D_new, T_mat, rcond=None)
        
        if verify(U_new, V_new, W_new, D_rank):
            best_rank = D_rank
            best_U, best_V, best_W = U_new, V_new, W_new
    
    # Strategy 2: Try removing one column at a time and re-solving
    # For each column r, check if it can be expressed as a combination of others
    for attempt in range(best_rank):
        if time.time() > deadline - 20:
            break
        
        # Try removing column 'attempt'
        keep = [i for i in range(best_rank) if i != attempt]
        U_sub = best_U[:, keep]
        V_sub = best_V[:, keep]
        
        D_sub = np.zeros((16*36, len(keep)))
        for r in range(len(keep)):
            D_sub[:, r] = np.kron(U_sub[:, r], V_sub[:, r])
        
        try:
            W_sub, residuals, _, _ = np.linalg.lstsq(D_sub, T_mat, rcond=None)
            
            if verify(U_sub, V_sub, W_sub, len(keep)):
                best_rank = len(keep)
                best_U, best_V, best_W = U_sub, V_sub, W_sub
        except:
            pass
    
    # Strategy 3: Try removing pairs of columns simultaneously
    # This can find reductions that single-column removal misses
    if time.time() < deadline - 25:
        for i in range(best_rank):
            if time.time() > deadline - 20:
                break
            for j in range(i + 1, best_rank):
                if time.time() > deadline - 18:
                    break
                keep = [k for k in range(best_rank) if k != i and k != j]
                U_sub = best_U[:, keep]
                V_sub = best_V[:, keep]
                
                D_sub = np.zeros((16*36, len(keep)))
                for r in range(len(keep)):
                    D_sub[:, r] = np.kron(U_sub[:, r], V_sub[:, r])
                
                try:
                    W_sub, residuals, _, _ = np.linalg.lstsq(D_sub, T_mat, rcond=None)
                    if verify(U_sub, V_sub, W_sub, len(keep)):
                        best_rank = len(keep)
                        best_U, best_V, best_W = U_sub, V_sub, W_sub
                        break  # Restart pair search with new best
                except:
                    pass
            if time.time() > deadline - 18:
                break
    
    # Strategy 4: Exploit the shared U structure between the two Strassen blocks
    # U[:, 0:49] == U[:, 49:98] since both use the same A matrix
    # Try to find a joint decomposition that uses fewer total terms
    if time.time() < deadline - 25:
        # The key insight: if U1_r == U2_r for all r, we can try to find
        # a representation where some terms serve both blocks
        # 
        # Alternative approach: try to find if any of the 49 Strassen terms
        # for block 1 can be "absorbed" into the block 2 terms by adjusting
        # V and W vectors appropriately
        #
        # More practically: try greedy removal with the constraint that
        # we remove at most one from each Strassen block pair
        for i in range(49):
            if time.time() > deadline - 20:
                break
            # Try removing term i from block 1 AND term i from block 2
            keep = [k for k in range(best_rank) if k != i and k != i + 49]
            if len(keep) < best_rank - 2:
                continue
            U_sub = best_U[:, keep]
            V_sub = best_V[:, keep]
            
            D_sub = np.zeros((16*36, len(keep)))
            for r in range(len(keep)):
                D_sub[:, r] = np.kron(U_sub[:, r], V_sub[:, r])
            
            try:
                W_sub, residuals, _, _ = np.linalg.lstsq(D_sub, T_mat, rcond=None)
                if verify(U_sub, V_sub, W_sub, len(keep)):
                    best_rank = len(keep)
                    best_U, best_V, best_W = U_sub, V_sub, W_sub
            except:
                pass
    
    # Final: ensure we have a valid solution
    if best_rank < 144:
        U_out = best_U
        V_out = best_V
        W_out = best_W
    else:
        U_out, V_out, W_out = U, V, W
    
    # Round to half-integers for cleanliness
    U_out = np.round(U_out * 2) / 2.0
    V_out = np.round(V_out * 2) / 2.0
    W_out = np.round(W_out * 2) / 2.0
    
    final_rank = U_out.shape[1]
    
    # Final verification
    if not verify(U_out, V_out, W_out, final_rank):
        # Fallback to the original 114
        U_out, V_out, W_out = U, V, W
        final_rank = 114
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
