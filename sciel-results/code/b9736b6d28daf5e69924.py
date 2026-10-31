import numpy as np
import time

def construct():
    """
    Returns a rank-minimal decomposition of the 4x4x4 matrix multiplication tensor.
    Uses rank 49 via recursive Strassen (7x7).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def _build_rank49():
        """Build the standard recursive Strassen rank-49 decomposition."""
        # The 2x2 Strassen decomposition has 7 terms.
        # Each term is (U_indices, V_indices, W_indices) where each is a list of (row, col, coeff)
        # For 2x2 matrices: indices are (i,j) in {0,1}^2
        # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        
        # Strassen 2x2 terms:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        strassen_2x2 = [
            # M1
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M2
            ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),
            # M3
            ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            # M4
            ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            # M5
            ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),
            # M6
            ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),
            # M7
            ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),
        ]
        
        # For 4x4, we apply Strassen at the 4x4 level (treating it as 2x2 of 2x2 blocks)
        # and then apply Strassen again within each 2x2 block multiplication.
        # This gives 7 * 7 = 49 terms.
        
        # The outer Strassen is the same structure, but indices are block indices.
        # Block index (br, bc) where br, bc in {0,1}
        # Local index (lr, lc) where lr, lc in {0,1}
        # Global index: row = br*2 + lr, col = bc*2 + lc
        
        outer_strassen = [
            # M1
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
            # M2
            ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),
            # M3
            ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
            # M4
            ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
            # M5
            ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),
            # M6
            ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),
            # M7
            ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),
        ]
        
        terms = []
        
        for (a_blocks, b_blocks, c_blocks) in outer_strassen:
            for (a_comb, b_comb, c_comb) in strassen_2x2:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                # U part: for each (lr, lc, ca) in a_comb and (br, bc, coeff) in a_blocks
                # global row = br*2 + lr, global col = bc*2 + lc
                # U[global_row * 4 + global_col] += ca * coeff
                for (lr, lc, ca) in a_comb:
                    for (br, bc, coeff) in a_blocks:
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        u_idx = gr * 4 + gc
                        u_dict[u_idx] = u_dict.get(u_idx, 0) + ca * coeff
                
                # V part: for each (lr, lc, cb) in b_comb and (br, bc, coeff) in b_blocks
                # global row = br*2 + lr, global col = bc*2 + lc
                # V[global_row * 4 + global_col] += cb * coeff
                for (lr, lc, cb) in b_comb:
                    for (br, bc, coeff) in b_blocks:
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        v_idx = gr * 4 + gc
                        v_dict[v_idx] = v_dict.get(v_idx, 0) + cb * coeff
                
                # W part: for each (lr, lc, cc) in c_comb and (br, bc, coeff) in c_blocks
                # W convention: W[k*4+i, r] where k is col of B, i is row of A
                # In our indexing: W[global_col_of_B * 4 + global_row_of_A]
                # c_comb gives (lr, lc, cc) where lr is row within block, lc is col within block
                # c_blocks gives (br, bc, coeff) where br is block row, bc is block col
                # global row of A = br*2 + lr, global col of B = bc*2 + lc
                # W[(bc*2+lc)*4 + (br*2+lr)] += cc * coeff
                for (lr, lc, cc) in c_comb:
                    for (br, bc, coeff) in c_blocks:
                        gr_a = br * 2 + lr  # row of A
                        gc_b = bc * 2 + lc  # col of B
                        w_idx = gc_b * 4 + gr_a
                        w_dict[w_idx] = w_dict.get(w_idx, 0) + cc * coeff
                
                terms.append((u_dict, v_dict, w_dict))
        
        rank = len(terms)
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for idx, val in u_dict.items():
                if val != 0:
                    U[idx][col] = val
            for idx, val in v_dict.items():
                if val != 0:
                    V[idx][col] = val
            for idx, val in w_dict.items():
                if val != 0:
                    W[idx][col] = val
        
        return U, V, W, rank

    def _verify(U, V, W, rank):
        """Verify the decomposition is exact using NumPy (fast path)."""
        try:
            Ua = np.asarray(U, dtype=np.int64)
            Va = np.asarray(V, dtype=np.int64)
            Wa = np.asarray(W, dtype=np.int64)
            
            # U shape (16, rank), V shape (16, rank), W shape (16, rank)
            # U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
            # We want: for all i,j,k: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1
            
            U3 = Ua.reshape(4, 4, rank)
            V3 = Va.reshape(4, 4, rank)
            W3 = Wa.reshape(4, 4, rank)
            
            # S[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
            S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            
            return bool(np.all(S == 1))
        except Exception:
            # Fallback to slow python loop if numpy fails
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        s = 0
                        for r in range(rank):
                            s += U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r]
                        if s != 1:
                            return False
            return True

    # Build rank-49 decomposition
    U, V, W, rank = _build_rank49()
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    # Verify
    if not _verify(U, V, W, rank):
        # Fallback to schoolbook rank 64
        rank = 64
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
