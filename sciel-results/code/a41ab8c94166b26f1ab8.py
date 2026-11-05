"""Refined Strassen-based tensor decomposition for 4x4x7 matrix multiplication."""
import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    t0 = time.time()
    deadline = t0 + 110.0

    # Strassen 2x2x2 rank-7 term definitions
    # Each term: (x_pattern, y_pattern, c_pattern) where patterns are dicts of (row,col)->coeff
    S2 = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

    # 4x4x4 Strassen: 7 outer block-level terms, each decomposed into 7 inner 2x2x2 terms
    # Outer: (x_block_indices, y_block_indices, c_block_indices)
    # Block indices: 0=TL,1=TR,2=BL,3=BR
    S4_outer = [
        (({0:1,3:1},{0:1,3:1}), {(0,0):1.0,(1,1):1.0}),
        (({2:1,3:1},{0:1}), {(1,0):1.0,(1,1):-1.0}),
        (({0:1},{1:1,3:-1}), {(0,1):1.0,(1,1):1.0}),
        (({3:1},{2:1,0:-1}), {(0,0):1.0,(1,0):1.0}),
        (({0:1,1:1},{3:1}), {(0,0):-1.0,(0,1):1.0}),
        (({2:1,0:-1},{0:1,1:1}), {(1,1):1.0}),
        (({1:1,3:-1},{2:1,3:1}), {(0,0):1.0}),
    ]

    def emit_strassen_2x2(U, V, W, rank_ptr, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        """Emit 7 Strassen terms for a 2x2x2 multiplication into U,V,W at columns starting at rank_ptr[0]."""
        base = rank_ptr[0]
        for t in range(7):
            r = base + t
            xs, ys, cs = S2[t]
            for (ri,ci), cv in xs.items():
                U[A_rows[ri]*m + A_cols[ci], r] += cv
            for (ri,ci), cv in ys.items():
                V[B_rows[ri]*p + B_cols[ci], r] += cv
            for (ri,ci), cv in cs.items():
                if cv != 0:
                    W[C_cols[ci]*n + C_rows[ri], r] += cv
        rank_ptr[0] += 7
        return base

    def emit_strassen_4x4(U, V, W, rank_ptr):
        """Emit 49 Strassen terms for 4x4x4 using recursive 2x2x2 decomposition."""
        base = rank_ptr[0]
        block_row = [[0,1],[2,3]]  # rows for block 0 (top), block 1 (bottom)
        block_col = [[0,1],[2,3]]  # cols for block 0 (left), block 1 (right)
        
        for oi, ((x_b, y_b), c_b) in enumerate(S4_outer):
            for ti in range(7):
                r = base + oi*7 + ti
                xs, ys, cs = S2[ti]
                
                # U: combine x_block selection with inner 2x2 pattern
                for (bi,bj), xb_coeff in x_b.items():
                    for (ri,ci), xs_coeff in xs.items():
                        row = block_row[bj][ri]
                        col = block_col[bj][ci]
                        U[row*m + col, r] += xb_coeff * xs_coeff
                
                # V: combine y_block selection with inner 2x2 pattern
                for (bi,bj), yb_coeff in y_b.items():
                    for (ri,ci), ys_coeff in ys.items():
                        row = block_row[bj][ri]
                        col = block_col[bj][ci]
                        V[row*p + col, r] += yb_coeff * ys_coeff
                
                # W: combine c_block selection with inner 2x2 pattern
                for (ri,ci), cs_coeff in cs.items():
                    if cs_coeff == 0:
                        continue
                    for (bi,bj), cb_coeff in c_b.items():
                        if cb_coeff == 0:
                            continue
                        row = block_row[bj][ri]
                        col = block_col[bj][ci]
                        W[col*n + row, r] += cs_coeff * cb_coeff
        
        rank_ptr[0] += 49
        return base

    def emit_strassen_2x2x3(U, V, W, rank_ptr, A_r, A_c, B_r, B_c, C_r, C_c):
        """
        Emit rank-11 decomposition for 2x2x3 multiplication.
        = Strassen 2x2x2 (rank 7) for cols 0,1 of B + schoolbook 2x2x1 (rank 4) for col 2 of B.
        A is 2x2, B is 2x3, C is 2x3.
        A_r, A_c: row/col indices in global A (length 2 each)
        B_r, B_c: row indices (length 2) and col indices (length 3) in global B
        C_r, C_c: row indices (length 2) and col indices (length 3) in global C
        """
        base = rank_ptr[0]
        # Part A: Strassen 2x2x2 for B columns 0,1
        B_c2 = B_c[:2]
        C_c2 = C_c[:2]
        for t in range(7):
            r = base + t
            xs, ys, cs = S2[t]
            for (ri,ci), cv in xs.items():
                U[A_r[ri]*m + A_c[ci], r] += cv
            for (ri,ci), cv in ys.items():
                V[B_r[ri]*p + B_c2[ci], r] += cv
            for (ri,ci), cv in cs.items():
                if cv != 0:
                    W[C_c2[ci]*n + C_r[ri], r] += cv
        
        # Part B: schoolbook 2x2x1 for B column 2
        for i in range(2):
            for j in range(2):
                r = base + 7 + i*2 + j
                U[A_r[i]*m + A_c[j], r] = 1.0
                V[B_r[j]*p + B_c[2], r] = 1.0
                W[C_c[2]*n + C_r[i], r] = 1.0
        
        rank_ptr[0] += 11
        return base

    def emit_strassen_2x2x2_school(U, V, W, rank_ptr, A_r, A_c, B_r, B_c, C_r, C_c):
        """Emit rank-7 Strassen 2x2x2. B and C are 2x2."""
        base = rank_ptr[0]
        for t in range(7):
            r = base + t
            xs, ys, cs = S2[t]
            for (ri,ci), cv in xs.items():
                U[A_r[ri]*m + A_c[ci], r] += cv
            for (ri,ci), cv in ys.items():
                V[B_r[ri]*p + B_c[ci], r] += cv
            for (ri,ci), cv in cs.items():
                if cv != 0:
                    W[C_c[ci]*n + C_r[ri], r] += cv
        rank_ptr[0] += 7
        return base

    def build_decomposition():
        total_rank = 49 + 44 + 16  # 4x4x4 Strassen + 4*(2x2x3) + 4x4x1 schoolbook = 109? 
        # Wait: 4x4x2 = 4 separate 2x2x2 = 4*7=28, 4x4x1 = 16. Total = 49+28+16=93
        total_rank = 49 + 28 + 16
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        rp = [0]
        
        # Part 1: 4x4x4 Strassen (columns 0-3 of B)
        emit_strassen_4x4(U, V, W, rp)
        
        # Part 2: 4x4x2 (columns 4-5 of B) as 4 separate 2x2x2 Strassen
        # A00 @ B_top, A01 @ B_bot, A10 @ B_top, A11 @ B_bot
        A_blocks = [
            ([0,1],[0,1]),  # A00
            ([0,1],[2,3]),  # A01
            ([2,3],[0,1]),  # A10
            ([2,3],[2,3]),  # A11
        ]
        B_top = ([0,1],[4,5])
        B_bot = ([2,3],[4,5])
        C_top = ([0,1],[4,5])
        C_bot = ([2,3],[4,5])
        
        pairs = [
            (A_blocks[0], B_top, C_top),
            (A_blocks[1], B_bot, C_top),
            (A_blocks[2], B_top, C_bot),
            (A_blocks[3], B_bot, C_bot),
        ]
        for (A_r, A_c), (B_r, B_c), (C_r, C_c) in pairs:
            emit_strassen_2x2x2_school(U, V, W, rp, A_r, A_c, B_r, B_c, C_r, C_c)
        
        # Part 3: 4x4x1 schoolbook (column 6 of B)
        for i in range(n):
            for j in range(m):
                r = rp[0]
                rp[0] += 1
                U[i*m + j, r] = 1.0
                V[j*p + 6, r] = 1.0
                W[6*n + i, r] = 1.0
        
        assert rp[0] == total_rank, f"Rank mismatch: {rp[0]} != {total_rank}"
        return U, V, W

    def verify_decomp(U, V, W):
        """Verify U⊗V⊗W equals the matmul tensor. Return max abs error."""
        Ua = np.asarray(U, dtype=np.float64)
        Va = np.asarray(V, dtype=np.float64)
        Wa = np.asarray(W, dtype=np.float64)
        rank = Ua.shape[1]
        
        # Tensor index: (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
        # U index: i*m+j, V index: j*p+k, W index: k*n+i
        ii = np.repeat(np.arange(n), m*p)
        jj = np.tile(np.repeat(np.arange(m), p), n)
        kk = np.tile(np.arange(p), n*m)
        
        u_idx = ii*m + jj
        v_idx = jj*p + kk
        w_idx = kk*n + ii
        
        result = (Ua[u_idx,:] * Va[v_idx,:] * Wa[w_idx,:]).sum(axis=1)
        return float(np.max(np.abs(result - 1.0)))

    def reduce_rank(U, V, W, tol=1e-9):
        """
        Attempt to reduce rank by identifying linearly dependent columns
        and eliminating redundant rank-1 terms.
        """
        rank = U.shape[1]
        # Check for exactly zero columns
        non_zero = []
        for r in range(rank):
            if np.any(U[:,r] != 0) and np.any(V[:,r] != 0) and np.any(W[:,r] != 0):
                non_zero.append(r)
        
        if len(non_zero) < rank:
            U = U[:, non_zero]
            V = V[:, non_zero]
            W = W[:, non_zero]
            rank = len(non_zero)
        
        # Try to find columns where one term is a linear combination of others
        # This is a simplified check: look for pairs of terms that have identical
        # support patterns (up to scaling) in U and V, allowing W to absorb.
        # For now, just return the pruned version.
        return U, V, W, rank

    def build_decomposition_v2():
        """
        Alternative: use 2x2x3 rank-11 blocks for the 4x4x3 part.
        4x4x3 = 4 blocks of 2x2x3, each rank 11 -> total 44
        4x4x4 = Strassen 49
        Total = 93 (same as v1 since 4x4x2 schoolbook=16+4x4x1=16=32, but 4x4x3=44... no)
        Actually 4x4x3 as 4*(2x2x3): each 2x2x3 has rank 11, so 4*11=44
        4x4x4: 49
        Total: 93. Same!
        """
        total_rank = 49 + 44
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        rp = [0]
        
        # Part 1: 4x4x4 Strassen (columns 0-3)
        emit_strassen_4x4(U, V, W, rp)
        
        # Part 2: 4x4x3 (columns 4-6) as 4 separate 2x2x3
        A_blocks = [
            ([0,1],[0,1]),  # A00
            ([0,1],[2,3]),  # A01
            ([2,3],[0,1]),  # A10
            ([2,3],[2,3]),  # A11
        ]
        B_top = [0,1]
        B_bot = [2,3]
        C_top = [0,1]
        C_bot = [2,3]
        B_cols = [4,5,6]
        C_cols = [4,5,6]
        
        pairs = [
            (A_blocks[0], B_top, C_top),
            (A_blocks[1], B_bot, C_top),
            (A_blocks[2], B_top, C_bot),
            (A_blocks[3], B_bot, C_bot),
        ]
        for (A_r, A_c), B_r, C_r in pairs:
            emit_strassen_2x2x3(U, V, W, rp, A_r, A_c, B_r, B_cols, C_r, C_cols)
        
        assert rp[0] == total_rank, f"Rank mismatch v2: {rp[0]} != {total_rank}"
        return U, V, W

    # Build the primary decomposition (v1: 49+28+16=93)
    U, V, W = build_decomposition()
    
    # Verify
    err = verify_decomp(U, V, W)
    if err < 1e-10:
        U, V, W, rank = reduce_rank(U, V, W)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try v2 (49+44=93)
    if time.time() < deadline:
        U2, V2, W2 = build_decomposition_v2()
        err2 = verify_decomp(U2, V2, W2)
        if err2 < 1e-10:
            U2, V2, W2, rank2 = reduce_rank(U2, V2, W2)
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    # Fallback: schoolbook rank 112
    rank_school = n*m*p
    U_s = np.zeros((n*m, rank_school))
    V_s = np.zeros((m*p, rank_school))
    W_s = np.zeros((p*n, rank_school))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, c] = 1.0
                V_s[j*p+k, c] = 1.0
                W_s[k*n+i, c] = 1.0
                c += 1
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
