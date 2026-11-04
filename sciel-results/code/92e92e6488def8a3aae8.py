import numpy as np
import time

def construct():
    start = time.time()
    n, m, p = 4, 4, 6
    DEADLINE = 140.0
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        for idx, coeff in u_dict.items():
            u_col[idx] += coeff
        for idx, coeff in v_dict.items():
            v_col[idx] += coeff
        for idx, coeff in w_dict.items():
            w_col[idx] += coeff
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def add_term_ijk(u_dict, v_dict, w_dict):
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] += coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] += coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] += coeff
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def strassen_2x2(oi, oj, ok):
        # m1 = (x00+x11)(y00+y11)
        add_term_ijk(
            {(oi, oj): 1, (oi+1, oj+1): 1},
            {(oj, ok): 1, (oj+1, ok+1): 1},
            {(oi, ok): 1, (oi+1, ok+1): 1}
        )
        # m2 = (x10+x11)y00
        add_term_ijk(
            {(oi+1, oj): 1, (oi+1, oj+1): 1},
            {(oj, ok): 1},
            {(oi+1, ok): 1, (oi+1, ok+1): -1}
        )
        # m3 = x00(y01-y11)
        add_term_ijk(
            {(oi, oj): 1},
            {(oj, ok+1): 1, (oj+1, ok+1): -1},
            {(oi, ok+1): 1, (oi+1, ok+1): 1}
        )
        # m4 = x11(y10-y00)
        add_term_ijk(
            {(oi+1, oj+1): 1},
            {(oj+1, ok): 1, (oj, ok): -1},
            {(oi, ok): 1, (oi+1, ok): 1}
        )
        # m5 = (x00+x01)y11
        add_term_ijk(
            {(oi, oj): 1, (oi, oj+1): 1},
            {(oj+1, ok+1): 1},
            {(oi, ok): -1, (oi, ok+1): 1}
        )
        # m6 = (x10-x00)(y00+y01)
        add_term_ijk(
            {(oi+1, oj): 1, (oi, oj): -1},
            {(oj, ok): 1, (oj, ok+1): 1},
            {(oi+1, ok+1): 1}
        )
        # m7 = (x01-x11)(y10+y11)
        add_term_ijk(
            {(oi, oj+1): 1, (oi+1, oj+1): -1},
            {(oj+1, ok): 1, (oj+1, ok+1): 1},
            {(oi, ok): 1}
        )
    
    def strassen_4x4(oi, oj, ok):
        # 4x4 split into 2x2 blocks
        # A = [[A00, A01], [A10, A11]] each 2x2
        # B = [[B00, B01], [B10, B11]] each 2x2
        # C = [[C00, C01], [C10, C11]] each 2x2
        # Block offsets: A00 at (oi,oj), A01 at (oi,oj+2), A10 at (oi+2,oj), A11 at (oi+2,oj+2)
        # Same for B and C with appropriate column offsets
        
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        _strassen_super(oi, oj, ok,
                        u_blocks={(0,0): 1, (2,2): 1},
                        v_blocks={(0,0): 1, (2,2): 1},
                        w_blocks={(0,0): 1, (2,2): 1})
        
        # M2 = (A10+A11)B00 -> C10, C11
        _strassen_super(oi, oj, ok,
                        u_blocks={(2,0): 1, (2,2): 1},
                        v_blocks={(0,0): 1},
                        w_blocks={(2,0): 1, (2,2): -1})
        
        # M3 = A00(B01-B11) -> C01, C11
        _strassen_super(oi, oj, ok,
                        u_blocks={(0,0): 1},
                        v_blocks={(0,2): 1, (2,2): -1},
                        w_blocks={(0,2): 1, (2,2): 1})
        
        # M4 = A11(B10-B00) -> C00, C10
        _strassen_super(oi, oj, ok,
                        u_blocks={(2,2): 1},
                        v_blocks={(2,0): 1, (0,0): -1},
                        w_blocks={(0,0): 1, (2,0): 1})
        
        # M5 = (A00+A01)B11 -> C00, C01
        _strassen_super(oi, oj, ok,
                        u_blocks={(0,0): 1, (0,2): 1},
                        v_blocks={(2,2): 1},
                        w_blocks={(0,0): -1, (0,2): 1})
        
        # M6 = (A10-A00)(B00+B01) -> C11
        _strassen_super(oi, oj, ok,
                        u_blocks={(2,0): 1, (0,0): -1},
                        v_blocks={(0,0): 1, (0,2): 1},
                        w_blocks={(2,2): 1})
        
        # M7 = (A01-A11)(B10+B11) -> C00
        _strassen_super(oi, oj, ok,
                        u_blocks={(0,2): 1, (2,2): -1},
                        v_blocks={(2,0): 1, (2,2): 1},
                        w_blocks={(0,0): 1})
    
    def _strassen_super(oi, oj, ok, u_blocks, v_blocks, w_blocks):
        """
        Apply Strassen to a super-block where each sub-block is 2x2.
        u_blocks: {(bi,bj): coeff} where bi,bj in {0,1} index the 2x2 sub-blocks of A
        v_blocks: {(bi,bj): coeff} where bi,bj in {0,1} index the 2x2 sub-blocks of B
        w_blocks: {(bi,bj): coeff} where bi,bj in {0,1} index the 2x2 sub-blocks of C
        """
        for inner_s in range(7):
            # Get the inner Strassen step coefficients
            if inner_s == 0:
                iu = {(0,0): 1, (1,1): 1}
                iv = {(0,0): 1, (1,1): 1}
                iw = {(0,0): 1, (1,1): 1}
            elif inner_s == 1:
                iu = {(1,0): 1, (1,1): 1}
                iv = {(0,0): 1}
                iw = {(1,0): 1, (1,1): -1}
            elif inner_s == 2:
                iu = {(0,0): 1}
                iv = {(0,1): 1, (1,1): -1}
                iw = {(0,1): 1, (1,1): 1}
            elif inner_s == 3:
                iu = {(1,1): 1}
                iv = {(1,0): 1, (0,0): -1}
                iw = {(0,0): 1, (1,0): 1}
            elif inner_s == 4:
                iu = {(0,0): 1, (0,1): 1}
                iv = {(1,1): 1}
                iw = {(0,0): -1, (0,1): 1}
            elif inner_s == 5:
                iu = {(1,0): 1, (0,0): -1}
                iv = {(0,0): 1, (0,1): 1}
                iw = {(1,1): 1}
            else:  # inner_s == 6
                iu = {(0,1): 1, (1,1): -1}
                iv = {(1,0): 1, (1,1): 1}
                iw = {(0,0): 1}
            
            # Build the full term
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            for (bi, bj), u_coeff in u_blocks.items():
                for (ii, jj), i_coeff in iu.items():
                    # Global A index: row = oi + bi*2 + ii, col = oj + bj*2 + jj
                    gi = oi + bi * 2 + ii
                    gj = oj + bj * 2 + jj
                    key = (gi, gj)
                    u_dict[key] = u_dict.get(key, 0) + u_coeff * i_coeff
            
            for (bi, bj), v_coeff in v_blocks.items():
                for (ii, jj), i_coeff in iv.items():
                    # Global B index: row = oj + bi*2 + ii, col = ok + bj*2 + jj
                    gj = oj + bi * 2 + ii
                    gk = ok + bj * 2 + jj
                    key = (gj, gk)
                    v_dict[key] = v_dict.get(key, 0) + v_coeff * i_coeff
            
            for (bi, bj), w_coeff in w_blocks.items():
                for (ii, jj), i_coeff in iw.items():
                    # Global C index: row = oi + bi*2 + ii, col = ok + bj*2 + jj
                    gi = oi + bi * 2 + ii
                    gk = ok + bj * 2 + jj
                    key = (gi, gk)
                    w_dict[key] = w_dict.get(key, 0) + w_coeff * i_coeff
            
            # Remove zero entries
            u_dict = {k: v for k, v in u_dict.items() if v != 0}
            v_dict = {k: v for k, v in v_dict.items() if v != 0}
            w_dict = {k: v for k, v in w_dict.items() if v != 0}
            
            if u_dict and v_dict and w_dict:
                add_term_ijk(u_dict, v_dict, w_dict)
    
    # Part 1: 4x4 * 4x4 (columns 0-3 of B) using recursive Strassen, rank 49
    strassen_4x4(0, 0, 0)
    
    # Part 2: 4x4 * 4x2 (columns 4-5 of B) using 4 independent 2x2*2x2 Strassen, rank 28
    # A rows 0-1, cols 0-3 * B rows 0-1, cols 4-5 -> C rows 0-1, cols 4-5
    strassen_2x2(0, 0, 4)
    # A rows 0-1, cols 2-3 * B rows 2-3, cols 4-5 -> C rows 0-1, cols 4-5
    strassen_2x2(0, 2, 4)
    # A rows 2-3, cols 0-1 * B rows 0-1, cols 4-5 -> C rows 2-3, cols 4-5
    strassen_2x2(2, 0, 4)
    # A rows 2-3, cols 2-3 * B rows 2-3, cols 4-5 -> C rows 2-3, cols 4-5
    strassen_2x2(2, 2, 4)
    
    rank = len(U_cols)
    
    # Convert to arrays
    U = np.array(U_cols).T  # shape (n*m, rank)
    V = np.array(V_cols).T  # shape (m*p, rank)
    W = np.array(W_cols).T  # shape (p*n, rank)
    
    # Verify
    try:
        rng = np.random.default_rng(42)
        for trial in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            A_flat = A.reshape(n * m)
            B_flat = B.reshape(m * p)
            C_true = A @ B
            C_true_flat = C_true.T.reshape(p * n)
            a = U.T @ A_flat
            b = V.T @ B_flat
            C_scheme_flat = W.T @ (a * b)
            if not np.allclose(C_scheme_flat, C_true_flat, atol=1e-10):
                raise ValueError(f"Verification failed at trial {trial}")
    except Exception:
        # If verification fails, fall back to schoolbook
        U_fb = np.zeros((n * m, n * m * p), dtype=float)
        V_fb = np.zeros((m * p, n * m * p), dtype=float)
        W_fb = np.zeros((p * n, n * m * p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U_fb[i * m + j, r] = 1
                    V_fb[j * p + k, r] = 1
                    W_fb[k * n + i, r] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    if time.time() - start > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
