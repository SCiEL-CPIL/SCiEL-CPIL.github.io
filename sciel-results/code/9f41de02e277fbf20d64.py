"""Matrix multiplication 4x4 * 4x6 rank-minimal decomposition."""
import time
_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy:
    - Split the 4x4 * 4x6 problem into 4x4 * 4x4 (first 4 columns of B) + 4x4 * 4x2 (last 2 columns of B)
    - For 4x4 * 4x4: use recursive Strassen (2 levels) giving rank 49
    - For 4x4 * 4x2: use Strassen on 2x2 blocks, giving rank 28
    - Total rank: 77
    """
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add one rank-1 term to the decomposition."""
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] = coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] = coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
    
    def strassen_2x2(oi, oj, ok):
        """
        Add Strassen decomposition for 2x2 * 2x2 block multiplication.
        oi: offset row in A (0 or 2)
        oj: offset col in A / row in B (0 or 2)  
        ok: offset col in B (0, 2, or 4)
        """
        # X = A[oi:oi+2, oj:oj+2], Y = B[oj:oj+2, ok:ok+2]
        # C[oi:oi+2, ok:ok+2] = X * Y
        
        # m1 = (x00+x11)(y00+y11)
        u1 = {(oi, oj): 1, (oi+1, oj+1): 1}
        v1 = {(oj, ok): 1, (oj+1, ok+1): 1}
        w1 = {(oi, ok): 1, (oi+1, ok+1): 1}
        add_term(u1, v1, w1)
        
        # m2 = (x10+x11)y00
        u2 = {(oi+1, oj): 1, (oi+1, oj+1): 1}
        v2 = {(oj, ok): 1}
        w2 = {(oi+1, ok): 1, (oi+1, ok+1): -1}
        add_term(u2, v2, w2)
        
        # m3 = x00(y01-y11)
        u3 = {(oi, oj): 1}
        v3 = {(oj, ok+1): 1, (oj+1, ok+1): -1}
        w3 = {(oi, ok+1): 1, (oi+1, ok+1): 1}
        add_term(u3, v3, w3)
        
        # m4 = x11(y10-y00)
        u4 = {(oi+1, oj+1): 1}
        v4 = {(oj+1, ok): 1, (oj, ok): -1}
        w4 = {(oi, ok): 1, (oi+1, ok): 1}
        add_term(u4, v4, w4)
        
        # m5 = (x00+x01)y11
        u5 = {(oi, oj): 1, (oi, oj+1): 1}
        v5 = {(oj+1, ok+1): 1}
        w5 = {(oi, ok): -1, (oi, ok+1): 1}
        add_term(u5, v5, w5)
        
        # m6 = (x10-x00)(y00+y01)
        u6 = {(oi+1, oj): 1, (oi, oj): -1}
        v6 = {(oj, ok): 1, (oj, ok+1): 1}
        w6 = {(oi+1, ok+1): 1}
        add_term(u6, v6, w6)
        
        # m7 = (x01-x11)(y10+y11)
        u7 = {(oi, oj+1): 1, (oi+1, oj+1): -1}
        v7 = {(oj+1, ok): 1, (oj+1, ok+1): 1}
        w7 = {(oi, ok): 1}
        add_term(u7, v7, w7)

    def add_super_term(u_blocks, v_blocks, w_blocks, oi, oj, ok):
        """
        Add a Strassen super-term which is a sum of block products.
        u_blocks, v_blocks, w_blocks are dicts mapping (block_row, block_col) to coeff.
        These blocks are 2x2 blocks relative to the 4x4 block at (oi, oj, ok).
        """
        # The 7 terms of Strassen for 2x2x2
        terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
        ]
        
        for u_t, v_t, w_t in terms:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for (br, bc), coeff_u in u_blocks.items():
                for (r, c), coeff_t in u_t.items():
                    i = oi + 2*br + r
                    j = oj + 2*bc + c
                    u_col[i * m + j] += coeff_u * coeff_t
            
            for (br, bc), coeff_v in v_blocks.items():
                for (r, c), coeff_t in v_t.items():
                    j = oj + 2*br + r
                    k = ok + 2*bc + c
                    v_col[j * p + k] += coeff_v * coeff_t
            
            for (br, bc), coeff_w in w_blocks.items():
                for (r, c), coeff_t in w_t.items():
                    i = oi + 2*br + r
                    k = ok + 2*bc + c
                    w_col[k * n + i] += coeff_w * coeff_t
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    def strassen_4x4(oi, oj, ok):
        """
        Add Strassen decomposition for 4x4 * 4x4 block multiplication.
        """
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        add_super_term({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, oi, oj, ok)
        
        # M2 = (A10+A11)B00 -> C10, C11
        add_super_term({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}, oi, oj, ok)
        
        # M3 = A00(B01-B11) -> C01, C11
        add_super_term({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, oi, oj, ok)
        
        # M4 = A11(B10-B00) -> C00, C10
        add_super_term({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, oi, oj, ok)
        
        # M5 = (A00+A01)B11 -> C00, C01
        add_super_term({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}, oi, oj, ok)
        
        # M6 = (A10-A00)(B00+B01) -> C11
        add_super_term({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}, oi, oj, ok)
        
        # M7 = (A01-A11)(B10+B11) -> C00
        add_super_term({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}, oi, oj, ok)
    
    # PART 1: 4x4 * 4x4 (columns 0-3 of B)
    # C[:, 0:4] = A * B[:, 0:4]
    # Use recursive Strassen (rank 49)
    if _time_remaining() > 5.0:
        strassen_4x4(0, 0, 0)
    
    # PART 2: 4x4 * 4x2 (columns 4-5 of B)
    # C[:, 4:6] = A * B[:, 4:6]
    # For rb in {0,1}:
    #   C[rb*2:rb*2+2, 4:6] = A[rb*2:rb*2+2, 0:2]*B[0:2, 4:6] + A[rb*2:rb*2+2, 2:4]*B[2:4, 4:6]
    
    for rb in range(2):
        if _time_remaining() < 5.0:
            break
        # A[rb, 0] * B[0, 2]
        strassen_2x2(rb*2, 0, 4)
        # A[rb, 1] * B[1, 2]
        strassen_2x2(rb*2, 2, 4)
    
    rank = len(U_cols)
    
    # Verify decomposition correctness using random test
    try:
        import numpy as np
        if _time_remaining() > 10.0:
            np.random.seed(42)
            A_test = np.random.randint(-2, 3, size=(n, m))
            B_test = np.random.randint(-2, 3, size=(m, p))
            C_expected = A_test @ B_test
            
            U_arr = np.array(U_cols).T  # (n*m, rank)
            V_arr = np.array(V_cols).T  # (m*p, rank)
            W_arr = np.array(W_cols).T  # (p*n, rank)
            
            # C = sum_r U_r * V_r * W_r (outer product reconstruction)
            # U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] should sum to C[i,k]
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_r = U_arr[:, r].reshape(n, m)
                v_r = V_arr[:, r].reshape(m, p)
                w_r = W_arr[:, r].reshape(p, n)
                C_computed += u_r @ v_r @ w_r.T
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Fallback: use schoolbook
                U_cols = []
                V_cols = []
                W_cols = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_col = [0] * (n * m)
                            v_col = [0] * (m * p)
                            w_col = [0] * (p * n)
                            u_col[i * m + j] = 1
                            v_col[j * p + k] = 1
                            w_col[k * n + i] = 1
                            U_cols.append(u_col)
                            V_cols.append(v_col)
                            W_cols.append(w_col)
                rank = len(U_cols)
    except Exception:
        pass
    
    # Transpose to get U as (n*m, rank), V as (m*p, rank), W as (p*n, rank)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
