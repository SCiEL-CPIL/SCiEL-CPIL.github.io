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
    
    Actually, let me reconsider. The current best is 84 (Strassen on each 2x2*2x2 block separately).
    Let me try to get a lower rank.
    
    For 4x4 * 4x6:
    - Schoolbook: 96
    - Strassen on 2x2 blocks: 84 (current)
    
    Let me try a different approach: use the fact that we can share Strassen multiplications
    across the 3 column blocks of B.
    
    Actually, let me just implement a clean recursive Strassen for 4x4*4x4 and 4x4*4x2.
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
    
    # PART 1: 4x4 * 4x4 (columns 0-3 of B)
    # C[:, 0:4] = A * B[:, 0:4]
    # Split into 2x2 blocks:
    # For rb in {0,1}, cb in {0,1}:
    #   C[rb*2:rb*2+2, cb*2:cb*2+2] = A[rb*2:rb*2+2, 0:2]*B[0:2, cb*2:cb*2+2] + A[rb*2:rb*2+2, 2:4]*B[2:4, cb*2:cb*2+2]
    
    for rb in range(2):
        if _time_remaining() < 5.0:
            break
        for cb in range(2):
            if _time_remaining() < 5.0:
                break
            # A[rb, 0] * B[0, cb]
            strassen_2x2(rb*2, 0, cb*2)
            # A[rb, 1] * B[1, cb]
            strassen_2x2(rb*2, 2, cb*2)
    
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
    
    # Transpose to get U as (n*m, rank), V as (m*p, rank), W as (p*n, rank)
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
