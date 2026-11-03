"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Rank-84 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Uses Strassen's 7-multiplication algorithm on 2x2 blocks.
    
    A (4x4) split into 2x2 blocks: A[rb][cb] for rb,cb in {0,1}
    B (4x6) split into 2x3 blocks: B[rb][cb] for rb in {0,1}, cb in {0,1,2}
    C (4x6) split into 2x3 blocks: C[rb][cb] for rb in {0,1}, cb in {0,1,2}
    
    C[rb][cb] = A[rb][0] * B[0][cb] + A[rb][1] * B[1][cb]
    
    Each 2x2*2x2 product uses Strassen (7 mults), giving 14 per output block.
    6 output blocks * 14 = 84 total rank.
    """
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def strassen_2x2_2x2(oi, oj, ok):
        """
        Add 7 Strassen terms for 2x2*2x2 product.
        oi: offset row in A/C (0 or 2)
        oj: offset col in A / row in B (0 or 2)  
        ok: offset col in B/C (0, 2, or 4)
        
        X = A[oi:oi+2, oj:oj+2], Y = B[oj:oj+2, ok:ok+2]
        Output C[oi:oi+2, ok:ok+2] = X * Y
        """
        # Elements:
        # x00=A[oi,oj], x01=A[oi,oj+1], x10=A[oi+1,oj], x11=A[oi+1,oj+1]
        # y00=B[oj,ok], y01=B[oj,ok+1], y10=B[oj+1,ok], y11=B[oj+1,ok+1]
        # C00=C[oi,ok], C01=C[oi,ok+1], C10=C[oi+1,ok], C11=C[oi+1,ok+1]
        
        # m1 = (x00+x11)(y00+y11) -> C00+=1, C11+=1
        # m2 = (x10+x11)y00 -> C10+=1, C11+=-1
        # m3 = x00(y01-y11) -> C01+=1, C11+=1
        # m4 = x11(y10-y00) -> C00+=1, C10+=1
        # m5 = (x00+x01)y11 -> C00+=-1, C01+=1
        # m6 = (x10-x00)(y00+y01) -> C11+=1
        # m7 = (x01-x11)(y10+y11) -> C00+=1
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # m1
            ([(oi, oj, 1), (oi+1, oj+1, 1)],
             [(oj, ok, 1), (oj+1, ok+1, 1)],
             [(oi, ok, 1), (oi+1, ok+1, 1)]),
            # m2
            ([(oi+1, oj, 1), (oi+1, oj+1, 1)],
             [(oj, ok, 1)],
             [(oi+1, ok, 1), (oi+1, ok+1, -1)]),
            # m3
            ([(oi, oj, 1)],
             [(oj, ok+1, 1), (oj+1, ok+1, -1)],
             [(oi, ok+1, 1), (oi+1, ok+1, 1)]),
            # m4
            ([(oi+1, oj+1, 1)],
             [(oj+1, ok, 1), (oj, ok, -1)],
             [(oi, ok, 1), (oi+1, ok, 1)]),
            # m5
            ([(oi, oj, 1), (oi, oj+1, 1)],
             [(oj+1, ok+1, 1)],
             [(oi, ok, -1), (oi, ok+1, 1)]),
            # m6
            ([(oi+1, oj, 1), (oi, oj, -1)],
             [(oj, ok, 1), (oj, ok+1, 1)],
             [(oi+1, ok+1, 1)]),
            # m7
            ([(oi, oj+1, 1), (oi+1, oj+1, -1)],
             [(oj+1, ok, 1), (oj+1, ok+1, 1)],
             [(oi, ok, 1)]),
        ]
        
        for u_c, v_c, w_c in terms:
            uc = [0.0] * (n * m)
            vc = [0.0] * (m * p)
            wc = [0.0] * (p * n)
            
            for (i, j, c) in u_c:
                uc[i * m + j] = c
            for (j, k, c) in v_c:
                vc[j * p + k] = c
            for (i, k, c) in w_c:
                wc[k * n + i] = c
            
            U_cols.append(uc)
            V_cols.append(vc)
            W_cols.append(wc)
    
    # For each output block position (rb, cb):
    # C[rb][cb] = A[rb][0]*B[0][cb] + A[rb][1]*B[1][cb]
    for rb in range(2):
        for cb in range(3):
            if _time_remaining() < 5.0:
                break
            # First term: A[rb][0] * B[0][cb]
            # A[rb][0] starts at row rb*2, col 0
            # B[0][cb] starts at row 0, col cb*2
            strassen_2x2_2x2(rb * 2, 0, cb * 2)
            # Second term: A[rb][1] * B[1][cb]
            # A[rb][1] starts at row rb*2, col 2
            # B[1][cb] starts at row 2, col cb*2
            strassen_2x2_2x2(rb * 2, 2, cb * 2)
    
    rank = len(U_cols)
    
    # Build final matrices: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    U_mat = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_mat = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_mat = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
