"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _bclr19_factors():
    """
    Return (U, V, W) for 3x3 matrix multiplication with rank 19 (BCLR 1978).
    U: 9x19, V: 9x19, W: 9x19
    Row indices: A row-major (a0..a8), B row-major (b0..b8), C row-major (c0..c8)
    """
    U = [[0]*19 for _ in range(9)]
    V = [[0]*19 for _ in range(9)]
    W = [[0]*19 for _ in range(9)]
    
    # BCLR 1978 rank-19 decomposition
    # Products m0..m18
    
    # m0 = (a0+a3+a6)*(b0+b3+b6)
    U[0][0]=1; U[3][0]=1; U[6][0]=1
    V[0][0]=1; V[3][0]=1; V[6][0]=1
    W[0][0]=1; W[1][0]=1; W[2][0]=1
    W[3][0]=1; W[4][0]=1; W[5][0]=1
    W[6][0]=1; W[7][0]=1; W[8][0]=1
    
    # m1 = (a0+a3-a6)*(b0+b3-b6)
    U[0][1]=1; U[3][1]=1; U[6][1]=-1
    V[0][1]=1; V[3][1]=1; V[6][1]=-1
    W[0][1]=1; W[1][1]=1; W[2][1]=-1
    W[3][1]=1; W[4][1]=1; W[5][1]=-1
    W[6][1]=-1; W[7][1]=-1; W[8][1]=1
    
    # m2 = (a0-a3+a6)*(b0-b3+b6)
    U[0][2]=1; U[3][2]=-1; U[6][2]=1
    V[0][2]=1; V[3][2]=-1; V[6][2]=1
    W[0][2]=1; W[1][2]=-1; W[2][2]=1
    W[3][2]=-1; W[4][2]=1; W[5][2]=-1
    W[6][2]=1; W[7][2]=-1; W[8][2]=1
    
    # m3 = (a1+a4+a7)*(b1+b4+b7)
    U[1][3]=1; U[4][3]=1; U[7][3]=1
    V[1][3]=1; V[4][3]=1; V[7][3]=1
    W[0][3]=1; W[1][3]=1; W[2][3]=1
    W[3][3]=1; W[4][3]=1; W[5][3]=1
    W[6][3]=1; W[7][3]=1; W[8][3]=1
    
    # m4 = (a1+a4-a7)*(b1+b4-b7)
    U[1][4]=1; U[4][4]=1; U[7][4]=-1
    V[1][4]=1; V[4][4]=1; V[7][4]=-1
    W[0][4]=1; W[1][4]=1; W[2][4]=-1
    W[3][4]=1; W[4][4]=1; W[5][4]=-1
    W[6][4]=-1; W[7][4]=-1; W[8][4]=1
    
    # m5 = (a1-a4+a7)*(b1-b4+b7)
    U[1][5]=1; U[4][5]=-1; U[7][5]=1
    V[1][5]=1; V[4][5]=-1; V[7][5]=1
    W[0][5]=1; W[1][5]=-1; W[2][5]=1
    W[3][5]=-1; W[4][5]=1; W[5][5]=-1
    W[6][5]=1; W[7][5]=-1; W[8][5]=1
    
    # m6 = (a2+a5+a8)*(b2+b5+b8)
    U[2][6]=1; U[5][6]=1; U[8][6]=1
    V[2][6]=1; V[5][6]=1; V[8][6]=1
    W[0][6]=1; W[1][6]=1; W[2][6]=1
    W[3][6]=1; W[4][6]=1; W[5][6]=1
    W[6][6]=1; W[7][6]=1; W[8][6]=1
    
    # m7 = (a2+a5-a8)*(b2+b5-b8)
    U[2][7]=1; U[5][7]=1; U[8][7]=-1
    V[2][7]=1; V[5][7]=1; V[8][7]=-1
    W[0][7]=1; W[1][7]=1; W[2][7]=-1
    W[3][7]=1; W[4][7]=1; W[5][7]=-1
    W[6][7]=-1; W[7][7]=-1; W[8][7]=1
    
    # m8 = (a2-a5+a8)*(b2-b5+b8)
    U[2][8]=1; U[5][8]=-1; U[8][8]=1
    V[2][8]=1; V[5][8]=-1; V[8][8]=1
    W[0][8]=1; W[1][8]=-1; W[2][8]=1
    W[3][8]=-1; W[4][8]=1; W[5][8]=-1
    W[6][8]=1; W[7][8]=-1; W[8][8]=1
    
    # m9 = (a0+a3+a6)*(b1+b4+b7)
    U[0][9]=1; U[3][9]=1; U[6][9]=1
    V[1][9]=1; V[4][9]=1; V[7][9]=1
    W[1][9]=1; W[4][9]=1; W[7][9]=1
    
    # m10 = (a0+a3-a6)*(b1+b4-b7)
    U[0][10]=1; U[3][10]=1; U[6][10]=-1
    V[1][10]=1; V[4][10]=1; V[7][10]=-1
    W[1][10]=1; W[4][10]=1; W[7][10]=-1
    
    # m11 = (a0-a3+a6)*(b1-b4+b7)
    U[0][11]=1; U[3][11]=-1; U[6][11]=1
    V[1][11]=1; V[4][11]=-1; V[7][11]=1
    W[1][11]=1; W[4][11]=-1; W[7][11]=1
    
    # m12 = (a1+a4+a7)*(b0+b3+b6)
    U[1][12]=1; U[4][12]=1; U[7][12]=1
    V[0][12]=1; V[3][12]=1; V[6][12]=1
    W[0][12]=1; W[3][12]=1; W[6][12]=1
    
    # m13 = (a1+a4-a7)*(b0+b3-b6)
    U[1][13]=1; U[4][13]=1; U[7][13]=-1
    V[0][13]=1; V[3][13]=1; V[6][13]=-1
    W[0][13]=1; W[3][13]=1; W[6][13]=-1
    
    # m14 = (a1-a4+a7)*(b0-b3+b6)
    U[1][14]=1; U[4][14]=-1; U[7][14]=1
    V[0][14]=1; V[3][14]=-1; V[6][14]=1
    W[0][14]=1; W[3][14]=-1; W[6][14]=1
    
    # m15 = (a2+a5+a8)*(b0+b3+b6)
    U[2][15]=1; U[5][15]=1; U[8][15]=1
    V[0][15]=1; V[3][15]=1; V[6][15]=1
    W[0][15]=1; W[3][15]=1; W[6][15]=1
    
    # m16 = (a2+a5-a8)*(b0+b3-b6)
    U[2][16]=1; U[5][16]=1; U[8][16]=-1
    V[0][16]=1; V[3][16]=1; V[6][16]=-1
    W[0][16]=1; W[3][16]=1; W[6][16]=-1
    
    # m17 = (a2-a5+a8)*(b0-b3+b6)
    U[2][17]=1; U[5][17]=-1; U[8][17]=1
    V[0][17]=1; V[3][17]=-1; V[6][17]=1
    W[0][17]=1; W[3][17]=-1; W[6][17]=1
    
    # m18 = (a4+b0+b7-b1)*(b4+a0+a7-a1)  -- This is the tricky one
    # Actually, let me reconsider. The standard BCLR 19 uses:
    # m18 = (a4 + b0 + b7 - b1) * (b4 + a0 + a7 - a1)
    # But this mixes A and B entries in both factors, which is not a valid bilinear form.
    # 
    # Let me use a different verified formulation. The correct BCLR 19th product is:
    # m18 = (a4 + a1 + a7 - b1) -- no, that's wrong too.
    #
    # I realize I cannot reliably reproduce the exact BCLR 19 from memory.
    # Let me return None to signal failure and use fallback.
    return None


def _verify_bilinear(U, V, W, n, m, p, num_tests=500, seed=42):
    """Verify that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta(i,j,k) for all i,j,k."""
    import random
    random.seed(seed)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Test all i,j,k combinations (or a subset if too many)
    total = n * m * p
    if total <= 500:
        indices = [(i,j,k) for i in range(n) for j in range(m) for k in range(p)]
    else:
        indices = [(random.randint(0,n-1), random.randint(0,m-1), random.randint(0,p-1)) 
                   for _ in range(num_tests)]
    
    for i, j, k in indices:
        val = 0.0
        for r in range(len(U[0])):
            val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        if abs(val - 1.0) > 1e-10:
            return False
    return True


def _build_strassen_2x2():
    """
    Return (U, V, W) for 2x2 matrix multiplication with rank 7 (Strassen).
    U: 4x7, V: 4x7, W: 4x4
    Row indices: row-major for 2x2 matrices.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Strassen 7 products
    # M1 = (a00+a11)*(b00+b11)
    U[0][0]=1; U[1][0]=1
    V[0][0]=1; V[1][0]=1
    W[0][0]=1; W[3][0]=1
    
    # M2 = (a10+a11)*b00
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[2][1]=1; W[3][1]=-1
    
    # M3 = a00*(b01-b11)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[1][2]=1; W[3][2]=1
    
    # M4 = a11*(b10-b00)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[0][3]=1; W[2][3]=1
    
    # M5 = (a00+a01)*b11
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[0][4]=-1; W[1][4]=1
    
    # M6 = (a10-a00)*(b00+b01)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    
    # M7 = (a01-a11)*(b10+b11)
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1
    
    return U, V, W


def _build_3x3_schoolbook():
    """
    Return (U, V, W) for 3x3 matrix multiplication with rank 27 (schoolbook).
    U: 9x27, V: 9x27, W: 9x27
    """
    U = [[0]*27 for _ in range(9)]
    V = [[0]*27 for _ in range(9)]
    W = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for k in range(3):
            for j in range(3):
                U[i*3+k][col] = 1
                V[k*3+j][col] = 1
                W[j*3+i][col] = 1
                col += 1
    
    return U, V, W


def _kron_combine(U1, V1, W1, U2, V2, W2):
    """
    Combine two tensor decompositions via Kronecker product.
    If T1 has rank r1 and T2 has rank r2, the product has rank r1*r2.
    
    U1: (n1*m1) x r1, V1: (m1*p1) x r1, W1: (p1*n1) x r1
    U2: (n2*m2) x r2, V2: (m2*p2) x r2, W2: (p2*n2) x r2
    
    Result: U: (n1*n2 * m1*m2) x (r1*r2), etc.
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    n1m1 = len(U1)
    m1p1 = len(V1)
    p1n1 = len(W1)
    n2m2 = len(U2)
    m2p2 = len(V2)
    p2n2 = len(W2)
    
    # n1 = n1m1 / m1, m1 = m1p1 / p1, etc.
    # We need to figure out n1, m1, p1, n2, m2, p2
    # From dimensions: n1*m1 = n1m1, m1*p1 = m1p1, p1*n1 = p1n1
    # So n1 = n1m1 * p1n1 / (m1p1 * m1) ... this is circular.
    # Let's just use the structure directly.
    
    # For our specific case: outer is 2x2 blocks (n1=m1=p1=2), inner is 3x3 (n2=m2=p2=3)
    # So n1=2, m1=2, p1=2, n2=3, m2=3, p2=3
    # n1*m1=4, m1*p1=4, p1*n1=4, n2*m2=9, m2*p2=9, p2*n2=9
    
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 3, 3, 3
    
    N = n1 * n2  # total rows = 6
    M = m1 * m2  # total cols = 6
    P = p1 * p2  # total = 6
    
    rank = r1 * r2
    U = [[0]*rank for _ in range(N*M)]
    V = [[0]*rank for _ in range(M*P)]
    W = [[0]*rank for _ in range(P*N)]
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            col = r1_idx * r2 + r2_idx
            
            # U: index (i1*n2+i2) * (m1*m2) + (k1*m2+k2)
            # U1[i1*m1+k1, r1_idx] * U2[i2*m2+k2, r2_idx]
            for i1 in range(n1):
                for k1 in range(m1):
                    u1_val = U1[i1*m1+k1][r1_idx]
                    if u1_val == 0:
                        continue
                    for i2 in range(n2):
                        for k2 in range(m2):
                            u2_val = U2[i2*m2+k2][r2_idx]
                            if u2_val == 0:
                                continue
                            row_u = (i1*n2 + i2) * M + (k1*m2 + k2)
                            U[row_u][col] += u1_val * u2_val
            
            # V: index (k1*m2+k2) * (p1*p2) + (j1*p2+j2)
            # V1[k1*p1+j1, r1_idx] * V2[k2*p2+j2, r2_idx]
            for k1 in range(m1):
                for j1 in range(p1):
                    v1_val = V1[k1*p1+j1][r1_idx]
                    if v1_val == 0:
                        continue
                    for k2 in range(m2):
                        for j2 in range(p2):
                            v2_val = V2[k2*p2+j2][r2_idx]
                            if v2_val == 0:
                                continue
                            row_v = (k1*m2 + k2) * P + (j1*p2 + j2)
                            V[row_v][col] += v1_val * v2_val
            
            # W: index (j1*p2+j2) * (n1*n2) + (i1*n2+i2)
            # W1[j1*n1+i1, r1_idx] * W2[j2*n2+i2, r2_idx]
            for j1 in range(p1):
                for i1 in range(n1):
                    w1_val = W1[j1*n1+i1][r1_idx]
                    if w1_val == 0:
                        continue
                    for j2 in range(p2):
                        for i2 in range(n2):
                            w2_val = W2[j2*n2+i2][r2_idx]
                            if w2_val == 0:
                                continue
                            row_w = (j1*p2 + j2) * N + (i1*n2 + i2)
                            W[row_w][col] += w1_val * w2_val
    
    return U, V, W


def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 6x6 matrix by a 6x6 matrix.
    Uses 2x2 blocks of 3x3 matrices: Strassen (rank 7) for outer 2x2 block multiplication,
    and schoolbook (rank 27) for inner 3x3 multiplication.
    Total rank = 7 * 27 = 189.
    
    Also attempts BCLR rank-19 for inner 3x3, giving rank 7*19=133 if verified.
    """
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS  # all 6
    
    # Strategy 1: Try BCLR 19 for inner 3x3 + Strassen 7 for outer 2x2
    # Total rank = 7 * 19 = 133
    U_outer, V_outer, W_outer = _build_strassen_2x2()
    
    # Try to get BCLR 19
    bclr_result = _bclr19_factors()
    
    if bclr_result is not None:
        U_inner, V_inner, W_inner = bclr_result
        # Verify the inner decomposition
        if _verify_bilinear(U_inner, V_inner, W_inner, 3, 3, 3, num_tests=300):
            U, V, W = _kron_combine(U_outer, V_outer, W_outer, U_inner, V_inner, W_inner)
            if _verify_bilinear(U, V, W, n, m, p, num_tests=500):
                if time.time() < deadline:
                    return {"U": U, "V": V, "W": W}
    
    # Strategy 2: Strassen 7 outer + schoolbook 27 inner = rank 189
    U_inner_sb, V_inner_sb, W_inner_sb = _build_3x3_schoolbook()
    U, V, W = _kron_combine(U_outer, V_outer, W_outer, U_inner_sb, V_inner_sb, W_inner_sb)
    
    if _verify_bilinear(U, V, W, n, m, p, num_tests=500):
        if time.time() < deadline:
            return {"U": U, "V": V, "W": W}
    
    # Strategy 3: Fall back to original block Strassen (3x3 blocks of 2x2)
    rank = 189
    U_fb = [[0] * rank for _ in range(n * m)]
    V_fb = [[0] * rank for _ in range(m * p)]
    W_fb = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_block in range(3):
        for j_block in range(3):
            for k_block in range(3):
                def u_row(a, b, ib=i_block, kb=k_block):
                    return (ib * 2 + a) * m + (kb * 2 + b)
                def v_row(c, d, kb=k_block, jb=j_block):
                    return (kb * 2 + c) * p + (jb * 2 + d)
                def w_row(e, f, ib=i_block, jb=j_block):
                    return (jb * 2 + f) * n + (ib * 2 + e)
                
                # M1
                U_fb[u_row(0,0)][col]=1; U_fb[u_row(1,1)][col]=1
                V_fb[v_row(0,0)][col]=1; V_fb[v_row(1,1)][col]=1
                W_fb[w_row(0,0)][col]=1; W_fb[w_row(1,1)][col]=1
                col+=1
                # M2
                U_fb[u_row(1,0)][col]=1; U_fb[u_row(1,1)][col]=1
                V_fb[v_row(0,0)][col]=1
                W_fb[w_row(1,0)][col]=1; W_fb[w_row(1,1)][col]=-1
                col+=1
                # M3
                U_fb[u_row(0,0)][col]=1
                V_fb[v_row(0,1)][col]=1; V_fb[v_row(1,1)][col]=-1
                W_fb[w_row(0,1)][col]=1; W_fb[w_row(1,1)][col]=1
                col+=1
                # M4
                U_fb[u_row(1,1)][col]=1
                V_fb[v_row(1,0)][col]=1; V_fb[v_row(0,0)][col]=-1
                W_fb[w_row(0,0)][col]=1; W_fb[w_row(1,0)][col]=1
                col+=1
                # M5
                U_fb[u_row(0,0)][col]=1; U_fb[u_row(0,1)][col]=1
                V_fb[v_row(1,1)][col]=1
                W_fb[w_row(0,0)][col]=-1; W_fb[w_row(0,1)][col]=1
                col+=1
                # M6
                U_fb[u_row(1,0)][col]=1; U_fb[u_row(0,0)][col]=-1
                V_fb[v_row(0,0)][col]=1; V_fb[v_row(0,1)][col]=1
                W_fb[w_row(1,1)][col]=1
                col+=1
                # M7
                U_fb[u_row(0,1)][col]=1; U_fb[u_row(1,1)][col]=-1
                V_fb[v_row(1,0)][col]=1; V_fb[v_row(1,1)][col]=1
                W_fb[w_row(0,0)][col]=1
                col+=1
    
    if _verify_bilinear(U_fb, V_fb, W_fb, n, m, p, num_tests=200):
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    # Final fallback: schoolbook rank 216
    rank_fallback = n * m * p
    U_f = [[0] * rank_fallback for _ in range(n * m)]
    V_f = [[0] * rank_fallback for _ in range(m * p)]
    W_f = [[0] * rank_fallback for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j][col] = 1
                V_f[j * p + k][col] = 1
                W_f[k * n + i][col] = 1
                col += 1
    return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
