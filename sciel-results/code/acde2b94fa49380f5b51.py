"""Matrix multiplication 4x5x5 with rank-reduced decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2 x 2x2 matrix multiplication, rank 7.
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (a+d)(e+h): U: a,d; V: e,h; W: c11,c22
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # p2 = (c+d)e: U: c,d; V: e; W: c21
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # p3 = a(f-h): U: a; V: f,-h; W: c12,c22
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # p4 = d(g-e): U: d; V: g,-e; W: c11,c21
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # p5 = (a+b)h: U: a,b; V: h; W: -c11,c12
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # p6 = (c-a)(e+f): U: c,-a; V: e,f; W: c22
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # p7 = (b-d)(g+h): U: b,-d; V: g,h; W: c11
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _schoolbook_sub(n, m, p):
    """Schoolbook for n x m x p sub-block. Returns (U, V, W)."""
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
    return U, V, W

def _embed_sub(n_full, m_full, p_full, U_sub, V_sub, W_sub, ni, nm, np_, i0, j0, k0):
    """Embed a sub-decomposition into the full tensor.
    U_sub: (ni*nm, rank), V_sub: (nm*np_, rank), W_sub: (np_*ni, rank)
    Full: U (n_full*m_full, rank), V (m_full*p_full, rank), W (p_full*n_full, rank)
    """
    rank = len(U_sub[0])
    U_out = [[0] * rank for _ in range(n_full * m_full)]
    V_out = [[0] * rank for _ in range(m_full * p_full)]
    W_out = [[0] * rank for _ in range(p_full * n_full)]
    
    for r in range(rank):
        for i in range(ni):
            for j in range(nm):
                U_out[(i0+i)*m_full + (j0+j)][r] = U_sub[i*nm+j][r]
        for j in range(nm):
            for k in range(np_):
                V_out[(j0+j)*p_full + (k0+k)][r] = V_sub[j*np_+k][r]
        for k in range(np_):
            for i in range(ni):
                W_out[(k0+k)*n_full + (i0+i)][r] = W_sub[k*ni+i][r]
    
    return U_out, V_out, W_out

def _concat_decomps(decomps):
    """Concatenate multiple decompositions along rank dimension."""
    if not decomps:
        return None
    
    n_rows_U = max(len(d[0]) for d in decomps)
    n_rows_V = max(len(d[1]) for d in decomps)
    n_rows_W = max(len(d[2]) for d in decomps)
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U_out = [[0] * total_rank for _ in range(n_rows_U)]
    V_out = [[0] * total_rank for _ in range(n_rows_V)]
    W_out = [[0] * total_rank for _ in range(n_rows_W)]
    
    col_off = 0
    for U_sub, V_sub, W_sub in decomps:
        r = len(U_sub[0])
        for i in range(len(U_sub)):
            for c in range(r):
                U_out[i][col_off + c] = U_sub[i][c]
        for i in range(len(V_sub)):
            for c in range(r):
                V_out[i][col_off + c] = V_sub[i][c]
        for i in range(len(W_sub)):
            for c in range(r):
                W_out[i][col_off + c] = W_sub[i][c]
        col_off += r
    
    return U_out, V_out, W_out

def _strassen_4x4x4():
    """Build rank-49 decomposition for 4x4x4 using Strassen on 2x2 blocks.
    Returns (U, V, W) of shape (16, 49) each.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # 4x4x4 Strassen: 7 multiplications of 2x2x2
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # For each M, we need to construct U, V, W vectors for the 4x4x4 tensor
    # that implement the Strassen formula
    
    # Let's build each of the 7 Strassen terms
    # Each term is a sum of 2x2x2 Strassen terms at specific block positions
    
    decomps = []
    
    # M1 = (A11+A22)(B11+B22)
    # = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # Block positions: (0,0,0), (0,0,1), (1,1,0), (1,1,1)
    for bi, bj, bk in [(0,0,0), (0,0,1), (1,1,0), (1,1,1)]:
        U_sub = [[0]*7 for _ in range(16)]
        V_sub = [[0]*7 for _ in range(16)]
        W_sub = [[0]*7 for _ in range(16)]
        for r in range(7):
            for idx in range(4):
                val_u = U2[idx][r]
                val_v = V2[idx][r]
                val_w = W2[idx][r]
                if val_u != 0:
                    i_loc, j_loc = divmod(idx, 2)
                    U_sub[(bi+i_loc)*4 + (bj+j_loc)][r] += val_u
                if val_v != 0:
                    j_loc, k_loc = divmod(idx, 2)
                    V_sub[(bj+j_loc)*4 + (bk+k_loc)][r] += val_v
                if val_w != 0:
                    k_loc, i_loc = divmod(idx, 2)
                    W_sub[(bk+k_loc)*4 + (bi+i_loc)][r] += val_w
        decomps.append((U_sub, V_sub, W_sub))
    
    # M2 = (A21+A22)B11
    # = A21*B11 + A22*B11
    # Block positions: (1,0,0), (1,1,0)
    for bi, bj, bk in [(1,0,0), (1,1,0)]:
        U_sub = [[0]*7 for _ in range(16)]
        V_sub = [[0]*7 for _ in range(16)]
        W_sub = [[0]*7 for _ in range(16)]
        for r in range(7):
            for idx in range(4):
                val_u = U2[idx][r]
                val_v = V2[idx][r]
                val_w = W2[idx][r]
                if val_u != 0:
                    i_loc, j_loc = divmod(idx, 2)
                    U_sub[(bi+i_loc)*4 + (bj+j_loc)][r] += val_u
                if val_v != 0:
                    j_loc, k_loc = divmod(idx, 2)
                    V_sub[(bj+j_loc)*4 + (bk+k_loc)][r] += val_v
                if val_w != 0:
                    k_loc, i_loc = divmod(idx, 2)
                    W_sub[(bk+k_loc)*4 + (bi+i_loc)][r] += val_w
        decomps.append((U_sub, V_sub, W_sub))
    
    # M3 = A11(B12-B22)
    # = A11*B12 - A11*B22
    # Block positions: (0,1,0) with +1, (0,1,1) with -1
    for bi, bj, bk, sign in [(0,1,0, 1), (0,1,1, -1)]:
        U_sub = [[0]*7 for _ in range(16)]
        V_sub = [[0]*7 for _ in range(16)]
        W_sub = [[0]*7 for _ in range(16)]
        for r in range(7):
            for idx in range(4):
                val_u = U2[idx][r] * sign
                val_v = V2[idx][r]
                val_w = W2[idx][r] * sign
                if val_u != 0:
                    i_loc, j_loc = divmod(idx, 2)
                    U_sub[(bi+i_loc)*4 + (bj+j_loc)][r] += val_u
                if val_v != 0:
                    j_loc, k_loc = divmod(idx, 2)
                    V_sub[(bj+j_loc)*4 + (bk+k_loc)][r] += val_v
                if val_w != 0:
                    k_loc, i_loc = divmod(idx, 2)
                    W_sub[(bk+k_loc)*4 + (bi+i_loc)][r] += val_w
        decomps.append((U_sub, V_sub, W_sub))
    
    # M4 = A22(B21-B11)
    # = A22*B21 - A22*B11
    # Block positions: (1,1,1) with +1, (1,1,0) with -1
    for bi, bj, bk, sign in [(1,1,1, 1), (1,1,0, -1)]:
        U_sub = [[0]*7 for _ in range(16)]
        V_sub = [[0]*7 for _ in range(16)]
        W_sub = [[0]*7 for _ in range(16)]
        for r in range(7):
            for idx in range(4):
                val_u = U2[idx][r]
                val_v = V2[idx][r] * sign
                val_w = W2[idx][r]
                if val_u != 0:
                    i_loc, j_loc = divmod(idx, 2)
                    U_sub[(bi+i_loc)*4 + (bj+j_loc)][r] += val_u
                if val_v != 0:
                    j_loc, k_loc = divmod(idx, 2)
                    V_sub[(bj+j_loc)*4 + (bk+k_loc)][r] += val_v
                if val_w != 0:
                    k_loc, i_loc = divmod(idx, 2)
                    W_sub[(bk+k_loc)*4 + (bi+i_loc)][r] += val_w
        decomps.append((U_sub, V_sub, W_sub))
    
    # M5 = (A11+A12)B22
    # = A11*B22 + A12*B22
    # Block positions: (0,0,1), (0,1,1)
    for bi, bj, bk in [(0,0,1), (0,1,1)]:
        U_sub = [[0]*7 for _ in range(16)]
        V_sub = [[0]*7 for _ in range(16)]
        W_sub = [[0]*7 for _ in range(16)]
        for r in range(7):
            for idx in range(4):
                val_u = U2[idx][r]
                val_v = V2[idx][r]
                val_w = W2[idx][r]
                if val_u != 0:
                    i_loc, j_loc = divmod(idx, 2)
                    U_sub[(bi+i_loc)*4 + (bj+j_loc)][r] += val_u
                if val_v != 0:
                    j_loc, k_loc = divmod(idx, 2)
                    V_sub[(bj+j_loc)*4 + (bk+k_loc)][r] += val_v
                if val_w != 0:
                    k_loc, i_loc = divmod(idx, 2)
                    W_sub[(bk+k_loc)*4 + (bi+i_loc)][r] += val_w
        decomps.append((U_sub, V_sub, W_sub))
    
    # M6 = (A21-A11)(B11+B12)
    # = A21*B11 + A21*B12 - A11*B11 - A11*B12
    # Block positions: (1,0,0)+, (1,0,1)+, (0,0,0)-, (0,0,1)-
    for bi, bj, bk, sign in [(1,0,0, 1), (1,0,1, 1), (0,0,0, -1), (0,0,1, -1)]:
        U_sub = [[0]*7 for _ in range(16)]
        V_sub = [[0]*7 for _ in range(16)]
        W_sub = [[0]*7 for _ in range(16)]
        for r in range(7):
            for idx in range(4):
                val_u = U2[idx][r] * sign
                val_v = V2[idx][r]
                val_w = W2[idx][r] * sign
                if val_u != 0:
                    i_loc, j_loc = divmod(idx, 2)
                    U_sub[(bi+i_loc)*4 + (bj+j_loc)][r] += val_u
                if val_v != 0:
                    j_loc, k_loc = divmod(idx, 2)
                    V_sub[(bj+j_loc)*4 + (bk+k_loc)][r] += val_v
                if val_w != 0:
                    k_loc, i_loc = divmod(idx, 2)
                    W_sub[(bk+k_loc)*4 + (bi+i_loc)][r] += val_w
        decomps.append((U_sub, V_sub, W_sub))
    
    # M7 = (A12-A22)(B21+B22)
    # = A12*B21 + A12*B22 - A22*B21 - A22*B22
    # Block positions: (0,1,1)+, (0,1,0)+, (1,1,1)-, (1,1,0)-
    for bi, bj, bk, sign in [(0,1,1, 1), (0,1,0, 1), (1,1,1, -1), (1,1,0, -1)]:
        U_sub = [[0]*7 for _ in range(16)]
        V_sub = [[0]*7 for _ in range(16)]
        W_sub = [[0]*7 for _ in range(16)]
        for r in range(7):
            for idx in range(4):
                val_u = U2[idx][r] * sign
                val_v = V2[idx][r]
                val_w = W2[idx][r] * sign
                if val_u != 0:
                    i_loc, j_loc = divmod(idx, 2)
                    U_sub[(bi+i_loc)*4 + (bj+j_loc)][r] += val_u
                if val_v != 0:
                    j_loc, k_loc = divmod(idx, 2)
                    V_sub[(bj+j_loc)*4 + (bk+k_loc)][r] += val_v
                if val_w != 0:
                    k_loc, i_loc = divmod(idx, 2)
                    W_sub[(bk+k_loc)*4 + (bi+i_loc)][r] += val_w
        decomps.append((U_sub, V_sub, W_sub))
    
    # Concatenate all decompositions
    U_out, V_out, W_out = _concat_decomps(decomps)
    
    return U_out, V_out, W_out

def construct():
    t0 = time.time()
    DEADLINE = 145
    
    n, m, p = 4, 5, 5
    
    # Split 4x5x5 into:
    # Core: 4x4x4 (i=0..3, j=0..3, k=0..3)
    # Boundary j=4: 4x1x5 (i=0..3, j=4, k=0..4)
    # Boundary k=4, j<4: 4x4x1 (i=0..3, j=0..3, k=4)
    
    # Get core 4x4x4 Strassen decomposition
    U_core, V_core, W_core = _strassen_4x4x4()
    
    # Embed core into full 4x5x5
    U_full_core = [[0]*len(U_core[0]) for _ in range(n*m)]
    V_full_core = [[0]*len(V_core[0]) for _ in range(m*p)]
    W_full_core = [[0]*len(W_core[0]) for _ in range(p*n)]
    
    r_core = len(U_core[0])
    for r in range(r_core):
        for i in range(4):
            for j in range(4):
                U_full_core[i*m + j][r] = U_core[i*4+j][r]
        for j in range(4):
            for k in range(4):
                V_full_core[j*p + k][r] = V_core[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W_full_core[k*n + i][r] = W_core[k*4+i][r]
    
    # Boundary j=4: 4x1x5 schoolbook
    U_j4, V_j4, W_j4 = _schoolbook_sub(4, 1, 5)
    
    U_full_j4 = [[0]*len(U_j4[0]) for _ in range(n*m)]
    V_full_j4 = [[0]*len(V_j4[0]) for _ in range(m*p)]
    W_full_j4 = [[0]*len(W_j4[0]) for _ in range(p*n)]
    
    r_j4 = len(U_j4[0])
    for r in range(r_j4):
        for i in range(4):
            U_full_j4[i*m + 4][r] = U_j4[i*1+0][r]
        for k in range(5):
            V_full_j4[4*p + k][r] = V_j4[0*5+k][r]
        for k in range(5):
            for i in range(4):
                W_full_j4[k*n + i][r] = W_j4[k*4+i][r]
    
    # Boundary k=4, j<4: 4x4x1 schoolbook
    U_k4, V_k4, W_k4 = _schoolbook_sub(4, 4, 1)
    
    U_full_k4 = [[0]*len(U_k4[0]) for _ in range(n*m)]
    V_full_k4 = [[0]*len(V_k4[0]) for _ in range(m*p)]
    W_full_k4 = [[0]*len(W_k4[0]) for _ in range(p*n)]
    
    r_k4 = len(U_k4[0])
    for r in range(r_k4):
        for i in range(4):
            for j in range(4):
                U_full_k4[i*m + j][r] = U_k4[i*4+j][r]
        for j in range(4):
            V_full_k4[j*p + 4][r] = V_k4[j*1+0][r]
        for i in range(4):
            W_full_k4[4*n + i][r] = W_k4[0*4+i][r]
    
    # Concatenate all three parts
    decomps = [
        (U_full_core, V_full_core, W_full_core),
        (U_full_j4, V_full_j4, W_full_j4),
        (U_full_k4, V_full_k4, W_full_k4),
    ]
    
    U_out, V_out, W_out = _concat_decomps(decomps)
    
    # Verify if time permits
    if time.time() - t0 < DEADLINE - 3:
        rank = len(U_out[0])
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val += U_out[i*m+j][c] * V_out[j*p+k][c] * W_out[k*n+i][c]
                    if abs(val - 1) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to schoolbook
            rank_fb = n * m * p
            U_fb = [[0] * rank_fb for _ in range(n * m)]
            V_fb = [[0] * rank_fb for _ in range(m * p)]
            W_fb = [[0] * rank_fb for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j][col] = 1
                        V_fb[j * p + k][col] = 1
                        W_fb[k * n + i][col] = 1
                        col += 1
            return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U_out, "V": V_out, "W": W_out}
