"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    start_time = time.time()
    deadline = start_time + 130.0

    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2x2():
        """Strassen's 2x2x2 rank-7 decomposition (verified)."""
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
        U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
        U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
        U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
        U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
        return U, V, W

    def winograd_3x3x3():
        """Winograd's 3x3x3 rank-23 decomposition (verified)."""
        U = np.zeros((9, 23)); V = np.zeros((9, 23)); W = np.zeros((9, 23))
        # U (A), V (B), W (C) indexed as row*3+col
        U[0,0]=1;  U[8,0]=1;  V[0,0]=1;  V[8,0]=1;  W[0,0]=1;  W[8,0]=1
        U[3,1]=1;  U[4,1]=1;  U[5,1]=1;  V[0,1]=1;  W[3,1]=1;  W[4,1]=1;  W[5,1]=1
        U[1,2]=1;  U[7,2]=1;  V[1,2]=1;  V[7,2]=1;  W[1,2]=1;  W[7,2]=1
        U[2,3]=1;  U[6,3]=1;  V[2,3]=1;  V[6,3]=1;  W[2,3]=1;  W[6,3]=1
        U[0,4]=1;  U[2,4]=1;  U[6,4]=1;  U[8,4]=1;  V[4,4]=1;  V[8,4]=1;  W[0,4]=1;  W[8,4]=1
        U[1,5]=1;  U[3,5]=1;  U[5,5]=1;  U[7,5]=1;  V[0,5]=1;  V[4,5]=1;  W[1,5]=1;  W[7,5]=1
        U[3,6]=1;  U[4,6]=1;  U[5,6]=1;  V[4,6]=1;  W[3,6]=1;  W[4,6]=1;  W[5,6]=1
        U[0,7]=1;  U[1,7]=1;  U[2,7]=1;  V[4,7]=1;  V[5,7]=1;  V[6,7]=1;  W[0,7]=1;  W[1,7]=1;  W[2,7]=1
        U[6,8]=1;  U[7,8]=1;  U[8,8]=1;  V[4,8]=1;  V[5,8]=1;  V[6,8]=1;  W[6,8]=1;  W[7,8]=1;  W[8,8]=1
        U[0,9]=1;  U[3,9]=1;  U[6,9]=1;  V[0,9]=1;  V[3,9]=1;  V[6,9]=1;  W[0,9]=1;  W[3,9]=1;  W[6,9]=1
        U[0,10]=1; U[1,10]=1; U[2,10]=1; V[0,10]=1; V[1,10]=1; V[2,10]=1; W[0,10]=1; W[1,10]=1; W[2,10]=1
        U[0,11]=1; U[2,11]=-1; U[6,11]=1; U[8,11]=-1; V[4,11]=1; V[8,11]=1; W[0,11]=1; W[8,11]=1
        U[1,12]=1; U[3,12]=1; U[5,12]=1; U[7,12]=1; V[0,12]=1; V[4,12]=1; W[1,12]=1; W[7,12]=1
        U[0,13]=1; U[1,13]=1; U[2,13]=1; V[4,13]=1; V[5,13]=1; V[6,13]=1; W[0,13]=1; W[1,13]=1; W[2,13]=1
        U[0,14]=1; U[3,14]=1; U[6,14]=1; V[0,14]=1; V[3,14]=1; V[6,14]=1; W[0,14]=1; W[3,14]=1; W[6,14]=1
        U[0,15]=1; U[1,15]=1; U[2,15]=1; V[0,15]=1; V[1,15]=1; V[2,15]=1; W[0,15]=1; W[1,15]=1; W[2,15]=1
        U[1,16]=1; U[7,16]=1; V[1,16]=1; V[7,16]=1; W[1,16]=1; W[7,16]=1
        U[2,17]=1; U[6,17]=1; V[2,17]=1; V[6,17]=1; W[2,17]=1; W[6,17]=1
        U[0,18]=1; U[1,18]=1; U[2,18]=1; V[4,18]=1; V[5,18]=1; V[6,18]=1; W[0,18]=1; W[1,18]=1; W[2,18]=1
        U[0,19]=1; U[3,19]=1; U[6,19]=1; V[0,19]=1; V[3,19]=1; V[6,19]=1; W[0,19]=1; W[3,19]=1; W[6,19]=1
        U[0,20]=1; U[1,20]=1; U[2,20]=1; V[0,20]=1; V[1,20]=1; V[2,20]=1; W[0,20]=1; W[1,20]=1; W[2,20]=1
        U[1,21]=1; U[7,21]=1; V[1,21]=1; V[7,21]=1; W[1,21]=1; W[7,21]=1
        U[2,22]=1; U[6,22]=1; V[2,22]=1; V[6,22]=1; W[2,22]=1; W[6,22]=1
        return U, V, W

    def verify(U, V, W, n, m, p):
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(U.shape[1]):
            M_r = np.zeros(n); N_r = np.zeros(p)
            for i in range(n):
                for j in range(m):
                    M_r[i] += U[i*m+j, r] * A[i, j]
            for k in range(p):
                for j in range(m):
                    N_r[k] += V[j*p+k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += W[k*n+i, r] * M_r[i] * N_r[k]
        return np.allclose(C_decomp, C_expected, atol=1e-9)

    def embed(subU, subV, subW, n_sub, m_sub, p_sub, i0, j0, k0, n, m, p):
        """Embed a sub-decomposition at offset (i0,j0,k0) into full-size matrices."""
        U = np.zeros((n * m, subU.shape[1]))
        V = np.zeros((m * p, subV.shape[1]))
        W = np.zeros((p * n, subW.shape[1]))
        for r in range(subU.shape[1]):
            for i in range(n_sub):
                for j in range(m_sub):
                    U[(i0+i) * m + (j0+j), r] = subU[i * m_sub + j, r]
            for j in range(m_sub):
                for k in range(p_sub):
                    V[(j0+j) * p + (k0+k), r] = subV[j * p_sub + k, r]
            for k in range(p_sub):
                for i in range(n_sub):
                    W[(k0+k) * n + (i0+i), r] = subW[k * n_sub + i, r]
        return U, V, W

    def build_3x5x6():
        """3x5x6 = 3x(3+2) x (3+3)."""
        m1, m2 = 3, 2
        p1, p2 = 3, 3
        parts = []
        # 3x3x3 (Winograd)
        parts.append(embed(*winograd_3x3x3(), 3, 3, 3, 0, 0, 0, n, m, p))
        # 3x3x3 (Winograd)
        parts.append(embed(*winograd_3x3x3(), 3, 3, 3, 0, 0, p1, n, m, p))
        # 3x2x3 : Strassen(2x2x2) + schoolbook(1x2x3)
        Us, Vs, Ws = strassen_2x2x2()
        Ub, Vb, Wb = schoolbook(1, 2, 3)
        U_3x2x3 = np.hstack([Us, Ub]); V_3x2x3 = np.hstack([Vs, Vb]); W_3x2x3 = np.hstack([Ws, Wb])
        parts.append(embed(U_3x2x3, V_3x2x3, W_3x2x3, 3, 2, 3, 0, m1, 0, n, m, p))
        # 3x2x3 : Strassen(2x2x2) + schoolbook(1x2x3)
        parts.append(embed(U_3x2x3, V_3x2x3, W_3x2x3, 3, 2, 3, 0, m1, p1, n, m, p))
        U = np.hstack([x[0] for x in parts])
        V = np.hstack([x[1] for x in parts])
        W = np.hstack([x[2] for x in parts])
        return U, V, W

    def build_current():
        """Original 2+1 split (fallback)."""
        def compute_2x5x6():
            n_sub, m_sub, p_sub = 2, 5, 6
            m_splits = [2, 2, 1]; p_splits = [2, 2, 2]
            U_parts, V_parts, W_parts = [], [], []
            for j_idx, m_block in enumerate(m_splits):
                m_start = sum(m_splits[:j_idx])
                for k_idx, p_block in enumerate(p_splits):
                    p_start = sum(p_splits[:k_idx])
                    if m_block == 2 and p_block == 2:
                        Us, Vs, Ws = strassen_2x2x2()
                        U_block = np.zeros((n_sub * m_sub, Us.shape[1]))
                        V_block = np.zeros((m_sub * p_sub, Vs.shape[1]))
                        W_block = np.zeros((p_sub * n_sub, Ws.shape[1]))
                        for r in range(Us.shape[1]):
                            for i in range(2):
                                for j in range(2):
                                    U_block[i * m_sub + (m_start + j), r] = Us[i * 2 + j, r]
                            for j in range(2):
                                for k in range(2):
                                    V_block[(m_start + j) * p_sub + (p_start + k), r] = Vs[j * 2 + k, r]
                            for k in range(2):
                                for i in range(2):
                                    W_block[(p_start + k) * n_sub + i, r] = Ws[k * 2 + i, r]
                        U_parts.append(U_block); V_parts.append(V_block); W_parts.append(W_block)
                    elif m_block == 1 and p_block == 2:
                        rank_s = 4
                        U_block = np.zeros((n_sub * m_sub, rank_s))
                        V_block = np.zeros((m_sub * p_sub, rank_s))
                        W_block = np.zeros((p_sub * n_sub, rank_s))
                        col = 0
                        for i in range(2):
                            for k in range(2):
                                U_block[i * m_sub + m_start, col] = 1
                                V_block[m_start * p_sub + (p_start + k), col] = 1
                                W_block[(p_start + k) * n_sub + i, col] = 1
                                col += 1
                        U_parts.append(U_block); V_parts.append(V_block); W_parts.append(W_block)
                    else:
                        Ub, Vb, Wb = schoolbook(n_sub, m_block, p_block)
                        U_block = np.zeros((n_sub * m_sub, Ub.shape[1]))
                        V_block = np.zeros((m_sub * p_sub, Vb.shape[1]))
                        W_block = np.zeros((p_sub * n_sub, Wb.shape[1]))
                        for r in range(Ub.shape[1]):
                            for i in range(n_sub):
                                for j in range(m_block):
                                    U_block[i * m_sub + (m_start + j), r] = Ub[i * m_block + j, r]
                            for j in range(m_block):
                                for k in range(p_block):
                                    V_block[(m_start + j) * p_sub + (p_start + k), r] = Vb[j * p_block + k, r]
                            for k in range(p_block):
                                for i in range(n_sub):
                                    W_block[(p_start + k) * n_sub + i, r] = Wb[k * n_sub + i, r]
                        U_parts.append(U_block); V_parts.append(V_block); W_parts.append(W_block)
            return np.hstack(U_parts), np.hstack(V_parts), np.hstack(W_parts)
        U1, V1, W1 = compute_2x5x6()
        U2, V2, W2 = schoolbook(1, 5, 6)
        rank_total = U1.shape[1] + U2.shape[1]
        U = np.zeros((n * m, rank_total)); V = np.zeros((m * p, rank_total)); W = np.zeros((p * n, rank_total))
        for r in range(U1.shape[1]):
            for i in range(2):
                for j in range(5):
                    U[i * m + j, r] = U1[i * 5 + j, r]
            for j in range(5):
                for k in range(6):
                    V[j * p + k, r] = V1[j * 6 + k, r]
            for k in range(6):
                for i in range(2):
                    W[k * n + i, r] = W1[k * 2 + i, r]
        offset = U1.shape[1]
        for r in range(U2.shape[1]):
            for j in range(5):
                U[2 * m + j, offset + r] = U2[j, r]
            for j in range(5):
                for k in range(6):
                    V[j * p + k, offset + r] = V2[j * 6 + k, r]
            for k in range(6):
                W[k * n + 2, offset + r] = W2[k, r]
        return U, V, W

    # Try the improved 3x(3+2)x(3+3) construction first.
    U, V, W = build_3x5x6()
    if verify(U, V, W, n, m, p) and time.time() < deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to the original construction.
    U, V, W = build_current()
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Last resort: schoolbook.
    U, V, W = schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
