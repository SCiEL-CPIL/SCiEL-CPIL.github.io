import numpy as np
import time

def construct():
    t0 = time.time()
    tlimit = t0 + 115

    nn, mm, pp = 4, 5, 6
    NU = nn * mm          # 20
    NV = mm * pp          # 30
    NW = pp * nn          # 24

    STR = [
        ([[1,0],[0,1]], [[1,0],[0,1]], [[1,0],[0,1]]),
        ([[0,0],[1,1]], [[1,0],[0,0]], [[0,1],[0,-1]]),
        ([[1,0],[0,0]], [[0,1],[0,-1]], [[0,0],[1,1]]),
        ([[0,0],[0,1]], [[-1,0],[1,0]], [[1,1],[0,0]]),
        ([[1,1],[0,0]], [[0,0],[0,1]], [[-1,0],[1,0]]),
        ([[-1,0],[1,0]], [[1,1],[0,0]], [[0,0],[0,1]]),
        ([[0,1],[0,-1]], [[0,0],[1,1]], [[1,0],[0,0]]),
    ]

    def build_decomp(partitions):
        """Build decomposition given a list of (i_range, j_range, k_range, method)"""
        terms = []
        for (i_start, i_end, j_start, j_end, k_start, k_end, method) in partitions:
            if method == 'strassen':
                # 2x2x2 Strassen block
                for (u_mat, v_mat, w_mat) in STR:
                    u_vec = {}
                    v_vec = {}
                    w_vec = {}
                    for li in range(2):
                        for lj in range(2):
                            c = u_mat[li][lj]
                            if c != 0:
                                gi = i_start + li
                                gj = j_start + lj
                                u_vec[(gi, gj)] = c
                    for lj in range(2):
                        for lk in range(2):
                            c = v_mat[lj][lk]
                            if c != 0:
                                gj = j_start + lj
                                gk = k_start + lk
                                v_vec[(gj, gk)] = c
                    for lk in range(2):
                        for li in range(2):
                            c = w_mat[lk][li]
                            if c != 0:
                                gk = k_start + lk
                                gi = i_start + li
                                w_vec[(gk, gi)] = c
                    terms.append((u_vec, v_vec, w_vec))
            elif method == 'schoolbook':
                for gi in range(i_start, i_end):
                    for gj in range(j_start, j_end):
                        for gk in range(k_start, k_end):
                            u_vec = {((gi, gj)): 1.0}
                            v_vec = {((gj, gk)): 1.0}
                            w_vec = {((gk, gi)): 1.0}
                            terms.append((u_vec, v_vec, w_vec))
        
        rank = len(terms)
        U = np.zeros((NU, rank), dtype=np.float64)
        V = np.zeros((NV, rank), dtype=np.float64)
        W = np.zeros((NW, rank), dtype=np.float64)
        
        for col, (u_vec, v_vec, w_vec) in enumerate(terms):
            for (gi, gj), c in u_vec.items():
                U[gi * mm + gj, col] += c
            for (gj, gk), c in v_vec.items():
                V[gj * pp + gk, col] += c
            for (gk, gi), c in w_vec.items():
                W[gk * nn + gi, col] += c
        
        return U, V, W, rank

    def verify(U, V, W, rank):
        max_err = 0.0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    ui = U[i * mm + j, :rank]
                    vj = V[j * pp + k, :rank]
                    wk = W[k * nn + i, :rank]
                    s = np.dot(ui, vj * wk)
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    # Partition 1: Original (4x4x6 as 12 Strassen + 4x1x6 schoolbook)
    p1 = []
    for bi in range(2):
        for bj in range(2):
            for bk in range(3):
                p1.append((2*bi, 2*bi+2, 2*bj, 2*bj+2, 2*bk, 2*bk+2, 'strassen'))
    for gi in range(4):
        for gk in range(6):
            p1.append((gi, gi+1, 4, 5, gk, gk+1, 'schoolbook'))
    
    U1, V1, W1, r1 = build_decomp(p1)
    err1 = verify(U1, V1, W1, r1)

    best_U, best_V, best_W, best_rank = U1, V1, W1, r1
    best_err = err1

    # Try other partitions if time permits
    if time.time() < tlimit - 5:
        # Partition 2: 4x5x6 = 4x3x6 + 4x2x6
        p2 = []
        # 4x3x6: split into 4x3x2 + 4x3x4
        # 4x3x2: 2 blocks of 2x3x2
        for bi in range(2):
            # 2x3x2: split j into 2+1
            p2.append((2*bi, 2*bi+2, 0, 2, 0, 2, 'strassen'))  # 2x2x2
            for gj in range(2, 3):
                for gk in range(2):
                    p2.append((2*bi, 2*bi+2, gj, gj+1, gk, gk+1, 'schoolbook'))  # 2x1x2
            # Wait, this is getting complicated. Let me just use the original.
        
        # Skip complex partitions, stick with original

    if best_err > 1e-9:
        # Fallback: pure schoolbook
        rs = nn * mm * pp
        Us = np.zeros((NU, rs), dtype=np.float64)
        Vs = np.zeros((NV, rs), dtype=np.float64)
        Ws = np.zeros((NW, rs), dtype=np.float64)
        c2 = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    Us[i * mm + j, c2] = 1.0
                    Vs[j * pp + k, c2] = 1.0
                    Ws[k * nn + i, c2] = 1.0
                    c2 += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    return {"U": best_U[:, :best_rank].tolist(), "V": best_V[:, :best_rank].tolist(), "W": best_W[:, :best_rank].tolist()}
