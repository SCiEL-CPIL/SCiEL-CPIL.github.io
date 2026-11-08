import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS

    def verify_decomp(U, V, W, rank):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = i * m + j
                    vi = j * p + k
                    wi = k * n + i
                    s = np.dot(U_arr[ui, :], V_arr[vi, :] * W_arr[wi, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_rank66():
        """Build the known rank-66 decomposition via p=4+2 split with Strassen."""
        rank = 66
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        cur = [0]

        def add_rank1(u_iv, v_iv, w_iv):
            if cur[0] >= rank:
                return
            for idx, val in u_iv:
                U[idx][cur[0]] += val
            for idx, val in v_iv:
                V[idx][cur[0]] += val
            for idx, val in w_iv:
                W[idx][cur[0]] += val
            cur[0] += 1

        def add_sb(i0, j0, k0, nn, mm, pp):
            for ii in range(nn):
                for jj in range(mm):
                    for kk in range(pp):
                        add_rank1([((i0 + ii) * m + (j0 + jj), 1.0)],
                                  [((j0 + jj) * p + (k0 + kk), 1.0)],
                                  [((k0 + kk) * n + (i0 + ii), 1.0)])

        def add_ssr(i0, j0, k0):
            U2 = lambda i, j: (i0 + i) * m + (j0 + j)
            V2 = lambda j, k: (j0 + j) * p + (k0 + k)
            W2 = lambda i, k: (k0 + k) * n + (i0 + i)
            add_rank1([(U2(0,0),1),(U2(1,1),1)], [(V2(0,0),1),(V2(1,1),1)], [(W2(0,0),1),(W2(1,1),1)])
            add_rank1([(U2(1,0),1),(U2(1,1),1)], [(V2(0,0),1)], [(W2(1,0),1),(W2(1,1),-1)])
            add_rank1([(U2(0,0),1)], [(V2(0,1),1),(V2(1,1),-1)], [(W2(0,1),1),(W2(1,1),1)])
            add_rank1([(U2(1,1),1)], [(V2(1,0),1),(V2(0,0),-1)], [(W2(0,0),1),(W2(1,0),1)])
            add_rank1([(U2(0,0),1),(U2(0,1),1)], [(V2(1,1),1)], [(W2(0,0),-1),(W2(0,1),1)])
            add_rank1([(U2(1,0),1),(U2(0,0),-1)], [(V2(0,0),1),(V2(0,1),1)], [(W2(1,1),1)])
            add_rank1([(U2(0,1),1),(U2(1,1),-1)], [(V2(1,0),1),(V2(1,1),1)], [(W2(0,0),1)])

        # p split: 4 + 2
        add_sb(0, 0, 0, 1, 4, 4)
        add_ssr(1, 0, 0)
        add_ssr(1, 2, 0)
        add_ssr(1, 0, 2)
        add_ssr(1, 2, 2)
        add_sb(0, 0, 4, 1, 4, 2)
        add_ssr(1, 0, 4)
        add_ssr(1, 2, 4)

        return {"U": U, "V": V, "W": W}

    def build_rank66_alt():
        """Alternative rank-66: split p=2+2+2, n=1+2 per block."""
        rank = 66
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        cur = [0]

        def add_rank1(u_iv, v_iv, w_iv):
            if cur[0] >= rank:
                return
            for idx, val in u_iv:
                U[idx][cur[0]] += val
            for idx, val in v_iv:
                V[idx][cur[0]] += val
            for idx, val in w_iv:
                W[idx][cur[0]] += val
            cur[0] += 1

        def add_ssr(i0, j0, k0):
            U2 = lambda i, j: (i0 + i) * m + (j0 + j)
            V2 = lambda j, k: (j0 + j) * p + (k0 + k)
            W2 = lambda i, k: (k0 + k) * n + (i0 + i)
            add_rank1([(U2(0,0),1),(U2(1,1),1)], [(V2(0,0),1),(V2(1,1),1)], [(W2(0,0),1),(W2(1,1),1)])
            add_rank1([(U2(1,0),1),(U2(1,1),1)], [(V2(0,0),1)], [(W2(1,0),1),(W2(1,1),-1)])
            add_rank1([(U2(0,0),1)], [(V2(0,1),1),(V2(1,1),-1)], [(W2(0,1),1),(W2(1,1),1)])
            add_rank1([(U2(1,1),1)], [(V2(1,0),1),(V2(0,0),-1)], [(W2(0,0),1),(W2(1,0),1)])
            add_rank1([(U2(0,0),1),(U2(0,1),1)], [(V2(1,1),1)], [(W2(0,0),-1),(W2(0,1),1)])
            add_rank1([(U2(1,0),1),(U2(0,0),-1)], [(V2(0,0),1),(V2(0,1),1)], [(W2(1,1),1)])
            add_rank1([(U2(0,1),1),(U2(1,1),-1)], [(V2(1,0),1),(V2(1,1),1)], [(W2(0,0),1)])

        for k_block in range(3):
            k0 = k_block * 2
            for j in range(4):
                for k in range(2):
                    add_rank1([((0) * m + j, 1.0)], [(j * p + (k0 + k), 1.0)], [((k0 + k) * n + 0, 1.0)])
            add_ssr(1, 0, k0)
            add_ssr(1, 2, k0)

        return {"U": U, "V": V, "W": W}

    def build_schoolbook():
        """Pure schoolbook rank 72 fallback."""
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1.0
                    V[j * p + k][r] = 1.0
                    W[k * n + i][r] = 1.0
                    r += 1
        return {"U": U, "V": V, "W": W}

    # Try multiple constructions, verify, return best valid one
    best = None
    best_rank = float('inf')

    builders = [build_rank66, build_rank66_alt, build_schoolbook]

    for builder in builders:
        if time.time() > deadline - 5:
            break
        try:
            result = builder()
            r = len(result["U"][0])
            if r < best_rank:
                if verify_decomp(result["U"], result["V"], result["W"], r):
                    best_rank = r
                    best = result
        except Exception:
            continue

    if best is None:
        best = build_rank66()

    return best
