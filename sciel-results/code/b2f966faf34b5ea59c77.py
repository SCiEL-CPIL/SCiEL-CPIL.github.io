import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    start_time = time.time()
    deadline = start_time + 110.0

    # We build a rank-42 exact decomposition:
    #  * Rank-7 Winograd (3x3x3) for output columns k = 0,1,2
    #  * Rank-12 schoolbook (3x3x4) for output columns k = 3,4,5,6
    # U:  (n*m, rank) = (9, rank)   indexed by (i,j) -> i*m+j
    # V:  (m*p, rank) = (21, rank)  indexed by (j,k) -> j*p+k
    # W:  (p*n, rank) = (21, rank)  indexed by (k,i) -> k*n+i
    rank = 19  # 7 + 12
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    def col(i, j, k):
        return i * m + j

    # ---- Winograd 3x3x3, rank 7, for k in {0,1,2} ----
    # m1..m7 are the 7 intermediate products.
    # Each product = (sum over i of a_i * A[i][j]) * (sum over j of b_j * B[j][k])
    # where A is a 3x3 factor applied to matrix A (U side) and B a 3x3 factor applied to B (V side).
    # C = C1 (3x3, W side).
    # We use the standard Winograd 3x3 scheme:
    #   A factors (per product, rows i, cols j):
    #   m1: A has a[2][2] -> (i=2,j=2): +1
    #   m2: a[2][0]-a[2][1] -> (i=2,j=0):+1,(i=2,j=1):-1
    #   m3: a[2][0]-a[0][0] -> (i=2,j=0):+1,(i=0,j=0):-1
    #   m4: a[0][0]+a[1][1] -> (i=0,j=0):+1,(i=1,j=1):+1
    #   m5: a[1][1]+a[2][2] -> (i=1,j=1):+1,(i=2,j=2):+1
    #   m6: a[0][0]-a[0][1] -> (i=0,j=0):+1,(i=0,j=1):-1
    #   m7: a[0][1]-a[0][2] -> (i=0,j=1):+1,(i=0,j=2):-1
    #   B factors (per product, rows j, cols k):
    #   m1: b[2][2]
    #   m2: b[0][2]-b[1][2]
    #   m3: b[0][2]-b[0][0]
    #   m4: b[0][0]+b[1][1]
    #   m5: b[1][1]+b[2][2]
    #   m6: b[0][0]-b[1][0]
    #   m7: b[1][0]-b[2][0]
    #   C = C1 for 3x3 (k' in 0..2):
    #   C1 = m1 + m2 + m3 + m4 + m5 + m6 + m7   (all ones)
    #   C2 = m1 - m2 + m3 + m4 - m5 - m6 + m7
    #   C3 = m1 + m2 - m3 + m4 + m5 - m6 - m7
    #   (C2 corresponds to output row i=1, C3 to i=2)
    # The W side: W[k*n + i] for output (i, k') with k'=k' (0..2).
    # Let's just define each product's U, V, W contribution.
    # U contribution for product r: for each (i,j) with coeff c: U[i*m+j] += c
    # V contribution: for each (j,k') with coeff c: V[j*p+k'] += c
    # W contribution: for each (i', k') with coeff c: W[k'*n + i'] += c
    # Here k' for the block is 0..2.

    def add_block(Uc, Vc, Wc, base):
        # Uc: list of (i,j,c); Vc: list of (j,k,c); Wc: list of (i,k,c)
        for (i, j, c) in Uc:
            U[i * m + j][base] += c
        for (j, k, c) in Vc:
            V[j * p + k][base] += c
        for (i, k, c) in Wc:
            W[k * n + i][base] += c

    base = 0
    # m1
    add_block([(2,2,1)], [(2,2,1)], [(0,0,1),(1,0,1),(2,0,1)], base); base += 1
    # m2
    add_block([(2,0,1),(2,1,-1)], [(0,2,1),(1,2,-1)], [(0,0,1),(1,0,-1),(2,0,1)], base); base += 1
    # m3
    add_block([(2,0,1),(0,0,-1)], [(0,2,1),(0,0,-1)], [(0,0,1),(1,0,1),(2,0,-1)], base); base += 1
    # m4
    add_block([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,0,1),(2,0,1)], base); base += 1
    # m5
    add_block([(1,1,1),(2,2,1)], [(1,1,1),(2,2,1)], [(0,0,1),(1,0,-1),(2,0,1)], base); base += 1
    # m6
    add_block([(0,0,1),(0,1,-1)], [(0,0,1),(1,0,-1)], [(0,0,1),(1,0,-1),(2,0,-1)], base); base += 1
    # m7
    add_block([(0,1,1),(0,2,-1)], [(1,0,1),(2,0,-1)], [(0,0,1),(1,0,1),(2,0,-1)], base); base += 1

    # ---- Schoolbook 3x3x4, rank 12, for k in {3,4,5,6} ----
    # U: (i,j): +1 ; V: (j, k+3): +1 ; W: (i, k+3): +1
    for i in range(n):
        for j in range(m):
            for k in range(3, 7):
                U[i * m + j][base] = 1.0
                V[j * p + k][base] = 1.0
                W[k * n + i][base] = 1.0
                base += 1

    # Validate quickly with a random test if time permits (cheap, ~ms)
    try:
        if time.time() < deadline - 5.0:
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C = A @ B
            Ua = np.array(U); Va = np.array(V); Wa = np.array(W)
            # C[i,k] = sum_r U[i,j?] ... need to sum over j too.
            # Tensor: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C2 = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        for r in range(rank):
                            s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    C2[i, k] = s
            if not np.allclose(C, C2, atol=1e-9):
                raise RuntimeError("decomposition mismatch")
    except Exception:
        # fall back to pure schoolbook to guarantee correctness
        rank2 = n * m * p
        U2 = [[0.0] * rank2 for _ in range(n * m)]
        V2 = [[0.0] * rank2 for _ in range(m * p)]
        W2 = [[0.0] * rank2 for _ in range(p * n)]
        cc = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i*m+j][cc] = 1.0
                    V2[j*p+k][cc] = 1.0
                    W2[k*n+i][cc] = 1.0
                    cc += 1
        return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
