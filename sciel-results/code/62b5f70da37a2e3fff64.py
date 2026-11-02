"""Search for a low-rank exact tensor decomposition of (2,2,8) matrix multiplication."""
import time

DIMENSIONS = [2, 2, 8]
N, M, P = DIMENSIONS
NM = N * M   # 4
MP = M * P   # 16
PN = P * N   # 16


def verify(U, V, W, n=N, m=M, p=P):
    """Verify that U, V, W form an exact decomposition. Returns True/False."""
    nm, mp, pn = n * m, m * p, p * n
    if not U or not V or not W:
        return False
    rank = len(U[0]) if U else 0
    if len(U) != nm or len(V) != mp or len(W) != pn:
        return False
    for row in U:
        if len(row) != rank:
            return False
    for row in V:
        if len(row) != rank:
            return False
    for row in W:
        if len(row) != rank:
            return False
    # Build a set of valid (a,b,c) indices
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i * m + j, j * p + k, k * n + i))
    # Check all entries
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                s = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                expected = 1 if (a, b, c) in valid else 0
                if s != expected:
                    return False
    return True


def schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    n, m, p = N, M, P
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


def strassen_decomposition_228():
    """
    Construct a rank-28 decomposition for M_{2,2,8} using 4 blocks of Strassen M_{2,2,2}.
    
    Index conventions:
    - U: shape (4, 28), index a = i*2 + j  (i in {0,1}, j in {0,1})
    - V: shape (16, 28), index b = j*8 + k  (j in {0,1}, k in {0,...,7})
    - W: shape (16, 28), index c = k*2 + i  (k in {0,...,7}, i in {0,1})
    
    For block b (b in {0,1,2,3}), we use columns k = 2b and k = 2b+1.
    
    Strassen terms for a 2x2 x 2x2 block:
    Local U indices: 0=(i=0,j=0), 1=(i=0,j=1), 2=(i=1,j=0), 3=(i=1,j=1)
    Local V indices: 0=(j=0,k=0), 1=(j=0,k=1), 2=(j=1,k=0), 3=(j=1,k=1)
    Local W indices: 0=(k=0,i=0), 1=(k=1,i=0), 2=(k=0,i=1), 3=(k=1,i=1)
    """
    n, m, p = N, M, P
    rank = 28
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]

    # Strassen 7 terms: (u_vec, v_vec, w_vec)
    # u_vec indexed by (i,j): [A00, A01, A10, A11]
    # v_vec indexed by (j,k): [B00, B01, B10, B11]
    # w_vec indexed by (k,i): [C00_k0, C01_k0, C00_k1, C01_k1] -> actually (k,i) pairs
    # W local index: 0=(k=0,i=0), 1=(k=1,i=0), 2=(k=0,i=1), 3=(k=1,i=1)
    
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    # M2 = (A10+A11)B00 -> contributes to C10
    # M3 = A00(B01-B11) -> contributes to C01
    # M4 = A11(B10-B00) -> contributes to C00 and C10
    # M5 = (A00+A01)B11 -> contributes to C00 and C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00 and C11
    
    terms = [
        # (u, v, w)
        # M1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: u=[0,0,1,1], v=[1,0,0,0], w=[0,0,1,-1]
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3: u=[1,0,0,0], v=[0,1,0,-1], w=[0,1,0,1]
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4: u=[0,0,0,1], v=[-1,0,1,0], w=[1,0,1,0]
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5: u=[1,1,0,0], v=[0,0,0,1], w=[-1,1,0,0]
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: u=[-1,0,1,0], v=[1,1,0,0], w=[0,0,0,1]
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: u=[0,1,0,-1], v=[0,0,1,1], w=[1,0,0,0]
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    num_blocks = p // 2  # 4 blocks

    for block_idx in range(num_blocks):
        for term_idx in range(7):
            col = block_idx * 7 + term_idx
            u, v, w = terms[term_idx]

            # Fill U: local index l corresponds to (i,j) where l = i*2+j
            # l=0: i=0,j=0; l=1: i=0,j=1; l=2: i=1,j=0; l=3: i=1,j=1
            for l in range(4):
                i_val = l // 2
                j_val = l % 2
                a_idx = i_val * m + j_val  # = l since m=2
                U[a_idx][col] = u[l]

            # Fill V: local index l corresponds to (j, k_local) where k_local in {0,1}
            # l=0: j=0,k_local=0; l=1: j=0,k_local=1; l=2: j=1,k_local=0; l=3: j=1,k_local=1
            # Global k = 2*block_idx + k_local
            for l in range(4):
                j_val = l // 2
                k_local = l % 2
                k_global = 2 * block_idx + k_local
                b_idx = j_val * p + k_global
                V[b_idx][col] = v[l]

            # Fill W: local index l corresponds to (k_local, i) where k_local in {0,1}
            # l=0: k_local=0,i=0; l=1: k_local=1,i=0; l=2: k_local=0,i=1; l=3: k_local=1,i=1
            # Global k = 2*block_idx + k_local
            for l in range(4):
                k_local = l // 2
                i_val = l % 2
                k_global = 2 * block_idx + k_local
                c_idx = k_global * n + i_val
                W[c_idx][col] = w[l]

    return U, V, W


def construct():
    """Entry point: return the best decomposition found."""
    deadline = time.time() + 140  # safety margin before 150s kill

    # Try the known rank-28 Strassen decomposition
    try:
        U, V, W = strassen_decomposition_228()
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # Fallback to schoolbook
    U, V, W = schoolbook()
    return {"U": U, "V": V, "W": W}
