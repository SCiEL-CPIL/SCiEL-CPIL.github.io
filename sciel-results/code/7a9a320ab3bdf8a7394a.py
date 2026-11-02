import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 4, 8

    def verify_decomposition(U, V, W, rank):
        """Verify exact decomposition: sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 iff j==k else 0."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = float(np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]))
                    target = 1.0 if j == k else 0.0
                    if abs(s - target) > 1e-6:
                        return False
        return True

    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return (U, V, W), rank

    # Start with schoolbook as fallback
    best_result, best_rank = schoolbook_decomposition(n, m, p)

    def tensor_product(U1, V1, W1, U2, V2, W2):
        """Kronecker tensor product of two decompositions.
        T1 for M_{a,b,c}, T2 for M_{d,e,f} -> T1⊗T2 for M_{ad,be,cf}, rank r1*r2."""
        U = np.kron(U1, U2)
        V = np.kron(V1, V2)
        W = np.kron(W1, W2)
        return U, V, W

    def strassen_222():
        """Rank-7 Strassen decomposition for M_{2,2,2}.
        U:(4,7) idx a=0,b=1,c=2,d=3 ; V:(4,7) idx e=0,f=1,g=2,h=3 ; W:(4,7) idx p00=0,p01=1,p10=2,p11=3."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # m1 = (a+b)(e+f)
        U[0,0]=1; U[1,0]=1; V[0,0]=1; V[1,0]=1
        W[0,0]=1; W[1,0]=-1; W[2,0]=1
        # m2 = (c+d)(e+f)
        U[2,1]=1; U[3,1]=1; V[0,1]=1; V[1,1]=1
        W[1,1]=1; W[3,1]=1
        # m3 = a(h-f)
        U[0,2]=1; V[3,2]=1; V[1,2]=-1
        W[2,2]=-1; W[3,2]=-1
        # m4 = d(g-e)
        U[3,3]=1; V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        # m5 = (a+e)(h+f)
        U[0,4]=1; V[0,4]=1; V[3,4]=1; V[1,4]=1
        W[0,4]=-1; W[1,4]=1; W[3,4]=1
        # m6 = (c+a)(f-h)
        U[2,5]=1; U[0,5]=1; V[1,5]=1; V[3,5]=-1
        W[1,5]=1; W[2,5]=1
        # m7 = (b+d)(g-c)
        U[1,6]=1; U[3,6]=1; V[2,6]=1; V[0,6]=-1
        W[0,6]=1; W[3,6]=1
        return U, V, W

    def winograd_212():
        """Rank-5 Winograd decomposition for M_{2,1,2} (row 1x2 * column 2x1 -> 1x2).
        U:(2,5) idx a=0,b=1 ; V:(2,5) idx e=0,g=1 ; W:(2,5) idx p0=0,p1=1."""
        U = np.zeros((2, 5))
        V = np.zeros((2, 5))
        W = np.zeros((2, 5))
        # m1 = (a+b) e
        U[0,0]=1; U[1,0]=1; V[0,0]=1
        W[0,0]=1; W[1,0]=1
        # m2 = (a-b) g
        U[0,1]=1; U[1,1]=-1; V[1,1]=1
        W[0,1]=1; W[1,1]=-1
        # m3 = a e
        U[0,2]=1; V[0,2]=1
        W[0,2]=1
        # m4 = a g
        U[0,3]=1; V[1,3]=1
        W[1,3]=1
        # m5 = b g
        U[1,4]=1; V[1,4]=1
        W[1,4]=1
        return U, V, W

    def trivial_112():
        """Rank-2 decomposition for M_{1,1,2} (1x1 * 1x2 -> 1x2)."""
        U = np.array([[1.0, 1.0]])
        V = np.array([[1.0, 0.0], [0.0, 1.0]])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        return U, V, W

    def trivial_114():
        """Rank-4 decomposition for M_{1,1,4} (1x1 * 1x4 -> 1x4)."""
        U = np.ones((1, 4))
        V = np.eye(4)
        W = np.eye(4)
        return U, V, W

    def schoolbook(n, m, p):
        U = np.zeros((n*m, n*m*p)); V = np.zeros((m*p, n*m*p)); W = np.zeros((p*n, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1; V[j*p+k, col] = 1; W[k*n+i, col] = 1; col += 1
        return U, V, W

    def try_combo(combo, label):
        """combo: list of (U,V,W). Combine via tensor product, verify, return rank if valid."""
        U, V, W = combo[0]
        for (U2, V2, W2) in combo[1:]:
            U, V, W = tensor_product(U, V, W, U2, V2, W2)
        rank = U.shape[1]
        if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
            return None
        if verify_decomposition(U, V, W, rank):
            return rank, (U, V, W)
        return None

    # ---- Known tensor-product constructions ----
    S = strassen_222()
    Wg = winograd_212()
    T112 = trivial_112()
    T114 = trivial_114()

    # M_{2,4,4} = S ⊗ Wg  (rank 35)
    c244 = try_combo([S, Wg], "244")
    # M_{3,4,8} = M_{2,4,4} ⊗ T112  (rank 70)
    if c244 is not None:
        c348 = try_combo([S, Wg, T112], "348_via_244")
        if c348 is not None and c348[0] < best_rank:
            best_rank = c348[0]; best_result = c348[1]

    # M_{3,2,4} = S ⊗ S (rank 49); M_{1,2,2}=T112 (rank 2) -> M_{3,4,8} = S⊗S⊗T112 (rank 98)
    c348b = try_combo([S, S, T112], "348_via_324")
    if c348b is not None and c348b[0] < best_rank:
        best_rank = c348b[0]; best_result = c348b[1]

    # M_{3,4,2} = S⊗S (rank 49); M_{1,1,4}=T114 (rank 4) -> rank 196
    c348c = try_combo([S, S, T114], "348_via_342")
    if c348c is not None and c348c[0] < best_rank:
        best_rank = c348c[0]; best_result = c348c[1]

    # M_{3,2,2}=S⊗Wg (35); M_{1,2,2}=T112 (2); M_{1,1,2}=T112 (2) -> rank 140
    c348d = try_combo([S, Wg, T112, T112], "348_via_322")
    if c348d is not None and c348d[0] < best_rank:
        best_rank = c348d[0]; best_result = c348d[1]

    # ---- Direct search with CORRECTED W solver ----
    # Constraint: sum_c M[i,k,c] W[k*n+i,c] = 1 if j==k else 0.
    # Build Q (24 x rank) with row (i,k) = M[i,k,:]; find W (24 x rank) s.t. Q W^T = I_24.
    def direct_search(target_rank, max_attempts, seed_base, scale_choices):
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline - 8:
                return None
            seed = seed_base + attempt * 7919 + target_rank
            np.random.seed(seed)
            scale = scale_choices[attempt % len(scale_choices)]
            U = np.random.randint(-scale, scale+1, size=(n*m, target_rank)).astype(float)
            V = np.random.randint(-scale, scale+1, size=(m*p, target_rank)).astype(float)

            # M[i,k,c] = sum_j U[i*m+j,c] V[j*p+k,c]
            M = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for c in range(target_rank):
                        M[i, k, c] = sum(U[i*m+j, c] * V[j*p+k, c] for j in range(m))

            # Q row (i,k) = M[i,k,:]
            Q = np.zeros((n*p, target_rank))
            for i in range(n):
                for k in range(p):
                    Q[i*p+k, :] = M[i, k, :]

            # Solve Q W^T = I  ->  W = (Q^T Q)^{-1} Q^T  (least squares)
            try:
                QWt, _, _, _ = np.linalg.lstsq(Q, np.eye(n*p), rcond=None)
            except np.linalg.LinAlgError:
                continue
            # QWt is (rank x 24); W = QWt^T (24 x rank)
            W = QWt.T

            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2

            if verify_decomposition(U_r, V_r, W_r, target_rank):
                return (U_r, V_r, W_r)
        return None

    # Search ranks below the best known construction
    search_ranks = list(range(min(best_rank - 1, 95), max(20, best_rank - 60), -1))
    for target_rank in search_ranks:
        if time.time() - start_time > deadline - 8:
            break
        for sc_choice in ([1,2], [2,3], [3,4], [1,3]):
            if time.time() - start_time > deadline - 8:
                break
            result = direct_search(target_rank, max_attempts=25, seed_base=5000, scale_choices=sc_choice)
            if result is not None:
                best_result = result
                best_rank = target_rank
                search_ranks = list(range(target_rank - 1, max(20, target_rank - 40), -1))
                break

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
