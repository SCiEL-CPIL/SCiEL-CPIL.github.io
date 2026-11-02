import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = 135  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n*m, m*p, p*n  # 12, 32, 24

    def verify_decomposition(U, V, W):
        """Vectorized exact verification."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        rank = U.shape[1]
        # Build full tensor check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Vectorize over c
        # U: (nm, rank), V: (mp, rank), W: (pn, rank)
        # We need: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == (1 if j==k else 0)
        # Reshape approach:
        # For each c, define Uc = U[:,c] (nm,), Vc = V[:,c] (mp,), Wc = W[:,c] (pn,)
        # The contribution to (i,j,k) from c is Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
        # Sum over c.
        
        # Precompute index arrays
        i_idx = np.repeat(np.arange(n), m*p)  # length n*m*p
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n*m)
        
        U_idx = i_idx * m + j_idx  # (n*m*p,)
        V_idx = j_idx * p + k_idx
        W_idx = k_idx * n + i_idx
        
        # For each c: U[U_idx, c] * V[V_idx, c] * W[W_idx, c]
        # Stack over c: shape (n*m*p, rank)
        # U[U_idx] is (n*m*p, rank) if we index rows
        U_exp = U[U_idx, :]  # (n*m*p, rank)
        V_exp = V[V_idx, :]
        W_exp = W[W_idx, :]
        
        # Element-wise product and sum over rank axis
        prod = U_exp * V_exp * W_exp  # (n*m*p, rank)
        sums = prod.sum(axis=1)  # (n*m*p,)
        
        # Target: 1 if j==k else 0
        target = (j_idx == k_idx).astype(float)
        
        return np.allclose(sums, target, atol=1e-9)

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

    def tensor_product(U1, V1, W1, U2, V2, W2):
        U = np.kron(U1, U2)
        V = np.kron(V1, V2)
        W = np.kron(W1, W2)
        return U, V, W

    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0,0]=1; U[1,0]=1; V[0,0]=1; V[1,0]=1
        W[0,0]=1; W[1,0]=-1; W[2,0]=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; V[1,1]=1
        W[1,1]=1; W[3,1]=1
        U[0,2]=1; V[3,2]=1; V[1,2]=-1
        W[2,2]=-1; W[3,2]=-1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1
        W[0,3]=1; W[2,3]=1
        U[0,4]=1; V[0,4]=1; V[3,4]=1; V[1,4]=1
        W[0,4]=-1; W[1,4]=1; W[3,4]=1
        U[2,5]=1; U[0,5]=1; V[1,5]=1; V[3,5]=-1
        W[1,5]=1; W[2,5]=1
        U[1,6]=1; U[3,6]=1; V[2,6]=1; V[0,6]=-1
        W[0,6]=1; W[3,6]=1
        return U, V, W

    def winograd_212():
        U = np.zeros((2, 5))
        V = np.zeros((2, 5))
        W = np.zeros((2, 5))
        U[0,0]=1; U[1,0]=1; V[0,0]=1
        W[0,0]=1; W[1,0]=1
        U[0,1]=1; U[1,1]=-1; V[1,1]=1
        W[0,1]=1; W[1,1]=-1
        U[0,2]=1; V[0,2]=1
        W[0,2]=1
        U[0,3]=1; V[1,3]=1
        W[1,3]=1
        U[1,4]=1; V[1,4]=1
        W[1,4]=1
        return U, V, W

    def trivial_112():
        U = np.array([[1.0, 1.0]])
        V = np.array([[1.0, 0.0], [0.0, 1.0]])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        return U, V, W

    # Start with schoolbook as fallback
    best_result, best_rank = schoolbook_decomposition(n, m, p)

    def try_combo(combo):
        U, V, W = combo[0]
        for (U2, V2, W2) in combo[1:]:
            U, V, W = tensor_product(U, V, W, U2, V2, W2)
        rank = U.shape[1]
        if U.shape[0] != nm or V.shape[0] != mp or W.shape[0] != pn:
            return None
        if verify_decomposition(U, V, W):
            return rank, (U, V, W)
        return None

    S = strassen_222()
    Wg = winograd_212()
    T112 = trivial_112()

    # M_{3,4,8} = S ⊗ Wg ⊗ T112 (rank 70)
    c348 = try_combo([S, Wg, T112])
    if c348 is not None and c348[0] < best_rank:
        best_rank = c348[0]; best_result = c348[1]

    # M_{3,4,8} = S ⊗ S ⊗ T112 (rank 98)
    c348b = try_combo([S, S, T112])
    if c348b is not None and c348b[0] < best_rank:
        best_rank = c348b[0]; best_result = c348b[1]

    # Direct search with full least-squares W solver
    def direct_search(target_rank, max_attempts, seed_base, scale_choices):
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline - 8:
                return None
            seed = seed_base + attempt * 7919 + target_rank
            rng = np.random.RandomState(seed)
            scale = scale_choices[attempt % len(scale_choices)]
            U = rng.randint(-scale, scale+1, size=(nm, target_rank)).astype(float)
            V = rng.randint(-scale, scale+1, size=(mp, target_rank)).astype(float)

            # M[i,k,c] = sum_j U[i*m+j,c] V[j*p+k,c]
            M = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for c in range(target_rank):
                        M[i, k, c] = sum(U[i*m+j, c] * V[j*p+k, c] for j in range(m))

            # Q (pn x rank): row (k*n+i) = M[i,k,:]
            Q = np.zeros((pn, target_rank))
            for i in range(n):
                for k in range(p):
                    Q[k*n+i, :] = M[i, k, :]

            # Solve Q W^T = I_pn -> W^T = pinv(Q) @ I_pn = pinv(Q)
            try:
                Wt = np.linalg.lstsq(Q, np.eye(pn), rcond=None)[0]  # (rank, pn)
            except np.linalg.LinAlgError:
                continue
            W = Wt.T  # (pn, rank)

            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2

            if verify_decomposition(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
        return None

    # Search ranks below best known
    search_ranks = list(range(min(best_rank - 1, 69), max(20, best_rank - 50), -1))
    for target_rank in search_ranks:
        if time.time() - start_time > deadline - 8:
            break
        for sc_choice in ([1,2], [2,3], [3,4], [1,3], [2,4]):
            if time.time() - start_time > deadline - 8:
                break
            result = direct_search(target_rank, max_attempts=30, seed_base=5000, scale_choices=sc_choice)
            if result is not None:
                best_result = result
                best_rank = target_rank
                search_ranks = list(range(target_rank - 1, max(20, target_rank - 40), -1))
                break

    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
