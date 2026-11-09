# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def verify(U, V, W, n, m, p, tol=1e-6):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        rank = U.shape[1]
        T_decomp = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            U_r = U[:, r].reshape(n, m)
            V_r = V[:, r].reshape(m, p)
            W_r = W[:, r].reshape(p, n)
            T_decomp += U_r * V_r * W_r
        return np.allclose(T_decomp, np.ones((n, m, p)), atol=tol)

    # Attempt to find a rank-49 decomposition for 4x4x4
    # This is a heuristic search
    best_444 = None
    if time.time() < deadline - 30:
        # Try a few random attempts for 4x4x4 rank 49-63
        for target_rank in range(49, 64):
            if time.time() > deadline - 30:
                break
            for _ in range(10):
                if time.time() > deadline - 30:
                    break
                U = np.random.choice([0, 1], size=(16, target_rank))
                V = np.random.choice([0, 1], size=(16, target_rank))
                W = np.random.choice([0, 1], size=(16, target_rank))
                if verify(U, V, W, 4, 4, 4):
                    best_444 = (U, V, W, target_rank)
                    break
            if best_444:
                break

    if best_444:
        U_444, V_444, W_444, r_444 = best_444
        # Construct 4x5x5 from 4x4x4, 4x4x1, 4x1x5
        # 4x4x1 rank 16
        U_441 = np.zeros((20, 16), dtype=int)
        V_441 = np.zeros((25, 16), dtype=int)
        W_441 = np.zeros((20, 16), dtype=int)
        col = 0
        for i in range(4):
            for j in range(4):
                U_441[i * 5 + j, col] = 1
                V_441[j * 5 + 4, col] = 1
                W_441[4 * 4 + i, col] = 1
                col += 1
        
        # 4x1x5 rank 20
        U_415 = np.zeros((20, 20), dtype=int)
        V_415 = np.zeros((25, 20), dtype=int)
        W_415 = np.zeros((20, 20), dtype=int)
        col = 0
        for i in range(4):
            for k in range(5):
                U_415[i * 5 + 4, col] = 1
                V_415[4 * 5 + k, col] = 1
                W_415[k * 4 + i, col] = 1
                col += 1

        # Combine
        # 4x4x4 part needs to be embedded in 4x5x5
        # U_444 is 16xR. We need 20xR.
        U_full_444 = np.zeros((20, r_444), dtype=int)
        V_full_444 = np.zeros((25, r_444), dtype=int)
        W_full_444 = np.zeros((20, r_444), dtype=int)
        # Map indices
        # U: i*5+j for i,j in 0..3
        for i in range(4):
            for j in range(4):
                U_full_444[i * 5 + j, :] = U_444[i * 4 + j, :]
        # V: j*5+k for j,k in 0..3
        for j in range(4):
            for k in range(4):
                V_full_444[j * 5 + k, :] = V_444[j * 4 + k, :]
        # W: k*4+i for k,i in 0..3
        for k in range(4):
            for i in range(4):
                W_full_444[k * 4 + i, :] = W_444[k * 4 + i, :]

        U_total = np.hstack([U_full_444, U_441, U_415])
        V_total = np.hstack([V_full_444, V_441, V_415])
        W_total = np.hstack([W_full_444, W_441, W_415])
        
        if verify(U_total, V_total, W_total, 4, 5, 5):
            return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

    # Fallback to schoolbook
    U, V, W, rank = schoolbook(4, 5, 5)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
