# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def compute_tensor(U, V, W):
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                BC = Vr[j, :, :] * Wr[:, i, :]
                T[i, j, :] = np.sum(Ur[i, j, :] * BC, axis=1)
        return T

    def verify(U, V, W):
        T = compute_tensor(U, V, W)
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    # Try rank 12 construction (stub, likely fails)
    # We skip actual construction as it's theoretically impossible
    # based on direct sum rank additivity.
    
    # Quick search for rank 35
    # Limit to 2 seconds to ensure we return quickly
    search_time_limit = 2.0
    best_result = None
    
    # Only search if we have time
    if time.time() < deadline - search_time_limit - 5:
        for rank in [35]:
            if time.time() > deadline - search_time_limit:
                break
            # Random search for rank 35
            # Use integer/half-integer entries
            # Scale 0.5
            # Try a few iterations
            max_iters = 5000
            for it in range(max_iters):
                if time.time() > deadline - search_time_limit:
                    break
                U = np.random.randint(-4, 5, size=(n*m, rank)) * 0.5
                V = np.random.randint(-4, 5, size=(m*p, rank)) * 0.5
                W = np.random.randint(-4, 5, size=(p*n, rank)) * 0.5
                
                T = compute_tensor(U, V, W)
                err = np.sum((T - 1.0) ** 2)
                
                if err < 1e-9:
                    if verify(U, V, W):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        break
            if best_result:
                break

    if best_result:
        return best_result
    
    return schoolbook()
# EVOLVE-BLOCK-END
