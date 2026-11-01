"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    def build_target_triples():
        """Return list of (a,b,c) where a=i*m+j, b=j*p+k, c=k*n+i"""
        triples = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    triples.append((a, b, c))
        return triples

    triples = build_target_triples()
    n_triples = len(triples)  # 20

    def check_decomposition(U, V, W, rank):
        """Check if decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for (a, b, c) in triples:
            val = 0.0
            for r in range(rank):
                val += U_arr[a, r] * V_arr[b, r] * W_arr[c, r]
            if abs(val - 1.0) > 1e-9:
                return False
        return True

    def build_schoolbook():
        """Schoolbook decomposition, rank = 20."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for (a, b, c) in triples:
            U[a, idx] = 1.0
            V[b, idx] = 1.0
            W[c, idx] = 1.0
            idx += 1
        return U, V, W, rank

    def als_decompose(r, max_iter=500, tol=1e-14, seed=None):
        """Try to find a rank-r decomposition using ALS with random restarts."""
        if seed is not None:
            np.random.seed(seed)

        for attempt in range(10):
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1

            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                # Update U
                U_new = np.zeros((n*m, r))
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        A_mat = np.zeros((p, r))
                        for k in range(p):
                            A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                        b_vec = np.ones(p)
                        U_new[a, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                U = U_new

                # Update V
                V_new = np.zeros((m*p, r))
                for j in range(m):
                    for k in range(p):
                        b = j * p + k
                        A_mat = np.zeros((n, r))
                        for i in range(n):
                            A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                        b_vec = np.ones(n)
                        V_new[b, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                V = V_new

                # Update W
                W_new = np.zeros((p*n, r))
                for k in range(p):
                    for i in range(n):
                        c = k * n + i
                        A_mat = np.zeros((m, r))
                        for j in range(m):
                            A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                        b_vec = np.ones(m)
                        W_new[c, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                W = W_new

            # Verify
            if check_decomposition(U.tolist(), V.tolist(), W.tolist(), r):
                return U, V, W
        return None

    # Try ranks from 17 upward
    for r in range(17, 21):
        if time.time() > deadline:
            break
        result = als_decompose(r, max_iter=300, seed=r*100)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fall back to schoolbook rank 20
    U, V, W, _ = build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
