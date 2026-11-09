import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS

    # Build the 3-mode tensor T[i,j,k] = 1 for all i,j,k
    T = np.ones((n, m, p))

    # We want to find a rank-r decomposition: T = sum_c U[:,c] * V[:,c] * W[:,c]
    # U is (n*m, r), V is (m*p, r), W is (p*n, r)

    # Start with the trivial rank = n*m*p = 60
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    for c in range(rank):
        i = c // (m * p)
        j = (c // p) % m
        k = c % p
        U[i * m + j, c] = 1
        V[j * p + k, c] = 1
        W[k * n + i, c] = 1

    # Verify the trivial solution
    def verify(U, V, W, n, m, p):
        """Check if the decomposition exactly reproduces the tensor."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        # Reconstruct T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        return ok

    # Try to reduce rank using random initializations + ALS
    best_U, best_V, best_W, best_rank = U, V, W, rank

    def try_als(rank_candidate, max_iter=50, seed=None):
        """Try to find a rank-rank_candidate decomposition using ALS."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        U = rng.randn(n * m, rank_candidate) * 0.01
        V = rng.randn(m * p, rank_candidate) * 0.01
        W = rng.randn(p * n, rank_candidate) * 0.01

        # Reshape for easier indexing
        def reshape_U(U_mat):
            return U_mat.reshape(n, m, rank_candidate)
        def reshape_V(V_mat):
            return V_mat.reshape(m, p, rank_candidate)
        def reshape_W(W_mat):
            return W_mat.reshape(p, n, rank_candidate)

        for iteration in range(max_iter):
            if time.time() > deadline:
                return None

            Ur = reshape_U(U)
            Vr = reshape_V(V)
            Wr = reshape_W(W)

            # Update W: for each (k, i), W[k,i,:] = sum_j T[i,j,k] * U[i,j,:] * V[j,k,:]
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        Wr[k, i, :] += T[i, j, k] * Ur[i, j, :] * Vr[j, k, :]
            W = Wr.reshape(p * n, rank_candidate)

            # Update U: for each (i, j), U[i,j,:] = sum_k T[i,j,k] * V[j,k,:] * W[k,i,:]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        Ur[i, j, :] += T[i, j, k] * Vr[j, k, :] * Wr[k, i, :]
            U = Ur.reshape(n * m, rank_candidate)

            # Update V: for each (j, k), V[j,k,:] = sum_i T[i,j,k] * U[i,j,:] * W[k,i,:]
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        Vr[j, k, :] += T[i, j, k] * Ur[i, j, :] * Wr[k, i, :]
            V = Vr.reshape(m * p, rank_candidate)

        return U, V, W

    # Try ranks from 30 up to 59
    for rank_candidate in range(30, rank):
        if time.time() > deadline:
            break

        found = False
        for trial in range(20):
            if time.time() > deadline:
                break
            result = try_als(rank_candidate, max_iter=30, seed=trial * 1000 + rank_candidate)
            if result is None:
                break
            U_c, V_c, W_c = result
            if verify(U_c, V_c, W_c, n, m, p):
                best_U, best_V, best_W, best_rank = U_c, V_c, W_c, rank_candidate
                found = True
                break
        if found:
            break

    # Convert to lists of integers or half-integers if possible
    def to_exact(U_mat):
        U_list = []
        for row in U_mat:
            r = []
            for val in row:
                # Round to nearest half-integer
                v = round(val * 2) / 2.0
                if abs(v - round(v)) < 1e-9:
                    r.append(int(round(v)))
                else:
                    r.append(v)
            U_list.append(r)
        return U_list

    return {"U": to_exact(best_U), "V": to_exact(best_V), "W": to_exact(best_W)}
