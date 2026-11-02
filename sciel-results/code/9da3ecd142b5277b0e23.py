import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n

    # Schoolbook solution (rank = n*m*p = 105)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    def compute_error(U, V, W, num_samples=800):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.dot(U[a,:]*V[b,:], W[c,:])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        rng = np.random.RandomState(42)
        for _ in range(num_samples):
            a = rng.randint(nm)
            b = rng.randint(mp)
            c = rng.randint(pn)
            i_a, j_a = divmod(a, m)
            j_b, k_b = divmod(b, p)
            i_c, k_c = divmod(c, n)
            if not (i_a == i_c and j_a == j_b and k_b == k_c):
                val = np.dot(U[a,:]*V[b,:], W[c,:])
                err = abs(val)
                if err > max_err:
                    max_err = err
        return max_err

    def snap(X):
        return np.round(X * 2) / 2

    # Attempt to find lower-rank decompositions
    for rank in range(104, 50, -1):
        if time.time() > deadline:
            break
        for trial in range(5):
            if time.time() > deadline:
                break
            U = np.random.randn(nm, rank) * 0.3
            V = np.random.randn(mp, rank) * 0.3
            W = np.random.randn(pn, rank) * 0.3
            for it in range(50):
                if time.time() > deadline:
                    break
                # Update U
                VtV, WtW = V.T @ V, W.T @ W
                M = VtV * WtW + 1e-5 * np.eye(rank)
                rhs = np.zeros((nm, rank))
                for i in range(n):
                    for j in range(m):
                        a = i*m + j
                        for k in range(p):
                            rhs[a,:] += V[j*p+k,:] * W[k*n+i,:]
                try:
                    U = np.linalg.solve(M.T, rhs.T).T
                except:
                    U = np.linalg.lstsq(M.T, rhs.T, rcond=None)[0].T
                # Update V
                UtU, WtW2 = U.T @ U, W.T @ W
                M2 = UtU * WtW2 + 1e-5 * np.eye(rank)
                rhs2 = np.zeros((mp, rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            rhs2[j*p+k,:] += U[i*m+j,:] * W[k*n+i,:]
                try:
                    V = np.linalg.solve(M2.T, rhs2.T).T
                except:
                    V = np.linalg.lstsq(M2.T, rhs2.T, rcond=None)[0].T
                # Update W
                UtU2, VtV2 = U.T @ U, V.T @ V
                M3 = UtU2 * VtV2 + 1e-5 * np.eye(rank)
                rhs3 = np.zeros((pn, rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            rhs3[k*n+i,:] += U[i*m+j,:] * V[j*p+k,:]
                try:
                    W = np.linalg.solve(M3.T, rhs3.T).T
                except:
                    W = np.linalg.lstsq(M3.T, rhs3.T, rcond=None)[0].T
            err = compute_error(U, V, W, 500)
            if err < 1e-4:
                Us, Vs, Ws = snap(U), snap(V), snap(W)
                errs = compute_error(Us, Vs, Ws, 500)
                if errs < 1e-4:
                    best_U, best_V, best_W = Us, Vs, Ws
                    best_rank = rank
                    break
        if best_rank < rank_school:
            break

    return {"U": [[float(x) for x in r] for r in best_U],
            "V": [[float(x) for x in r] for r in best_V],
            "W": [[float(x) for x in r] for r in best_W]}
