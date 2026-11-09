import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 130

    n, m, p = 2, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 6, 24, 16

    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    best_rank = n * m * p  # schoolbook = 48
    best_U = None
    best_V = None
    best_W = None

    # Schoolbook decomposition
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Winograd-style approach: treat the p=8 dimension as a convolution
    # and try to find a low-rank decomposition using structured transforms.
    # For a 2x3x8 multiplication, we can think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a set of 2*8 = 16 output values, each a sum over 3 terms.
    #
    # The key insight: if we can find a transform on the k-dimension that
    # diagonalizes or simplifies the multiplication, we can reduce the rank.
    #
    # Let's try a direct approach: search for rank R decompositions using
    # a structured initialization based on Winograd transforms.
    #
    # For a 1x1x8 convolution (simplest case), Winograd uses a 4-point transform
    # to reduce from 8 multiplications to 4. We can try similar ideas here.
    #
    # Actually, let's try a more systematic search. The matrix multiplication
    # tensor for 2x3x8 has a specific structure. Let's try to find the
    # minimum rank by searching over possible ranks.

    # Try ranks from a reasonable lower bound up to schoolbook
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(6, 24, 16) = 24
    # But for matrix multiplication, the rank is at least max(n*m, m*p, p*n)
    # Actually, for M[n,m] x M[m,p], the rank is at least max(n*m, m*p, p*n)
    # which is 24 here. Let's try from 24 upward.

    # Use a smarter search: try to find exact half-integer solutions
    for R in range(24, best_rank + 1):
        if time.time() > deadline:
            break

        # Try multiple random initializations with different strategies
        found = False
        for trial in range(30):
            if time.time() > deadline:
                break

            # Different initialization strategies
            if trial < 10:
                # Random initialization
                np.random.seed(42 + R * 100 + trial)
                U = np.random.randn(nm, R)
                V = np.random.randn(mp, R)
                W = np.random.randn(pn, R)
            elif trial < 20:
                # Structured initialization based on Winograd-like patterns
                # Try to initialize with block structure
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                # For each (i,j,k) triple, try to share rank components
                # This is heuristic - try to create overlapping patterns
                np.random.seed(42 + R * 100 + trial)
                for r in range(R):
                    # Randomly select which (i,j,k) triples share this rank
                    for i in range(n):
                        for j in range(m):
                            if np.random.random() < 0.5:
                                U[i * m + j, r] = np.random.choice([-1, 0, 1])
                        for k in range(p):
                            if np.random.random() < 0.5:
                                V[j * p + k, r] = np.random.choice([-1, 0, 1])
                            if np.random.random() < 0.5:
                                W[k * n + i, r] = np.random.choice([-1, 0, 1])
            else:
                # Sparse initialization
                np.random.seed(42 + R * 100 + trial)
                U = np.zeros((nm, R))
                V = np.zeros((mp, R))
                W = np.zeros((pn, R))
                for r in range(R):
                    # Each rank component touches a few entries
                    n_u = np.random.randint(1, 4)
                    n_v = np.random.randint(1, 4)
                    n_w = np.random.randint(1, 4)
                    u_idx = np.random.choice(nm, n_u, replace=False)
                    v_idx = np.random.choice(mp, n_v, replace=False)
                    w_idx = np.random.choice(pn, n_w, replace=False)
                    U[u_idx, r] = np.random.choice([-1, 1], n_u)
                    V[v_idx, r] = np.random.choice([-1, 1], n_v)
                    W[w_idx, r] = np.random.choice([-1, 1], n_w)

            # ALS iterations
            converged = False
            for it in range(100):
                if time.time() > deadline:
                    break

                # Update U
                for i in range(n):
                    for j in range(m):
                        M_mat = np.zeros((p, R))
                        for k in range(p):
                            M_mat[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.ones(p)
                        x, residuals, rank_m, sv = np.linalg.lstsq(M_mat, b, rcond=None)
                        U[i * m + j, :] = x

                # Update V
                for j in range(m):
                    for k in range(p):
                        M_mat = np.zeros((n, R))
                        for i in range(n):
                            M_mat[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        x, residuals, rank_m, sv = np.linalg.lstsq(M_mat, b, rcond=None)
                        V[j * p + k, :] = x

                # Update W
                for i in range(n):
                    for k in range(p):
                        M_mat = np.zeros((m, R))
                        for j in range(m):
                            M_mat[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        x, residuals, rank_m, sv = np.linalg.lstsq(M_mat, b, rcond=None)
                        W[k * n + i, :] = x

                # Check convergence
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            residual += abs(val - 1.0)
                if residual < 1e-6:
                    converged = True
                    break

            if converged:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2

                # Verify
                valid = True
                for ii in range(nm):
                    for jj in range(mp):
                        for kk in range(pn):
                            val = np.sum(U_r[ii, :] * V_r[jj, :] * W_r[kk, :])
                            expected = 0.0
                            for i in range(n):
                                for j in range(m):
                                    for k in range(p):
                                        if ii == i * m + j and jj == j * p + k and kk == k * n + i:
                                            expected = 1.0
                            if abs(val - expected) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break

                if valid:
                    best_rank = R
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break

        if found:
            break

    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]

    return {"U": U_list, "V": V_list, "W": W_list}
