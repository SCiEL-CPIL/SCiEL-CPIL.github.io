"""Matrix multiplication rank minimization for 3x5 * 5x6."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Build the support of the matrix multiplication tensor
    # Tensor T[i,j,k] = 1 if (i,j,k) corresponds to a valid (a,b,c) with
    # i = a*m+b, j = b*p+c, k = c*n+a
    support = []
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i_idx = a * m + b
                j_idx = b * p + c
                k_idx = c * n + a
                support.append((i_idx, j_idx, k_idx))

    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i_idx, j_idx, k_idx in support:
        U_school[i_idx, col] = 1.0
        V_school[j_idx, col] = 1.0
        W_school[k_idx, col] = 1.0
        col += 1

    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Helper: verify a decomposition is exact
    def verify(U, V, W, tol=1e-9):
        # Check all support entries equal 1
        for i_idx, j_idx, k_idx in support:
            val = np.dot(U[i_idx, :] * V[j_idx, :], W[k_idx, :])
            if abs(val - 1.0) > tol:
                return False
        # Check all off-support entries are 0
        # Iterate over all possible (i,j,k) triples
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i_idx = a * m + b
                    j_idx = b * p + c
                    k_idx = c * n + a
                    # This is a support point, already checked
        # Check off-support: any (i,j,k) where the indices don't form a valid (a,b,c)
        for i in range(nm):
            a, b = divmod(i, m)
            for j in range(mp):
                b2, c = divmod(j, p)
                for k in range(pn):
                    c2, a2 = divmod(k, n)
                    if b != b2 or c != c2 or a != a2:
                        val = np.dot(U[i, :] * V[j, :], W[k, :])
                        if abs(val) > tol:
                            return False
        return True

    # Helper: compute the full tensor error for a decomposition
    def compute_errors(U, V, W):
        errors = []
        for i_idx, j_idx, k_idx in support:
            val = np.dot(U[i_idx, :] * V[j_idx, :], W[k_idx, :])
            errors.append(val - 1.0)
        # Sample off-support
        random.seed(42)
        for _ in range(200):
            i = random.randint(0, nm - 1)
            j = random.randint(0, mp - 1)
            k = random.randint(0, pn - 1)
            a, b = divmod(i, m)
            b2, c = divmod(j, p)
            c2, a2 = divmod(k, n)
            if b != b2 or c != c2 or a != a2:
                val = np.dot(U[i, :] * V[j, :], W[k, :])
                errors.append(val)
        return np.array(errors)

    # Try to find lower-rank decompositions
    # Start from rank_school - 1 and go down
    for r in range(rank_school - 1, 30, -1):
        if time.time() > deadline:
            break

        found = False
        num_attempts = 8

        for attempt in range(num_attempts):
            if time.time() > deadline:
                break

            # Initialize with a structured random assignment
            U_try = np.zeros((nm, r))
            V_try = np.zeros((mp, r))
            W_try = np.zeros((pn, r))

            # Assign each support element to a random rank component
            assignments = np.random.randint(0, r, len(support))
            for idx, (i_idx, j_idx, k_idx) in enumerate(support):
                r_idx = assignments[idx]
                U_try[i_idx, r_idx] = 1.0
                V_try[j_idx, r_idx] = 1.0
                W_try[k_idx, r_idx] = 1.0

            # Add small random noise
            noise_scale = 0.05
            U_try += noise_scale * np.random.randn(nm, r)
            V_try += noise_scale * np.random.randn(mp, r)
            W_try += noise_scale * np.random.randn(pn, r)

            # Iterative refinement using alternating least squares
            # For each rank component, solve for the best U, V, W columns
            for step in range(200):
                if time.time() > deadline:
                    break

                # Update U: for each support point (i,j,k), we want
                # sum_r U[i,r] * V[j,r] * W[k,r] = 1
                # For fixed V, W, this is linear in U[i,:]
                # But it's coupled across rank components, so use gradient descent

                # Compute gradients
                # d/dU[i,r] sum_s U[i,s]*V[j,s]*W[k,s] - 1
                # = V[j,r]*W[k,r] for the support point (i,j,k)

                lr = 0.01
                U_grad = np.zeros_like(U_try)
                V_grad = np.zeros_like(V_try)
                W_grad = np.zeros_like(W_try)

                for i_idx, j_idx, k_idx in support:
                    recon = np.dot(U_try[i_idx, :] * V_try[j_idx, :], W_try[k_idx, :])
                    err = recon - 1.0
                    U_grad[i_idx, :] += err * V_try[j_idx, :] * W_try[k_idx, :]
                    V_grad[j_idx, :] += err * U_try[i_idx, :] * W_try[k_idx, :]
                    W_grad[k_idx, :] += err * U_try[i_idx, :] * V_try[j_idx, :]

                # Off-support gradient (sampled)
                random.seed(step)
                for _ in range(50):
                    i = random.randint(0, nm - 1)
                    j = random.randint(0, mp - 1)
                    k = random.randint(0, pn - 1)
                    a, b = divmod(i, m)
                    b2, c = divmod(j, p)
                    c2, a2 = divmod(k, n)
                    if b != b2 or c != c2 or a != a2:
                        recon = np.dot(U_try[i, :] * V_try[j, :], W_try[k, :])
                        U_grad[i, :] += recon * V_try[j, :] * W_try[k, :]
                        V_grad[j, :] += recon * U_try[i, :] * W_try[k, :]
                        W_grad[k, :] += recon * U_try[i, :] * V_try[j, :]

                U_try -= lr * U_grad
                V_try -= lr * V_grad
                W_try -= lr * W_grad

                # Check convergence
                max_err = 0.0
                for i_idx, j_idx, k_idx in support:
                    recon = np.dot(U_try[i_idx, :] * V_try[j_idx, :], W_try[k_idx, :])
                    max_err = max(max_err, abs(recon - 1.0))

                if max_err < 1e-10:
                    # Full verification
                    if verify(U_try, V_try, W_try):
                        best_U, best_V, best_W = U_try, V_try, W_try
                        best_rank = r
                        found = True
                        break

            if found:
                break

        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
