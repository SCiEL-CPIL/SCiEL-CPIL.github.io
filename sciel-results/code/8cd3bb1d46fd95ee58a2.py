"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Schoolbook as fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    best_result = schoolbook()
    best_rank = n * m * p

    # Precompute index arrays for vectorized operations
    # For each (i,j,k) triple, we need:
    #   u_idx = i*m + j
    #   v_idx = j*p + k
    #   w_idx = k*n + i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    triples = np.array(triples, dtype=np.int64)
    ntrip = len(triples)

    def compute_sum(U, V, W, rank):
        """Compute the tensor contraction sum for all (i,j,k) triples."""
        # U[u_idx, :], V[v_idx, :], W[w_idx, :]
        # sum_c U[u,c] * V[v,c] * W[w,c]
        U_sub = U[triples[:, 0], :]  # (ntrip, rank)
        V_sub = V[triples[:, 1], :]  # (ntrip, rank)
        W_sub = W[triples[:, 2], :]  # (ntrip, rank)
        return np.sum(U_sub * V_sub * W_sub, axis=1)  # (ntrip,)

    def compute_error(sums):
        """Compute max absolute error from sums."""
        return np.max(np.abs(sums - 1.0))

    def smooth_objective(sums, beta=100.0):
        """Smooth surrogate for max|s-1| using log-sum-exp."""
        d = sums - 1.0
        # Use soft-max of |d| via log-sum-exp
        abs_d = np.abs(d)
        max_abs = np.max(abs_d)
        if max_abs < 1e-15:
            return 0.0
        # log-sum-exp for stability
        shifted = abs_d - max_abs
        lse = max_abs + np.log(np.sum(np.exp(beta * shifted))) / beta
        return lse

    def gradient_step(U, V, W, rank, lr, beta=100.0):
        """One gradient step on the smooth objective."""
        sums = compute_sum(U, V, W, rank)
        d = sums - 1.0
        abs_d = np.abs(d)
        max_abs = np.max(abs_d)
        if max_abs < 1e-15:
            return U, V, W, 0.0

        # Soft weights: exp(beta * (|d| - max_abs)) / sum
        shifted = abs_d - max_abs
        weights = np.exp(beta * shifted)
        weights /= np.sum(weights)

        # Gradient of smooth objective w.r.t. U[u,c]:
        # d_obj/d_U[u,c] = sum_k weights[k] * sign(d[k]) * V[v,c] * W[w,c]
        # where k indexes the triple with u_idx = u
        # We need to accumulate per row of U, V, W

        U_sub = U[triples[:, 0], :]  # (ntrip, rank)
        V_sub = V[triples[:, 1], :]  # (ntrip, rank)
        W_sub = W[triples[:, 2], :]  # (ntrip, rank)

        sign_d = np.sign(d)
        wgt = weights * sign_d  # (ntrip,)

        # d_obj/d_U[u,c] = sum over triples with u_idx=u of wgt[k] * V[v,c] * W[w,c]
        # Use np.add.at for accumulation
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)

        # For each triple k, contribution to dU[triples[k,0], c] += wgt[k] * V_sub[k,c] * W_sub[k,c]
        contrib_U = wgt[:, None] * V_sub * W_sub  # (ntrip, rank)
        contrib_V = wgt[:, None] * U_sub * W_sub  # (ntrip, rank)
        contrib_W = wgt[:, None] * U_sub * V_sub  # (ntrip, rank)

        np.add.at(dU, triples[:, 0], contrib_U)
        np.add.at(dV, triples[:, 1], contrib_V)
        np.add.at(dW, triples[:, 2], contrib_W)

        U = U - lr * dU
        V = V - lr * dV
        W = W - lr * dW

        return U, V, W, compute_error(sums)

    # Try ranks from 55 down
    for rank in range(55, 35, -1):
        if time.time() > deadline - 15:
            break

        found = False
        for trial in range(30):
            if time.time() > deadline - 15:
                break
            if found:
                break

            np.random.seed(rank * 1000 + trial)
            # Scale initialization: for the product to be ~1, we need U*V*W ~ 1
            # With rank terms, each term is U*V*W. If U,V,W ~ 1/sqrt(rank), product ~ 1/rank
            # We want sum of rank terms ~ 1, so each ~ 1/rank
            # So U,V,W ~ 1/rank^(1/3)
            scale = rank ** (-1.0 / 3.0) * 2.0
            U = np.random.randn(n * m, rank) * scale
            V = np.random.randn(m * p, rank) * scale
            W = np.random.randn(p * n, rank) * scale

            # Adaptive learning rate schedule
            lr = 0.05
            min_lr = 0.001
            best_err = np.inf
            stalled = 0

            for iteration in range(500):
                if time.time() > deadline - 15:
                    break

                U, V, W, err = gradient_step(U, V, W, rank, lr, beta=50.0)

                if err < best_err:
                    best_err = err
                    stalled = 0
                    # Decay learning rate slowly
                    if lr > min_lr:
                        lr *= 0.995
                else:
                    stalled += 1
                    # If stalled, reduce learning rate
                    if stalled > 20:
                        lr = max(min_lr, lr * 0.9)
                        stalled = 0

                if err < 1e-8:
                    break

            # Final polish: use exact gradient with smaller step
            if err < 1e-4 and time.time() < deadline - 15:
                for polish_iter in range(100):
                    if time.time() > deadline - 15:
                        break
                    lr_p = 0.001 * (0.95 ** polish_iter)
                    U, V, W, err = gradient_step(U, V, W, rank, lr_p, beta=200.0)
                    if err < 1e-10:
                        break

            # Check if converged
            if err < 1e-6:
                # Convert to fractions
                U_frac = [[float(U[i, c]) for c in range(rank)] for i in range(n * m)]
                V_frac = [[float(V[i, c]) for c in range(rank)] for i in range(m * p)]
                W_frac = [[float(W[i, c]) for c in range(rank)] for i in range(p * n)]

                # Round to nearest half-integer
                U_frac = [[round(x * 2) / 2 for x in row] for row in U_frac]
                V_frac = [[round(x * 2) / 2 for x in row] for row in V_frac]
                W_frac = [[round(x * 2) / 2 for x in row] for row in W_frac]

                # Verify with rounded values
                U_r = np.array(U_frac)
                V_r = np.array(V_frac)
                W_r = np.array(W_frac)
                sums_r = compute_sum(U_r, V_r, W_r, rank)
                err_rounded = compute_error(sums_r)

                if err_rounded < 1e-6:
                    best_result = {"U": U_frac, "V": V_frac, "W": W_frac}
                    best_rank = rank
                    found = True
                    break

    return best_result
# EVOLVE-BLOCK-END
