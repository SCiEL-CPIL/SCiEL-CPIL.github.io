import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Build schoolbook decomposition as fallback
    U_school = [[0] * schoolbook_rank for _ in range(n * m)]
    V_school = [[0] * schoolbook_rank for _ in range(m * p)]
    W_school = [[0] * schoolbook_rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def verify_exact(U_mat, V_mat, W_mat, rank):
        """Verify exact decomposition using Fractions"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
                    if s != Fraction(1):
                        return False
        return True

    def verify_float(U_f, V_f, W_f, rank, tol=1e-8):
        """Fast float verification"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_f[i * m + j][r] * V_f[j * p + k][r] * W_f[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def compute_error_and_grad(U_f, V_f, W_f, rank):
        """Compute error and gradients for U, V, W"""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.sum(U_f[i * m + j] * V_f[j * p + k] * W_f[k * n + i])

        err = np.sum((T - 1.0) ** 2)

        dU = np.zeros_like(U_f)
        dV = np.zeros_like(V_f)
        dW = np.zeros_like(W_f)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = 2.0 * (T[i, j, k] - 1.0)
                    dU[i * m + j] += diff * V_f[j * p + k] * W_f[k * n + i]
                    dV[j * p + k] += diff * U_f[i * m + j] * W_f[k * n + i]
                    dW[k * n + i] += diff * U_f[i * m + j] * V_f[j * p + k]

        return err, dU, dV, dW

    def optimize_rank(R, seed, max_iters=3000, tol=1e-12):
        """Try to find a rank R decomposition using gradient descent with momentum"""
        np.random.seed(seed)

        U_f = np.random.randn(n * m, R) * 0.5
        V_f = np.random.randn(m * p, R) * 0.5
        W_f = np.random.randn(p * n, R) * 0.5

        lr = 0.05
        momentum = 0.9
        vU = np.zeros_like(U_f)
        vV = np.zeros_like(V_f)
        vW = np.zeros_like(W_f)

        prev_err = np.inf
        plateau_count = 0
        err = np.inf

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break

            err, dU, dV, dW = compute_error_and_grad(U_f, V_f, W_f, R)

            if err < prev_err * 0.995:
                lr = min(lr * 1.05, 0.2)
                plateau_count = 0
            else:
                plateau_count += 1
                if plateau_count > 30:
                    lr *= 0.9
                    if lr < 1e-7:
                        break

            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW

            U_f += vU
            V_f += vV
            W_f += vW

            prev_err = err

            if err < tol:
                break

        if err < tol * 100:
            # Try exact rational recovery
            U_rat = [[Fraction(0) for _ in range(R)] for _ in range(n * m)]
            V_rat = [[Fraction(0) for _ in range(R)] for _ in range(m * p)]
            W_rat = [[Fraction(0) for _ in range(R)] for _ in range(p * n)]

            for i in range(n * m):
                for r in range(R):
                    if abs(U_f[i, r]) > 1e-15:
                        U_rat[i][r] = Fraction(U_f[i, r]).limit_denominator(10**8)
                    else:
                        U_rat[i][r] = Fraction(0)
            for i in range(m * p):
                for r in range(R):
                    if abs(V_f[i, r]) > 1e-15:
                        V_rat[i][r] = Fraction(V_f[i, r]).limit_denominator(10**8)
                    else:
                        V_rat[i][r] = Fraction(0)
            for i in range(p * n):
                for r in range(R):
                    if abs(W_f[i, r]) > 1e-15:
                        W_rat[i][r] = Fraction(W_f[i, r]).limit_denominator(10**8)
                    else:
                        W_rat[i][r] = Fraction(0)

            if verify_exact(U_rat, V_rat, W_rat, R):
                return U_rat, V_rat, W_rat, R

        return None

    def build_block_decomposition():
        """
        Build a block-based decomposition using Strassen for 2x2x2 blocks.
        This gives rank 34 as a starting point.
        """
        U_terms = []
        V_terms = []
        W_terms = []

        def add_term(u_vals, v_vals, w_vals):
            U_list = [Fraction(u_vals.get((i, j), 0)) for i in range(n) for j in range(m)]
            V_list = [Fraction(v_vals.get((j, k), 0)) for j in range(m) for k in range(p)]
            W_list = [Fraction(w_vals.get((k, i), 0)) for k in range(p) for i in range(n)]
            U_terms.append(U_list)
            V_terms.append(V_list)
            W_terms.append(W_list)

        # Block 1: C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2] * B[2, 0:2]
        # Strassen for 2x2x2: rank 7
        add_term({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1})
        add_term({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1})
        add_term({(0, 0): 1}, {(1, 0): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1})
        add_term({(1, 1): 1}, {(0, 1): 1, (1, 0): -1}, {(0, 0): 1, (1, 0): 1})
        add_term({(0, 0): 1, (1, 0): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1})
        add_term({(0, 1): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}, {(1, 1): 1})
        add_term({(1, 0): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}, {(0, 0): 1})

        # A[0:2,2] * B[2,0:2]: rank 4
        add_term({(0, 2): 1}, {(2, 0): 1}, {(0, 0): 1})
        add_term({(0, 2): 1}, {(2, 1): 1}, {(0, 1): 1})
        add_term({(1, 2): 1}, {(2, 0): 1}, {(1, 0): 1})
        add_term({(1, 2): 1}, {(2, 1): 1}, {(1, 1): 1})

        # Block 2: C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2] * B[2, 2:4]
        # Strassen for 2x2x2: rank 7
        add_term({(0, 0): 1, (1, 1): 1}, {(0, 2): 1, (1, 3): 1}, {(2, 0): 1, (3, 1): 1})
        add_term({(1, 0): 1, (1, 1): 1}, {(0, 2): 1}, {(2, 1): 1})
        add_term({(0, 0): 1}, {(1, 2): 1, (1, 3): -1}, {(3, 0): 1, (3, 1): 1})
        add_term({(1, 1): 1}, {(0, 3): 1, (1, 2): -1}, {(2, 0): 1, (2, 1): 1})
        add_term({(0, 0): 1, (1, 0): 1}, {(1, 3): 1}, {(2, 0): -1, (3, 0): 1})
        add_term({(0, 1): 1, (0, 0): -1}, {(0, 2): 1, (1, 2): 1}, {(3, 1): 1})
        add_term({(1, 0): 1, (1, 1): -1}, {(0, 3): 1, (1, 3): 1}, {(2, 0): 1})

        # A[0:2,2] * B[2,2:4]: rank 4
        add_term({(0, 2): 1}, {(2, 2): 1}, {(2, 0): 1})
        add_term({(0, 2): 1}, {(2, 3): 1}, {(3, 0): 1})
        add_term({(1, 2): 1}, {(2, 2): 1}, {(2, 1): 1})
        add_term({(1, 2): 1}, {(2, 3): 1}, {(3, 1): 1})

        # Block 3: C[2, 0:2] = A[2, 0:2] * B[0:2, 0:2] + A[2, 2] * B[2, 0:2]
        # A[2,0:2] * B[0:2,0:2]: rank 4
        add_term({(2, 0): 1}, {(0, 0): 1}, {(0, 2): 1})
        add_term({(2, 0): 1}, {(0, 1): 1}, {(1, 2): 1})
        add_term({(2, 1): 1}, {(1, 0): 1}, {(0, 2): 1})
        add_term({(2, 1): 1}, {(1, 1): 1}, {(1, 2): 1})

        # A[2,2] * B[2,0:2]: rank 2
        add_term({(2, 2): 1}, {(2, 0): 1}, {(0, 2): 1})
        add_term({(2, 2): 1}, {(2, 1): 1}, {(1, 2): 1})

        # Block 4: C[2, 2:4] = A[2, 0:2] * B[0:2, 2:4] + A[2, 2] * B[2, 2:4]
        # A[2,0:2] * B[0:2,2:4]: rank 4
        add_term({(2, 0): 1}, {(0, 2): 1}, {(2, 2): 1})
        add_term({(2, 0): 1}, {(0, 3): 1}, {(3, 2): 1})
        add_term({(2, 1): 1}, {(1, 2): 1}, {(2, 2): 1})
        add_term({(2, 1): 1}, {(1, 3): 1}, {(3, 2): 1})

        # A[2,2] * B[2,2:4]: rank 2
        add_term({(2, 2): 1}, {(2, 2): 1}, {(2, 2): 1})
        add_term({(2, 2): 1}, {(2, 3): 1}, {(3, 2): 1})

        rank = len(U_terms)

        # Convert to list-of-lists format
        U = [list(row) for row in U_terms]
        V = [list(row) for row in V_terms]
        W = [list(row) for row in W_terms]

        # Transpose: U is (n*m, rank), so U[i*m+j][r]
        U_final = [[U[r][i * m + j] for r in range(rank)] for i in range(n) for j in range(m)]
        V_final = [[V[r][j * p + k] for r in range(rank)] for j in range(m) for k in range(p)]
        W_final = [[W[r][k * n + i] for r in range(rank)] for k in range(p) for i in range(n)]

        return U_final, V_final, W_final, rank

    # Try gradient descent for various ranks
    best_result = None

    # Try ranks from 33 down to 30
    for R in [33, 32, 31, 30]:
        if time.time() > deadline - 30:
            break

        found = False
        for seed in range(10):
            if time.time() > deadline - 15:
                break

            result = optimize_rank(R, seed, max_iters=2000, tol=1e-12)
            if result:
                best_result = result
                found = True
                break

        if found:
            break

    # If gradient descent didn't find anything, try block decomposition
    if best_result is None:
        U_block, V_block, W_block, rank_block = build_block_decomposition()
        if verify_exact(U_block, V_block, W_block, rank_block):
            best_result = (U_block, V_block, W_block, rank_block)

    # Final fallback: schoolbook
    if best_result is None:
        best_result = (U_school, V_school, W_school, schoolbook_rank)

    U_res, V_res, W_res, rank_res = best_result

    # Convert Fractions to floats for output
    U_out = [[float(x) for x in row] for row in U_res]
    V_out = [[float(x) for x in row] for row in V_res]
    W_out = [[float(x) for x in row] for row in W_res]

    return {"U": U_out, "V": V_out, "W": W_out}
