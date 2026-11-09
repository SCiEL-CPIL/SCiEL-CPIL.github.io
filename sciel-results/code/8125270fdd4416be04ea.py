import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    def get_schoolbook():
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

    # Build the multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        return T
    T = build_tensor()

    def check_decomposition(U, V, W_mat, tol=1e-9):
        """Check if the decomposition is valid using einsum (fast)."""
        R = U.shape[1]
        result = np.einsum('ac,bc,dc->abd', U, V, W_mat)
        return float(np.max(np.abs(result - T))) < tol

    def solve_for_W(U, V, R):
        """Given U and V, solve for W using least squares."""
        A_sys = np.zeros((n * m * p, R * n * p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i * m * p + j * p + k
                    for r in range(R):
                        col = r * n * p + k * n + i
                        A_sys[row, col] = U[i * m + j, r] * V[j * p + k, r]
        b_vec = np.ones(n * m * p)
        W_flat, _, _, _ = np.linalg.lstsq(A_sys, b_vec, rcond=None)
        W_mat = np.zeros((p * n, R))
        for r in range(R):
            for i in range(n):
                for k in range(p):
                    W_mat[k * n + i, r] = W_flat[r * n * p + k * n + i]
        return W_mat

    def polish(U, V, W_mat, R, max_iter=200):
        """High-precision polish: alternate exact W-solve with gradient step on U,V."""
        best_err = float(np.max(np.abs(np.einsum('ac,bc,dc->abd', U, V, W_mat) - T)))
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Solve W exactly
            W_mat = solve_for_W(U, V, R)
            # Compute residual
            residual = np.einsum('ac,bc,dc->abd', U, V, W_mat) - T
            err = float(np.max(np.abs(residual)))
            if err < best_err:
                best_err = err
            if err < 1e-10:
                break
            # Gradient step on U and V (fix W)
            # d/dU[a,c] sum_b,d residual[a,b,d] = sum_b,d residual[a,b,d] V[b,c] W[d,c]
            grad_U = np.einsum('abd,bc,dc->ac', residual, V, W_mat)
            grad_V = np.einsum('abd,ac,dc->bc', residual, U, W_mat)
            # Adaptive step size
            scale = err / (np.linalg.norm(grad_U) ** 2 + np.linalg.norm(grad_V) ** 2 + 1e-12)
            lr = min(0.5, scale)
            U = U - lr * grad_U
            V = V - lr * grad_V
        return U, V, W_mat, best_err

    def snap_to_rationals(U, V, W_mat, R):
        """Try to snap entries to small rational values (half-integers)."""
        def snap_mat(M):
            Ms = M.copy()
            for idx in np.ndindex(M.shape):
                x = M[idx]
                # Try small rationals with denominator up to 4
                best = x
                bestd = abs(x)
                for den in [1, 2, 3, 4]:
                    num = round(x * den)
                    val = num / den
                    if abs(val - x) < bestd:
                        bestd = abs(val - x)
                        best = val
                Ms[idx] = best
            return Ms
        return snap_mat(U), snap_mat(V), snap_mat(W_mat)

    # Coefficient set for structured search
    coeffs = [-1, -0.5, 0, 0.5, 1]

    best_rank = n * m * p
    best_result = None

    # Try ranks from 9 upwards
    for R in range(9, n * m * p):
        if time.time() > deadline - 20:
            break

        found = False
        np.random.seed(42 + R)

        for restart in range(300):
            if time.time() > deadline - 10:
                break

            # Different initialization strategies
            if restart % 4 == 0:
                # Sparse random with small coefficients
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                for c in range(R):
                    for i in range(n):
                        for j in range(m):
                            if np.random.random() < 0.4:
                                U[i * m + j, c] = np.random.choice(coeffs)
                    for j in range(m):
                        for k in range(p):
                            if np.random.random() < 0.4:
                                V[j * p + k, c] = np.random.choice(coeffs)
            elif restart % 4 == 1:
                # Structured: each term covers a block of triples
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
                for c in range(R):
                    # Assign a primary triple
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    # Add some cross terms
                    for _ in range(2):
                        i2 = np.random.randint(n)
                        j2 = np.random.randint(m)
                        k2 = np.random.randint(p)
                        if (i2, j2, k2) != (i, j, k):
                            U[i2 * m + j2, c] = np.random.choice([-1, 1])
                            V[j2 * p + k2, c] = np.random.choice([-1, 1])
            elif restart % 4 == 2:
                # Dense random
                U = np.random.randn(n * m, R)
                V = np.random.randn(m * p, R)
            else:
                # Hybrid: start from schoolbook-like then perturb
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                for c in range(R):
                    i = c % n
                    j = (c // n) % m
                    k = (c // (n * m)) % p
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    U += 0.2 * np.random.randn(n * m, 1)[:, c:c+1] * 0.1
                    V += 0.2 * np.random.randn(m * p, 1)[:, c:c+1] * 0.1

            # ALS iterations: solve W, then gradient step on U,V
            for it in range(40):
                if time.time() > deadline - 5:
                    break

                W_mat = solve_for_W(U, V, R)

                if it % 4 == 0:
                    if check_decomposition(U, V, W_mat, tol=1e-8):
                        # Polish for high precision
                        U, V, W_mat, err = polish(U, V, W_mat, R, max_iter=100)
                        if check_decomposition(U, V, W_mat, tol=1e-9):
                            if R < best_rank:
                                best_rank = R
                                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W_mat.tolist()}
                            found = True
                            break

                # Gradient step on U, V
                residual = np.einsum('ac,bc,dc->abd', U, V, W_mat) - T
                grad_U = np.einsum('abd,bc,dc->ac', residual, V, W_mat)
                grad_V = np.einsum('abd,ac,dc->bc', residual, U, W_mat)
                err = float(np.max(np.abs(residual)))
                scale = err / (np.linalg.norm(grad_U) ** 2 + np.linalg.norm(grad_V) ** 2 + 1e-12)
                lr = min(0.5, scale)
                U = U - lr * grad_U
                V = V - lr * grad_V

            if found:
                break

        if found:
            break

    if best_result is not None and best_rank < n * m * p:
        return best_result

    return get_schoolbook()
# EVOLVE-BLOCK-END
