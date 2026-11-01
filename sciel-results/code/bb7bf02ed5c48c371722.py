"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145

    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def tensor_residual(U, V, W):
        """Compute residual efficiently using vectorized operations."""
        rank = U.shape[1]
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    total += (s - (1.0 if i == k else 0.0)) ** 2
        return total

    def is_half_integer(arr, tol=1e-8):
        """Check if all entries are integer or half-integer."""
        flat = arr.flatten()
        if len(flat) == 0:
            return True
        v = flat * 2
        return np.all(np.abs(v - np.round(v)) < tol)

    def round_to_half(arr):
        return np.round(arr * 2) / 2

    def check_and_return(U, V, W, tol=1e-6):
        """Check if solution is valid half-integer and return if so."""
        U_r = round_to_half(U)
        V_r = round_to_half(V)
        W_r = round_to_half(W)
        if tensor_residual(U_r, V_r, W_r) < tol:
            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        # Also try integer rounding
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if tensor_residual(U_i, V_i, W_i) < tol:
            return {"U": U_i.tolist(), "V": V_i.tolist(), "W": W_i.tolist()}
        return None

    def polish(U, V, W, target_rank, max_iter=200, tol=1e-10):
        """High-precision polish pass using ALS with tighter tolerance."""
        prev_err = tensor_residual(U, V, W)
        for iteration in range(max_iter):
            if time.time() > deadline - 3:
                break

            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = np.array([1.0 if i == k else 0.0 for k in range(p)])
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, b, rcond=1e-15)[0]
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, target_rank))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = np.array([1.0 if i == k else 0.0 for i in range(n)])
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M.T, b, rcond=1e-15)[0]
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    b = np.array([1.0 if i == k else 0.0 for _ in range(m)])
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M.T, b, rcond=1e-15)[0]
                    except:
                        pass

            err = tensor_residual(U, V, W)
            if err < tol:
                break
            if abs(err - prev_err) < 1e-14:
                break
            prev_err = err

        return U, V, W, prev_err

    # Try to find lower rank solutions
    for target_rank in [17, 16, 15, 14]:
        if time.time() > deadline - 40:
            break

        for attempt in range(300):
            if time.time() > deadline - 15:
                break

            # Different initialization strategies
            if attempt % 5 == 0:
                # Random Gaussian initialization
                np.random.seed(attempt * 1000 + target_rank)
                U = np.random.randn(n * m, target_rank) * 0.5
                V = np.random.randn(m * p, target_rank) * 0.5
                W = np.random.randn(p * n, target_rank) * 0.5
            elif attempt % 5 == 1:
                # Sparse random initialization
                np.random.seed(42 + target_rank * 1000 + attempt)
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    for _ in range(np.random.randint(1, n*m+1)):
                        U[np.random.randint(n*m), r] = np.random.choice([-1.0, 1.0])
                    for _ in range(np.random.randint(1, m*p+1)):
                        V[np.random.randint(m*p), r] = np.random.choice([-1.0, 1.0])
                    for _ in range(np.random.randint(1, p*n+1)):
                        W[np.random.randint(p*n), r] = np.random.choice([-1.0, 1.0])
            elif attempt % 5 == 2:
                # Structured initialization based on schoolbook with merged columns
                np.random.seed(777 + target_rank * 100 + attempt)
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            c = (i * m * p + j * p + k) % target_rank
                            U[i * m + j, c] += 1.0
                            V[j * p + k, c] += 1.0
                            W[k * n + i, c] += 1.0
                U += np.random.randn(n * m, target_rank) * 0.1
                V += np.random.randn(m * p, target_rank) * 0.1
                W += np.random.randn(p * n, target_rank) * 0.1
            elif attempt % 5 == 3:
                # Half-integer random initialization
                np.random.seed(999 + target_rank * 500 + attempt)
                U = np.random.randint(-2, 3, size=(n*m, target_rank)) / 2.0
                V = np.random.randint(-2, 3, size=(m*p, target_rank)) / 2.0
                W = np.random.randint(-2, 3, size=(p*n, target_rank)) / 2.0
            else:
                # Near-schoolbook with random column merging
                np.random.seed(1234 + target_rank * 200 + attempt)
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                # Assign each schoolbook term to a random column
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            c = np.random.randint(target_rank)
                            U[i * m + j, c] += 1.0
                            V[j * p + k, c] += 1.0
                            W[k * n + i, c] += 1.0
                # Small perturbation
                U += np.random.randn(n * m, target_rank) * 0.05
                V += np.random.randn(m * p, target_rank) * 0.05
                W += np.random.randn(p * n, target_rank) * 0.05

            # Use alternating least squares
            prev_err = float('inf')
            for iteration in range(500):
                if time.time() > deadline - 8:
                    break

                # Update U: for each (i,j), solve for U[i,j,:]
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.array([1.0 if i == k else 0.0 for k in range(p)])
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(M.T, b, rcond=None)[0]
                        except:
                            pass

                # Update V: for each (j,k), solve for V[j,k,:]
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.array([1.0 if i == k else 0.0 for i in range(n)])
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(M.T, b, rcond=None)[0]
                        except:
                            pass

                # Update W: for each (k,i), solve for W[k,i,:]
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.array([1.0 if i == k else 0.0 for _ in range(m)])
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(M.T, b, rcond=None)[0]
                        except:
                            pass

                # Check convergence
                err = tensor_residual(U, V, W)
                if err < 1e-10:
                    # Polish pass with tighter tolerance
                    U, V, W, err = polish(U, V, W, target_rank, max_iter=100, tol=1e-12)
                    if err < 1e-8:
                        result = check_and_return(U, V, W, tol=1e-5)
                        if result is not None:
                            return result
                    break
                if abs(err - prev_err) < 1e-13:
                    break
                prev_err = err

    return schoolbook()

# EVOLVE-BLOCK-END
