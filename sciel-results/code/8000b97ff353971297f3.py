import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p  # 72

    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B

        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    u_val = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                    v_val = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                    C_computed[i][k] += W[k*n+i][r] * u_val * v_val

        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
        return True

    def compute_error(U, V, W, A, B):
        """Compute the error of the decomposition for given matrices."""
        rank = len(U[0]) if U else 0
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    u_val = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                    v_val = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                    C_computed[i][k] += W[k*n+i][r] * u_val * v_val
        C_expected = A @ B
        return np.max(np.abs(C_computed - C_expected))

    def random_decomposition(rank, seed=42):
        """Create a random decomposition of given rank."""
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        return U, V, W

    def refine_decomposition(U, V, W, A, B, max_iters=1000, lr=0.01, tol=1e-12):
        """Refine a decomposition using gradient descent."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Compute error
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for r in range(rank):
                        u_val = sum(U[i*m+j, r] * A[i][j] for j in range(m))
                        v_val = sum(V[j*p+k, r] * B[j][k] for j in range(m))
                        C_computed[i][k] += W[k*n+i, r] * u_val * v_val
            C_expected = A @ B
            error = C_computed - C_expected
            max_err = np.max(np.abs(error))

            if max_err < tol:
                break

            # Compute gradients (simplified - use numerical differentiation for small steps)
            # This is expensive but works for small problems
            eps = 1e-7
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)

            # Only compute a subset of gradients to save time
            num_samples = min(50, U.size)
            np.random.seed(it)
            indices_U = np.random.choice(U.size, num_samples, replace=False)
            indices_V = np.random.choice(V.size, num_samples, replace=False)
            indices_W = np.random.choice(W.size, num_samples, replace=False)

            for idx in indices_U:
                i, j, r = np.unravel_index(idx, U.shape)
                U[idx] += eps
                C_comp = np.zeros((n, p))
                for ii in range(n):
                    for kk in range(p):
                        for rr in range(rank):
                            u_val = sum(U[ii*m+jj, rr] * A[ii][jj] for jj in range(m))
                            v_val = sum(V[jj*p+kk, rr] * B[jj][kk] for jj in range(m))
                            C_comp[ii][kk] += W[kk*n+ii, rr] * u_val * v_val
                err_plus = np.max(np.abs(C_comp - C_expected))
                U[idx] -= eps
                grad_U[idx] = (err_plus - max_err) / eps

            for idx in indices_V:
                j, k, r = np.unravel_index(idx, V.shape)
                V[idx] += eps
                C_comp = np.zeros((n, p))
                for ii in range(n):
                    for kk in range(p):
                        for rr in range(rank):
                            u_val = sum(U[ii*m+jj, rr] * A[ii][jj] for jj in range(m))
                            v_val = sum(V[jj*p+kk, rr] * B[jj][kk] for jj in range(m))
                            C_comp[ii][kk] += W[kk*n+ii, rr] * u_val * v_val
                err_plus = np.max(np.abs(C_comp - C_expected))
                V[idx] -= eps
                grad_V[idx] = (err_plus - max_err) / eps

            for idx in indices_W:
                k, i, r = np.unravel_index(idx, W.shape)
                W[idx] += eps
                C_comp = np.zeros((n, p))
                for ii in range(n):
                    for kk in range(p):
                        for rr in range(rank):
                            u_val = sum(U[ii*m+jj, rr] * A[ii][jj] for jj in range(m))
                            v_val = sum(V[jj*p+kk, rr] * B[jj][kk] for jj in range(m))
                            C_comp[ii][kk] += W[kk*n+ii, rr] * u_val * v_val
                err_plus = np.max(np.abs(C_comp - C_expected))
                W[idx] -= eps
                grad_W[idx] = (err_plus - max_err) / eps

            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

        return U, V, W, max_err

    # Try to find lower-rank decompositions
    best_result = make_schoolbook()
    best_rank = schoolbook_rank

    # Use fixed test matrices for verification
    np.random.seed(123)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)

    # Try ranks from 60 down to 50
    for target_rank in range(60, 45, -1):
        if time.time() > deadline:
            break

        # Try multiple random initializations
        for attempt in range(3):
            if time.time() > deadline:
                break

            U, V, W = random_decomposition(target_rank, seed=42 + attempt * 7)
            U, V, W, err = refine_decomposition(U, V, W, A_test, B_test, max_iters=200, lr=0.05)

            if err < 1e-8:
                # Verify with different random matrices
                np.random.seed(999)
                A_verify = np.random.rand(n, m)
                B_verify = np.random.rand(m, p)
                err_verify = compute_error(U, V, W, A_verify, B_verify)

                if err_verify < 1e-8:
                    # Convert to lists
                    U_list = [[float(U[i, r]) for r in range(target_rank)] for i in range(n * m)]
                    V_list = [[float(V[i, r]) for r in range(target_rank)] for i in range(m * p)]
                    W_list = [[float(W[i, r]) for r in range(target_rank)] for i in range(p * n)]

                    if verify(U_list, V_list, W_list):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        break

        if best_rank < 60:
            break

    return best_result
# EVOLVE-BLOCK-END
