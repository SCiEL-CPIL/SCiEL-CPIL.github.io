"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    def make_schoolbook():
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

    best_result = make_schoolbook()
    best_rank = n * m * p

    def tensor_dot(U, V, W, r):
        """Compute the tensor contraction efficiently using einsum."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # We want T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Reshape U to (n, m, r), V to (m, p, r), W to (p, n, r)
        U3 = U.reshape(n, m, r)
        V3 = V.reshape(m, p, r)
        W3 = W.reshape(p, n, r)
        # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return T

    def check_decomposition(U, V, W, r, tol=1e-9):
        """Check if the decomposition is exact."""
        T = tensor_dot(U, V, W, r)
        # Each entry should be 1
        return np.allclose(T, 1.0, atol=tol, rtol=0)

    def gradient_descent(U, V, W, r, steps=500, lr=0.01, tol=1e-9):
        """Perform gradient descent to refine the decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()

        for step in range(steps):
            if time.time() > deadline:
                break

            # Compute current tensor
            T = tensor_dot(U, V, W, r)
            residual = T - 1.0  # We want T[i,j,k] = 1

            # Compute gradients
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)

            # grad_U[i,j,c] = sum_{k} residual[i,j,k] * V3[j,k,c] * W3[k,i,c]
            grad_U = np.einsum('ijk,jkc,kic->ijc', residual, V3, W3)
            grad_V = np.einsum('ijk,ijc,kic->jkc', residual, U3, W3)
            grad_W = np.einsum('ijk,ijc,jkc->kic', residual, U3, V3)

            # Update
            U -= lr * grad_U.reshape(n*m, r)
            V -= lr * grad_V.reshape(m*p, r)
            W -= lr * grad_W.reshape(p*n, r)

            # Check convergence
            if np.max(np.abs(residual)) < tol:
                break

        return U, V, W

    # Try structured approaches first
    # Approach 1: Try to use known fast matrix multiplication patterns
    # For 4x4 * 4x7, we can try to exploit structure

    # Approach 2: Random search with gradient descent
    np.random.seed(42)

    for r in range(40, 112):
        if time.time() > deadline:
            break

        for trial in range(5):
            if time.time() > deadline:
                break

            # Random initialization
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1

            # Run gradient descent
            U, V, W = gradient_descent(U, V, W, r, steps=300, lr=0.05)

            # Check if we found a valid decomposition
            if check_decomposition(U, V, W, r, tol=1e-9):
                best_rank = r
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        else:
            continue
        break

    if time.time() > deadline:
        return best_result

    return best_result
# EVOLVE-BLOCK-END
