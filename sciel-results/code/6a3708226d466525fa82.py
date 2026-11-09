"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    def verify(U, V, W, n, m, p, tol=1e-5):
        """Verify decomposition correctness."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # Check all valid entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check some random invalid entries
        np.random.seed(999)
        for _ in range(200):
            a = np.random.randint(n*m)
            b = np.random.randint(m*p)
            d = np.random.randint(p*n)
            i1, j1 = divmod(a, m)
            j2, k2 = divmod(b, p)
            k3, i3 = divmod(d, n)
            if not (j1 == j2 and k2 == k3 and i3 == i1):
                val = np.dot(U[a, :], V[b, :] * W[d, :])
                if abs(val) > tol:
                    return False
        return True

    def find_decomp(n, m, p, target_rank, max_iter=5000, seed=42, deadline=None):
        """Optimize a rank-target tensor decomposition using gradient descent."""
        np.random.seed(seed)
        U = np.random.randn(n*m, target_rank) * 0.5
        V = np.random.randn(m*p, target_rank) * 0.5
        W = np.random.randn(p*n, target_rank) * 0.5

        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0

        def loss_and_grad(U, V, W):
            T_approx = np.zeros((n*m, m*p, p*n))
            for c in range(target_rank):
                T_approx += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])

            diff = T_approx - T
            loss = np.sum(diff**2)

            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            for c in range(target_rank):
                grad_U[:,c] = np.sum(diff * V[:,c][None,:,None] * W[:,c][None,None,:], axis=(1,2))
                grad_V[:,c] = np.sum(diff * U[:,c][:,None,None] * W[:,c][None,None,:], axis=(0,2))
                grad_W[:,c] = np.sum(diff * U[:,c][:,None,None] * V[:,c][None,:,None], axis=(0,1))

            return loss, (grad_U, grad_V, grad_W)

        lr = 0.003
        best_loss = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            loss, (gU, gV, gW) = loss_and_grad(U, V, W)

            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            if best_loss < 1e-14:
                break

            U -= lr * gU
            V -= lr * gV
            W -= lr * gW

            if it % 200 == 0:
                lr *= 0.9

        return best_U, best_V, best_W, best_loss

    def try_kronecker_decomp(n, m, p, deadline=None):
        """
        Try to decompose the tensor using Kronecker product structure.
        For 3x3x5, we can try to exploit the 3x3 block structure.
        """
        if time.time() > deadline:
            return None

        # The tensor T[i,j,k] = 1 for all i,j,k in range
        # We can think of this as a tensor product of smaller tensors
        # For 3x3x5, let's try to decompose into blocks

        # Strategy: Try to find a decomposition where the tensor can be written
        # as a sum of Kronecker products of lower-rank tensors

        # For 3x3 matrix multiplication, the rank is known to be 19 (or 20?)
        # For 3x3x5, we can try to use the structure

        # Let's try a simpler approach: use the known rank for 3x3 and extend
        # The 3x3 matrix multiplication tensor has rank 19 (Bini et al.)
        # But let's try to find a decomposition for 3x3x5

        # Actually, let's try a different approach:
        # Decompose the 5 dimension into 3+2 and use block structure

        # For now, let's try to use the gradient descent approach with better initialization
        # based on the Kronecker structure

        # Initialize using Kronecker products
        # T = sum_c U_c ⊗ V_c ⊗ W_c

        # Let's try to use the fact that for 3x3, we can use the 19-term decomposition
        # and extend it to 3x3x5

        # Actually, let's just use the gradient descent with better initialization
        # based on the block structure

        # Try rank 40 first (below schoolbook 45)
        for target_rank in [40, 41, 42, 43, 44]:
            if time.time() > deadline:
                break
            for seed in [42, 123, 456, 789, 1000, 2023]:
                if time.time() > deadline:
                    break
                U, V, W, loss = find_decomp(n, m, p, target_rank, max_iter=3000, seed=seed, deadline=deadline)
                if loss < 1e-8:
                    if verify(U, V, W, n, m, p):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return None

    # First, try the Kronecker-based approach
    result = try_kronecker_decomp(n, m, p, deadline)
    if result is not None:
        return result

    # Fall back to schoolbook
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
# EVOLVE-BLOCK-END
