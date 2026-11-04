"""Rewritten program: search for lower-rank exact tensor decomposition of 2x5*5x5 matmul."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS
TENSOR_SIZE = N * M * P  # 50

# Deadline guard
START_TIME = time.time()
DEADLINE = 110.0  # seconds, safety margin before 120s limit


def _time_left():
    return DEADLINE - (time.time() - START_TIME)


def _build_tensor():
    """Build the matrix multiplication tensor T[i,j,k] = 1 iff i in [0,n), j in [0,m), k in [0,p).
    Actually the tensor is T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    We represent it as a 3D array of shape (n*m, m*p, p*n) with 1s at the right places."""
    T = np.zeros((N * M, M * P, P * N), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i * M + j, j * P + k, k * N + i] = 1.0
    return T


def _verify(U, V, W, T, tol=1e-8):
    """Check that sum_c U[:,c]*V[:,c]*W[:,c] == T element-wise."""
    rank = U.shape[1]
    # Compute tensor product: for each c, outer product U[:,c] @ V[:,c] @ W[:,c]
    # This is expensive for large rank, so do it in chunks
    result = np.zeros_like(T)
    for c in range(rank):
        result += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return np.allclose(result, T, atol=tol, rtol=tol)


def _try_rank(rank, T, max_iter=2000, lr=0.01, n_restarts=3):
    """Try to find a rank-r decomposition via gradient descent on a smooth loss."""
    best_loss = float('inf')
    best_params = None

    for restart in range(n_restarts):
        if _time_left() < 5.0:
            break

        # Random initialization
        U = np.random.randn(N * M, rank) * 0.1
        V = np.random.randn(M * P, rank) * 0.1
        W = np.random.randn(P * N, rank) * 0.1

        # Scale to match the scale of T (entries are 0 or 1)
        scale = (T.sum() / rank) ** (1.0 / 3.0) if rank > 0 else 1.0
        U *= scale
        V *= scale
        W *= scale

        # Gradient descent with adaptive learning rate
        current_lr = lr
        prev_loss = float('inf')
        patience = 0
        max_patience = 200

        for it in range(max_iter):
            if _time_left() < 2.0:
                break

            # Compute loss: ||T - sum_c U_c V_c W_c||^2
            # For efficiency, compute in a vectorized way
            # tensor_product = sum_c U[:,c] outer V[:,c] outer W[:,c]
            # This is O(rank * n*m * m*p * p*n) which can be large
            # For rank up to ~50 and sizes 10x25x10=2500, that's 50*2500=125k per step, fine.

            # Compute the predicted tensor
            pred = np.zeros_like(T)
            for c in range(rank):
                pred += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])

            diff = pred - T
            loss = np.sum(diff * diff)

            if loss < best_loss:
                best_loss = loss
                best_params = (U.copy(), V.copy(), W.copy())

            # Check convergence
            if loss < 1e-10:
                break

            if abs(prev_loss - loss) < 1e-12:
                patience += 1
                current_lr *= 0.9
            else:
                patience = 0

            if patience > max_patience:
                break

            prev_loss = loss

            # Compute gradients
            # dL/dU[:,c] = sum_{j,k} diff[i,j,k] * V[j,c] * W[k,c] for each i
            # dL/dV[:,c] = sum_{i,k} diff[i,j,k] * U[i,c] * W[k,c] for each j
            # dL/dW[:,c] = sum_{i,j} diff[i,j,k] * U[i,c] * V[j,c] for each k

            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)

            for c in range(rank):
                # dU[:,c]_i = sum_{j,k} diff[i,j,k] * V[j,c] * W[k,c]
                # = (diff * (V[:,c] outer W[:,c])) summed over j,k axes
                vw = np.outer(V[:, c], W[:, c])  # shape (m*p, p*n)
                dU[:, c] = np.einsum('ijk,jk->i', diff, vw)

                # dV[:,c]_j = sum_{i,k} diff[i,j,k] * U[i,c] * W[k,c]
                uw = np.outer(U[:, c], W[:, c])  # shape (n*m, p*n)
                dV[:, c] = np.einsum('ijk,ik->j', diff, uw)

                # dW[:,c]_k = sum_{i,j} diff[i,j,k] * U[i,c] * V[j,c]
                uv = np.outer(U[:, c], V[:, c])  # shape (n*m, m*p)
                dW[:, c] = np.einsum('ijk,ij->k', diff, uv)

            # Gradient step
            U -= current_lr * dU
            V -= current_lr * dV
            W -= current_lr * dW

        if best_loss < 1e-8:
            break

    if best_loss < 1e-8 and best_params is not None:
        U, V, W = best_params
        # Verify with exact integer/half-integer check
        if _verify(U, V, W, T, tol=1e-6):
            return U, V, W, best_loss
    return None, None, None, best_loss


def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2.0


def _try_rank_rounded(rank, T, max_iter=3000, lr=0.005, n_restarts=5):
    """Try rank-r with rounding to half-integers at the end."""
    best_loss = float('inf')
    best_params = None

    for restart in range(n_restarts):
        if _time_left() < 5.0:
            break

        U = np.random.randn(N * M, rank) * 0.5
        V = np.random.randn(M * P, rank) * 0.5
        W = np.random.randn(P * N, rank) * 0.5

        current_lr = lr
        prev_loss = float('inf')
        patience = 0
        max_patience = 300

        for it in range(max_iter):
            if _time_left() < 2.0:
                break

            pred = np.zeros_like(T)
            for c in range(rank):
                pred += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])

            diff = pred - T
            loss = np.sum(diff * diff)

            if loss < best_loss:
                best_loss = loss
                best_params = (U.copy(), V.copy(), W.copy())

            if loss < 1e-10:
                break

            if abs(prev_loss - loss) < 1e-13:
                patience += 1
                current_lr *= 0.95
            else:
                patience = 0

            if patience > max_patience:
                break

            prev_loss = loss

            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)

            for c in range(rank):
                vw = np.outer(V[:, c], W[:, c])
                dU[:, c] = np.einsum('ijk,jk->i', diff, vw)

                uw = np.outer(U[:, c], W[:, c])
                dV[:, c] = np.einsum('ijk,ik->j', diff, uw)

                uv = np.outer(U[:, c], V[:, c])
                dW[:, c] = np.einsum('ijk,ij->k', diff, uv)

            U -= current_lr * dU
            V -= current_lr * dV
            W -= current_lr * dW

        # Try rounding the best solution found so far
        if best_params is not None:
            Ur = _round_to_half(best_params[0])
            Vr = _round_to_half(best_params[1])
            Wr = _round_to_half(best_params[2])
            if _verify(Ur, Vr, Wr, T, tol=1e-6):
                return Ur, Vr, Wr, 0.0

    if best_params is not None:
        Ur = _round_to_half(best_params[0])
        Vr = _round_to_half(best_params[1])
        Wr = _round_to_half(best_params[2])
        if _verify(Ur, Vr, Wr, T, tol=1e-6):
            return Ur, Vr, Wr, 0.0

    return None, None, None, best_loss


def construct():
    T = _build_tensor()

    # Schoolbook rank is 50. Try to find lower rank.
    # Known lower bounds: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Also, for 2x5*5x5, the rank is at least 2*5=10 (from first dim) but more tightly
    # the rank must be at least the max of the three dimensions products... actually
    # the rank is at least max(n, m, p) * min(n, m, p) in some cases, but let's just search.

    # Try ranks from a reasonable lower bound up to 50
    # Lower bound: at least max(N*M, M*P, P*N) / something... actually rank >= max(N*M, M*P, P*N) is not right.
    # The rank must satisfy: rank * rank >= N*M*P? No.
    # Actually, the tensor has "support" of N*M*P = 50 nonzero entries.
    # Each rank-1 term contributes to at most (number of nonzero in U_c) * (in V_c) * (in W_c) entries.
    # A trivial lower bound: rank >= 50 / (max possible support per term). If each term has at most 1 nonzero in each factor, support=1, so rank>=50.
    # But if we allow multiple nonzeros, we can do better.

    # Let's try ranks from 25 up to 49 (since 50 is schoolbook)
    for rank in range(25, 50):
        if _time_left() < 10.0:
            break

        # First try without rounding
        U, V, W, loss = _try_rank(rank, T, max_iter=1000, lr=0.01, n_restarts=2)
        if U is not None:
            # Convert to lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        # Try with rounding
        U, V, W, loss = _try_rank_rounded(rank, T, max_iter=1500, lr=0.005, n_restarts=3)
        if U is not None:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # If we couldn't find lower rank, fall back to schoolbook
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    column = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][column] = 1
                V[j * P + k][column] = 1
                W[k * N + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
