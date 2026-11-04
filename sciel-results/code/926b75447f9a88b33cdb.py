"""Tensor decomposition for 2x3 by 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]


def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(actual - expected) > tol:
                    return False
    return True


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using tensor operations."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(
                    U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :]
                )
                if abs(actual - expected) > tol:
                    return False
    return True


def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor T[i,j,k] = 1."""
    T = np.ones((n, m, p), dtype=float)
    return T


def _als_decompose(n, m, p, rank, max_iters=100, tol=1e-10):
    """
    Use ALS (Alternating Least Squares) to decompose the multiplication tensor.
    The tensor T[i,j,k] = 1 for all valid i,j,k.
    We seek U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    # Initialize randomly
    rng = np.random.RandomState(42)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)

    # Scale
    scale = (rank ** 0.5)
    U /= scale
    V /= scale
    W /= scale

    for iteration in range(max_iters):
        # Update U: for each row (i*m+j), solve least squares
        for i in range(n):
            for j in range(m):
                # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                # This is: U[i*m+j, :] @ (V[j*p+k,:] * W[k*n+i,:])^T = 1 for all k
                # Build matrix A_k_c = V[j*p+k, c] * W[k*n+i, c] for k in range(p)
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve A^T U[i*m+j, :] = 1 (vector of ones)
                # Actually: sum_c U[i*m+j,c] * A[k,c] = 1 for all k
                # So: A @ U[i*m+j, :] = ones(p)
                try:
                    U[i * m + j, :], residuals, rank_l, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                except np.linalg.LinAlgError:
                    break

        # Update V: for each row (j*p+k), solve least squares
        for j in range(m):
            for k in range(p):
                # sum_c V[j*p+k, c] * U[i*m+j, c] * W[k*n+i, c] = 1 for all i
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :], residuals, rank_l, sv = np.linalg.lstsq(A, np.ones(n), rcond=None)
                except np.linalg.LinAlgError:
                    break

        # Update W: for each row (k*n+i), solve least squares
        for k in range(p):
            for i in range(n):
                # sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :], residuals, rank_l, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    break

        # Check convergence
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    loss += (val - 1.0) ** 2

        if loss < tol:
            break

    # Check if we found a valid decomposition
    loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                loss += (val - 1.0) ** 2

    if loss < 1e-8:
        return U.tolist(), V.tolist(), W.tolist()
    return None


def _try_als_decomposition(n, m, p, rank, max_iters=200):
    """Try ALS decomposition for a given rank."""
    result = _als_decompose(n, m, p, rank, max_iters=max_iters)
    if result is not None:
        U, V, W = result
        # Verify
        if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-6):
            # Round to reasonable precision
            U_r = [[round(float(x), 8) for x in row] for row in U]
            V_r = [[round(float(x), 8) for x in row] for row in V]
            W_r = [[round(float(x), 8) for x in row] for row in W]
            if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-5):
                return {"U": U_r, "V": V_r, "W": W_r}
    return None


def _gradient_descent_decompose(n, m, p, rank, max_iters=500, lr=0.01):
    """
    Use gradient descent to find a rank decomposition.
    """
    rng = np.random.RandomState(123)
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)

    scale = (rank ** 0.5)
    U /= scale
    V /= scale
    W /= scale

    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    for iteration in range(max_iters):
        # Compute loss and gradients
        loss = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err = val - 1.0
                    loss += err * err
                    for c in range(rank):
                        grad_U[i * m + j, c] += 2 * err * V[j * p + k, c] * W[k * n + i, c]
                        grad_V[j * p + k, c] += 2 * err * U[i * m + j, c] * W[k * n + i, c]
                        grad_W[k * n + i, c] += 2 * err * U[i * m + j, c] * V[j * p + k, c]

        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

        if best_loss < 1e-12:
            break

        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        lr *= 0.995

    if best_loss < 1e-9:
        U_r = [[round(float(x), 8) for x in row] for row in best_U]
        V_r = [[round(float(x), 8) for x in row] for row in best_V]
        W_r = [[round(float(x), 8) for x in row] for row in best_W]
        if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-5):
            return {"U": U_r, "V": V_r, "W": W_r}
    return None


def _structured_block_decomposition(n, m, p):
    """
    Try to exploit the block structure: 3x6 = [3x3 | 3x3].
    This is equivalent to multiplying 2x3 by [C1 C2] where C1, C2 are 3x3.
    We can write this as: [2x3 * C1, 2x3 * C2]
    So we need two 2x3x3 multiplications.
    
    Rank of 2x3x3: schoolbook is 6.
    If we use two independent 2x3x3 decompositions, total rank is 2 * R(2,3,3).
    """
    # First, find or use schoolbook for 2x3x3
    def _schoolbook_233():
        nn, mm, pp = 2, 3, 3
        rank = nn * mm * pp  # 18
        U = [[0] * rank for _ in range(nn * mm)]
        V = [[0] * rank for _ in range(mm * pp)]
        W = [[0] * rank for _ in range(pp * nn)]
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j][col] = 1
                    V[j * pp + k][col] = 1
                    W[k * nn + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find better than schoolbook for 2x3x3 using ALS
    best_233 = None
    for r in range(6, 19):
        res = _try_als_decomposition(2, 3, 3, r, max_iters=100)
        if res is not None:
            best_233 = res
            break

    if best_233 is None:
        best_233 = _schoolbook_233()

    r233 = len(best_233['U'][0])

    # Now construct 2x3x6 using two copies
    # Output is 2x6 = [2x3 | 2x3]
    # U: 2x3 matrix (rows), so U has 2*3 = 6 rows
    # V: 3x6 matrix (rows), so V has 3*6 = 18 rows
    # W: 6x2 matrix (rows), so W has 6*2 = 12 rows

    # For the first block (columns 0-2 of output):
    # U_same as 233, V_uses first 3 cols of B, W_uses first 3 rows of output
    # For the second block (columns 3-5 of output):
    # U_same as 233, V_uses last 3 cols of B, W_uses last 3 rows of output

    U_total = [[0] * (2 * r233) for _ in range(6)]
    V_total = [[0] * (2 * r233) for _ in range(18)]
    W_total = [[0] * (2 * r233) for _ in range(12)]

    # First block
    for i in range(6):
        for c in range(r233):
            U_total[i][c] = best_233['U'][i][c]
    for j in range(3):
        for k in range(3):
            for c in range(r233):
                V_total[j * 6 + k][c] = best_233['V'][j * 3 + k][c]
    for k in range(3):
        for i in range(2):
            for c in range(r233):
                W_total[k * 2 + i][c] = best_233['W'][k * 2 + i][c]

    # Second block
    offset = r233
    for i in range(6):
        for c in range(r233):
            U_total[i][offset + c] = best_233['U'][i][c]
    for j in range(3):
        for k in range(3):
            for c in range(r233):
                V_total[j * 6 + (3 + k)][offset + c] = best_233['V'][j * 3 + k][c]
    for k in range(3):
        for i in range(2):
            for c in range(r233):
                W_total[(3 + k) * 2 + i][offset + c] = best_233['W'][k * 2 + i][c]

    # Verify
    if _verify_decomposition(U_total, V_total, W_total, 2, 3, 6, tol=1e-6):
        return {"U": U_total, "V": V_total, "W": W_total}

    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    best_result = None
    best_rank = n * m * p  # schoolbook

    # Try structured block decomposition first (exploits 3x6 = [3x3|3x3])
    if time.time() < deadline - 30.0:
        result = _structured_block_decomposition(n, m, p)
        if result is not None:
            r = len(result['U'][0])
            if r < best_rank:
                best_rank = r
                best_result = result

    # Try ALS for various ranks
    # Lower bound: max(n*m, m*p, p*n) = max(6, 18, 12) = 18
    min_rank = max(n * m, m * p, p * n)

    for rank in range(min_rank, best_rank + 1):
        if time.time() > deadline - 20.0:
            break

        result = _try_als_decomposition(n, m, p, rank, max_iters=150)
        if result is not None:
            r = len(result['U'][0])
            if r < best_rank:
                best_rank = r
                best_result = result
            break

    # Try gradient descent for remaining ranks
    for rank in range(min_rank, best_rank + 1):
        if time.time() > deadline - 10.0:
            break

        result = _gradient_descent_decompose(n, m, p, rank, max_iters=300)
        if result is not None:
            r = len(result['U'][0])
            if r < best_rank:
                best_rank = r
                best_result = result
            break

    if best_result is None:
        return _schoolbook_decomposition(n, m, p)

    return best_result
