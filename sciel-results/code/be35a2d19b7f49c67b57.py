"""Matrix multiplication 4x4 by 4x7 - tensor decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    # Verify by computing the actual multiplication for random matrices
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # Reconstruct A and B factors
            A_c = uc.reshape(n, m)
            B_c = vc.reshape(m, p)
            # The contribution to C from this rank-1 term
            C_c = wc.reshape(p, n).T
            C_computed += A_c @ B_c * C_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True


def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
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


def _compute_W_from_UV(U, V, n, m, p):
    """
    Given U (n*m x r) and V (m*p x r), compute W analytically.
    W[k*n+i, c] = 1 / sum_j U[i*m+j, c] * V[j*p+k, c]
    Returns None if any denominator is zero.
    """
    r = U.shape[1]
    W = np.zeros((p * n, r))
    for c in range(r):
        uc = U[:, c]
        vc = V[:, c]
        for k in range(p):
            for i in range(n):
                denom = 0.0
                for j in range(m):
                    denom += uc[i * m + j] * vc[j * p + k]
                if abs(denom) < 1e-12:
                    return None
                W[k * n + i, c] = 1.0 / denom
    return W


def _check_nice_values(M, half_integer=False):
    """Check if matrix M has integer or half-integer values."""
    if half_integer:
        M2 = M * 2
        return np.allclose(M2, np.round(M2), atol=1e-6)
    else:
        return np.allclose(M, np.round(M), atol=1e-6)


def _try_structured_search(n, m, p, start_time, deadline=DEADLINE_SECONDS):
    """
    Try to find a lower-rank decomposition using structured patterns.
    Key insight: For fixed U and V, W is determined by:
      W[k*n+i, c] = 1 / sum_j U[i*m+j, c] * V[j*p+k, c]
    
    We search over structured U, V patterns and check if W has nice values.
    """
    best_result = _build_schoolbook(n, m, p)
    schoolbook_rank = n * m * p

    np.random.seed(42)

    # Try various ranks from just below schoolbook down
    for target_rank in range(schoolbook_rank - 1, max(40, schoolbook_rank // 2), -1):
        if time.time() - start_time > deadline - 5:
            break

        # Generate diverse candidate patterns
        for attempt in range(80):
            if time.time() - start_time > deadline - 5:
                break

            # Strategy: Use sparse patterns with small integer entries
            # Mix of random and structured approaches

            if attempt < 30:
                # Random sparse patterns
                U_rand = np.zeros((n * m, target_rank))
                V_rand = np.zeros((m * p, target_rank))

                for c in range(target_rank):
                    # Each column has a few non-zero entries
                    num_nonzero_u = np.random.randint(2, 5)
                    num_nonzero_v = np.random.randint(2, 5)

                    u_indices = np.random.choice(n * m, num_nonzero_u, replace=False)
                    U_rand[u_indices, c] = np.random.choice([-1, 1], num_nonzero_u)

                    v_indices = np.random.choice(m * p, num_nonzero_v, replace=False)
                    V_rand[v_indices, c] = np.random.choice([-1, 1], num_nonzero_v)
            elif attempt < 60:
                # Block-structured patterns
                U_rand = np.zeros((n * m, target_rank))
                V_rand = np.zeros((m * p, target_rank))

                for c in range(target_rank):
                    # Assign blocks
                    block_size_u = np.random.randint(1, 4)
                    block_start_u = np.random.randint(0, n * m - block_size_u + 1)
                    U_rand[block_start_u:block_start_u + block_size_u, c] = np.random.choice([-1, 1], block_size_u)

                    block_size_v = np.random.randint(1, 4)
                    block_start_v = np.random.randint(0, m * p - block_size_v + 1)
                    V_rand[block_start_v:block_start_v + block_size_v, c] = np.random.choice([-1, 1], block_size_v)
            else:
                # Mixed: some columns structured, some random
                U_rand = np.zeros((n * m, target_rank))
                V_rand = np.zeros((m * p, target_rank))

                for c in range(target_rank):
                    if np.random.random() < 0.5:
                        # Sparse random
                        num_nonzero_u = np.random.randint(1, 4)
                        num_nonzero_v = np.random.randint(1, 4)

                        u_indices = np.random.choice(n * m, num_nonzero_u, replace=False)
                        U_rand[u_indices, c] = np.random.choice([-1, 1], num_nonzero_u)

                        v_indices = np.random.choice(m * p, num_nonzero_v, replace=False)
                        V_rand[v_indices, c] = np.random.choice([-1, 1], num_nonzero_v)
                    else:
                        # Dense small entries
                        U_rand[:, c] = np.random.choice([-1, 0, 1], n * m)
                        V_rand[:, c] = np.random.choice([-1, 0, 1], m * p)

            # Compute W analytically
            W_rand = _compute_W_from_UV(U_rand, V_rand, n, m, p)
            if W_rand is None:
                continue

            # Check if W has nice values (integers or half-integers)
            if not _check_nice_values(W_rand, half_integer=True):
                continue

            # Also check U and V have nice values (they should be integers by construction)
            if not _check_nice_values(U_rand, half_integer=False):
                continue
            if not _check_nice_values(V_rand, half_integer=False):
                continue

            # Convert to lists
            U_list = [[int(round(U_rand[i, c])) for c in range(target_rank)] for i in range(n * m)]
            V_list = [[int(round(V_rand[i, c])) for c in range(target_rank)] for i in range(m * p)]
            W_list = [[int(round(W_rand[i, c] * 2)) / 2.0 for c in range(target_rank)] for i in range(p * n)]

            # Verify the decomposition
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}

    return best_result


def _try_gradient_descent(n, m, p, start_time, deadline=DEADLINE_SECONDS):
    """
    Try gradient descent to minimize the rank while maintaining exact decomposition.
    Use a smooth surrogate: minimize sum of squared errors + regularization on rank.
    """
    best_result = _build_schoolbook(n, m, p)
    schoolbook_rank = n * m * p

    # Start from a slightly perturbed schoolbook
    base_rank = min(schoolbook_rank, 80)

    np.random.seed(123)

    for target_rank in [base_rank, base_rank - 10, base_rank - 20, base_rank - 30]:
        if target_rank < 30 or time.time() - start_time > deadline - 10:
            break

        # Initialize with random values
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5

        # Gradient descent
        lr = 0.01
        error = float('inf')
        for iteration in range(1500):
            if time.time() - start_time > deadline - 5:
                break

            # Compute current error and gradients efficiently
            error = 0.0
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)

            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Target: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                        current_sum = 0.0
                        for c in range(target_rank):
                            term = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            current_sum += term
                            # Gradients
                            grad_U[i * m + j, c] += (current_sum - 1) * V[j * p + k, c] * W[k * n + i, c]
                            grad_V[j * p + k, c] += (current_sum - 1) * U[i * m + j, c] * W[k * n + i, c]
                            grad_W[k * n + i, c] += (current_sum - 1) * U[i * m + j, c] * V[j * p + k, c]
                        error += (current_sum - 1) ** 2

            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

            # Check convergence
            if error < 1e-6:
                break

        # Check if we got close to a valid solution
        if error < 1e-4:
            # Try to snap to nice values
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2

            U_list = [[U_snapped[i, c] for c in range(target_rank)] for i in range(n * m)]
            V_list = [[V_snapped[i, c] for c in range(target_rank)] for i in range(m * p)]
            W_list = [[W_snapped[i, c] for c in range(target_rank)] for i in range(p * n)]

            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}

    return best_result


def _try_strassen_style(n, m, p, start_time, deadline=DEADLINE_SECONDS):
    """
    Try Strassen-style structured decomposition.
    For 4x4 matrices, Strassen uses 7 multiplications.
    For 4x4 x 4x7, we can try to adapt this structure.
    
    The idea: partition the matrices into 2x2 blocks and use
    Strassen's formulas to reduce the number of scalar multiplications.
    """
    best_result = _build_schoolbook(n, m, p)
    schoolbook_rank = n * m * p

    # For 4x4 x 4x4, Strassen gives rank 7 (vs 64 schoolbook)
    # For 4x4 x 4x7, we can extend this
    # Let's try a systematic search over small integer patterns
    
    np.random.seed(999)
    
    # Try ranks that are multiples of small numbers
    for target_rank in [49, 48, 47, 46, 45, 44, 42, 40, 38, 36, 35, 32, 30, 28, 25, 24, 21, 20, 18, 16, 15, 14, 12, 10, 9, 8, 7]:
        if target_rank >= schoolbook_rank:
            continue
        if time.time() - start_time > deadline - 5:
            break
        
        # Try multiple random initializations
        for init in range(20):
            if time.time() - start_time > deadline - 5:
                break
            
            # Initialize with small random integers
            U = np.random.randint(-2, 3, size=(n * m, target_rank))
            V = np.random.randint(-2, 3, size=(m * p, target_rank))
            
            # Compute W analytically
            W = _compute_W_from_UV(U.astype(float), V.astype(float), n, m, p)
            if W is None:
                continue
            
            # Check if W has nice values
            if not _check_nice_values(W, half_integer=True):
                continue
            
            # Verify
            U_list = [[int(U[i, c]) for c in range(target_rank)] for i in range(n * m)]
            V_list = [[int(V[i, c]) for c in range(target_rank)] for i in range(m * p)]
            W_list = [[int(round(W[i, c] * 2)) / 2.0 for c in range(target_rank)] for i in range(p * n)]
            
            if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS

    # Start with schoolbook as fallback
    best = _build_schoolbook(n, m, p)

    # Try structured search first (faster)
    try:
        result = _try_structured_search(n, m, p, start_time)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                if len(result["U"][0]) < len(best["U"][0]):
                    best = result
    except Exception:
        pass

    # Try Strassen-style search
    if time.time() - start_time < DEADLINE_SECONDS - 30:
        try:
            result = _try_strassen_style(n, m, p, start_time)
            if result is not None:
                if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                    if len(result["U"][0]) < len(best["U"][0]):
                        best = result
        except Exception:
            pass

    # If we have time, try gradient descent
    if time.time() - start_time < DEADLINE_SECONDS - 20:
        try:
            result = _try_gradient_descent(n, m, p, start_time)
            if result is not None:
                if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                    # Only update if this is better (lower rank)
                    if len(result["U"][0]) < len(best["U"][0]):
                        best = result
        except Exception:
            pass

    # Check deadline
    if time.time() - start_time > DEADLINE_SECONDS:
        pass  # Just return what we have

    return best
