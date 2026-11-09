"""Tensor decomposition search for 3x4 * 4x6 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import linprog

N, M, P = 3, 4, 6
DEADLINE = 130.0  # safety margin before 150s kill


def verify_fast(U, V, W, rank):
    """Faster verification using numpy."""
    n, m, p = N, M, P
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    return np.allclose(T, 1.0, atol=1e-9)


def schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = N, M, P
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


def try_gradient_descent(target_rank, max_iters, time_budget, seed=42):
    """Try gradient descent to minimize ||T - 1||_F^2 for given rank."""
    n, m, p = N, M, P
    start = time.time()
    rng = np.random.RandomState(seed)

    best_res = float('inf')
    best_U = best_V = best_W = None

    for attempt in range(10):
        if time.time() - start > time_budget or time.time() > DEADLINE:
            break

        # Initialize with random values, scaled appropriately
        scale = 1.0 / np.sqrt(target_rank)
        U = rng.randn(n * m, target_rank) * scale
        V = rng.randn(m * p, target_rank) * scale
        W = rng.randn(p * n, target_rank) * scale

        lr = 0.1
        for it in range(max_iters):
            if time.time() - start > time_budget or time.time() > DEADLINE:
                break

            Ur = U.reshape(n, m, target_rank)
            Vr = V.reshape(m, p, target_rank)
            Wr = W.reshape(p, n, target_rank)

            # Compute T[i,j,k] = sum_c Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
            T = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
            res = T - 1.0
            res_sq = np.sum(res * res)

            if res_sq < best_res:
                best_res = res_sq
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()

            if res_sq < 1e-12:
                break

            # Vectorized Gradient Computation
            grad_U_3d = np.einsum('ijk,jkc,kic->ijc', res, Vr, Wr)
            grad_U = grad_U_3d.reshape(n * m, target_rank)

            grad_V_3d = np.einsum('ijk,ijc,kic->jkc', res, Ur, Wr)
            grad_V = grad_V_3d.reshape(m * p, target_rank)

            grad_W_3d = np.einsum('ijk,ijc,jkc->kic', res, Ur, Vr)
            grad_W = grad_W_3d.reshape(p * n, target_rank)

            # Update with momentum-like decay
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W

            # Reduce learning rate periodically
            if it % 50 == 49:
                lr *= 0.9

    if best_res < 1e-10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    return None


def try_lp_search(target_rank, time_budget, seed=42):
    """
    Try to find a decomposition of rank target_rank by solving a linear program.
    
    We fix one tensor (say W) to random values, then the problem becomes linear
    in the entries of U and V. We solve for U and V such that
    sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all (i,j,k).
    
    However, this is still nonlinear because of the product U*V.
    
    Alternative: We can use the fact that if we fix V and W, the equations are
    linear in U. But we need to satisfy all (i,j,k) equations simultaneously.
    
    A better approach: For a fixed rank r, we can try to find U, V, W by
    solving a system of polynomial equations. We use homotopy or numerical methods.
    
    Since that's complex, let's try a different approach: we'll use the
    "Strassen-like" structured approach.
    
    For now, let's try random initialization + gradient descent with more attempts.
    """
    n, m, p = N, M, P
    start = time.time()
    rng = np.random.RandomState(seed)

    # Try multiple random initializations with gradient descent
    for attempt in range(20):
        if time.time() - start > time_budget or time.time() > DEADLINE:
            break

        remaining = time_budget - (time.time() - start)
        if remaining < 1:
            break

        # Use a smaller time budget per attempt
        per_attempt_budget = min(remaining, 2.0)
        result = try_gradient_descent(target_rank, 500, per_attempt_budget, seed=seed + attempt)
        if result is not None:
            return result

    return None


def try_structured_search(time_budget):
    """Try structured/analytic approaches for lower rank."""
    n, m, p = N, M, P
    start = time.time()

    # Try ranks from 70 down to 50
    for target_rank in [70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50]:
        if time.time() - start > time_budget or time.time() > DEADLINE:
            break
        remaining = time_budget - (time.time() - start)
        if remaining < 3:
            break
        result = try_lp_search(target_rank, min(10, remaining))
        if result is not None:
            return result

    return None


def construct():
    start_time = time.time()

    # Start with schoolbook as fallback
    best = schoolbook()

    # Try to find better decompositions
    remaining = min(80, DEADLINE - (time.time() - start_time) - 10)
    if remaining > 5:
        result = try_structured_search(remaining)
        if result is not None:
            best = result

    # Verify the best result
    if not verify_fast(best["U"], best["V"], best["W"], len(best["U"][0])):
        # Fallback to schoolbook
        best = schoolbook()

    return best
