import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy einsum."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]

    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)

    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)

    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _compute_gradients(U, V, W, n, m, p):
    """Compute gradients of ||T - 1||^2 w.r.t. U, V, W."""
    rank = U.shape[1]
    U_tensor = U.reshape(n, m, rank)
    V_tensor = V.reshape(m, p, rank)
    W_tensor = W.reshape(p, n, rank)

    T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
    residual = T - 1.0

    # dL/dU[i,j,c] = 2 * sum_k residual[i,j,k] * V[j,k,c] * W[k,i,c]
    dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', residual, V_tensor, W_tensor)
    # dL/dV[j,k,c] = 2 * sum_i residual[i,j,k] * U[i,j,c] * W[k,i,c]
    dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', residual, U_tensor, W_tensor)
    # dL/dW[k,i,c] = 2 * sum_j residual[i,j,k] * U[i,j,c] * V[j,k,c]
    dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', residual, U_tensor, V_tensor)

    return dU.reshape(n * m, rank), dV.reshape(m * p, rank), dW.reshape(p * n, rank)

def _gradient_descent(U, V, W, n, m, p, deadline, max_steps=300):
    """Adam-like optimizer to drive ||T - 1|| to zero."""
    rank = U.shape[1]
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()

    # Adam state
    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr = 0.1

    for step in range(max_steps):
        if time.time() > deadline - 10:
            break

        dU, dV, dW = _compute_gradients(U, V, W, n, m, p)
        loss = float(np.sum((np.einsum('ijc,jkc,kic->ijk',
                U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank)) - 1.0)**2))

        if loss < 1e-18:
            break

        t = step + 1
        # Adam updates
        mU = b1*mU + (1-b1)*dU; vU = b2*vU + (1-b2)*dU**2
        mUh = mU/(1-b1**t); vUh = vU/(1-b2**t)
        mV = b1*mV + (1-b1)*dV; vV = b2*vV + (1-b2)*dV**2
        mVh = mV/(1-b1**t); vVh = vV/(1-b2**t)
        mW = b1*mW + (1-b1)*dW; vW = b2*vW + (1-b2)*dW**2
        mWh = mW/(1-b1**t); vWh = vW/(1-b2**t)

        U -= lr * mUh / (np.sqrt(vUh) + eps)
        V -= lr * mVh / (np.sqrt(vVh) + eps)
        W -= lr * mWh / (np.sqrt(vWh) + eps)

    # Final check
    T = np.einsum('ijc,jkc,kic->ijk',
                  U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
    if np.allclose(T, 1.0, atol=1e-9):
        return U, V, W
    return None

def _try_random_search(n, m, p, target_rank, num_trials, deadline):
    """Random init + gradient refinement to find valid decompositions."""
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break

        # Random initialization
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)

        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)

        # Quick error check before investing in optimization
        U_tensor = U.reshape(n, m, target_rank)
        V_tensor = V.reshape(m, p, target_rank)
        W_tensor = W.reshape(p, n, target_rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_tensor, V_tensor, W_tensor)
        error = np.max(np.abs(T - 1.0))

        if error < 5.0:
            result = _gradient_descent(U, V, W, n, m, p, deadline, max_steps=200)
            if result is not None:
                return result

    return None

def _try_structured_patterns(n, m, p, deadline, best_rank):
    """Try various structured patterns that might give lower rank."""
    candidate_ranks = [99, 98, 97, 96, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]

    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue

        result = _try_random_search(n, m, p, target_rank, 15, deadline)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Leave 15s margin for safety

    # Schoolbook decomposition
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school

    # Try to find better decompositions
    result = _try_structured_patterns(n, m, p, deadline, best_rank)
    if result is not None:
        U_try, V_try, W_try = result
        if U_try.shape[1] < best_rank:
            if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                best_rank = U_try.shape[1]

    # If time remains, try more random searches with different seeds
    if time.time() < deadline - 20:
        for seed in [123, 456, 789, 101, 202]:
            if time.time() > deadline - 10:
                break
            np.random.seed(seed)
            result = _try_random_search(n, m, p, best_rank - 1, 10, deadline)
            if result is not None:
                U_try, V_try, W_try = result
                if _verify_decomposition(U_try, V_try, W_try, n, m, p):
                    best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                    best_rank = U_try.shape[1]
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
