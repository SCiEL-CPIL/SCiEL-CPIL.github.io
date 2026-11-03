import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition as dict of lists."""
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

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy einsum."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]

    U_tensor = Ua.reshape(n, m, rank)
    V_tensor = Va.reshape(m, p, rank)
    W_tensor = Wa.reshape(p, n, rank)

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

    dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', residual, V_tensor, W_tensor)
    dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', residual, U_tensor, W_tensor)
    dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', residual, U_tensor, V_tensor)

    return dU.reshape(n * m, rank), dV.reshape(m * p, rank), dW.reshape(p * n, rank)

def _gradient_descent(U, V, W, n, m, p, deadline, max_steps=500, lr=0.05):
    """Adam-like optimizer to drive ||T - 1|| to zero."""
    rank = U.shape[1]
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()

    mU = np.zeros_like(U); vU = np.zeros_like(U)
    mV = np.zeros_like(V); vV = np.zeros_like(V)
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    b1, b2, eps = 0.9, 0.999, 1e-8

    for step in range(max_steps):
        if time.time() > deadline - 5:
            break

        dU, dV, dW = _compute_gradients(U, V, W, n, m, p)
        
        t = step + 1
        mU = b1*mU + (1-b1)*dU; vU = b2*vU + (1-b2)*dU**2
        mUh = mU/(1-b1**t); vUh = vU/(1-b2**t)
        mV = b1*mV + (1-b1)*dV; vV = b2*vV + (1-b2)*dV**2
        mVh = mV/(1-b1**t); vVh = vV/(1-b2**t)
        mW = b1*mW + (1-b1)*dW; vW = b2*vW + (1-b2)*dW**2
        mWh = mW/(1-b1**t); vWh = vW/(1-b2**t)

        U -= lr * mUh / (np.sqrt(vUh) + eps)
        V -= lr * mVh / (np.sqrt(vVh) + eps)
        W -= lr * mWh / (np.sqrt(vWh) + eps)

        # Check convergence
        if step % 10 == 0:
            T = np.einsum('ijc,jkc,kic->ijk',
                          U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
            if np.allclose(T, 1.0, atol=1e-9):
                return U, V, W

    T = np.einsum('ijc,jkc,kic->ijk',
                  U.reshape(n,m,rank), V.reshape(m,p,rank), W.reshape(p,n,rank))
    if np.allclose(T, 1.0, atol=1e-9):
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin

    # 1. Schoolbook Decomposition (Rank 100)
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

    # 2. Attempt to find Rank 99 Decomposition
    # Strategy: Initialize with Schoolbook, but zero out one term and randomize slightly,
    # then use gradient descent to see if the remaining 99 terms can adjust to cover the gap.
    # This is a heuristic. If it fails, we return the schoolbook.
    
    target_rank = best_rank - 1
    if target_rank > 0 and time.time() < deadline - 20:
        # Try a few random "deletions"
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()
            
            # Remove one column (term)
            term_to_remove = np.random.randint(0, rank_school)
            U_try = np.delete(U_try, term_to_remove, axis=1)
            V_try = np.delete(V_try, term_to_remove, axis=1)
            W_try = np.delete(W_try, term_to_remove, axis=1)
            
            # Add small noise to help escape local minima
            noise_level = 0.1
            U_try += np.random.randn(*U_try.shape) * noise_level
            V_try += np.random.randn(*V_try.shape) * noise_level
            W_try += np.random.randn(*W_try.shape) * noise_level
            
            result = _gradient_descent(U_try, V_try, W_try, n, m, p, deadline, max_steps=300, lr=0.01)
            if result is not None:
                U_res, V_res, W_res = result
                if _verify_fast(U_res, V_res, W_res, n, m, p):
                    best_U, best_V, best_W = U_res, V_res, W_res
                    best_rank = target_rank
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
