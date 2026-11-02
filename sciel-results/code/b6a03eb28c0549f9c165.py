"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Schoolbook decomposition: rank = n*m*p = 48
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    def verify(U, V, W, tol=1e-8):
        """Verify tensor decomposition for matmul tensor exactly."""
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        T_calc = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_calc, T, atol=tol, rtol=0.0)

    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Precompute target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def try_rank_reduction(r, num_trials, max_steps=80):
        """Local search for rank-r decomposition via Adam-like gradient descent."""
        if r >= rank0:
            return None

        for trial in range(num_trials):
            if time.time() > deadline - 20:
                break
            # Initialize from a random subset of schoolbook terms plus noise
            idx = np.random.choice(rank0, size=r, replace=False)
            U = U0[:, idx].copy() + 0.05 * np.random.randn(n * m, r)
            V = V0[:, idx].copy() + 0.05 * np.random.randn(m * p, r)
            W = W0[:, idx].copy() + 0.05 * np.random.randn(p * n, r)

            # Momentum
            mU = np.zeros_like(U); mV = np.zeros_like(V); mW = np.zeros_like(W)
            vU = np.zeros_like(U); vV = np.zeros_like(V); vW = np.zeros_like(W)
            b1, b2 = 0.9, 0.999
            eps = 1e-8
            best_loss = np.inf
            for step in range(max_steps):
                if time.time() > deadline - 20:
                    break
                t = step + 1
                T_calc = np.einsum('ak,bk,ck->abc', U, V, W)
                residual = T_calc - T
                loss = np.sum(residual * residual)
                if loss < best_loss:
                    best_loss = loss
                if loss < 1e-10:
                    if verify(U, V, W, tol=1e-7):
                        return U, V, W
                    break
                # Gradients
                grad_U = np.einsum('abc,bk,ck->ak', residual, V, W)
                grad_V = np.einsum('abc,ak,ck->bk', residual, U, W)
                grad_W = np.einsum('abc,ak,bk->ck', residual, U, V)
                # Adam update
                mU = b1 * mU + (1 - b1) * grad_U; vU = b2 * vU + (1 - b2) * grad_U * grad_U
                mV = b1 * mV + (1 - b1) * grad_V; vV = b2 * vV + (1 - b2) * grad_V * grad_V
                mW = b1 * mW + (1 - b1) * grad_W; vW = b2 * vW + (1 - b2) * grad_W * grad_W
                mUh = mU / (1 - b1 ** t); vUh = vU / (1 - b2 ** t)
                mVh = mV / (1 - b1 ** t); vVh = vV / (1 - b2 ** t)
                mWh = mW / (1 - b1 ** t); vWh = vW / (1 - b2 ** t)
                lr = 0.1
                U -= lr * mUh / (np.sqrt(vUh) + eps)
                V -= lr * mVh / (np.sqrt(vVh) + eps)
                W -= lr * mWh / (np.sqrt(vWh) + eps)
        return None

    # Attempt rank 47, then 46 if time permits
    if time.time() < deadline - 40:
        res = try_rank_reduction(rank0 - 1, num_trials=20, max_steps=60)
        if res is not None:
            best_rank = rank0 - 1
            best_U, best_V, best_W = res
            print(f"Found rank {best_rank} decomposition!")

    if best_rank == rank0 and time.time() < deadline - 20:
        res = try_rank_reduction(rank0 - 2, num_trials=10, max_steps=50)
        if res is not None:
            best_rank = rank0 - 2
            best_U, best_V, best_W = res
            print(f"Found rank {best_rank} decomposition!")

    if best_rank < rank0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
