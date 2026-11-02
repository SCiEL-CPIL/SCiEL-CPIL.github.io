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

    # Precompute target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_exact(U, V, W, tol=1e-10):
        """Exact verification of tensor decomposition."""
        T_calc = np.einsum('ak,bk,ck->abc', U, V, W)
        diff = np.abs(T_calc - T)
        return np.max(diff) < tol

    def polish(U, V, W, max_iters=200, tol=1e-12):
        """High-precision polish pass using Newton-like steps."""
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            T_calc = np.einsum('ak,bk,ck->abc', U, V, W)
            residual = T_calc - T
            loss = np.sum(residual * residual)
            if loss < tol:
                if verify_exact(U, V, W, tol=1e-10):
                    return True
                break
            # Compute gradients
            grad_U = np.einsum('abc,bk,ck->ak', residual, V, W)
            grad_V = np.einsum('abc,ak,ck->bk', residual, U, W)
            grad_W = np.einsum('abc,ak,bk->ck', residual, U, V)
            # Adaptive learning rate based on gradient magnitude
            gnorm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if gnorm < 1e-15:
                break
            lr = 0.1 / (gnorm + 1e-8)
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
        return verify_exact(U, V, W, tol=1e-10)

    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Try rank 12: structured decomposition
    def try_rank_12_structured():
        """Try rank-12 decomposition with shared structure."""
        r = 12
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # For each output row i and each k, one term
        for i in range(n):
            for k in range(p):
                r_idx = i * p + k
                for j in range(m):
                    U[i * m + j, r_idx] = 1.0
                    V[j * p + k, r_idx] = 1.0
                    W[k * n + i, r_idx] = 1.0
        
        if verify_exact(U, V, W):
            return U, V, W
        return None

    if time.time() < deadline - 30:
        res = try_rank_12_structured()
        if res is not None:
            best_rank = 12
            best_U, best_V, best_W = res
            print(f"Found rank {best_rank} decomposition!")
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Improved gradient-based search
    def try_gradient_search(target_rank, num_trials=8, max_steps=60):
        """Improved gradient descent with better initialization and polishing."""
        for trial in range(num_trials):
            if time.time() > deadline - 20:
                break
            
            # Better initialization: mix of schoolbook and random
            if trial < num_trials // 2:
                # Initialize from schoolbook subset
                idx = np.random.choice(rank0, size=target_rank, replace=False)
                U = U0[:, idx].copy()
                V = V0[:, idx].copy()
                W = W0[:, idx].copy()
                # Add small perturbation
                U += 0.02 * np.random.randn(n * m, target_rank)
                V += 0.02 * np.random.randn(m * p, target_rank)
                W += 0.02 * np.random.randn(p * n, target_rank)
            else:
                # Pure random initialization with scaling
                scale = 1.0 / np.sqrt(target_rank)
                U = scale * np.random.randn(n * m, target_rank)
                V = scale * np.random.randn(m * p, target_rank)
                W = scale * np.random.randn(p * n, target_rank)

            mU = np.zeros_like(U); mV = np.zeros_like(V); mW = np.zeros_like(W)
            vU = np.zeros_like(U); vV = np.zeros_like(V); vW = np.zeros_like(W)
            b1, b2 = 0.9, 0.999
            eps = 1e-8
            best_loss = np.inf
            best_U_trial = U.copy()
            best_V_trial = V.copy()
            best_W_trial = W.copy()
            
            for step in range(max_steps):
                if time.time() > deadline - 20:
                    break
                t = step + 1
                T_calc = np.einsum('ak,bk,ck->abc', U, V, W)
                residual = T_calc - T
                loss = np.sum(residual * residual)
                if loss < best_loss:
                    best_loss = loss
                    best_U_trial = U.copy()
                    best_V_trial = V.copy()
                    best_W_trial = W.copy()
                if loss < 1e-14:
                    if verify_exact(U, V, W):
                        return U, V, W
                    break
                grad_U = np.einsum('abc,bk,ck->ak', residual, V, W)
                grad_V = np.einsum('abc,ak,ck->bk', residual, U, W)
                grad_W = np.einsum('abc,ak,bk->ck', residual, U, V)
                mU = b1 * mU + (1 - b1) * grad_U; vU = b2 * vU + (1 - b2) * grad_U * grad_U
                mV = b1 * mV + (1 - b1) * grad_V; vV = b2 * vV + (1 - b2) * grad_V * grad_V
                mW = b1 * mW + (1 - b1) * grad_W; vW = b2 * vW + (1 - b2) * grad_W * grad_W
                mUh = mU / (1 - b1 ** t); vUh = vU / (1 - b2 ** t)
                mVh = mV / (1 - b1 ** t); vVh = vV / (1 - b2 ** t)
                mWh = mW / (1 - b1 ** t); vWh = vW / (1 - b2 ** t)
                # Adaptive learning rate
                lr = 0.05 / (1 + step * 0.01)
                U -= lr * mUh / (np.sqrt(vUh) + eps)
                V -= lr * mVh / (np.sqrt(vVh) + eps)
                W -= lr * mWh / (np.sqrt(vWh) + eps)
            
            # Try polishing the best trial result
            if best_loss < 1e-6:
                if polish(best_U_trial, best_V_trial, best_W_trial, max_iters=100):
                    return best_U_trial, best_V_trial, best_W_trial
        return None

    # Try various target ranks with improved search
    for target in [13, 14, 15, 16]:
        if time.time() > deadline - 30:
            break
        res = try_gradient_search(target, num_trials=6, max_steps=50)
        if res is not None:
            best_rank = target
            best_U, best_V, best_W = res
            print(f"Found rank {best_rank} decomposition!")
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Final attempt: try to polish the schoolbook to see if we can reduce
    # by identifying redundant terms
    if time.time() < deadline - 10:
        # Try to find a lower-rank subset of schoolbook that works
        for target in [24, 20, 18, 16]:
            if time.time() > deadline - 10:
                break
            # Greedy subset selection
            idx = np.random.choice(rank0, size=target, replace=False)
            U_sub = U0[:, idx]
            V_sub = V0[:, idx]
            W_sub = W0[:, idx]
            if verify_exact(U_sub, V_sub, W_sub):
                best_rank = target
                best_U, best_V, best_W = U_sub.copy(), V_sub.copy(), W_sub.copy()
                print(f"Found rank {best_rank} decomposition via subset!")
                return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fall back to schoolbook
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
