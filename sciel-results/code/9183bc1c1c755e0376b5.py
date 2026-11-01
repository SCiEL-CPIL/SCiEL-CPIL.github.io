"""Search for a rank-minimal exact tensor decomposition of 2x3 * 3x9 multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]


def _to_halfint(v):
    """Round to nearest half-integer and verify exactness."""
    h = np.round(2.0 * v) / 2.0
    if not np.allclose(v, h, atol=1e-9):
        return None
    return h


def construct():
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n  # 6, 27, 18

    # Build the target multiplication tensor T[i,j,k] with index layout:
    # U rows indexed by (i,j) -> i*m + j
    # V rows indexed by (j,k) -> j*p + k
    # W rows indexed by (k,i) -> k*n + i  (trace convention W[k*n+i, r])
    # Tensor: T[(i*m+j), (j*p+k), (k*n+i)] = 1
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Best known: schoolbook rank = n*m*p = 54
    best_rank = n * m * p
    best_U = np.zeros((nm, best_rank))
    best_V = np.zeros((mp, best_rank))
    best_W = np.zeros((pn, best_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j, c] = 1.0
                best_V[j * p + k, c] = 1.0
                best_W[k * n + i, c] = 1.0

    def _verify(U, V, W, tol=1e-7):
        """Check if U,V,W exactly decompose T."""
        r = U.shape[1]
        # Reconstruct tensor: T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Efficient: for each r, outer product
        recon = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(r):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(recon - T)) < tol

    def _try_round_and_verify(U, V, W):
        """Try rounding to half-integers and verify."""
        U_h = _to_halfint(U)
        V_h = _to_halfint(V)
        W_h = _to_halfint(W)
        if U_h is None or V_h is None or W_h is None:
            return None
        if _verify(U_h, V_h, W_h, tol=1e-9):
            return (U_h, V_h, W_h)
        return None

    def _optimize_rank(rank, seed, max_iters=2000, lr=0.01):
        """Try to find a decomposition of given rank via gradient descent."""
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, rank) * 0.5
        V = rng.randn(mp, rank) * 0.5
        W = rng.randn(pn, rank) * 0.5

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            # Gradient of loss = 0.5 * ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
            # dU[a,r] = -sum_{b,c} T[a,b,c]*V[b,r]*W[c,r] + sum_{b,c} recon[a,b,c]*V[b,r]*W[c,r]
            recon = np.zeros((nm, mp, pn), dtype=np.float64)
            for r in range(rank):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            diff = recon - T

            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for r in range(rank):
                # dU[:,r] = sum_{b,c} diff[:,b,c] * V[b,r] * W[c,r]
                dU[:, r] = np.einsum('abc,b,c->a', diff, V[:, r], W[:, r])
                dV[:, r] = np.einsum('abc,a,c->b', diff, U[:, r], W[:, r])
                dW[:, r] = np.einsum('abc,a,b->c', diff, U[:, r], V[:, r])

            U -= lr * dU
            V -= lr * dV
            W -= lr * dW

            # Periodic check
            if it % 100 == 0:
                err = np.max(np.abs(recon - T))
                if err < 1e-6:
                    result = _try_round_and_verify(U, V, W)
                    if result is not None:
                        return result

        # Final check
        err = np.max(np.abs(recon - T))
        if err < 1e-6:
            result = _try_round_and_verify(U, V, W)
            if result is not None:
                return result
        return None

    def _optimize_rank_adaptive(rank, seed, max_iters=5000):
        """Adaptive learning rate gradient descent."""
        rng = np.random.RandomState(seed)
        U = rng.randn(nm, rank) * 0.5
        V = rng.randn(mp, rank) * 0.5
        W = rng.randn(pn, rank) * 0.5

        lr = 0.05
        prev_loss = np.inf

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            recon = np.zeros((nm, mp, pn), dtype=np.float64)
            for r in range(rank):
                recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            diff = recon - T
            loss = 0.5 * np.sum(diff * diff)

            # Adaptive LR
            if loss > prev_loss:
                lr *= 0.5
            elif it > 0 and (loss - prev_loss) < -1e-8:
                lr = min(lr * 1.05, 0.1)
            prev_loss = loss

            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for r in range(rank):
                dU[:, r] = np.einsum('abc,b,c->a', diff, V[:, r], W[:, r])
                dV[:, r] = np.einsum('abc,a,c->b', diff, U[:, r], W[:, r])
                dW[:, r] = np.einsum('abc,a,b->c', diff, U[:, r], V[:, r])

            U -= lr * dU
            V -= lr * dV
            W -= lr * dW

            if it % 200 == 0:
                err = np.max(np.abs(recon - T))
                if err < 1e-6:
                    result = _try_round_and_verify(U, V, W)
                    if result is not None:
                        return result

        # Final
        recon = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        err = np.max(np.abs(recon - T))
        if err < 1e-6:
            result = _try_round_and_verify(U, V, W)
            if result is not None:
                return result
        return None

    # Try ranks from lower bound upward
    # Lower bound: max(nm, mp, pn) = max(6, 27, 18) = 27
    # But for matrix mult, known lower bounds are higher.
    # For 2x3*3x9, let's try from reasonable lower bounds.
    lower_bound = max(nm, mp, pn)  # 27
    # Actually, the tensor rank of matrix multiplication M_{n,m,p} >= max(n*m, m*p, p*n)
    # but also >= max(n, m, p) * min(n, m, p) ... let's just try from 27 upward.
    # However, we know schoolbook is 54. Let's try a few ranks.

    ranks_to_try = list(range(lower_bound, best_rank + 1))
    # Try more promising ranks first (lower ranks)
    for rank in ranks_to_try:
        if time.time() > deadline - 15:
            break
        # Multiple seeds for each rank
        for seed in range(20):
            if time.time() > deadline - 15:
                break
            result = _optimize_rank_adaptive(rank, seed, max_iters=3000)
            if result is not None:
                U_h, V_h, W_h = result
                if rank < best_rank:
                    best_rank = rank
                    best_U, best_V, best_W = U_h, V_h, W_h
                    break  # Found this rank, try lower

    # Also try a structured approach: exploit the small dimensions
    # For 2x3*3x9, we can try to find structure
    if time.time() < deadline - 20:
        # Try rank = 36 (2x3x6, since 9 = 3*3, maybe we can group)
        # Actually let's try a smarter initialization
        for rank in [30, 33, 36, 40, 45, 48, 50, 51, 52, 53]:
            if time.time() > deadline - 15:
                break
            if rank >= best_rank:
                continue
            for seed in range(30):
                if time.time() > deadline - 15:
                    break
                result = _optimize_rank_adaptive(rank, seed, max_iters=4000)
                if result is not None:
                    U_h, V_h, W_h = result
                    if rank < best_rank:
                        best_rank = rank
                        best_U, best_V, best_W = U_h, V_h, W_h
                        break

    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
