import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin

    n, m, p = 2, 3, 3

    # Rank 17 solution (known valid)
    rank17 = 17
    U17 = np.zeros((n * m, rank17))
    V17 = np.zeros((m * p, rank17))
    W17 = np.zeros((p * n, rank17))

    # Part 1: Strassen for 2x2x2 sub-block
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1

    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1

    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1

    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1

    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1

    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1

    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1

    # Part 2: Cross terms A[:,2] * B[2,:]
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1

    # Part 3: Cross terms A[:,:2] * B[:,2]
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1

    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True

    # Verify rank-17 solution
    if not _verify(U17, V17, W17, n, m, p, rank17):
        # Fallback to schoolbook
        U_fb = np.zeros((n * m, n * m * p))
        V_fb = np.zeros((m * p, n * m * p))
        W_fb = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    best_rank = rank17
    best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()

    # Build the multiplication tensor T[a, b, c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    # Try to find rank-16 solution
    rank16 = 16

    def _try_random_restart(U_init, V_init, W_init, max_iter, lr_schedule='cosine'):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()

        # Adaptive learning rate
        base_lr = 0.05

        for it in range(max_iter):
            if time.time() > deadline:
                return None, None, None, float('inf')

            # Compute residual
            residual = np.zeros((n*m, m*p, p*n))
            for r in range(rank16):
                residual += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])

            diff = residual - T
            loss = np.sum(diff**2)

            if loss < 1e-10:
                return U, V, W, loss

            # Compute gradients
            dU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
            dV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
            dW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)

            # Adaptive learning rate
            if lr_schedule == 'cosine':
                lr = base_lr * (0.5 * (1 + np.cos(np.pi * it / max_iter)))
            else:
                lr = base_lr

            # Gradient norm-based scaling
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm > 1e-8:
                scale = min(1.0, 0.5 / grad_norm)
                lr *= scale

            U -= lr * dU
            V -= lr * dV
            W -= lr * dW

            # Periodic random perturbation to escape local minima
            if it % 100 == 50:
                U += 0.01 * np.random.randn(n*m, rank16)
                V += 0.01 * np.random.randn(m*p, rank16)
                W += 0.01 * np.random.randn(p*n, rank16)

        return U, V, W, loss

    # Try multiple random restarts
    np.random.seed(12345)
    found = False

    for restart in range(20):
        if time.time() > deadline:
            break

        # Different initialization strategies
        if restart < 5:
            # Pure random
            U_init = np.random.randn(n*m, rank16) * 0.5
            V_init = np.random.randn(m*p, rank16) * 0.5
            W_init = np.random.randn(p*n, rank16) * 0.5
        elif restart < 10:
            # Scaled random
            U_init = np.random.randn(n*m, rank16) * 0.3
            V_init = np.random.randn(m*p, rank16) * 0.3
            W_init = np.random.randn(p*n, rank16) * 0.3
        elif restart < 15:
            # Sparse random - set many entries to 0
            U_init = np.random.randn(n*m, rank16) * 0.5
            V_init = np.random.randn(m*p, rank16) * 0.5
            W_init = np.random.randn(p*n, rank16) * 0.5
            mask = np.random.rand(n*m, rank16) > 0.3
            U_init[~mask] = 0
            mask = np.random.rand(m*p, rank16) > 0.3
            V_init[~mask] = 0
            mask = np.random.rand(p*n, rank16) > 0.3
            W_init[~mask] = 0
        else:
            # Random with different scales
            s = 0.2 + 0.4 * np.random.rand()
            U_init = np.random.randn(n*m, rank16) * s
            V_init = np.random.randn(m*p, rank16) * s
            W_init = np.random.randn(p*n, rank16) * s

        U_res, V_res, W_res, loss = _try_random_restart(U_init, V_init, W_res if False else W_init, 3000, 'cosine')

        if loss < 1e-10:
            # Snap to nearest 0.5 grid and verify
            U_snap = np.round(U_res * 2) / 2
            V_snap = np.round(V_res * 2) / 2
            W_snap = np.round(W_res * 2) / 2

            if _verify(U_snap, V_snap, W_snap, n, m, p, rank16):
                best_rank = rank16
                best_U, best_V, best_W = U_snap, V_snap, W_snap
                found = True
                break

        # Also try with integer snapping
        if loss < 1e-8:
            U_int = np.round(U_res)
            V_int = np.round(V_res)
            W_int = np.round(W_res)

            if _verify(U_int, V_int, W_int, n, m, p, rank16):
                best_rank = rank16
                best_U, best_V, best_W = U_int, V_int, W_int
                found = True
                break

    if not found:
        # Try a more structured approach: take the rank-17 solution and try to merge two similar columns
        # by projecting onto rank 16
        # This is essentially a low-rank approximation followed by refinement

        # For now, return rank 17
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
