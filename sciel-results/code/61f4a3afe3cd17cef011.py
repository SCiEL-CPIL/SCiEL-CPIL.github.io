"""Find a low-rank exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p  # 48

    # Generate schoolbook solution as fallback
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = schoolbook_rank

    # Build target tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    # Lower bound on rank
    lower_bound = max(nm, mp, pn)  # 24

    rng = np.random.RandomState(42)

    for rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline:
            break

        max_attempts = 300 if rank <= 30 else 500
        found = False

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Random initialization
            U = rng.randn(nm, rank)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)

            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() > deadline:
                    break

                # Update U: target[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r].T)
                # M[b*pn+c, r] = V[b,r]*W[c,r]
                M = (V[:, :, None] * W[None, :, :]).reshape(mp * pn, rank)
                target_2d = target.reshape(nm, mp * pn)
                try:
                    MMt = M.T @ M
                    M_t = M.T @ target_2d.T
                    U = np.linalg.solve(MMt, M_t).T
                except:
                    try:
                        U = np.linalg.lstsq(M, target_2d.T, rcond=None)[0].T
                    except:
                        break

                # Update V
                N = (U[:, :, None] * W[None, :, :]).reshape(nm * pn, rank)
                target_2d_v = target.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    NNt = N.T @ N
                    N_t = N.T @ target_2d_v.T
                    V = np.linalg.solve(NNt, N_t).T
                except:
                    try:
                        V = np.linalg.lstsq(N, target_2d_v.T, rcond=None)[0].T
                    except:
                        break

                # Update W
                P = (U[:, :, None] * V[None, :, :]).reshape(nm * mp, rank)
                target_2d_w = target.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    PPt = P.T @ P
                    P_t = P.T @ target_2d_w.T
                    W = np.linalg.solve(PPt, P_t).T
                except:
                    try:
                        W = np.linalg.lstsq(P, target_2d_w.T, rcond=None)[0].T
                    except:
                        break

                # Check convergence every 20 iters
                if it % 20 == 19:
                    decomposed = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(decomposed - target) / np.linalg.norm(target)
                    if err < 1e-12:
                        converged = True
                        break

            # Final verification
            decomposed = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(decomposed, target, atol=1e-9, rtol=1e-9):
                best_rank = rank
                # Try to round to cleaner values
                U_r = np.round(U, 4)
                V_r = np.round(V, 4)
                W_r = np.round(W, 4)
                decomp_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                if np.allclose(decomp_r, target, atol=1e-6, rtol=1e-6):
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                else:
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break

        if found:
            break
        if time.time() > deadline:
            break

    return best_result
