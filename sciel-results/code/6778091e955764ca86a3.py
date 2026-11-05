"""Program to find rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 3, 6
    # Target: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

    def verify(U, V, W, tol=1e-8):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def residual(U, V, W):
        """Compute max absolute residual."""
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    max_res = max(max_res, abs(s - 1.0))
        return max_res

    def newton_polish(U, V, W, max_iters=200, tol=1e-12):
        """Newton's method on the residual to polish the solution."""
        rank = len(U[0])
        # Flatten variables
        def flatten(U, V, W):
            return (np.array(U).flatten(), np.array(V).flatten(), np.array(W).flatten())

        def unflatten(u_flat, v_flat, w_flat):
            U = u_flat.reshape(n * m, rank)
            V = v_flat.reshape(m * p, rank)
            W = w_flat.reshape(p * n, rank)
            return U, V, W

        u_flat, v_flat, w_flat = flatten(U, V, W)
        x = np.concatenate([u_flat, v_flat, w_flat])

        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            U, V, W = unflatten(x[:n*m*rank], x[n*m*rank:n*m*rank+m*p*rank], x[n*m*rank+m*p*rank:])
            # Compute residual and Jacobian
            res = np.zeros(n * m * p)
            jac = np.zeros((n * m * p, len(x)))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            u_val = U[i * m + j, r]
                            v_val = V[j * p + k, r]
                            w_val = W[k * n + i, r]
                            s += u_val * v_val * w_val
                            # Jacobian entries
                            u_idx = (i * m + j) * rank + r
                            v_idx = n * m * rank + (j * p + k) * rank + r
                            w_idx = (n * m + m * p) * rank + (k * n + i) * rank + r
                            jac[idx, u_idx] = v_val * w_val
                            jac[idx, v_idx] = u_val * w_val
                            jac[idx, w_idx] = u_val * v_val
                        res[idx] = s - 1.0
                        idx += 1

            if np.max(np.abs(res)) < tol:
                break

            # Solve least squares
            try:
                dx, _, _, _ = np.linalg.lstsq(jac, -res, rcond=None)
            except:
                break

            # Damped step
            step = 1.0
            for _ in range(20):
                x_new = x + step * dx
                U_new, V_new, W_new = unflatten(
                    x_new[:n*m*rank],
                    x_new[n*m*rank:n*m*rank+m*p*rank],
                    x_new[n*m*rank+m*p*rank:])
                new_res = residual(U_new, V_new, W_new)
                if new_res < np.max(np.abs(res)):
                    x = x_new
                    break
                step *= 0.5
            else:
                break

        U, V, W = unflatten(x[:n*m*rank], x[n*m*rank:n*m*rank+m*p*rank], x[n*m*rank+m*p*rank:])
        return U.tolist(), V.tolist(), W.tolist()

    def greedy_search(max_rank, time_budget):
        """Greedy search: add terms one at a time to reduce residual."""
        start = time.time()
        best_U, best_V, best_W = None, None, None
        best_res = float('inf')

        # Initialize with schoolbook-like random structure
        for attempt in range(3):
            if time.time() > start + time_budget or time.time() > deadline - 20:
                break

            np.random.seed(42 + attempt)
            rank = max_rank
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))

            # Greedy: for each term, find best (u,v,w) to reduce residual
            for r in range(rank):
                if time.time() > start + time_budget or time.time() > deadline - 20:
                    break

                # Compute current residual
                res = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for rr in range(r):
                                s += U[i * m + j, rr] * V[j * p + k, rr] * W[k * n + i, rr]
                            res[i, j, k] = 1.0 - s

                # Find best rank-1 update to minimize ||res - u⊗v⊗w||
                # Use iterative approach: fix two, optimize third
                best_improvement = -1
                best_u, best_v, best_w = None, None, None

                # Try several random initializations for this term
                for trial in range(50):
                    if time.time() > start + time_budget or time.time() > deadline - 20:
                        break

                    u = np.random.randn(n * m)
                    v = np.random.randn(m * p)
                    w = np.random.randn(p * n)

                    # Alternating optimization
                    for _ in range(30):
                        if time.time() > start + time_budget or time.time() > deadline - 20:
                            break
                        # Optimize u: minimize ||res - u * (v⊗w reshaped)||^2
                        # For each (i,j): res[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
                        # Fix v,w: for each (i,j), u[i*m+j] = argmin sum_k (res[i,j,k] - u*v[j*p+k]*w[k*n+i])^2
                        # This is: u[i*m+j] = sum_k res[i,j,k]*v[j*p+k]*w[k*n+i] / sum_k (v[j*p+k]*w[k*n+i])^2
                        for i in range(n):
                            for j in range(m):
                                num = 0.0
                                den = 0.0
                                for k in range(p):
                                    vk = v[j * p + k]
                                    wk = w[k * n + i]
                                    num += res[i, j, k] * vk * wk
                                    den += vk * vk * wk * wk
                                if abs(den) > 1e-15:
                                    u[i * m + j] = num / den

                        # Optimize v similarly
                        for j in range(m):
                            for k in range(p):
                                num = 0.0
                                den = 0.0
                                for i in range(n):
                                    ui = u[i * m + j]
                                    wi = w[k * n + i]
                                    num += res[i, j, k] * ui * wi
                                    den += ui * ui * wi * wi
                                if abs(den) > 1e-15:
                                    v[j * p + k] = num / den

                        # Optimize w similarly
                        for k in range(p):
                            for i in range(n):
                                num = 0.0
                                den = 0.0
                                for j in range(m):
                                    uj = u[i * m + j]
                                    vj = v[j * p + k]
                                    num += res[i, j, k] * uj * vj
                                    den += uj * uj * vj * vj
                                if abs(den) > 1e-15:
                                    w[k * n + i] = num / den

                    # Evaluate improvement
                    new_res = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for rr in range(r):
                                    s += U[i * m + j, rr] * V[j * p + k, rr] * W[k * n + i, rr]
                                s += u[i * m + j] * v[j * p + k] * w[k * n + i]
                                new_res[i, j, k] = 1.0 - s

                    improvement = np.sum(res**2) - np.sum(new_res**2)
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_u, best_v, best_w = u.copy(), v.copy(), w.copy()

                if best_u is not None:
                    U[:, r] = best_u
                    V[:, r] = best_v
                    W[:, r] = best_w

            # Check result
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            res_val = residual(U_list, V_list, W_list)
            if res_val < best_res:
                best_res = res_val
                best_U, best_V, best_W = U_list, V_list, W_list

            # Try to polish with Newton
            if best_res < 1.0 and time.time() < deadline - 15:
                U_pol, V_pol, W_pol = newton_polish(U_list, V_list, W_list)
                res_pol = residual(U_pol, V_pol, W_pol)
                if res_pol < best_res:
                    best_res = res_pol
                    best_U, best_V, best_W = U_pol, V_pol, W_pol

        if best_U is not None and best_res < 1e-8:
            return best_U, best_V, best_W, max_rank
        return None, None, None, max_rank

    # Try to find a low-rank solution
    best_result = None
    best_rank = float('inf')

    # Try ranks from low to high
    for target_rank in [30, 35, 40, 45, 46]:
        if time.time() > deadline - 30:
            break

        U, V, W, rank = greedy_search(target_rank, time_budget=20)
        if U is not None:
            # Verify
            if verify(U, V, W, tol=1e-8):
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
                break  # Found valid, stop

    if best_result is not None:
        return best_result

    # Fallback: rank 46 via two 3x3 blocks
    # Use a verified simple construction
    rank = 46
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Simple block construction: for each half of B (3x3), use a valid 3x3*3x3 decomposition
    # Use the known fact that we can decompose 3x3*3x3 with rank <= 23 (Laderman)
    # But let's use a simpler verified approach: rank 27 (schoolbook) per block = 54 total
    # Actually, let's just use rank 54 schoolbook as guaranteed fallback
    rank = n * m * p  # 54
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1

    # Verify fallback
    assert verify(U, V, W, tol=1e-10), "Schoolbook fallback failed!"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
