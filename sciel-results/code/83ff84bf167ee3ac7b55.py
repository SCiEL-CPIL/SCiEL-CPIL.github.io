import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin under 150s

    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify(U, V, W, tol=1e-9):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def get_rank_55_solution():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        col = 0
        # Part A: j=0
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        # Part B: j=1,2 with Strassen on 5 blocks
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2

            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1

            # M1
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1

            # M2
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1

            # M3
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1

            # M4
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1

            # M5
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1

            # M6
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1

            # M7
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1

        return U, V, W, rank

    U_best, V_best, W_best, rank_best = get_rank_55_solution()
    if not verify(U_best, V_best, W_best, tol=1e-8):
        # Fallback: schoolbook
        rank = nm * mp
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ALS polish for a given rank R, starting from random or given init
    def als_polish(R, U_init=None, V_init=None, W_init=None, iters=60, tol=1e-10):
        if U_init is None:
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
        else:
            U = np.array(U_init, dtype=float)
            V = np.array(V_init, dtype=float)
            W = np.array(W_init, dtype=float)

        for _ in range(iters):
            if time.time() > deadline - 5:
                break

            # Solve U given V,W
            M_VW = np.einsum('jr,kr->jkr', V, W).reshape(mp * pn, R)
            T_flat = T.reshape(nm, mp * pn)
            try:
                U = T_flat @ np.linalg.pinv(M_VW.T)
            except Exception:
                return None

            # Solve V given U,W
            M_UW = np.einsum('ir,kr->ikr', U, W).reshape(nm * pn, R)
            T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            try:
                V = T_flat_V @ np.linalg.pinv(M_UW.T)
            except Exception:
                return None

            # Solve W given U,V
            M_UV = np.einsum('ir,jr->ijr', U, V).reshape(nm * mp, R)
            T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                W = T_flat_W @ np.linalg.pinv(M_UV.T)
            except Exception:
                return None

            # Check error
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < tol:
                return U, V, W

        return None

    # Local gradient-based search for lower rank, using smooth residual objective
    def objective_and_grad(params, R):
        U = params[:nm * R].reshape(nm, R)
        V = params[nm * R:nm * R + mp * R].reshape(mp, R)
        W = params[nm * R + mp * R:].reshape(pn, R)

        err = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)

        # Efficient-ish: loop over (i,k) as in original
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)

        for i in range(n):
            if time.time() > deadline - 5:
                return float('inf'), np.zeros_like(params)
            for k in range(p):
                U_i = U_mat[i]
                V_k = V_mat[:, k]
                W_ki = W_mat[k, i]

                M_ik = (U_i * W_ki) @ V_k.T
                diff = M_ik - np.eye(m)
                err += np.sum(diff ** 2)

                dM_dU = np.diag(W_ki) @ V_k.T
                dM_dV = U_i @ np.diag(W_ki)

                grad_U_mat_i = 2 * (diff @ dM_dU.T)
                grad_V_mat_k = 2 * (dM_dV @ diff.T)

                grad_U_mat[i] += grad_U_mat_i
                grad_V_mat[:, k] += grad_V_mat_k

                for r in range(R):
                    dM_dWr = np.outer(U_i[:, r], V_k[:, r])
                    grad_W_mat[k, i, r] += 2 * np.sum(diff * dM_dWr)

        return err, np.concatenate([grad_U_mat.flatten(), grad_V_mat.flatten(), grad_W_mat.flatten()])

    def try_lower_rank(R):
        # Try a few random restarts with gradient method, then ALS polish
        for attempt in range(3):
            if time.time() > deadline - 10:
                return None

            scale = random.choice([0.5, 1.0, 2.0])
            U = np.random.randn(nm, R) * scale
            V = np.random.randn(mp, R) * scale
            W = np.random.randn(pn, R) * scale

            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                res = minimize(objective_and_grad, params, args=(R,), jac=True,
                               method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
                if res.fun < 1e-6:
                    U_sol = res.x[:nm * R].reshape(nm, R)
                    V_sol = res.x[nm * R:nm * R + mp * R].reshape(mp, R)
                    W_sol = res.x[nm * R + mp * R:].reshape(pn, R)

                    # ALS polish
                    polished = als_polish(R, U_sol, V_sol, W_sol, iters=40, tol=1e-10)
                    if polished is not None:
                        U_p, V_p, W_p = polished
                        if verify(U_p, V_p, W_p, tol=1e-9):
                            return U_p, V_p, W_p, R
            except Exception:
                pass

        return None

    # Search for lower rank
    for R in range(rank_best - 1, max(rank_best - 15, 35), -1):
        if time.time() > deadline - 15:
            break

        res = try_lower_rank(R)
        if res is not None:
            U_best, V_best, W_best, rank_best = res
            break

    # Final exact polish on fixed support: treat nonzeros as variables, solve linear system
    def exact_polish_fixed_support(U, V, W, R, tol=1e-12):
        # Collect support pattern (which entries are nonzero)
        # We'll solve for all entries as variables, but fix zero pattern
        # Build linear system: sum_r U[i,r] V[j,r] W[k,r] = T[i,j,k]
        # Variables: x = concat(U.flatten(), V.flatten(), W.flatten())
        # This is cubic, not linear. Instead, we can fix two factors and solve the third.
        # We'll fix U and V, solve for W linearly.
        # W[k,r] = ? such that for all i,j,k: T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
        # For fixed i,j,k, this is linear in W[k,:].
        # But it's coupled over r. We can set up a linear system for W.
        # Let A[i,j,k,r] = U[i,r] V[j,r]. Then T[i,j,k] = sum_r A[i,j,k,r] W[k,r].
        # This is a linear system in W.

        # Build system: for each (i,j,k), equation: sum_r A[i,j,k,r] W[k,r] = T[i,j,k]
        # Variables: W[k,r] for k=0..pn-1, r=0..R-1
        # Number of variables: pn * R
        # Number of equations: nm * mp * pn
        # This is overdetermined. Use least squares.

        A = np.einsum('ir,jr->ijkr', U, V)  # shape (nm, mp, pn, R)
        A_flat = A.reshape(nm * mp * pn, R)
        T_flat = T.reshape(nm * mp * pn)

        # We want W such that A_flat @ W_flat = T_flat, where W_flat is (pn*R,)
        # But A_flat is (nm*mp*pn, R), and W_flat is (pn*R,). This doesn't match.
        # Actually, for each k, we have a sub-system: for all i,j, sum_r A[i,j,k,r] W[k,r] = T[i,j,k]
        # So we can solve for each k separately.

        W_sol = np.zeros((pn, R))
        for k in range(pn):
            A_k = A[:, :, k, :]  # shape (nm, mp, R)
            A_k_flat = A_k.reshape(nm * mp, R)
            T_k = T[:, :, k].reshape(nm * mp)
            try:
                W_k = np.linalg.lstsq(A_k_flat, T_k, rcond=None)[0]
                W_sol[k] = W_k
            except Exception:
                return None

        # Check error
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W_sol)
        err = np.max(np.abs(T_recon - T))
        if err < tol:
            return U, V, W_sol

        return None

    # Try exact polish on current best
    polished = exact_polish_fixed_support(U_best, V_best, W_best, rank_best, tol=1e-12)
    if polished is not None:
        U_best, V_best, W_best = polished

    # Try to round to half-integers if possible
    U_round = np.round(U_best * 2) / 2
    V_round = np.round(V_best * 2) / 2
    W_round = np.round(W_best * 2) / 2
    if verify(U_round, V_round, W_round, tol=1e-9):
        U_best, V_best, W_best = U_round, V_round, W_round

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
