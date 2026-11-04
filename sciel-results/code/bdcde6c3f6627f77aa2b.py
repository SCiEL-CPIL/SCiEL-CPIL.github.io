"""Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DEADLINE = 140  # safety margin before 150s kill


def construct():
    start_time = time.time()
    n, m, p = 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24

    # Schoolbook: rank 120
    def schoolbook():
        R = n * m * p
        U = [[0.0] * R for _ in range(nm)]
        V = [[0.0] * R for _ in range(mp)]
        W = [[0.0] * R for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()

    def try_rank(R, time_budget):
        """Try to find exact rank-R decomposition via optimization with structured init."""
        deadline = time.time() + time_budget
        best_obj = float('inf')
        best_sol = None

        n_restarts = 0
        while time.time() < deadline - 3:
            n_restarts += 1
            if n_restarts > 30:
                break

            # Structured random initialization: each rank-1 term covers (i,j,k)
            # Use a mix of random and structured initialization
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))

            # For each rank component, assign a structured triple (i,j,k)
            # with some randomization
            for c in range(R):
                # Choose indices with some structure
                # Strategy: ensure coverage of all (i,j,k) triples
                # Use a base pattern with perturbation
                if c < n * m * p:
                    i = c // (m * p)
                    j = (c // p) % m
                    k = c % p
                else:
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)

                # Set primary entries
                U[i * m + j, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                V[j * p + k, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                W[k * n + i, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)

                # Add some off-diagonal entries for flexibility
                if random.random() < 0.3:
                    i2 = random.randint(0, n - 1)
                    j2 = random.randint(0, m - 1)
                    U[i2 * m + j2, c] = random.choice([-1, 1]) * random.uniform(0.1, 0.5)
                if random.random() < 0.3:
                    j2 = random.randint(0, m - 1)
                    k2 = random.randint(0, p - 1)
                    V[j2 * p + k2, c] = random.choice([-1, 1]) * random.uniform(0.1, 0.5)
                if random.random() < 0.3:
                    k2 = random.randint(0, p - 1)
                    i2 = random.randint(0, n - 1)
                    W[k2 * n + i2, c] = random.choice([-1, 1]) * random.uniform(0.1, 0.5)

            def obj_and_grad(x):
                Ux = x[:nm * R].reshape(nm, R)
                Vx = x[nm * R:nm * R + mp * R].reshape(mp, R)
                Wx = x[nm * R + mp * R:nm * R + mp * R + pn * R].reshape(pn, R)

                # Compute violations and gradients
                # T[i,j,k] = sum_r Ux[i*m+j, r] * Vx[j*p+k, r] * Wx[k*n+i, r]
                viol = np.zeros((n, m, p))
                dU = np.zeros((nm, R))
                dV = np.zeros((mp, R))
                dW = np.zeros((pn, R))

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += Ux[i * m + j, r] * Vx[j * p + k, r] * Wx[k * n + i, r]
                            viol[i, j, k] = s - 1.0
                            diff = 2.0 * viol[i, j, k]
                            for r in range(R):
                                dU[i * m + j, r] += diff * Vx[j * p + k, r] * Wx[k * n + i, r]
                                dV[j * p + k, r] += diff * Ux[i * m + j, r] * Wx[k * n + i, r]
                                dW[k * n + i, r] += diff * Ux[i * m + j, r] * Vx[j * p + k, r]

                fval = float(np.sum(viol * viol))
                gx = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return fval, gx

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                fval, gval = obj_and_grad(x0)
                res = minimize(lambda x: obj_and_grad(x)[0], x0,
                               jac=lambda x: obj_and_grad(x)[1],
                               method='L-BFGS-B',
                               options={'maxiter': 200, 'maxfun': 6000,
                                        'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50})
                fval = res.fun
                if fval < best_obj:
                    best_obj = fval
                    x_opt = res.x
                    U_opt = x_opt[:nm * R].reshape(nm, R)
                    V_opt = x_opt[nm * R:nm * R + mp * R].reshape(mp, R)
                    W_opt = x_opt[nm * R + mp * R:nm * R + mp * R + pn * R].reshape(pn, R)
                    best_sol = (U_opt, V_opt, W_opt)
            except Exception:
                pass

            if best_obj < 1e-14:
                break

        if best_sol is not None and best_obj < 1e-10:
            U_opt, V_opt, W_opt = best_sol
            # Verify exactness
            max_viol = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(R):
                            s += U_opt[i * m + j, r] * V_opt[j * p + k, r] * W_opt[k * n + i, r]
                        max_viol = max(max_viol, abs(s - 1.0))

            if max_viol < 1e-12:
                U_list = [[float(U_opt[i, r]) for r in range(R)] for i in range(nm)]
                V_list = [[float(V_opt[i, r]) for r in range(R)] for i in range(mp)]
                W_list = [[float(W_opt[i, r]) for r in range(R)] for i in range(pn)]
                return {"U": U_list, "V": V_list, "W": W_list}

        return None

    def try_rank_with_polish(R, time_budget):
        """Try rank-R with a high-precision polish pass after initial convergence."""
        deadline = time.time() + time_budget
        best_obj = float('inf')
        best_sol = None

        n_restarts = 0
        while time.time() < deadline - 5:
            n_restarts += 1
            if n_restarts > 20:
                break

            # Structured initialization
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            W = np.zeros((pn, R))

            for c in range(R):
                if c < n * m * p:
                    i = c // (m * p)
                    j = (c // p) % m
                    k = c % p
                else:
                    i = random.randint(0, n - 1)
                    j = random.randint(0, m - 1)
                    k = random.randint(0, p - 1)

                U[i * m + j, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                V[j * p + k, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                W[k * n + i, c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)

                # Add off-diagonal for flexibility
                if random.random() < 0.4:
                    i2 = random.randint(0, n - 1)
                    j2 = random.randint(0, m - 1)
                    U[i2 * m + j2, c] = random.choice([-1, 1]) * random.uniform(0.1, 0.6)
                if random.random() < 0.4:
                    j2 = random.randint(0, m - 1)
                    k2 = random.randint(0, p - 1)
                    V[j2 * p + k2, c] = random.choice([-1, 1]) * random.uniform(0.1, 0.6)
                if random.random() < 0.4:
                    k2 = random.randint(0, p - 1)
                    i2 = random.randint(0, n - 1)
                    W[k2 * n + i2, c] = random.choice([-1, 1]) * random.uniform(0.1, 0.6)

            def obj_and_grad(x):
                Ux = x[:nm * R].reshape(nm, R)
                Vx = x[nm * R:nm * R + mp * R].reshape(mp, R)
                Wx = x[nm * R + mp * R:nm * R + mp * R + pn * R].reshape(pn, R)

                viol = np.zeros((n, m, p))
                dU = np.zeros((nm, R))
                dV = np.zeros((mp, R))
                dW = np.zeros((pn, R))

                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(R):
                                s += Ux[i * m + j, r] * Vx[j * p + k, r] * Wx[k * n + i, r]
                            viol[i, j, k] = s - 1.0
                            diff = 2.0 * viol[i, j, k]
                            for r in range(R):
                                dU[i * m + j, r] += diff * Vx[j * p + k, r] * Wx[k * n + i, r]
                                dV[j * p + k, r] += diff * Ux[i * m + j, r] * Wx[k * n + i, r]
                                dW[k * n + i, r] += diff * Ux[i * m + j, r] * Vx[j * p + k, r]

                fval = float(np.sum(viol * viol))
                gx = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return fval, gx

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            try:
                fval, gval = obj_and_grad(x0)
                # First pass: coarse optimization
                res = minimize(lambda x: obj_and_grad(x)[0], x0,
                               jac=lambda x: obj_and_grad(x)[1],
                               method='L-BFGS-B',
                               options={'maxiter': 150, 'maxfun': 5000,
                                        'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40})
                fval = res.fun

                if fval < 1e-6:
                    # Polish pass: high-precision refinement
                    res2 = minimize(lambda x: obj_and_grad(x)[0], res.x,
                                    jac=lambda x: obj_and_grad(x)[1],
                                    method='L-BFGS-B',
                                    options={'maxiter': 300, 'maxfun': 10000,
                                             'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 80})
                    fval = res2.fun
                    x_opt = res2.x
                else:
                    x_opt = res.x

                if fval < best_obj:
                    best_obj = fval
                    U_opt = x_opt[:nm * R].reshape(nm, R)
                    V_opt = x_opt[nm * R:nm * R + mp * R].reshape(mp, R)
                    W_opt = x_opt[nm * R + mp * R:nm * R + mp * R + pn * R].reshape(pn, R)
                    best_sol = (U_opt, V_opt, W_opt)
            except Exception:
                pass

            if best_obj < 1e-16:
                break

        if best_sol is not None and best_obj < 1e-10:
            U_opt, V_opt, W_opt = best_sol
            # Verify exactness
            max_viol = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(R):
                            s += U_opt[i * m + j, r] * V_opt[j * p + k, r] * W_opt[k * n + i, r]
                        max_viol = max(max_viol, abs(s - 1.0))

            if max_viol < 1e-12:
                U_list = [[float(U_opt[i, r]) for r in range(R)] for i in range(nm)]
                V_list = [[float(V_opt[i, r]) for r in range(R)] for i in range(mp)]
                W_list = [[float(W_opt[i, r]) for r in range(R)] for i in range(pn)]
                return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Try ranks from low to high
    # Known lower bounds: for 4x5*5x6, the Strassen-style bound gives ~57-72
    # Try a range of ranks
    ranks_to_try = [50, 55, 60, 64, 68, 72, 76, 80, 85, 90, 95, 100, 110, 119]

    for R in ranks_to_try:
        if time.time() > DEADLINE - 10:
            break
        time_left = DEADLINE - time.time() - 5
        if time_left < 5:
            break

        # Use polish version for lower ranks (harder), plain for higher
        if R <= 72:
            result = try_rank_with_polish(R, min(time_left, 40))
        else:
            result = try_rank(R, min(time_left, 35))

        if result is not None:
            best = result
            break

    return best
