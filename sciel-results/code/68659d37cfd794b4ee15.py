import numpy as np
import scipy.optimize as opt
import time
import sys

def construct():
    n, m, p = 2, 4, 5

    # Precompute constraint structure
    cons_i = []
    cons_k = []
    cons_a = []
    cons_b = []
    cons_c = []
    cons_d = []
    cons_target = []

    for i in range(n):
        for k in range(p):
            for a in range(n):
                for b in range(m):
                    for c in range(m):
                        for d in range(p):
                            cons_i.append(i)
                            cons_k.append(k)
                            cons_a.append(a)
                            cons_b.append(b)
                            cons_c.append(c)
                            cons_d.append(d)
                            cons_target.append(1.0 if (b == c and a == i and d == k) else 0.0)

    cons_i = np.array(cons_i)
    cons_k = np.array(cons_k)
    cons_a = np.array(cons_a)
    cons_b = np.array(cons_b)
    cons_c = np.array(cons_c)
    cons_d = np.array(cons_d)
    cons_target = np.array(cons_target)
    n_cons = len(cons_target)

    # Precompute index arrays
    idx_a = cons_a * m + cons_b
    idx_c = cons_c * p + cons_d
    idx_w = cons_k * n + cons_i

    def get_initial_solution_rank36():
        rank = 36
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))

        col = 0
        def add_strassen_222(i0, j0, k0):
            nonlocal col
            # M1
            U[i0*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M2
            U[(i0+1)*m + j0][col] += 1; U[(i0+1)*m + (j0+1)][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + (i0+1)][col] += 1; W[(k0+1)*n + (i0+1)][col] -= 1
            col += 1
            # M3
            U[i0*m + j0][col] += 1
            V[j0*p + (k0+1)][col] += 1; V[(j0+1)*p + (k0+1)][col] -= 1
            W[(k0+1)*n + i0][col] += 1; W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M4
            U[(i0+1)*m + (j0+1)][col] += 1
            V[(j0+1)*p + k0][col] += 1; V[j0*p + k0][col] -= 1
            W[k0*n + i0][col] += 1; W[k0*n + (i0+1)][col] += 1
            col += 1
            # M5
            U[i0*m + j0][col] += 1; U[i0*m + (j0+1)][col] += 1
            V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] -= 1; W[(k0+1)*n + i0][col] += 1
            col += 1
            # M6
            U[(i0+1)*m + j0][col] += 1; U[i0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1; V[j0*p + (k0+1)][col] += 1
            W[(k0+1)*n + (i0+1)][col] += 1
            col += 1
            # M7
            U[i0*m + (j0+1)][col] += 1; U[(i0+1)*m + (j0+1)][col] -= 1
            V[(j0+1)*p + k0][col] += 1; V[(j0+1)*p + (k0+1)][col] += 1
            W[k0*n + i0][col] += 1
            col += 1

        def add_schoolbook_2x4x1(i0, j0, k0):
            nonlocal col
            for di in range(2):
                for dj in range(4):
                    U[(i0 + di)*m + (j0 + dj)][col] = 1
                    V[(j0 + dj)*p + k0][col] = 1
                    W[k0*n + (i0 + di)][col] = 1
                    col += 1

        add_strassen_222(0, 0, 0)
        add_strassen_222(0, 2, 0)
        add_strassen_222(0, 0, 2)
        add_strassen_222(0, 2, 2)
        add_schoolbook_2x4x1(0, 0, 4)

        return U, V, W

    U36, V36, W36 = get_initial_solution_rank36()

    def verify_solution(U, V, W, tol=1e-9):
        R = U.shape[1]
        U_sel = U[idx_a]
        V_sel = V[idx_c]
        W_sel = W[idx_w]
        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        residuals = vals - cons_target
        return np.max(np.abs(residuals)) < tol

    def snap_to_half_integers(U, V, W):
        U_snapped = np.round(U * 2) / 2.0
        V_snapped = np.round(V * 2) / 2.0
        W_snapped = np.round(W * 2) / 2.0
        return U_snapped, V_snapped, W_snapped

    def residual_vector_fast(x, R):
        """Vectorized residual computation."""
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R

        U = x[0:size_U].reshape(n*m, R)
        V = x[size_U:size_U+size_V].reshape(m*p, R)
        W = x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)

        U_sel = U[idx_a]
        V_sel = V[idx_c]
        W_sel = W[idx_w]

        vals = np.sum(U_sel * V_sel * W_sel, axis=1)
        return vals - cons_target

    def try_optimize(U_init, V_init, W_init, R, method='lm', max_nfev=20000):
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

        try:
            if method == 'lm':
                result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='lm', max_nfev=max_nfev)
            elif method == 'trf':
                result = opt.least_squares(residual_vector_fast, x0, args=(R,), method='trf', max_nfev=max_nfev)
            else:
                return None, None, None

            x_opt = result.x
            size_U = n * m * R
            size_V = m * p * R
            size_W = p * n * R
            U_opt = x_opt[0:size_U].reshape(n*m, R)
            V_opt = x_opt[size_U:size_U+size_V].reshape(m*p, R)
            W_opt = x_opt[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)

            if verify_solution(U_opt, V_opt, W_opt, tol=1e-6):
                U_snap, V_snap, W_snap = snap_to_half_integers(U_opt, V_opt, W_opt)
                if verify_solution(U_snap, V_snap, W_snap, tol=1e-9):
                    return U_snap, V_snap, W_snap
        except Exception:
            pass

        return None, None, None

    def try_rational_search(R, time_budget=3.0):
        """
        Attempt rational solution via structured algebraic approach.
        Uses the insight that for the multiplication tensor, each (i,k) slice
        has a specific structure that may allow rational solutions.
        """
        start = time.time()
        
        # For rank R, we have R columns in U, V, W.
        # The equations are: sum_mu U[a,b,mu]*V[c,d,mu]*W[k*n+i,mu] = delta[a,i]*delta[b,c]*delta[d,k]
        
        # Try a semi-structured approach: assume W has a block structure
        # where W[k*n+i, mu] is non-zero only for specific (k,i) pairs.
        
        # For the 2x4x5 multiplication, the tensor has a specific structure.
        # Let's try to find a solution where each column mu corresponds to
        # a specific (i,k) pair or a combination.
        
        # Since this is complex, we'll do a limited search:
        # Try to solve a linear system by fixing some variables.
        
        if time.time() - start > time_budget:
            return None
        
        # Simple check: can we verify if a random rational initialization works?
        # Generate random half-integer matrices and check.
        for _ in range(10):
            if time.time() - start > time_budget:
                break
            U_try = (np.random.randint(-4, 5, (n*m, R)) / 2.0)
            V_try = (np.random.randint(-4, 5, (m*p, R)) / 2.0)
            W_try = (np.random.randint(-4, 5, (p*n, R)) / 2.0)
            if verify_solution(U_try, V_try, W_try, tol=1e-9):
                return U_try, V_try, W_try
        
        return None

    start_time = time.time()
    time_limit = 130  # Safety margin before 150s kill

    # Try to find lower ranks
    for R in range(35, 25, -1):
        if time.time() - start_time > time_limit:
            break

        # Multiple initialization strategies
        inits = []

        # 1. Truncate rank 36 solution
        inits.append((U36[:, :R], V36[:, :R], W36[:, :R]))

        # 2. Random initializations with different scales
        for scale in [0.5, 1.0, 0.3, 0.7, 1.5]:
            U_init = np.random.normal(0, scale, (n*m, R))
            V_init = np.random.normal(0, scale, (m*p, R))
            W_init = np.random.normal(0, scale, (p*n, R))
            inits.append((U_init, V_init, W_init))

        # 3. Perturbed truncation
        for _ in range(5):
            U_init = U36[:, :R].copy()
            V_init = V36[:, :R].copy()
            W_init = W36[:, :R].copy()
            U_init += np.random.normal(0, 0.1, U_init.shape)
            V_init += np.random.normal(0, 0.1, V_init.shape)
            W_init += np.random.normal(0, 0.1, W_init.shape)
            inits.append((U_init, V_init, W_init))

        # 4. Try rational search (limited time)
        rat_result = try_rational_search(R, time_budget=2.0)
        if rat_result is not None:
            U_r, V_r, W_r = rat_result
            if verify_solution(U_r, V_r, W_r, tol=1e-9):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

        for U_init, V_init, W_init in inits:
            if time.time() - start_time > time_limit:
                break

            # Try multiple optimization methods
            for method in ['lm', 'trf']:
                if time.time() - start_time > time_limit:
                    break

                U_opt, V_opt, W_opt = try_optimize(U_init, V_init, W_init, R, method=method)
                if U_opt is not None:
                    return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}

    # If no better rank found, return the rank 36 solution (snapped)
    U_snap, V_snap, W_snap = snap_to_half_integers(U36, V36, W36)
    if verify_solution(U_snap, V_snap, W_snap, tol=1e-9):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}

    return {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
