import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit

    # Target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook fallback (rank 45)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1

    best_solution = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school

    # ============================================================
    # Strategy: Use the substitution method for tensor rank.
    # Fix U to a generic sparse form and solve for V and W.
    # ============================================================

    def try_substitution_method(rank):
        """
        Try to find a decomposition of given rank using the substitution method.
        We fix U to have a specific structure and solve for V, W.
        """
        if time.time() > deadline:
            return None

        # For each term r, we have a "support" pattern.
        # We'll try different random sparse patterns for U.
        # Key insight: For a valid decomposition, each (a,b) pair in U,V
        # that is non-zero must correspond to a (c) in W that completes the tensor entry.

        # Let's try a structured approach:
        # For 3x3x5, the tensor has 45 non-zero entries.
        # Each rank term covers at most min(#nonzeros in U_col, #nonzeros in V_col, #nonzeros in W_col)
        # product entries in the tensor.

        # Try various ranks from 44 down
        for target_rank in [rank, rank-1, rank-2, rank-3, rank-4, rank-5]:
            if target_rank < 40:
                break
            if time.time() > deadline - 5:
                break

            # For each target rank, try to find a valid decomposition
            # using random U patterns and solving the linear system

            for attempt in range(20):
                if time.time() > deadline - 5:
                    return None

                # Generate random sparse U matrix
                # Each column of U should have a small number of non-zeros
                # to make the problem tractable
                U = np.zeros((nm, target_rank))
                for r in range(target_rank):
                    # Choose 2-4 non-zero positions in each column
                    nnz = np.random.randint(2, 5)
                    positions = np.random.choice(nm, nnz, replace=False)
                    for pos in positions:
                        U[pos, r] = np.random.choice([-1, 0.5, 1, 1.5, 2])

                # For each column r, the non-zero entries in U determine which
                # (i,j) pairs are active. The corresponding (b,c) pairs in V,W
                # must satisfy the tensor equation.
                #
                # For a fixed r and fixed (a1, a2, ...) in U[:,r] non-zero:
                #   T[a1, b, c] = V[b,r] * W[c,r] for all b,c
                # This means V[:,r] and W[:,r] must be such that
                #   sum over a: U[a,r] * (outer product of V[:,r] and W[:,r]) matches T
                #
                # More precisely: for each (b,c), we need
                #   sum_a T[a,b,c] * ... 
                # Actually, the condition is:
                #   For all (a,b,c): T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                #
                # Fix U. Then for each r, define:
                #   S_r[b,c] = V[b,r] * W[c,r]
                # We need: T[a,b,c] = sum_r U[a,r] * S_r[b,c]
                #
                # This is a system of bilinear equations in V and W.
                # To make it linear, we can try to fix V as well, or use
                # the fact that S_r is a rank-1 matrix.
                #
                # Alternative: Fix U and V, solve for W.
                #   T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                #   For fixed a,b: T[a,b,:] = sum_r U[a,r]*V[b,r] * W[:,r]
                #   This is a linear system in W[:,r] for r where U[a,r]*V[b,r] != 0.
                #
                # Let's fix U and V randomly, then solve for W.

                V = np.zeros((mp, target_rank))
                for r in range(target_rank):
                    nnz = np.random.randint(2, 5)
                    positions = np.random.choice(mp, nnz, replace=False)
                    for pos in positions:
                        V[pos, r] = np.random.choice([-1, 0.5, 1, 1.5, 2])

                # Now solve for W.
                # For each c, and for each (a,b) where T[a,b,c] is non-zero:
                #   T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Let's build the linear system.
                # Unknowns: W[c, r] for c in range(pn), r in range(target_rank)
                # Total unknowns: pn * target_rank
                #
                # Equations: For each (a,b,c) with T[a,b,c] != 0:
                #   sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
                #
                # But this couples different c's? No! W[c,r] only appears
                # in equations where the output index is c.
                #
                # Actually, for a fixed c, the equation involves W[c,r] for all r.
                # And the coefficient of W[c,r] in the equation for (a,b,c) is U[a,r]*V[b,r].
                #
                # So for each c, we have a set of equations (one for each (a,b) with T[a,b,c]!=0)
                # in the unknowns W[c,0], W[c,1], ..., W[c,target_rank-1].
                #
                # The system for a fixed c is:
                #   sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]  for all (a,b) with T[a,b,c]!=0
                #
                # This is a linear system: M_c * W[c,:] = rhs_c
                # where M_c[(a,b), r] = U[a,r] * V[b,r] and rhs_c[(a,b)] = T[a,b,c]

                W = np.zeros((pn, target_rank))
                valid = True
                for c in range(pn):
                    # Find all (a,b) with T[a,b,c] != 0
                    pairs = []
                    rhs = []
                    for a in range(nm):
                        for b in range(mp):
                            if T[a,b,c] != 0:
                                pairs.append((a,b))
                                rhs.append(T[a,b,c])

                    if not pairs:
                        continue

                    # Build the linear system
                    # Number of equations = len(pairs)
                    # Number of unknowns = target_rank
                    if len(pairs) < target_rank:
                        # Underdetermined - try least squares
                        pass

                    A_mat = np.zeros((len(pairs), target_rank))
                    for idx, (a, b) in enumerate(pairs):
                        for r in range(target_rank):
                            A_mat[idx, r] = U[a, r] * V[b, r]

                    rhs_vec = np.array(rhs)

                    try:
                        # Solve using least squares
                        W_c, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, rhs_vec, rcond=None)
                        W[c, :] = W_c

                        # Check if the solution is exact
                        if len(residuals) > 0 and residuals[0] > 1e-10:
                            valid = False
                            break
                    except np.linalg.LinAlgError:
                        valid = False
                        break

                if not valid:
                    continue

                # Verify the full decomposition
                M_test = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                if np.allclose(M_test, T, atol=1e-8):
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2

                    M_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
                    if np.allclose(M_r, T, atol=1e-6):
                        return {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }, target_rank

        return None

    # Try the substitution method with rank 44, then 43, etc.
    for start_rank in [44, 43, 42]:
        if time.time() > deadline - 10:
            break

        result = try_substitution_method(start_rank)
        if result is not None:
            solution, achieved_rank = result
            if achieved_rank < best_rank:
                best_solution = solution
                best_rank = achieved_rank
                break  # Found a better solution, stop

    # Also try a direct optimization approach with fewer restarts but tighter tolerance
    if time.time() < deadline - 15:
        from scipy.optimize import minimize

        for target_rank in [best_rank - 1, best_rank - 2]:
            if target_rank < 40:
                break
            if time.time() > deadline - 10:
                break

            for restart in range(3):
                if time.time() > deadline - 8:
                    break

                # Initialize with the schoolbook solution and perturb
                U_init = np.zeros((nm, target_rank))
                V_init = np.zeros((mp, target_rank))
                W_init = np.zeros((pn, target_rank))

                # Copy first target_rank columns of schoolbook, with small perturbation
                for r in range(min(target_rank, rank_school)):
                    U_init[:, r] = U_school[:, r] + np.random.normal(0, 0.01, nm)
                    V_init[:, r] = V_school[:, r] + np.random.normal(0, 0.01, mp)
                    W_init[:, r] = W_school[:, r] + np.random.normal(0, 0.01, pn)

                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

                def error_func(params):
                    U_f = params[:nm * target_rank].reshape(nm, target_rank)
                    V_f = params[nm * target_rank:nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                    W_f = params[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)
                    M = np.einsum('ar,br,cr->abc', U_f, V_f, W_f, optimize=True)
                    return np.sum((M - T) ** 2)

                try:
                    res = minimize(error_func, params, method='L-BFGS-B',
                                   options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                except Exception:
                    continue

                if res.fun < 1e-10:
                    U_opt = res.x[:nm * target_rank].reshape(nm, target_rank)
                    V_opt = res.x[nm * target_rank:nm * target_rank + mp * target_rank].reshape(mp, target_rank)
                    W_opt = res.x[nm * target_rank + mp * target_rank:].reshape(pn, target_rank)

                    # Round to half-integers
                    U_r = np.round(U_opt * 2) / 2
                    V_r = np.round(V_opt * 2) / 2
                    W_r = np.round(W_opt * 2) / 2

                    M_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
                    if np.allclose(M_r, T, atol=1e-6):
                        best_solution = {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        best_rank = target_rank
                        break

    return best_solution
