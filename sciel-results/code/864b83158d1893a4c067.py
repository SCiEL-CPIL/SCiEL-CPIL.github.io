import numpy as np
import time
import itertools

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 3
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    # Target tensor: T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    # U is (n*m, r) = (4, r), V is (m*p, r) = (6, r), W is (p*n, r) = (6, r)

    def build_target():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T_target = build_target()

    def verify(U, V, W):
        """Check if U,V,W exactly decompose the tensor."""
        r = U.shape[1]
        T = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_target, atol=1e-9)

    def try_rank(r):
        """Try to find a rank-r decomposition with entries in {0, ±1, ±1/2}."""
        values = [0, 1, -1, 0.5, -0.5]

        # We need to find U (4xr), V (6xr), W (6xr) such that
        # sum_s U[a,s]*V[b,s]*W[c,s] = T[a,b,c] for all a,b,c

        # Strategy: enumerate structured patterns for U and V, then solve for W
        # For each c (0..5), the equation is linear in W[c,:]:
        # sum_s (U[a,s]*V[b,s]) * W[c,s] = T[a,b,c]
        # This gives a (4*6) x r linear system for W[c,:]

        # Generate candidate U and V matrices with sparse structure
        # Key insight: for rank 10, we can use a structured approach

        # Let's try a direct construction approach
        # The 12 non-zero entries of T are at:
        # (i*m+j, j*p+k, k*n+i) for i in {0,1}, j in {0,1}, k in {0,1,2}

        # Non-zero positions:
        nonzero_positions = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    nonzero_positions.append((a, b, c))

        # For rank r, we have r unknowns in each column of W
        # For each c, we need to solve: A_c * W[c,:] = t_c
        # where A_c[a*6+b, s] = U[a,s]*V[b,s]
        # and t_c[a*6+b] = T[a,b,c]

        # Build the full linear system for all c simultaneously
        # Total unknowns: 6*r (for W)
        # Total equations: 36 (4*6 combinations of a,b) but only 12 are non-zero
        # However, we need ALL 36 equations to be satisfied (12 = 1, 24 = 0)

        # Let's use a different approach: random search with verification
        # Generate random sparse U and V, then solve for W and verify

        import random

        # Use a seeded random for reproducibility
        rng = random.Random(42)

        best_residual = float('inf')
        best_result = None

        # For efficiency, precompute which (a,b) pairs correspond to which c
        ab_to_c = {}
        for (a, b, c) in nonzero_positions:
            ab_to_c[(a, b)] = c

        # Generate candidate decompositions
        for attempt in range(50000):
            if time.time() > deadline - 15:
                break

            # Generate sparse U and V
            U = np.zeros((4, r))
            V = np.zeros((6, r))

            for s in range(r):
                # U[:,s]: 1-2 non-zero entries
                n_nz_u = rng.choice([1, 1, 2, 2])
                rows_u = rng.sample(range(4), n_nz_u)
                for row in rows_u:
                    U[row, s] = rng.choice([1, -1, 0.5, -0.5])

                # V[:,s]: 1-2 non-zero entries
                n_nz_v = rng.choice([1, 1, 2, 2])
                rows_v = rng.sample(range(6), n_nz_v)
                for row in rows_v:
                    V[row, s] = rng.choice([1, -1, 0.5, -0.5])

            # Solve for W
            W = np.zeros((6, r))
            valid = True

            for c in range(6):
                # Build linear system: A_c * W[c,:] = t_c
                A_c = np.zeros((24, r))
                t_c = np.zeros(24)

                for a in range(4):
                    for b in range(6):
                        idx = a * 6 + b
                        A_c[idx, :] = U[a, :] * V[b, :]
                        if (a, b) in ab_to_c and ab_to_c[(a, b)] == c:
                            t_c[idx] = 1.0
                        else:
                            t_c[idx] = 0.0

                # Solve least squares
                try:
                    sol, residuals, rank_sol, sv = np.linalg.lstsq(A_c, t_c, rcond=None)
                    residual = np.linalg.norm(A_c @ sol - t_c)
                    if residual > 1e-9:
                        valid = False
                        break
                    W[c, :] = sol
                except:
                    valid = False
                    break

            if valid:
                # Verify full decomposition
                T = np.einsum('ar,br,cr->abc', U, V, W)
                residual = np.linalg.norm(T - T_target)
                if residual < 1e-9:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                elif residual < best_residual:
                    best_residual = residual
                    best_result = (U.copy(), V.copy(), W.copy())

        if best_result is not None and best_residual < 1e-6:
            U, V, W = best_result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return None

    # Try rank 10 first (known minimal rank for M(2,2,3))
    for rank in [10, 11, 12]:
        if time.time() > deadline - 20:
            break
        result = try_rank(rank)
        if result is not None:
            return result

    # Fallback: return schoolbook (rank 12)
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    return {"U": U_sb, "V": V_sb, "W": W_sb}
