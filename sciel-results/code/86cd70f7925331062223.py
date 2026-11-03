"""
Program for matrix multiplication 2x3 by 3x7.
Attempts to find a rank-minimal exact tensor decomposition.
Uses a structured peeling approach with exact rational arithmetic verification,
falling back to the schoolbook decomposition (rank 42) if no improvement is found.
"""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Build the target tensor T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
    # contributes to (i,k) entry of C
    # T_target[u, v, w] where u=i*m+j, v=j*p+k, w=k*n+i
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def check_decomposition_exact(U, V, W):
        """
        Check if the decomposition exactly matches the target tensor.
        U, V, W are lists of lists (integers or fractions).
        """
        # Convert to numpy for computation, but use exact check
        # Since entries should be integers/half-integers, we can multiply by 2 and check integers
        # Actually, let's just compute the tensor and check if it equals T_target
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        T_full = np.einsum('ur,vr,wr->uvw', U_np, V_np, W_np)
        return np.allclose(T_full, T_target, atol=1e-10)

    def check_decomposition_rational(U, V, W):
        """
        Exact check using fractions to avoid floating point issues.
        """
        # Compute each entry of the reconstructed tensor
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    total = Fraction(0, 1)
                    for r in range(len(U[0])):
                        total += Fraction(U[u][r]) * Fraction(V[v][r]) * Fraction(W[w][r])
                    expected = T_target[u, v, w]
                    if abs(float(total) - expected) > 1e-10:
                        return False
        return True

    def try_rank_peeling(T_res, max_rank, n_restarts=5, max_iter=30):
        """
        Peeling approach: iteratively find best rank-1 approximation to residual.
        Returns (U_list, V_list, W_list) or None if fails.
        """
        best_result = None
        for seed in range(n_restarts):
            if time.time() > deadline - 10.0:
                break
            np.random.seed(seed)
            residual = T_res.copy()
            U_list = []
            V_list = []
            W_list = []

            for rank_idx in range(max_rank):
                if time.time() > deadline - 10.0:
                    break

                # Find best rank-1 approx to residual via alternating maximization
                u = np.random.randn(nm)
                v = np.random.randn(mp)
                w = np.random.randn(pn)

                for it in range(max_iter):
                    # Update u
                    S_u = np.einsum('ijk,j,k->i', residual, v, w)
                    norm_u = np.linalg.norm(S_u)
                    if norm_u < 1e-15:
                        break
                    u_new = S_u / norm_u

                    # Update v
                    S_v = np.einsum('ijk,i,k->j', residual, u_new, w)
                    norm_v = np.linalg.norm(S_v)
                    if norm_v < 1e-15:
                        break
                    v_new = S_v / norm_v

                    # Update w
                    S_w = np.einsum('ijk,i,j->k', residual, u_new, v_new)
                    norm_w = np.linalg.norm(S_w)
                    if norm_w < 1e-15:
                        break
                    w_new = S_w / norm_w

                    # Check convergence
                    cur_norm = np.linalg.norm(u_new - u) + np.linalg.norm(v_new - v) + np.linalg.norm(w_new - w)
                    u, v, w = u_new, v_new, w_new

                    if cur_norm < 1e-12:
                        break

                # Compute coefficient
                alpha = np.einsum('ijk,i,j,k', residual, u, v, w)
                if abs(alpha) < 1e-12:
                    break

                residual = residual - alpha * np.einsum('i,j,k->ijk', u, v, w)
                U_list.append(alpha * u)
                V_list.append(v)
                W_list.append(w)

                # Check if residual is zero
                if np.linalg.norm(residual) < 1e-10:
                    break

            R = len(U_list)
            if R > max_rank:
                continue

            # Verify with floating point first
            U = np.column_stack(U_list)
            V = np.column_stack(V_list)
            W = np.column_stack(W_list)
            T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
            if np.allclose(T_full, T_target, atol=1e-6):
                # Try to convert to exact rational representation
                # Check if we can represent with small denominators
                U_frac = []
                V_frac = []
                W_frac = []
                success = True
                for row in U:
                    U_frac.append([Fraction(x).limit_denominator(1000) for x in row])
                for row in V:
                    V_frac.append([Fraction(x).limit_denominator(1000) for x in row])
                for row in W:
                    W_frac.append([Fraction(x).limit_denominator(1000) for x in row])

                if check_decomposition_rational(U_frac, V_frac, W_frac):
                    # Convert fractions to floats for return
                    U_ret = [[float(f) for f in row] for row in U_frac]
                    V_ret = [[float(f) for f in row] for row in V_frac]
                    W_ret = [[float(f) for f in row] for row in W_frac]
                    if best_result is None or R < best_result[0]:
                        best_result = (R, U_ret, V_ret, W_ret)

        if best_result:
            return best_result[1], best_result[2], best_result[3]
        return None

    # Try to find a decomposition of rank < 42
    # Lower bound is max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Known lower bound for (2,3,7) is 40, so try ranks 40, 41
    
    best_decomp = None
    for target_rank in [40, 41]:
        if time.time() > deadline - 20.0:
            break
        result = try_rank_peeling(T_target, target_rank, n_restarts=10, max_iter=50)
        if result:
            U, V, W = result
            best_decomp = (target_rank, U, V, W)
            break

    if best_decomp:
        rank, U, V, W = best_decomp
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook decomposition
        return get_schoolbook()
