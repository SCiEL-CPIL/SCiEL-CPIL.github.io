"""Matrix multiplication 2,2,14 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

_DIMS = (2, 2, 14)
_T0 = time.time()
_HARD_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 * 2x14 matrix multiplication.

    Strategy:
    1. Build the direct-sum of 7 Strassen rank-7 blocks -> rank 49.
    2. Attempt to find a lower-rank decomposition via randomized search
       (Bini-Capovani-Lotti-Rosenberg style) within the time budget.
    3. Return the best valid decomposition found.
    """
    n, m, p = _DIMS
    nm = n * m       # 4
    mp = m * p       # 28
    pn = p * n       # 28

    def _schoolbook():
        """Return the trivial rank-56 decomposition."""
        rank = n * m * p  # 56
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
        return U, V, W, rank

    def _strassen_blocks():
        """Return the rank-49 direct-sum-of-Strassen decomposition."""
        num_blocks = p // 2  # 7
        rank = num_blocks * 7  # 49
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))

        # Strassen rank-7 terms for 2x2 * 2x2
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        # B indices: 0=B00, 1=B01, 2=B10, 3=B11
        # C indices: 0=C00, 1=C01, 2=C10, 3=C11
        st = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]

        for t in range(num_blocks):
            for s in range(7):
                col = t * 7 + s
                a_c, b_c, c_c = st[s]
                for a_idx, coeff in a_c.items():
                    a_i, a_j = a_idx // m, a_idx % m
                    U[a_i * m + a_j, col] = coeff
                for b_idx, coeff in b_c.items():
                    b_j, b_col = b_idx // 2, b_idx % 2
                    b_k = 2 * t + b_col
                    V[b_j * p + b_k, col] = coeff
                for c_idx, coeff in c_c.items():
                    c_i, c_col = c_idx // 2, c_idx % 2
                    c_k = 2 * t + c_col
                    W[c_k * n + c_i, col] = coeff

        return U, V, W, rank

    def _verify(Uf, Vf, Wf, tol=1e-10):
        """Check that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all (i,j,k)."""
        # Vectorized check
        # For each (i,j,k): dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:]) == 1
        for i in range(n):
            for j in range(m):
                u_row = Uf[i * m + j, :]
                for k in range(p):
                    v_row = Vf[j * p + k, :]
                    w_row = Wf[k * n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def _try_lower_rank(target_rank, time_budget):
        """
        Attempt to find a rank-target_rank decomposition using randomized
        tensor decomposition search. Returns (U, V, W) or None.
        """
        if time.time() - _T0 > _HARD_LIMIT - 10.0:
            return None

        rng = np.random.RandomState(42)

        # We look for U (nm x r), V (mp x r), W (pn x r) such that
        # for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

        # Use a gradient-free approach: parameterize and use random restarts
        # with local refinement.

        best_error = float('inf')
        best_U, best_V, best_W = None, None, None

        max_restarts = 50
        max_iter = 200
        lr = 0.01

        for restart in range(max_restarts):
            if time.time() - _T0 > _HARD_LIMIT - 10.0:
                break

            # Random initialization
            U = rng.randn(nm, target_rank) * 0.5
            V = rng.randn(mp, target_rank) * 0.5
            W = rng.randn(pn, target_rank) * 0.5

            # Local refinement using coordinate descent / gradient steps
            for it in range(max_iter):
                if time.time() - _T0 > _HARD_LIMIT - 10.0:
                    break

                # Compute gradient w.r.t. U, V, W
                # Objective: sum_{i,j,k} (1 - sum_r U[i,j,r]*V[j,k,r]*W[k,i,r])^2
                # We do a simple coordinate update

                # Update U
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        # Gradient for U[u_idx, :]
                        grad_u = np.zeros(target_rank)
                        for k in range(p):
                            v_row = V[j * p + k, :]
                            w_row = W[k * n + i, :]
                            val = np.dot(U[u_idx, :], v_row * w_row)
                            err = 1.0 - val
                            grad_u += 2 * err * v_row * w_row
                        U[u_idx, :] -= lr * grad_u

                # Update V
                for j in range(m):
                    for k in range(p):
                        v_idx = j * p + k
                        grad_v = np.zeros(target_rank)
                        for i in range(n):
                            u_row = U[i * m + j, :]
                            w_row = W[k * n + i, :]
                            val = np.dot(u_row, V[v_idx, :] * w_row)
                            err = 1.0 - val
                            grad_v += 2 * err * u_row * w_row
                        V[v_idx, :] -= lr * grad_v

                # Update W
                for k in range(p):
                    for i in range(n):
                        w_idx = k * n + i
                        grad_w = np.zeros(target_rank)
                        for j in range(m):
                            u_row = U[i * m + j, :]
                            v_row = V[j * p + k, :]
                            val = np.dot(u_row, v_row * W[w_idx, :])
                            err = 1.0 - val
                            grad_w += 2 * err * u_row * v_row
                        W[w_idx, :] -= lr * grad_w

                # Periodic check
                if it % 20 == 0:
                    err = _max_error(U, V, W)
                    if err < best_error:
                        best_error = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_error < 1e-8:
                        # Try to snap to integers/half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        if _verify(U_snap, V_snap, W_snap, tol=1e-8):
                            return U_snap, V_snap, W_snap
                        if _verify(U, V, W, tol=1e-10):
                            return U, V, W

        # Final check
        if best_U is not None and _verify(best_U, best_V, best_W, tol=1e-8):
            return best_U, best_V, best_W
        return None

    def _max_error(Uf, Vf, Wf):
        """Compute max absolute error over all (i,j,k)."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = Uf[i * m + j, :]
                for k in range(p):
                    v_row = Vf[j * p + k, :]
                    w_row = Wf[k * n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def _to_list(mat):
        """Convert numpy matrix to plain list with integer entries where possible."""
        result = []
        for row in mat:
            new_row = []
            for x in row:
                rx = round(x)
                if abs(x - rx) < 1e-9:
                    new_row.append(int(rx))
                else:
                    # Check for half-integer
                    rh = round(x * 2) / 2.0
                    if abs(x - rh) < 1e-9:
                        new_row.append(float(rh))
                    else:
                        new_row.append(float(x))
            result.append(new_row)
        return result

    # Start with the known rank-49 solution
    U49, V49, W49, rank49 = _strassen_blocks()
    if _verify(U49, V49, W49):
        best_U, best_V, best_W, best_rank = U49, V49, W49, rank49
    else:
        # Fallback to schoolbook
        U49, V49, W49, rank49 = _schoolbook()
        best_U, best_V, best_W, best_rank = U49, V49, W49, rank49

    # Try to find lower rank
    # Theoretically, the rank of 2x2 * 2x14 is at least 7*7=49 by tensor rank
    # submultiplicativity, but it could be lower if there's cross-block sharing.
    # Let's try ranks from 48 down to 40 if time permits.
    for target_rank in range(48, 39, -1):
        if time.time() - _T0 > _HARD_LIMIT - 30.0:
            break
        result = _try_lower_rank(target_rank, 20.0)
        if result is not None:
            Uf, Vf, Wf = result
            if _verify(Uf, Vf, Wf, tol=1e-8):
                best_U, best_V, best_W, best_rank = Uf, Vf, Wf, target_rank
                break  # Found a lower rank, stop

    # Final verification
    if not _verify(best_U, best_V, best_W, tol=1e-8):
        # Emergency fallback
        Us, Vs, Ws, rs = _schoolbook()
        best_U, best_V, best_W, best_rank = Us, Vs, Ws, rs

    return {"U": _to_list(best_U), "V": _to_list(best_V), "W": _to_list(best_W)}

# EVOLVE-BLOCK-END
