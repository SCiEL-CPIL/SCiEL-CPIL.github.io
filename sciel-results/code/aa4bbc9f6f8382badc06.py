import numpy as np
import time

DIMENSIONS = [2, 3, 5]
_DEADLINE = 150.0
_START = time.monotonic()

def _time_left():
    return _DEADLINE - (time.monotonic() - _START)

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10

    # Build the target tensor T with T[I,J,K] = 1 iff (I,J,K) corresponds to (i,j,k)
    # I = i*m + j, J = j*p + k, K = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0

    def build_tensor(U, V, W):
        r = U.shape[1]
        T_approx = np.zeros_like(T)
        for c in range(r):
            T_approx += (U[:, c, None, None] *
                         V[:, None, c, None] *
                         W[:, None, None, c])
        return T_approx

    def verify(U, V, W, tol=1e-9):
        return np.allclose(build_tensor(U, V, W), T, atol=tol)

    # ---- Rank-28 decomposition (Strassen on two 2x2x2 blocks + schoolbook rest) ----
    terms = []

    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))

    def strassen_block(i0, i1, j0, j1, k0, k1):
        a_I = i0 * m + j0
        b_I = i0 * m + j1
        c_I = i1 * m + j0
        d_I = i1 * m + j1
        e_J = j0 * p + k0
        f_J = j0 * p + k1
        g_J = j1 * p + k0
        h_J = j1 * p + k1
        C00_K = k0 * n + i0
        C10_K = k0 * n + i1
        C01_K = k1 * n + i0
        C11_K = k1 * n + i1

        add_term({a_I: 1, d_I: 1}, {e_J: 1, h_J: 1}, {C00_K: 1, C11_K: 1})
        add_term({c_I: 1, d_I: 1}, {e_J: 1}, {C10_K: 1})
        add_term({a_I: 1}, {f_J: 1, h_J: -1}, {C01_K: 1})
        add_term({d_I: 1}, {g_J: 1, e_J: -1}, {C00_K: 1, C10_K: 1})
        add_term({a_I: 1, b_I: 1}, {h_J: 1}, {C00_K: -1, C01_K: 1})
        add_term({c_I: 1, a_I: -1}, {e_J: 1, f_J: 1}, {C11_K: 1})
        add_term({b_I: 1, d_I: -1}, {g_J: 1, h_J: 1}, {C00_K: 1})

    def schoolbook_term(i, j, k):
        I = i * m + j
        J = j * p + k
        K = k * n + i
        add_term({I: 1}, {J: 1}, {K: 1})

    strassen_block(0, 1, 0, 1, 0, 1)
    strassen_block(0, 1, 0, 1, 2, 3)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i in (0, 1)) and (j in (0, 1)) and (k in (0, 1, 2, 3)):
                    continue
                schoolbook_term(i, j, k)

    # Convert to dense
    rank = len(terms)
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))

    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            U[idx, c] = val
        for idx, val in v_dict.items():
            V[idx, c] = val
        for idx, val in w_dict.items():
            W[idx, c] = val

    if not verify(U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U, V, W
    best_rank = rank

    # ---- Attempt to find rank-27 via structured search ----
    # We use the matricization lower bound to guide the search.
    # The rank of any matricization is a lower bound on tensor rank.
    # For T as (dim_u, dim_v, dim_w), the U-V matricization has shape (dim_u, dim_v * dim_w) = (6, 150)
    # The V-W matricization has shape (dim_v, dim_w * dim_u) = (15, 60)
    # The W-U matricization has shape (dim_w, dim_u * dim_v) = (10, 90)
    # These ranks are 6, 15, and 10 respectively, so the lower bound is max(6, 15, 10) = 15.
    # This doesn't rule out rank 27, so we attempt a search.

    if _time_left() > 60:
        # Try a structured perturbation of the rank-28 solution
        # to see if we can merge two terms.
        # We look for pairs of terms that can be combined.
        for _ in range(3):
            if _time_left() < 30:
                break
            # Random perturbation
            np.random.seed(np.random.randint(0, 1000000))
            U_try = U.copy()
            V_try = V.copy()
            W_try = W.copy()

            # Try to merge two random terms
            c1, c2 = np.random.choice(rank, 2, replace=False)
            # Create a combined term
            U_new = (U_try[:, c1] + U_try[:, c2]) / 2
            V_new = (V_try[:, c1] + V_try[:, c2]) / 2
            W_new = (W_try[:, c1] + W_try[:, c2]) / 2

            # Remove the two terms
            mask = np.ones(rank, dtype=bool)
            mask[c1] = False
            mask[c2] = False
            U_reduced = U_try[:, mask]
            V_reduced = V_try[:, mask]
            W_reduced = W_try[:, mask]

            # Add the new term
            U_trial = np.hstack([U_reduced, U_new.reshape(-1, 1)])
            V_trial = np.hstack([V_reduced, V_new.reshape(-1, 1)])
            W_trial = np.hstack([W_reduced, W_new.reshape(-1, 1)])

            if verify(U_trial, V_trial, W_trial, tol=1e-8):
                best_U, best_V, best_W = U_trial, V_trial, W_trial
                best_rank = rank - 1
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
