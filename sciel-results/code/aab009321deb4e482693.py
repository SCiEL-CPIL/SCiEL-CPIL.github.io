import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6

    # Build the target tensor T[a,b,c] = 1 iff consistent
    T = np.zeros((n_a, n_b, n_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition using einsum."""
        rank = U.shape[1]
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return bool(np.allclose(recon, T, atol=tol))

    def snap_half(X):
        return np.round(X * 2) / 2.0

    # Build the known rank-17 solution
    def build_rank17():
        rank = 17
        U = np.zeros((n_a, rank))
        V = np.zeros((n_b, rank))
        W = np.zeros((n_c, rank))
        # Strassen for 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1})
        # a0=A[0,0], a1=A[0,1], a2=A[0,2], a3=A[1,0], a4=A[1,1], a5=A[1,2]
        # b0=B[0,0], b1=B[0,1], b2=B[0,2], b3=B[1,0], b4=B[1,1], b5=B[1,2], b6=B[2,0], b7=B[2,1], b8=B[2,2]
        # c0=C[0,0], c1=C[1,0], c2=C[0,1], c3=C[1,1], c4=C[0,2], c5=C[1,2]
        # M1 = (a0+a4)(b0+b4) -> c0, c3
        U[0, 0] = 1; U[4, 0] = 1; V[0, 0] = 1; V[4, 0] = 1; W[0, 0] = 1; W[3, 0] = 1
        # M2 = (a3+a4)b0 -> c1, c3
        U[3, 1] = 1; U[4, 1] = 1; V[0, 1] = 1; W[1, 1] = 1; W[3, 1] = -1
        # M3 = a0(b1-b4) -> c2, c3
        U[0, 2] = 1; V[1, 2] = 1; V[4, 2] = -1; W[2, 2] = 1; W[3, 2] = 1
        # M4 = a4(b3-b0) -> c0, c1
        U[4, 3] = 1; V[3, 3] = 1; V[0, 3] = -1; W[0, 3] = 1; W[1, 3] = 1
        # M5 = (a0+a1)b4 -> c0, c2
        U[0, 4] = 1; U[1, 4] = 1; V[4, 4] = 1; W[0, 4] = -1; W[2, 4] = 1
        # M6 = (a3-a0)(b0+b1) -> c3
        U[3, 5] = 1; U[0, 5] = -1; V[0, 5] = 1; V[1, 5] = 1; W[3, 5] = 1
        # M7 = (a1-a4)(b3+b4) -> c0
        U[1, 6] = 1; U[4, 6] = -1; V[3, 6] = 1; V[4, 6] = 1; W[0, 6] = 1
        # Schoolbook for remaining 10 terms
        # k=2, j in {0,1}: C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
        U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
        U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
        U[3, 9] = 1; V[2, 9] = 1; W[5, 9] = 1
        U[4, 10] = 1; V[5, 10] = 1; W[5, 10] = 1
        # j=2: C[i,k] = A[i,2]*B[2,k]
        U[2, 11] = 1; V[6, 11] = 1; W[0, 11] = 1
        U[2, 12] = 1; V[7, 12] = 1; W[2, 12] = 1
        U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
        U[5, 14] = 1; V[6, 14] = 1; W[1, 14] = 1
        U[5, 15] = 1; V[7, 15] = 1; W[3, 15] = 1
        U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
        return U, V, W

    U17, V17, W17 = build_rank17()
    if not verify(U17, V17, W17):
        raise ValueError("Rank-17 verification failed")

    best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    best_rank = 17

    # Fast SVD-based rank check: if tensor rank > 16, we can't do better than 17
    # (This is a lower bound; the true rank might be higher, but 17 is achievable)
    try:
        T_flat = T.reshape(n_a, n_b * n_c)
        sv = np.linalg.svd(T_flat, compute_uv=False)
        tensor_rank = int(np.sum(sv > 1e-10))
    except Exception:
        tensor_rank = 17

    # If tensor rank is 17 or more, rank 16 is impossible (tensor rank is a lower bound)
    if tensor_rank <= 16:
        # Try to find a rank-16 decomposition via exact linear solve
        # Strategy: try removing each column, then solve for a small set of free variables
        def try_rank16_remove_col(col, deadline):
            """Remove column `col` and try to solve for a rank-16 solution."""
            U16 = np.delete(U17, col, axis=1)
            V16 = np.delete(V17, col, axis=1)
            W16 = np.delete(W17, col, axis=1)

            # The removed column contributed to the tensor. We need to adjust the
            # remaining 16 columns to compensate. We'll try to find a small perturbation
            # to the remaining columns that fixes the tensor.
            #
            # The tensor equation is: sum_r U[:,r] V[:,r] W[:,r] = T
            # If we remove column `col`, the residual is:
            # R = T - sum_{r != col} U[:,r] V[:,r] W[:,r]
            # We need to find a rank-1 decomposition of R using the remaining columns,
            # i.e., find scalars or a new column that, when added, makes the sum = T.
            #
            # Actually, a simpler approach: we want to modify the 16 remaining columns
            # slightly so that their sum equals T. Let's set up a linear system.
            #
            # For each (a,b,c), we need:
            # sum_{r=0}^{15} U16[a,r] * V16[b,r] * W16[c,r] = T[a,b,c]
            #
            # This is a system of 306 equations in 16*21 = 336 unknowns.
            # We can try to solve it by fixing most variables and solving for a few free ones.
            #
            # Let's try: fix all U and V entries, and solve for W entries.
            # For each (a,b,c): sum_r U16[a,r] * V16[b,r] * W16[c,r] = T[a,b,c]
            # This is linear in W16. For fixed (a,b), the equation is:
            # sum_r (U16[a,r] * V16[b,r]) * W16[c,r] = T[a,b,c]
            # This is a linear system in W16[:,r] for r=0..15.
            #
            # Actually, for each c, W16[c, :] is a row of length 16.
            # The equation for (a,b,c) is: sum_r coeff[a,b,r] * W16[c,r] = T[a,b,c]
            # where coeff[a,b,r] = U16[a,r] * V16[b,r].
            #
            # This is a system of (n_a * n_b) = 54 equations for each c, with 16 unknowns.
            # That's overdetermined. Let's try a least-squares approach.
            #
            # For each c, we have 54 equations: A_c * W16[c, :] = t_c
            # where A_c[a,b,r] = U16[a,r] * V16[b,r], t_c[a,b] = T[a,b,c].
            #
            # Let's solve this least-squares problem.

            # Build the coefficient matrix for each c
            # A_c has shape (54, 16), t_c has shape (54,)
            A_list = []
            t_list = []
            for c in range(n_c):
                A_c = np.zeros((n_a * n_b, 16))
                t_c = np.zeros(n_a * n_b)
                for idx, (a, b) in enumerate(product(range(n_a), range(n_b))):
                    A_c[idx, :] = U16[a, :] * V16[b, :]
                    t_c[idx] = T[a, b, c]
                A_list.append(A_c)
                t_list.append(t_c)

            # Solve least squares for each c
            W16_new = np.zeros((n_c, 16))
            for c in range(n_c):
                try:
                    sol, residuals, rank_c, sv = np.linalg.lstsq(A_list[c], t_list[c], rcond=None)
                    W16_new[c, :] = sol
                except Exception:
                    W16_new[c, :] = W16[c, :]

            # Now check if this W16_new works with the original U16, V16
            U_s = snap_half(U16)
            V_s = snap_half(V16)
            W_s = snap_half(W16_new)

            if verify(U_s, V_s, W_s):
                return U_s, V_s, W_s

            # Try also solving for U instead of W
            # For each a, U16_new[a, :] is a row of length 16.
            # Equation: sum_r U16_new[a,r] * V16[b,r] * W16[c,r] = T[a,b,c]
            # For fixed (b,c): sum_r U16_new[a,r] * (V16[b,r] * W16[c,r]) = T[a,b,c]
            # This is linear in U16_new[a, :].

            U16_new = np.zeros((n_a, 16))
            for a in range(n_a):
                A_a = np.zeros((n_b * n_c, 16))
                t_a = np.zeros(n_b * n_c)
                for idx, (b, c) in enumerate(product(range(n_b), range(n_c))):
                    A_a[idx, :] = V16[b, :] * W16[c, :]
                    t_a[idx] = T[a, b, c]
                try:
                    sol, residuals, rank_a, sv = np.linalg.lstsq(A_a, t_a, rcond=None)
                    U16_new[a, :] = sol
                except Exception:
                    U16_new[a, :] = U16[a, :]

            U_s = snap_half(U16_new)
            V_s = snap_half(V16)
            W_s = snap_half(W16)
            if verify(U_s, V_s, W_s):
                return U_s, V_s, W_s

            # Try solving for V
            V16_new = np.zeros((n_b, 16))
            for b in range(n_b):
                A_b = np.zeros((n_a * n_c, 16))
                t_b = np.zeros(n_a * n_c)
                for idx, (a, c) in enumerate(product(range(n_a), range(n_c))):
                    A_b[idx, :] = U16[a, :] * W16[c, :]
                    t_b[idx] = T[a, b, c]
                try:
                    sol, residuals, rank_b, sv = np.linalg.lstsq(A_b, t_b, rcond=None)
                    V16_new[b, :] = sol
                except Exception:
                    V16_new[b, :] = V16[b, :]

            U_s = snap_half(U16)
            V_s = snap_half(V16_new)
            W_s = snap_half(W16)
            if verify(U_s, V_s, W_s):
                return U_s, V_s, W_s

            return None

        for col in range(17):
            if time.time() > deadline - 10:
                break
            result = try_rank16_remove_col(col, deadline)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = 16
                break

    # Local search: try to find a rank-16 solution by mutating the rank-17 decomposition
    # and then removing one column
    if best_rank == 17 and time.time() < deadline - 5:
        half_ints = [-1.0, -0.5, 0.0, 0.5, 1.0]
        np.random.seed(42)

        def local_search(deadline):
            """Try to find a rank-16 solution via local mutation + column removal."""
            max_iter = 500
            for it in range(max_iter):
                if time.time() > deadline:
                    return None

                # Start from the rank-17 solution
                U = U17.copy()
                V = V17.copy()
                W = W17.copy()

                # Make a few random mutations
                n_mut = np.random.randint(1, 4)
                for _ in range(n_mut):
                    which = np.random.randint(3)
                    if which == 0:
                        a, r = np.random.randint(n_a), np.random.randint(17)
                        U[a, r] = half_ints[np.random.randint(len(half_ints))]
                    elif which == 1:
                        b, r = np.random.randint(n_b), np.random.randint(17)
                        V[b, r] = half_ints[np.random.randint(len(half_ints))]
                    else:
                        c, r = np.random.randint(n_c), np.random.randint(17)
                        W[c, r] = half_ints[np.random.randint(len(half_ints))]

                # Try to remove each column
                for col in range(17):
                    U16 = np.delete(U, col, axis=1)
                    V16 = np.delete(V, col, axis=1)
                    W16 = np.delete(W, col, axis=1)

                    # Try to solve for W
                    W16_new = np.zeros((n_c, 16))
                    for c in range(n_c):
                        A_c = np.zeros((n_a * n_b, 16))
                        t_c = np.zeros(n_a * n_b)
                        for idx, (a, b) in enumerate(product(range(n_a), range(n_b))):
                            A_c[idx, :] = U16[a, :] * V16[b, :]
                            t_c[idx] = T[a, b, c]
                        try:
                            sol, residuals, rank_c, sv = np.linalg.lstsq(A_c, t_c, rcond=None)
                            W16_new[c, :] = sol
                        except Exception:
                            W16_new[c, :] = W16[c, :]

                    U_s = snap_half(U16)
                    V_s = snap_half(V16)
                    W_s = snap_half(W16_new)
                    if verify(U_s, V_s, W_s):
                        return U_s, V_s, W_s

                    # Try to solve for U
                    U16_new = np.zeros((n_a, 16))
                    for a in range(n_a):
                        A_a = np.zeros((n_b * n_c, 16))
                        t_a = np.zeros(n_b * n_c)
                        for idx, (b, c) in enumerate(product(range(n_b), range(n_c))):
                            A_a[idx, :] = V16[b, :] * W16[c, :]
                            t_a[idx] = T[a, b, c]
                        try:
                            sol, residuals, rank_a, sv = np.linalg.lstsq(A_a, t_a, rcond=None)
                            U16_new[a, :] = sol
                        except Exception:
                            U16_new[a, :] = U16[a, :]

                    U_s = snap_half(U16_new)
                    V_s = snap_half(V16)
                    W_s = snap_half(W16)
                    if verify(U_s, V_s, W_s):
                        return U_s, V_s, W_s

                    # Try to solve for V
                    V16_new = np.zeros((n_b, 16))
                    for b in range(n_b):
                        A_b = np.zeros((n_a * n_c, 16))
                        t_b = np.zeros(n_a * n_c)
                        for idx, (a, c) in enumerate(product(range(n_a), range(n_c))):
                            A_b[idx, :] = U16[a, :] * W16[c, :]
                            t_b[idx] = T[a, b, c]
                        try:
                            sol, residuals, rank_b, sv = np.linalg.lstsq(A_b, t_b, rcond=None)
                            V16_new[b, :] = sol
                        except Exception:
                            V16_new[b, :] = V16[b, :]

                    U_s = snap_half(U16)
                    V_s = snap_half(V16_new)
                    W_s = snap_half(W16)
                    if verify(U_s, V_s, W_s):
                        return U_s, V_s, W_s

            return None

        result = local_search(deadline - 2)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 16

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
