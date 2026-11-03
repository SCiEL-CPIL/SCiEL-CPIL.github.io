import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 7]


def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 28, 14

    start_time = time.time()
    deadline = start_time + 140

    # Build the target multiplication tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # ------------------------------------------------------------------
    # Step 1: Compute rank profiles (unfolding ranks) for all sub-tensors
    # This gives us a quick lower-bound check and helps identify structure
    # ------------------------------------------------------------------
    def _rank_profile(nn, mm, pp):
        """Compute ranks of the three unfoldings of the matmul tensor."""
        Tt = np.zeros((nn * mm, mm * pp, pp * nn), dtype=np.float64)
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    Tt[i * mm + j, j * pp + k, k * nn + i] = 1.0
        # Unfolding 1: (nn*mm) x (mm*pp*nn*pp) -> actually (nn*mm) x (mm*pp)
        U1 = Tt.reshape(nn * mm, -1)
        U2 = Tt.transpose(0, 2, 1).reshape(nn * mm * pp, -1)  # (nn*mm*pp) x (mm*pp)
        # Better: standard unfoldings
        # Mode-1 unfolding: (nn*mm) x (mm*pp * pp*nn)
        # Actually for a 3-tensor of shape (a,b,c):
        # Mode-1: a x (b*c), Mode-2: b x (a*c), Mode-3: c x (a*b)
        a, b, c = nn * mm, mm * pp, pp * nn
        M1 = Tt.reshape(a, b * c)
        M2 = Tt.transpose(1, 0, 2).reshape(b, a * c)
        M3 = Tt.transpose(2, 0, 1).reshape(c, a * b)
        r1 = np.linalg.matrix_rank(M1)
        r2 = np.linalg.matrix_rank(M2)
        r3 = np.linalg.matrix_rank(M3)
        return (r1, r2, r3)

    # Quick check: rank profile of our target
    # This is informational; the real search is below

    # ------------------------------------------------------------------
    # Step 2: Build a known-good baseline (schoolbook, rank 56)
    # ------------------------------------------------------------------
    def _build_schoolbook(nn, mm, pp):
        rank = nn * mm * pp
        U = np.zeros((nn * mm, rank), dtype=np.int64)
        V = np.zeros((mm * pp, rank), dtype=np.int64)
        W = np.zeros((pp * nn, rank), dtype=np.int64)
        col = 0
        for i in range(nn):
            for j in range(mm):
                for k in range(pp):
                    U[i * mm + j, col] = 1
                    V[j * pp + k, col] = 1
                    W[k * nn + i, col] = 1
                    col += 1
        return U, V, W

    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = n * m * p  # 56

    # ------------------------------------------------------------------
    # Step 3: Try to find lower-rank decompositions via randomized CP search
    # Strategy: for rank R from a reasonable upper bound down, try many
    # random integer/half-integer initializations and use L-BFGS to refine.
    # ------------------------------------------------------------------

    def _cp_residual(x, R, T_flat):
        U = x[:R * nm].reshape(nm, R)
        V = x[R * nm:R * nm + R * mp].reshape(mp, R)
        W = x[R * nm + R * mp:].reshape(pn, R)
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        residual = T_rec - T_flat
        return np.sum(residual * residual)

    def _cp_grad(x, R, T_flat):
        U = x[:R * nm].reshape(nm, R)
        V = x[R * nm:R * nm + R * mp].reshape(mp, R)
        W = x[R * nm + R * mp:].reshape(pn, R)
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        residual = T_rec - T_flat
        gU = np.einsum('ijk,jr,kr->ir', residual, V, W, optimize=True)
        gV = np.einsum('ijk,ir,kr->jr', residual, U, W, optimize=True)
        gW = np.einsum('ijk,ir,jr->kr', residual, U, V, optimize=True)
        return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])

    def _try_rank(R, n_restarts=20, time_budget=5.0):
        """Try to find an exact CP decomposition at rank R."""
        for restart in range(n_restarts):
            if time.time() > deadline - 3:
                return None

            # Random initialization with small integers/half-integers
            scale = 1.0
            U_init = np.random.randint(-2, 3, size=(nm, R)).astype(np.float64)
            V_init = np.random.randint(-2, 3, size=(mp, R)).astype(np.float64)
            W_init = np.random.randint(-2, 3, size=(pn, R)).astype(np.float64)

            # Sometimes use structured init: sparse random
            if restart % 3 == 1:
                U_init = np.zeros((nm, R))
                V_init = np.zeros((mp, R))
                W_init = np.zeros((pn, R))
                for r in range(R):
                    # Pick random non-zero entries
                    nzU = np.random.choice(nm, size=max(1, nm // 4), replace=False)
                    nzV = np.random.choice(mp, size=max(1, mp // 4), replace=False)
                    nzW = np.random.choice(pn, size=max(1, pn // 4), replace=False)
                    U_init[nzU, r] = np.random.choice([-1, 1], size=len(nzU))
                    V_init[nzV, r] = np.random.choice([-1, 1], size=len(nzV))
                    W_init[nzW, r] = np.random.choice([-1, 1], size=len(nzW))
            elif restart % 3 == 2:
                # Gaussian init
                U_init = np.random.randn(nm, R) * 0.5
                V_init = np.random.randn(mp, R) * 0.5
                W_init = np.random.randn(pn, R) * 0.5

            x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])

            try:
                res = minimize(
                    _cp_residual, x0, args=(R, T),
                    jac=_cp_grad, method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
                )

                if res.fun < 1e-10:
                    U_opt = res.x[:R * nm].reshape(nm, R)
                    V_opt = res.x[R * nm:R * nm + R * mp].reshape(mp, R)
                    W_opt = res.x[R * nm + R * mp:].reshape(pn, R)

                    # Verify exactness
                    T_rec = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt, optimize=True)
                    if np.max(np.abs(T_rec - T)) < 1e-8:
                        # Round to half-integers
                        U_r = np.round(U_opt * 2) / 2
                        V_r = np.round(V_opt * 2) / 2
                        W_r = np.round(W_opt * 2) / 2
                        T_rec_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r, optimize=True)
                        if np.max(np.abs(T_rec_r - T)) < 1e-8:
                            return (U_r, V_r, W_r, R)
                        else:
                            # Check if unrounded is close enough (shouldn't happen for exact)
                            return (U_opt, V_opt, W_opt, R)
            except Exception:
                pass

        return None

    # Try ranks from 50 down to 30
    # We know schoolbook is 56. Strassen-based should be around 46-50.
    # Let's try a range of ranks.
    for R in range(50, 25, -1):
        if time.time() > deadline - 10:
            break

        result = _try_rank(R, n_restarts=15, time_budget=6.0)
        if result is not None:
            U_new, V_new, W_new, rank_new = result
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = rank_new
            # Continue trying lower ranks
            continue

    # ------------------------------------------------------------------
    # Step 4: Also try a structured block approach as a fallback
    # Split 2x4x7 into blocks: A(2x4) = [A1 A2] (2x2 each), B(4x7)
    # C = A1*B_top + A2*B_bot where B_top, B_bot are 2x7
    # For each 2x2 * 2x7: split 7 cols into 2+2+2+1
    # 2x2*2x2: use rank 7 (Strassen) or rank 4 (schoolbook)
    # 2x2*2x1: use rank 2 (schoolbook) or rank 4?
    # ------------------------------------------------------------------
    def _build_strassen_block(n, m, p):
        """Strassen-based block decomposition."""
        # Strassen 2x2x2 (rank 7)
        U_s = np.array([
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, -1]
        ], dtype=np.int64)
        V_s = np.array([
            [0, 1, 1, 0, 0, 1, 0],
            [1, 0, 0, -1, 0, 0, 1],
            [0, 0, -1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0, 1]
        ], dtype=np.int64)
        W_s = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=np.int64)

        # Schoolbook 2x2x1 (rank 2)
        U_1 = np.array([[1, 0], [0, 0], [0, 1], [0, 0]], dtype=np.int64)
        V_1 = np.array([[1, 0], [0, 1]], dtype=np.int64)
        W_1 = np.array([[1, 1], [0, 0], [0, 1], [0, 0]], dtype=np.int64)
        # Wait, 2x2*2x1: output is 2x1, so pn=2, rank should be 2
        # U: 4x2, V: 2x2, W: 2x2
        # Actually for 2x2 * 2x1:
        # i=0,1; j=0,1; k=0
        # U[i*2+j, r], V[j*1+0, r], W[0*2+i, r]
        # Schoolbook: rank 2 (one per i)
        U_1 = np.zeros((4, 2), dtype=np.int64)
        V_1 = np.zeros((2, 2), dtype=np.int64)
        W_1 = np.zeros((2, 2), dtype=np.int64)
        for r in range(2):
            U_1[r * 2, r] = 1  # i=r, j=0
            U_1[r * 2 + 1, r] = 1  # i=r, j=1
            V_1[0, r] = 1
            V_1[1, r] = 1
            W_1[r, r] = 1

        total_rank = 2 * (3 * 7 + 1 * 2)  # 2 blocks, each with 3x(2x2x2) + 1x(2x2x1)
        U = np.zeros((8, total_rank), dtype=np.int64)
        V = np.zeros((28, total_rank), dtype=np.int64)
        W = np.zeros((14, total_rank), dtype=np.int64)

        col = 0
        for block in range(2):
            # A cols: 2*block, 2*block+1
            # B rows: 2*block, 2*block+1
            # B cols: 0..6, split as 2,2,2,1

            # Three 2x2x2 Strassen blocks
            for sub in range(3):
                for r in range(7):
                    for i in range(2):
                        for j in range(2):
                            U[i * 4 + (2 * block + j), col] += U_s[i * 2 + j, r]
                    for j in range(2):
                        for k in range(2):
                            V[(2 * block + j) * 7 + (sub * 2 + k), col] += V_s[j * 2 + k, r]
                    for k in range(2):
                        for i in range(2):
                            W[(sub * 2 + k) * 2 + i, col] += W_s[k * 2 + i, r]
                    col += 1

            # One 2x2x1 schoolbook block
            for r in range(2):
                for i in range(2):
                    for j in range(2):
                        U[i * 4 + (2 * block + j), col] += U_1[i * 2 + j, r]
                for j in range(2):
                    V[(2 * block + j) * 7 + 6, col] += V_1[j, r]
                for i in range(2):
                    W[6 * 2 + i, col] += W_1[i, r]
                col += 1

        return U, V, W

    U_str, V_str, W_str = _build_strassen_block(n, m, p)
    if _verify(U_str, V_str, W_str, n, m, p):
        if 2 * (3 * 7 + 1 * 2) < best_rank:
            best_U, best_V, best_W = U_str.copy(), V_str.copy(), W_str.copy()
            best_rank = 2 * (3 * 7 + 1 * 2)

    # ------------------------------------------------------------------
    # Step 5: Final verification and return
    # ------------------------------------------------------------------
    if _verify(best_U, best_V, best_W, n, m, p):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fallback: schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True
