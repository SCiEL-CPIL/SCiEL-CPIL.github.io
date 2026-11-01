"""Matrix multiplication 2x2, 2x7 exact tensor decomposition.

Correct tensor condition (bilinear, for ALL A,B):
    C[i,k] = sum_j A[i,j] B[j,k]
must equal
    sum_c (sum_j U[i*m+j,c] A[i,j]) (sum_j V[j*p+k,c] B[j,k]).

Matching coefficients of the independent monomials A[i,j]B[j',k] gives the
defining system (for every i, j, k):
    sum_c U[i*m+j,c] * V[j*p+k,c] == delta_{j,k}.

Note this is the *j,k* delta (NOT the i,k delta used by the previous seed),
which is why the previous verification was unsatisfiable below rank 28.
The W factor only routes each term to the output slot k*n+i and so is
determined once U,V are found: W[k*n+i,c] = 1 whenever the column c has a
non-zero contribution to (i,k).

Strategy:
  * Guaranteed fallback: schoolbook rank 28 (always exact).
  * Improvement: solve the 28 quadratic equations for R = 27, 26, ... using
    Gauss-Newton with many random starts (the system is very underdetermined),
    then round to the half-integer grid and verify EXACTLY with Fraction.
    Only an exact candidate is accepted; otherwise we keep the fallback.
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 7]


def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _build_W(U, V, n, m, p):
    R = len(U[0])
    W = [[0] * R for _ in range(p * n)]
    for c in range(R):
        for i in range(n):
            for k in range(p):
                if any(U[i * m + j][c] != 0 and V[j * p + k][c] != 0 for j in range(m)):
                    W[k * n + i][c] = 1
    return W


def _residual(U, V, n, m, p):
    """r_{i,j,k} = sum_c U[i*m+j,c] V[j*p+k,c] - delta_{j,k}."""
    R = len(U[0])
    resid = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i * m + j][c] * V[j * p + k][c]
                resid.append(s - (1.0 if j == k else 0.0))
    return resid


def _residual2(U, V, n, m, p):
    r = _residual(U, V, n, m, p)
    return sum(x * x for x in r)


def _gauss_newton_numpy(U, V, n, m, p, iters=100, tol=1e-12, momentum=0.9):
    """Efficient Gauss-Newton using NumPy for linear algebra."""
    R = len(U[0])
    nm, mp = n * m, m * p
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    
    # Precompute indices
    i_idx = np.array([[i for k in range(p)] for j in range(m) for i in range(n)]).reshape(-1)
    j_idx = np.array([[j for i in range(n)] for k in range(p) for j in range(m)]).reshape(-1)
    k_idx = np.array([[k for j in range(m)] for i in range(n) for k in range(p)]).reshape(-1)
    
    # Target: delta_{j,k}
    target = np.array([1.0 if j == k else 0.0 for j, k in zip(j_idx, k_idx)])
    
    prev_step = None
    for it in range(iters):
        # Compute residual
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] - delta_{j,k}
        U_rows = U_np[i_idx * m + j_idx]  # shape (eqs, R)
        V_rows = V_np[j_idx * p + k_idx]  # shape (eqs, R)
        prod = np.sum(U_rows * V_rows, axis=1)  # shape (eqs,)
        resid = prod - target  # shape (eqs,)
        sse = float(np.sum(resid ** 2))
        
        if sse < tol:
            break
        
        # Compute Jacobian (sparse structure)
        # d(resid)/d(U[i*m+j,c]) = V[j*p+k,c] for the (i,j,k) equation
        # d(resid)/d(V[j*p+k,c]) = U[i*m+j,c] for the (i,j,k) equation
        J_U = V_rows  # shape (eqs, R), each row corresponds to one equation
        J_V = U_rows   # shape (eqs, R)
        
        # Stack: J = [J_U, J_V] shape (eqs, nm*R + mp*R)
        # But we can solve the normal equations more efficiently
        # Normal equations: (J^T J) d = -J^T r
        # J^T J = [[sum_c U*c*U*c, sum_c U*c*V*c], [sum_c V*c*U*c, sum_c V*c*V*c]]
        
        # Compute J^T J blocks
        # Block (0,0): sum over equations of outer(U_rows[e], U_rows[e])
        # This is: U_rows.T @ U_rows shape (nm*R, nm*R)
        # But U_rows has shape (eqs, R), and we need to map back to (nm*R,)
        
        # Actually, let's build the full Jacobian for simplicity since dimensions are small
        nv = nm * R + mp * R
        J = np.zeros((len(resid), nv))
        
        for eq in range(len(resid)):
            i, j, k = i_idx[eq], j_idx[eq], k_idx[eq]
            u_row = i * m + j
            v_row = j * p + k
            for c in range(R):
                J[eq, u_row * R + c] = V_np[v_row, c]
                J[eq, nv - mp * R + v_row * R + c] = U_np[u_row, c]
        
        # Solve normal equations: (J^T J + lambda*I) d = -J^T r
        JtJ = J.T @ J
        Jtr = J.T @ resid
        nv_total = nv
        # Add small regularization for numerical stability
        JtJ += 1e-8 * np.eye(nv_total)
        
        # Solve
        try:
            d = np.linalg.solve(JtJ, -Jtr)
        except np.linalg.LinAlgError:
            d = np.linalg.lstsq(JtJ, -Jtr, rcond=None)[0]
        
        # Momentum
        if prev_step is not None:
            d = momentum * prev_step + (1 - momentum) * d
        prev_step = d
        
        # Update
        dU = d[:nm * R].reshape(nm, R)
        dV = d[nm * R:].reshape(mp, R)
        U_np = U_np + dU
        V_np = V_np + dV
    
    sse = _residual2(U_np.tolist(), V_np.tolist(), n, m, p)
    return U_np.tolist(), V_np.tolist(), sse


def _to_frac(x):
    if isinstance(x, Fraction):
        return x
    h = int(round(2.0 * float(x))) / 2.0
    return Fraction(h)


def _verify_exact(U, V, W, n, m, p, probes=12):
    """Exact check of the bilinear identity C = A*B for random integer A,B."""
    R = len(U[0])
    Uf = [[_to_frac(U[i][c]) for c in range(R)] for i in range(n * m)]
    Vf = [[_to_frac(V[j][c]) for c in range(R)] for j in range(m * p)]
    Wf = [[_to_frac(W[k][c]) for c in range(R)] for k in range(p * n)]

    for _ in range(probes):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        # expected C = A*B
        Cexp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        # reconstructed C
        Crec = [[0] * p for _ in range(n)]
        for c in range(R):
            ua = [sum(Uf[i * m + j][c] * A[i][j] for j in range(m)) for i in range(n)]
            vb = [sum(Vf[j * p + k][c] * B[j][k] for j in range(m)) for k in range(p)]
            for k in range(p):
                for i in range(n):
                    Crec[i][k] += Wf[k * n + i][c] * ua[i] * vb[k]
        for i in range(n):
            for k in range(p):
                if Crec[i][k] != Cexp[i][k]:
                    return False
    return True


def _round_half(M):
    return [[int(round(2.0 * float(M[i][c]))) / 2.0 for c in range(len(M[0]))]
            for i in range(len(M))]


def _structured_init(R, n, m, p, rng):
    """Try structured initializations that might be closer to solutions."""
    inits = []
    
    # 1. Random
    U = [[rng.gauss(0.0, 1.0) for _ in range(R)] for _ in range(n * m)]
    V = [[rng.gauss(0.0, 1.0) for _ in range(R)] for _ in range(m * p)]
    inits.append((U, V))
    
    # 2. Sparse random (many zeros)
    U = [[rng.choice([0, 0, 0, 1, -1]) for _ in range(R)] for _ in range(n * m)]
    V = [[rng.choice([0, 0, 0, 1, -1]) for _ in range(R)] for _ in range(m * p)]
    inits.append((U, V))
    
    # 3. Diagonal-like
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    for c in range(min(R, n * m)):
        U[c][c] = 1.0
    for c in range(min(R, m * p)):
        V[c][c] = 1.0
    inits.append((U, V))
    
    return inits


def _try_rank(R, n, m, p, rng, deadline, starts):
    best_sse = float('inf')
    best_UV = None
    
    for s in range(starts):
        if time.time() > deadline:
            break
        
        # Try multiple initialization strategies
        for init_idx, (U, V) in enumerate(_structured_init(R, n, m, p, rng)):
            if time.time() > deadline:
                break
            
            U2, V2, sse = _gauss_newton_numpy(U, V, n, m, p, iters=80)
            
            if sse < best_sse:
                best_sse = sse
                best_UV = (U2, V2)
            
            if sse < 1e-10:
                Ur = _round_half(U2)
                Vr = _round_half(V2)
                if _residual2(Ur, Vr, n, m, p) < 1e-9:
                    W = _build_W(Ur, Vr, n, m, p)
                    if _verify_exact(Ur, Vr, W, n, m, p):
                        return Ur, Vr, W
    
    # If we found a good candidate but didn't verify, try rounding the best
    if best_UV is not None and best_sse < 1e-8:
        U2, V2 = best_UV
        Ur = _round_half(U2)
        Vr = _round_half(V2)
        if _residual2(Ur, Vr, n, m, p) < 1e-9:
            W = _build_W(Ur, Vr, n, m, p)
            if _verify_exact(Ur, Vr, W, n, m, p):
                return Ur, Vr, W
    
    return None


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 135  # safety margin under the 150s wall-clock kill

    U, V, W = _build_schoolbook(n, m, p)
    rng = random.Random(12345)

    try:
        # Try ranks from 27 down to 12
        for R in range(n * m * p - 1, 11, -1):
            if time.time() > deadline:
                break
            
            # Allocate more time for higher ranks (more likely to succeed)
            time_per_rank = min(10.0, (deadline - time.time()) / max(1, n * m * p - R))
            max_starts = max(2, int(time_per_rank / 0.5))
            
            cand = _try_rank(R, n, m, p, rng, deadline, max_starts)
            if cand is not None:
                U, V, W = cand
                break
    except Exception:
        pass

    return {"U": U, "V": V, "W": W}
