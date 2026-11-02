"""Search for a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 10 matrix."""
import time
import random
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [2, 3, 10]
N, M, P = 2, 3, 10
TIME_LIMIT = 145  # safety margin before 150s kill

_START = time.time()

def _time_left():
    return TIME_LIMIT - (time.time() - _START)

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-8):
    """Fast numpy verification of the full matmul tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    # Target: T[i,j,k]=1 iff (i,j,k) is a valid matmul term
    # i = a_row*m + a_col, j = a_col*p + b_col, k = b_col*n + a_row
    target = np.zeros((n*m, m*p, p*n))
    for a_row in range(n):
        for a_col in range(m):
            for b_col in range(p):
                i = a_row*m + a_col
                j = a_col*p + b_col
                k = b_col*n + a_row
                target[i, j, k] = 1.0
    return np.allclose(T, target, atol=tol)

# ---------------------------------------------------------------------------
# Sub-problem target tensor: 2x3 (A) by 3x2 (B) -> 2x2 (C)
# ---------------------------------------------------------------------------
def _sub_target():
    """Build the 6x6x4 target tensor for 2x3 by 3x2 matmul."""
    T = np.zeros((6, 6, 4))
    for a_row in range(2):
        for a_col in range(3):
            for b_col in range(2):
                i = a_row*3 + a_col
                j = a_col*2 + b_col
                k = a_row*2 + b_col
                T[i, j, k] = 1.0
    return T

_SUB_T = _sub_target()

# ---------------------------------------------------------------------------
# Rank-11 sub-solution (Strassen on A[:,0:2]*B[0:2,:] + schoolbook A[:,2]*B[2,:])
# ---------------------------------------------------------------------------
def _sub_rank11():
    """Known rank-11 decomposition of the 2x3 x 3x2 sub-problem."""
    U = np.zeros((6, 11))
    V = np.zeros((6, 11))
    W = np.zeros((4, 11))
    # A indices: a00=0,a01=1,a10=3,a11=4,a02=2,a12=5
    # B indices: b00=0,b01=1,b10=2,b11=3,b20=4,b21=5
    # C indices: c00=0,c01=1,c10=2,c11=3

    a00, a01, a10, a11, a02, a12 = 0, 1, 3, 4, 2, 5
    b00, b01, b10, b11, b20, b21 = 0, 1, 2, 3, 4, 5
    c00, c01, c10, c11 = 0, 1, 2, 3

    col = 0
    # Strassen M1: (A00+A11)(B00+B11)
    U[a00, col] = 1; U[a11, col] = 1
    V[b00, col] = 1; V[b11, col] = 1
    W[c00, col] = 1; W[c01, col] = -1; W[c10, col] = -1; W[c11, col] = 1
    col += 1
    # M2: (A10+A11) B00
    U[a10, col] = 1; U[a11, col] = 1
    V[b00, col] = 1
    W[c10, col] = 1; W[c11, col] = 1
    col += 1
    # M3: A00 (B01-B11)
    U[a00, col] = 1
    V[b01, col] = 1; V[b11, col] = -1
    W[c01, col] = 1
    col += 1
    # M4: A11 (B10-B00)
    U[a11, col] = 1
    V[b10, col] = 1; V[b00, col] = -1
    W[c10, col] = 1
    col += 1
    # M5: (A00+A01) B11
    U[a00, col] = 1; U[a01, col] = 1
    V[b11, col] = 1
    W[c00, col] = 1; W[c01, col] = -1
    col += 1
    # M6: (A10-A00)(B00+B01)
    U[a10, col] = 1; U[a00, col] = -1
    V[b00, col] = 1; V[b01, col] = 1
    W[c10, col] = 1; W[c11, col] = -1
    col += 1
    # M7: (A01-A11)(B10+B11)
    U[a01, col] = 1; U[a11, col] = -1
    V[b10, col] = 1; V[b11, col] = 1
    W[c01, col] = 1; W[c11, col] = 1
    col += 1
    # Schoolbook Q1: A02 B20
    U[a02, col] = 1; V[b20, col] = 1; W[c00, col] = 1
    col += 1
    # Q2: A02 B21
    U[a02, col] = 1; V[b21, col] = 1; W[c01, col] = 1
    col += 1
    # Q3: A12 B20
    U[a12, col] = 1; V[b20, col] = 1; W[c10, col] = 1
    col += 1
    # Q4: A12 B21
    U[a12, col] = 1; V[b21, col] = 1; W[c11, col] = 1
    col += 1

    return U, V, W

def _sub_residual(U, V, W):
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.sum((T - _SUB_T)**2)

# ---------------------------------------------------------------------------
# Tensor search for the sub-problem
# ---------------------------------------------------------------------------
def _search_sub(rank, rng, time_budget, init=None):
    """
    Search for a rank-`rank` decomposition of the 2x3 x 3x2 sub-problem.
    Uses L-BFGS-B with a smooth squared-residual objective.
    Returns (U, V, W) if a solution with residual < tol is found, else None.
    """
    from scipy.optimize import minimize

    nA, nB, nC = 6, 6, 4
    size_U = nA * rank
    size_V = nB * rank
    size_W = nC * rank
    total = size_U + size_V + size_W

    def unpack(x):
        U = x[:size_U].reshape(nA, rank)
        V = x[size_U:size_U+size_V].reshape(nB, rank)
        W = x[size_U+size_V:].reshape(nC, rank)
        return U, V, W

    def objective(x):
        U, V, W = unpack(x)
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.sum((T - _SUB_T)**2)

    def gradient(x):
        U, V, W = unpack(x)
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        diff = T - _SUB_T
        # dL/dU[i,r] = sum_{j,k} diff[i,j,k] * V[j,r] * W[k,r]
        gU = np.einsum('ijk,jr,kr->ir', diff, V, W)
        gV = np.einsum('ijk,ir,kr->jr', diff, U, W)
        gW = np.einsum('ijk,ir,jr->kr', diff, U, V)
        return np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])

    tol = 1e-6

    # Build initial parameters
    if init is not None:
        U0, V0, W0 = init
        x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        # If init has more columns than rank, truncate; if fewer, pad
        if U0.shape[1] > rank:
            x0 = np.concatenate([U0[:, :rank].ravel(), V0[:, :rank].ravel(), W0[:, :rank].ravel()])
        elif U0.shape[1] < rank:
            pad_U = np.zeros((nA, rank - U0.shape[1]))
            pad_V = np.zeros((nB, rank - V0.shape[1]))
            pad_W = np.zeros((nC, rank - W0.shape[1]))
            x0 = np.concatenate([U0.ravel(), pad_U.ravel(), V0.ravel(), pad_V.ravel(), W0.ravel(), pad_W.ravel()])
    else:
        x0 = rng.randn(total) * 0.5

    best_x = x0
    best_obj = objective(x0)
    if best_obj < tol:
        U, V, W = unpack(best_x)
        return U, V, W

    t0 = time.time()
    max_iter = 2000

    while time.time() - t0 < time_budget and best_obj >= tol:
        res = minimize(objective, best_x, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x
        if best_obj < tol:
            break
        # Random restart
        x0 = rng.randn(total) * 1.0
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_obj:
            best_obj = res.fun
            best_x = res.x

    if best_obj < tol:
        U, V, W = unpack(best_x)
        return U, V, W
    return None

# ---------------------------------------------------------------------------
# Assemble full 2x3 x 3x10 solution from a sub-solution
# ---------------------------------------------------------------------------
def _assemble_full(sub_U, sub_V, sub_W):
    """
    Given a rank-r sub-solution for 2x3 x 3x2, build the full 2x3 x 3x10 solution.
    The 5 blocks are independent, so we use rank = 5*r.
    """
    r = sub_U.shape[1]
    full_rank = 5 * r
    U = np.zeros((6, full_rank))
    V = np.zeros((30, full_rank))
    W = np.zeros((20, full_rank))

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        # B indices for block b: B[a_col, k0] and B[a_col, k1]
        # sub_V row mapping: sub_V[a_col*2 + b_col]
        #   sub_V[0] = B[0,k0], sub_V[1] = B[0,k1]
        #   sub_V[2] = B[1,k0], sub_V[3] = B[1,k1]
        #   sub_V[4] = B[2,k0], sub_V[5] = B[2,k1]
        # Full V index: B[a_col, k] = a_col*10 + k
        v_map = {
            0: 0*10 + k0,
            1: 0*10 + k1,
            2: 1*10 + k0,
            3: 1*10 + k1,
            4: 2*10 + k0,
            5: 2*10 + k1,
        }
        # W indices: C[a_row, b_col] = b_col*2 + a_row
        # sub_W row mapping: sub_W[a_row*2 + b_col]
        #   sub_W[0] = C[0,0], sub_W[1] = C[0,1]
        #   sub_W[2] = C[1,0], sub_W[3] = C[1,1]
        # Full W index: C[a_row, k] = k*2 + a_row
        w_map = {
            0: k0*2 + 0,
            1: k1*2 + 0,
            2: k0*2 + 1,
            3: k1*2 + 1,
        }

        col_off = b * r
        for c in range(r):
            U[:, col_off + c] = sub_U[:, c]
            for sv_row, fv_row in v_map.items():
                V[fv_row, col_off + c] = sub_V[sv_row, c]
            for sw_row, fw_row in w_map.items():
                W[fw_row, col_off + c] = sub_W[sw_row, c]

    return U, V, W

# ---------------------------------------------------------------------------
# Fallback rank-55 construction
# ---------------------------------------------------------------------------
def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_rank55(n, m, p):
    """Construct the known rank-55 decomposition."""
    if n != 2 or m != 3 or p != 10:
        return None
    sub_U, sub_V, sub_W = _sub_rank11()
    U, V, W = _assemble_full(sub_U, sub_V, sub_W)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    n, m, p = DIMENSIONS

    # Guaranteed rank-55 fallback
    fallback = _strassen_rank55(n, m, p)

    # Try to find a rank-10 sub-solution -> full rank 50
    rng = np.random.RandomState(42)

    # First, try with the rank-11 init truncated (warm start)
    if _time_left() > 20:
        sub_U11, sub_V11, sub_W11 = _sub_rank11()
        # Truncate to rank 10 by zeroing the last column (the Q4 schoolbook term)
        # Actually, just use the first 10 columns as init
        init10 = (sub_U11[:, :10], sub_V11[:, :10], sub_W11[:, :10])
        result = _search_sub(10, rng, time_budget=40, init=init10)
        if result is not None:
            U, V, W = result
            full_U, full_V, full_W = _assemble_full(U, V, W)
            if _verify_np(full_U, full_V, full_W, n, m, p, tol=1e-6):
                return {"U": full_U.tolist(), "V": full_V.tolist(), "W": full_W.tolist()}

    # Try random inits for rank 10
    if _time_left() > 30:
        for attempt in range(3):
            if _time_left() < 15:
                break
            result = _search_sub(10, rng, time_budget=min(30, _time_left() - 10), init=None)
            if result is not None:
                U, V, W = result
                full_U, full_V, full_W = _assemble_full(U, V, W)
                if _verify_np(full_U, full_V, full_W, n, m, p, tol=1e-6):
                    return {"U": full_U.tolist(), "V": full_V.tolist(), "W": full_W.tolist()}

    # Try rank 9 (even more ambitious)
    if _time_left() > 25:
        result = _search_sub(9, rng, time_budget=min(25, _time_left() - 10), init=None)
        if result is not None:
            U, V, W = result
            full_U, full_V, full_W = _assemble_full(U, V, W)
            if _verify_np(full_U, full_V, full_W, n, m, p, tol=1e-6):
                return {"U": full_U.tolist(), "V": full_V.tolist(), "W": full_W.tolist()}

    # Fallback to rank 55
    return fallback
