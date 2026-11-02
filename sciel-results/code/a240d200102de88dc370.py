"""Tensor-decomposition search for 3x4 x 4x7 matrix multiplication.

Goal: rank-minimal EXACT decomposition of the multiplication tensor with
integer / half-integer entries.  We start from the schoolbook rank-84
solution and attempt to reduce the rank via a structured local search
(alternating least-squares on the 0/1 multiplication tensor followed by
exact rational reconstruction and verification).  We always fall back to
the guaranteed-valid schoolbook solution, so the returned object is always
a correct decomposition.
"""
import time
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """T[i, j, k] = 1 for the multiplication tensor (3D array)."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T


def _decompose_to_uv(T, n, m, p, rank, U, V, W):
    """Given U (n*m x rank), V (m*p x rank), compute W so that the
    reconstruction matches T as closely as possible (least squares)."""
    # Reshape U to (n, m, rank), V to (m, p, rank).
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    # For each (i, k): sum_j T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W[k,i,c]
    # => linear in W[k,i,c].
    # Build system: for each (i,k), unknowns are W[k,i,c] for c in range(rank).
    # Coefficient for W[k,i,c] is sum_j U3[i,j,c] * V3[j,k,c].
    # But this couples all c for the same (i,k).  We solve a block system.
    # Simpler: treat W as flat and solve a global least squares.
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is linear in W.  Build A (n*m*p x (p*n*rank)) is too big; instead
    # solve per (i,k) with rank unknowns.
    W = np.zeros((p * n, rank), dtype=np.float64)
    for i in range(n):
        for k in range(p):
            # Unknowns: W[k*n+i, c] for c in 0..rank-1
            # For each j: T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
            # => A_row[j, c] = U[i*m+j,c] * V[j*p+k,c],  b[j] = T[i,j,k]
            A = np.zeros((m, rank), dtype=np.float64)
            for j in range(m):
                A[j, :] = U[i * m + j, :] * V[j * p + k, :]
            b = T[i, k, :]  # T[i, j, k] over j -> shape (m,)
            # Solve A w = b in least squares.
            w, *_ = np.linalg.lstsq(A, b, rcond=None)
            W[k * n + i, :] = w
    return W


def _reconstruct(U, V, W, n, m, p):
    """Return the reconstructed tensor from U,V,W."""
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    for c in range(rank):
        T += U3[:, :, c][:, :, None] * V3[:, c, :][None, :, :] * W3[:, :, c][None, None, :]
    return T


def _round_half_int(A):
    """Round each entry to the nearest half-integer (multiple of 0.5)."""
    return np.round(A * 2.0) / 2.0


def _verify_exact(U, V, W, n, m, p, tol=1e-9):
    """Exact verification using Fractions.  Returns True if the decomposition
    exactly reproduces the multiplication tensor (entries 0 or 1)."""
    rank = U.shape[1]
    # Convert to Fraction grids.
    Uf = [[Fraction(int(round(U[i, c] * 2))) / 2 for c in range(rank)] for i in range(n * m)]
    Vf = [[Fraction(int(round(V[i, c] * 2))) / 2 for c in range(rank)] for i in range(m * p)]
    Wf = [[Fraction(int(round(W[i, c] * 2))) / 2 for c in range(rank)] for i in range(p * n)]
    U3 = [[Uf[i * m + j] for j in range(m)] for i in range(n)]
    V3 = [[Vf[j * p + k] for k in range(p)] for j in range(m)]
    W3 = [[Wf[k * n + i] for i in range(n)] for k in range(p)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U3[i][j][c] * V3[j][k][c] * W3[k][i][c]
                if s != 1:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank


def _try_rank(n, m, p, rank, T, deadline, seed):
    """Attempt to find a valid rank-`rank` decomposition.  Returns
    (U, V, W) as half-integer numpy arrays or None."""
    rng = np.random.default_rng(seed)
    best_err = None
    best = None
    n_iter = 40
    for it in range(n_iter):
        if time.time() > deadline:
            break
        # Random-ish structured init: start from a random orthonormal-ish basis.
        U = rng.standard_normal((n * m, rank)) * 0.5
        V = rng.standard_normal((m * p, rank)) * 0.5
        W = rng.standard_normal((p * n, rank)) * 0.5
        for _ in range(60):
            if time.time() > deadline:
                break
            W = _decompose_to_uv(T, n, m, p, rank, U, V, W)
            # Update U: for each (i, j), solve for U[i*m+j, c] given V, W.
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # T[i,j,k] = sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) summed over k
                    # => A_row[k, c] = V[j*p+k,c] * W[k*n+i,c], b[k] = T[i,j,k]
                    A = np.zeros((p, rank), dtype=np.float64)
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    b = T[i, j, :]
                    w, *_ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i * m + j, :] = w
            U = U_new
            # Update V: for each (j, k), solve for V[j*p+k, c] given U, W.
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank), dtype=np.float64)
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    b = T[:, j, k]
                    w, *_ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j * p + k, :] = w
            V = V_new
        # Evaluate error and try rounding.
        Rec = _reconstruct(U, V, W, n, m, p)
        err = float(np.linalg.norm(Rec - T))
        # Try rounding to half-integers and verify exactly.
        Ur = _round_half_int(U)
        Vr = _round_half_int(V)
        Wr = _round_half_int(W)
        if _verify_exact(Ur, Vr, Wr, n, m, p):
            return Ur, Vr, Wr
        if best_err is None or err < best_err:
            best_err = err
            best = (Ur, Vr, Wr)
    # Fallback: if the unrounded solution is very close, try a finer rational
    # reconstruction (denominators up to 8) and verify.
    if best is not None and best_err is not None and best_err < 1e-3:
        Ur, Vr, Wr = best
        for den in (2, 4, 8):
            Ur2 = np.round(U * den) / den
            Vr2 = np.round(V * den) / den
            Wr2 = np.round(W * den) / den
            # verify with general rational (den)
            if _verify_exact_general(Ur2, Vr2, Wr2, n, m, p):
                return Ur2, Vr2, Wr2
    return None


def _verify_exact_general(U, V, W, n, m, p):
    """Exact verification for entries that are multiples of 1/8."""
    rank = U.shape[1]
    def frac(x):
        v = int(round(x * 8))
        return Fraction(v, 8)
    Uf = [[frac(U[i, c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[frac(V[i, c]) for c in range(rank)] for i in range(m * p)]
    Wf = [[frac(W[i, c]) for c in range(rank)] for i in range(p * n)]
    U3 = [[Uf[i * m + j] for j in range(m)] for i in range(n)]
    V3 = [[Vf[j * p + k] for k in range(p)] for j in range(m)]
    W3 = [[Wf[k * n + i] for i in range(n)] for k in range(p)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U3[i][j][c] * V3[j][k][c] * W3[k][i][c]
                if s != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110.0  # safety margin vs 150s wall-clock kill

    T = _build_target_tensor(n, m, p)

    # Guaranteed-valid baseline.
    U0, V0, W0, rank0 = _schoolbook(n, m, p)

    best = (U0, V0, W0, rank0)

    # Try to beat the schoolbook rank, descending from rank0-1.
    for rank in range(rank0 - 1, max(rank0 - 6, 79), -1):
        if time.time() > deadline:
            break
        res = _try_rank(n, m, p, rank, T, deadline, seed=rank * 7919)
        if res is not None:
            U, V, W = res
            # Convert to lists of numbers (floats that are exact half-integers).
            best = (U, V, W, rank)
            break

    U, V, W, rank = best
    # Return as lists (floats) to match the interface.
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
