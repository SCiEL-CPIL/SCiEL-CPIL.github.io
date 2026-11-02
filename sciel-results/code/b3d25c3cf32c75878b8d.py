"""
Combined search for a rank-minimal exact decomposition of 3x4 * 4x4 multiplication.

Strategy:
  1. Fast randomized search: draw random U, V with small integer entries,
     solve for W via per-(i,k) linear systems, snap to half-integers, verify.
  2. Block-recursive (Strassen-like) structured search: partition indices into
     overlapping 2x2 / 2x3 blocks, generate bilinear combinations, solve for W,
     snap and verify.
  3. Linear-dependency reduction: try to eliminate one column at a time by
     checking if it is a linear combination of the others in the tensor-product
     sense (solving a linear system in the coefficients).

All methods return the best (U, V, W, rank) found.  The entry point is construct().
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
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
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Exact verification using Fraction arithmetic."""
    rank = len(U[0])
    Uf = [[Fraction(U[i][c]) for c in range(rank)] for i in range(len(U))]
    Vf = [[Fraction(V[i][c]) for c in range(rank)] for i in range(len(V))]
    Wf = [[Fraction(W[i][c]) for c in range(rank)] for i in range(len(W))]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i * m + j][c] * Vf[j * p + k][c] * Wf[k * n + i][c]
                if s != 1:
                    return False
    return True


def _snap_half(x, tol=1e-10):
    v = round(x * 2) / 2
    return v if abs(x - v) < tol else x


def _snap_matrix(M, tol=1e-10):
    return [[_snap_half(x, tol) for x in row] for row in M]


def _solve_for_W(U_np, V_np, n, m, p, rank):
    """
    Given U (n*m x rank) and V (m*p x rank), solve for W (p*n x rank).
    For each (i, k): for all j, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    This is m equations in rank unknowns for each (i, k).
    Returns W_np or None if the system is inconsistent.
    """
    nW = p * n
    W_np = np.zeros((nW, rank))
    for i in range(n):
        for k in range(p):
            A = np.zeros((m, rank))
            for j in range(m):
                for c in range(rank):
                    A[j, c] = U_np[i * m + j, c] * V_np[j * p + k, c]
            try:
                W_ik, res, rA, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
            except np.linalg.LinAlgError:
                return None
            if np.linalg.norm(A @ W_ik - np.ones(m)) > 1e-9:
                return None
            W_np[k * n + i, :] = W_ik
    return W_np


def _try_random_uv(n, m, p, rank, max_time=5.0, seed=None):
    """
    Randomly draw integer U, V with entries in {-2,-1,0,1,2}, solve for W,
    snap to half-integers, verify exactly.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    t0 = time.time()
    deadline = t0 + max_time
    nU, nV = n * m, m * p
    while time.time() < deadline - 1.0:
        U_np = np.random.randint(-2, 3, (nU, rank))
        V_np = np.random.randint(-2, 3, (nV, rank))
        W_np = _solve_for_W(U_np, V_np, n, m, p, rank)
        if W_np is None:
            continue
        Us = _snap_matrix(U_np.tolist())
        Vs = _snap_matrix(V_np.tolist())
        Ws = _snap_matrix(W_np.tolist())
        if _verify(Us, Vs, Ws, n, m, p):
            return Us, Vs, Ws, rank
    return None


def _try_structured_signs(n, m, p, rank, max_time=5.0):
    """
    U and V columns are random ±1 vectors; solve for W, snap, verify.
    """
    t0 = time.time()
    deadline = t0 + max_time
    nU, nV = n * m, m * p
    while time.time() < deadline - 1.0:
        U_np = np.zeros((nU, rank))
        V_np = np.zeros((nV, rank))
        for c in range(rank):
            U_np[:, c] = np.random.choice([-1, 1], nU)
            V_np[:, c] = np.random.choice([-1, 1], nV)
        W_np = _solve_for_W(U_np, V_np, n, m, p, rank)
        if W_np is None:
            continue
        Us = _snap_matrix(U_np.tolist())
        Vs = _snap_matrix(V_np.tolist())
        Ws = _snap_matrix(W_np.tolist())
        if _verify(Us, Vs, Ws, n, m, p):
            return Us, Vs, Ws, rank
    return None


def _generate_block_patterns(n, m, p, rank):
    """
    Generate structured U, V patterns inspired by Strassen-like block recursion.
    Partition the index sets into overlapping 2x2 or 2x3 blocks and assign
    bilinear coefficients.  Returns a list of (U_np, V_np) pairs.
    """
    patterns = []
    nU, nV = n * m, m * p

    # Pattern 1: 2x2 block tiling with Strassen-like coefficients
    # For 3x4 * 4x4, we can think of the 4x4 as two 2x4 blocks or 4x2 blocks.
    # We'll try a few fixed structured assignments.

    # Pattern: assign each column a "block role"
    # For rank r, we can think of r as sum of block contributions.
    # Each block of size a x b x c contributes a*b*c scalar multiplications in schoolbook,
    # but Strassen reduces 2x2x2 from 8 to 7.

    # Let's try: partition i in {0,1,2}, j in {0,1,2,3}, k in {0,1,2,3}
    # into blocks: (i in {0,1}, j in {0,1}, k in {0,1}), etc.

    # For simplicity, generate random structured patterns where each column
    # has a block-structured pattern in U and V.

    for trial in range(50):
        U_np = np.zeros((nU, rank))
        V_np = np.zeros((nV, rank))
        for c in range(rank):
            # Each column: random block structure
            # U column: for each i, assign a random 2D pattern over j
            for i in range(n):
                # Random ±1 or 0 for each j
                U_np[i * m:(i+1) * m, c] = np.random.choice([-1, 0, 1], m)
            for j in range(m):
                V_np[j * p:(j+1) * p, c] = np.random.choice([-1, 0, 1], p)
        patterns.append((U_np, V_np))

    return patterns


def _try_block_structured(n, m, p, rank, max_time=5.0):
    """
    Try block-structured patterns from _generate_block_patterns.
    """
    t0 = time.time()
    deadline = t0 + max_time
    nU, nV = n * m, m * p

    # Generate a batch of patterns
    patterns = _generate_block_patterns(n, m, p, rank)

    for U_np, V_np in patterns:
        if time.time() > deadline - 1.0:
            break
        W_np = _solve_for_W(U_np, V_np, n, m, p, rank)
        if W_np is None:
            continue
        Us = _snap_matrix(U_np.tolist())
        Vs = _snap_matrix(V_np.tolist())
        Ws = _snap_matrix(W_np.tolist())
        if _verify(Us, Vs, Ws, n, m, p):
            return Us, Vs, Ws, rank

    # Also try pure ±1 structured patterns
    while time.time() < deadline - 1.0:
        U_np = np.zeros((nU, rank))
        V_np = np.zeros((nV, rank))
        for c in range(rank):
            # Block-structured ±1: each 2x2 block of U gets a uniform sign
            for i in range(n):
                for j in range(m):
                    block_sign = np.random.choice([-1, 1])
                    U_np[i * m + j, c] = block_sign
            for j in range(m):
                for k in range(p):
                    block_sign = np.random.choice([-1, 1])
                    V_np[j * p + k, c] = block_sign
        W_np = _solve_for_W(U_np, V_np, n, m, p, rank)
        if W_np is None:
            continue
        Us = _snap_matrix(U_np.tolist())
        Vs = _snap_matrix(V_np.tolist())
        Ws = _snap_matrix(W_np.tolist())
        if _verify(Us, Vs, Ws, n, m, p):
            return Us, Vs, Ws, rank

    return None


def _try_reduce_by_dependency(n, m, p, U, V, W, max_time=5.0):
    """
    Try to eliminate one column at a time by checking if it is a linear combination
    of the others.  For column c, we check if there exist alpha_d (d != c) such that:
    sum_{d != c} alpha_d * U[:,d] * V[:,d] * W[:,d] = U[:,c] * V[:,c] * W[:,c]
    This is a linear system in alpha_d.
    """
    t0 = time.time()
    deadline = t0 + max_time
    current_rank = len(U[0])
    nU, nV, nW = n * m, m * p, p * n

    # Convert to numpy
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Try to eliminate each column
    for c_elim in range(current_rank - 1, -1, -1):
        if time.time() > deadline - 2.0:
            break

        # Build the system: for all (i,j,k),
        # sum_{d != c_elim} alpha_d * U[i*m+j,d] * V[j*p+k,d] * W[k*n+i,d]
        # = U[i*m+j,c_elim] * V[j*p+k,c_elim] * W[k*n+i,c_elim]

        # This is n*m*p equations in (current_rank - 1) unknowns.
        # For rank=48, that's 47 unknowns and 48 equations.

        n_eq = n * m * p
        n_unknowns = current_rank - 1

        A = np.zeros((n_eq, n_unknowns))
        b = np.zeros(n_eq)

        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = U_np[i * m + j, c_elim] * V_np[j * p + k, c_elim] * W_np[k * n + i, c_elim]
                    b[idx] = target
                    for d in range(current_rank):
                        if d == c_elim:
                            continue
                        A[idx, d if d < c_elim else d - 1] = (
                            U_np[i * m + j, d] * V_np[j * p + k, d] * W_np[k * n + i, d]
                        )
                    idx += 1

        try:
            alpha, res, rA, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue

        if np.linalg.norm(A @ alpha - b) > 1e-9:
            continue

        # Column c_elim is a linear combination of the others.
        # Remove it.
        U_new = [row[:c_elim] + row[c_elim + 1:] for row in U]
        V_new = [row[:c_elim] + row[c_elim + 1:] for row in V]
        W_new = [row[:c_elim] + row[c_elim + 1:] for row in W]

        if _verify(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new, current_rank - 1

    return None


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin (evaluator kills at 150s)
    n, m, p = DIMENSIONS

    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U, V, W, n * m * p

    # Try rank 47 with random search
    if time.time() < deadline - 30.0:
        res = _try_random_uv(n, m, p, best_rank - 1, max_time=15.0, seed=42)
        if res:
            best_U, best_V, best_W, best_rank = res

    # Try rank 46 with random search
    if best_rank > 46 and time.time() < deadline - 20.0:
        res = _try_random_uv(n, m, p, 46, max_time=10.0, seed=123)
        if res:
            best_U, best_V, best_W, best_rank = res

    # Try structured sign search for rank 47
    if best_rank > 47 and time.time() < deadline - 15.0:
        res = _try_structured_signs(n, m, p, 47, max_time=10.0)
        if res:
            best_U, best_V, best_W, best_rank = res

    # Try block-structured search for rank 47
    if best_rank > 47 and time.time() < deadline - 10.0:
        res = _try_block_structured(n, m, p, 47, max_time=8.0)
        if res:
            best_U, best_V, best_W, best_rank = res

    # Try block-structured search for rank 46
    if best_rank > 46 and time.time() < deadline - 5.0:
        res = _try_block_structured(n, m, p, 46, max_time=5.0)
        if res:
            best_U, best_V, best_W, best_rank = res

    # Try linear-dependency reduction from schoolbook
    if best_rank > 47 and time.time() < deadline - 3.0:
        res = _try_reduce_by_dependency(n, m, p, U, V, W, max_time=3.0)
        if res:
            best_U, best_V, best_W, best_rank = res

    # Final verification
    if not _verify(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = n * m * p

    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
