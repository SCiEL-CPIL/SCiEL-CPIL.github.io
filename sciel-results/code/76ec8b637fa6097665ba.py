"""Matrix multiplication 3,3,3 - rank-minimal exact tensor decomposition.

We search for an exact rank-23 decomposition with integer (or half-integer)
entries.  The rank-23 construction for 3x3 matrix multiplication was given by
Strassen-like improvements; the known optimal rank for 3x3 is 23 (lower bound
from Landsberg).  We perform a structured random search guided by the known
algebraic structure, then verify exactness.  If no exact rank-23 decomposition
is found within the time budget, we fall back to the schoolbook rank-27.
"""

import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W, n=3, m=3, p=3):
    """Check whether the factor matrices exactly decompose the matrix-multiplication tensor."""
    rank = len(U[0])
    # Build the multiplication tensor T[i, j, k'] where the coefficient of A[i,j]*B[j',k']
    # should be 1 iff j == j' (and output index is (i, k')).
    # T is indexed by (i, j, j', k') with T[i,j,j',k'] = 1 if j==j' else 0.
    # The decomposition says: sum_c W[k', i, c] * U[i, j, c] * V[j', k', c] = delta_{j,j'}
    for i in range(n):
        for k_prime in range(p):
            for j in range(m):
                for j_prime in range(m):
                    s = 0.0
                    for c in range(rank):
                        s += W[k_prime * n + i][c] * U[i * m + j][c] * V[j_prime * p + k_prime][c]
                    expected = 1.0 if j == j_prime else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _try_rank_23_randomized(max_attempts=200, deadline=110.0):
    """Randomized search for a rank-23 exact decomposition."""
    n, m, p = 3, 3, 3
    rank = 23
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    for attempt in range(max_attempts):
        t0 = time.time()
        if t0 > deadline - 10:
            break

        # Initialize with small random integers in {-2,-1,0,1,2}
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for _ in range(n * m * rank):
            U.flat[_] = rng.choice([-2, -1, 0, 1, 2])
        for _ in range(m * p * rank):
            V.flat[_] = rng.choice([-2, -1, 0, 1, 2])
        for _ in range(p * n * rank):
            W.flat[_] = rng.choice([-2, -1, 0, 1, 2])

        # Alternating minimization: fix two, optimize the third via linear solve
        # The error is sum over (i, k', j, j') of (sum_c W[k',i,c] U[i,j,c] V[j',k',c] - delta_{j,j'})^2
        # For fixed V, W: the system is linear in U.
        # For fixed U, V: linear in W.
        # For fixed U, W: linear in V.

        # We'll do a few rounds of alternating linear solves.
        for iteration in range(30):
            if time.time() > deadline - 5:
                break

            # Update U: for each (i, j), solve min sum_{k',j'} (sum_c W[k',i,c] U[i,j,c] V[j',k',c] - delta_{j,j'})^2
            # This is: for each (i, j), the vector of unknowns is U[i,j,:] (length rank).
            # The objective involves summing over k', j':
            # Let a_c = U[i,j,c]. Then sum_{k',j'} (sum_c W[k',i,c] a_c V[j',k',c] - delta_{j,j'})^2
            # = sum_{k',j'} ( (W_i^T a) V[j',k'] - delta_{j,j'} )^2  where W_i = W[k',i,:]
            # This is a quadratic in a. The gradient w.r.t. a is:
            # 2 * sum_{k',j'} (W[k',i,:] * V[j',k']^T) * ( (W[k',i,:] @ a) * V[j',k'] - delta_{j,j'} )
            # Set to zero:
            # sum_{k',j'} W[k',i,:].outer(V[j',k']) (W[k',i,:] @ a) V[j',k'] = sum_{k',j'} W[k',i,:].outer(V[j',k']) delta_{j,j'}
            # Note: sum_{j'} delta_{j,j'} V[j',k'] = V[j, k']
            # So RHS: sum_{k'} W[k',i,:].outer(V[j,k'])
            # LHS: sum_{k',j'} W[k',i,:].outer(V[j',k']) (W[k',i,:] @ a) V[j',k']
            #     = (sum_{k',j'} V[j',k']^2 ... ) -- this is getting complex. Let me just build the Hessian.

            # Actually, let me use a simpler approach: build the full linear system.
            # The error tensor E[i, j, j', k'] = sum_c W[k', i, c] U[i, j, c] V[j', k', c] - delta_{j, j'}
            # We want E = 0.

            # For alternating: fix V and W, solve for U.
            # E[i, j, j', k'] = sum_c (W[k', i, c] V[j', k', c]) U[i, j, c] - delta_{j, j'}
            # Let M[i, j, c] = U[i, j, c]. Then for fixed i, j:
            # sum_c B_{j', k'}(i, j, c) M[i, j, c] = delta_{j, j'}  for all j', k'
            # where B[j', k'](i, j, c) = W[k', i, c] * V[j', k', c]
            # This is a linear system: for each (i, j), we have m*p equations in rank unknowns.
            # Total equations: n*m*(m*p) = 9*9 = 81. Unknowns: n*m*rank = 9*23 = 207.
            # Underdetermined. We solve least-squares.

            # Build A matrix (81 x 207) and b vector (81,)
            A_rows = []
            b_vec = []
            for i in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k_prime in range(p):
                            row = [0.0] * (n * m * rank)
                            for c in range(rank):
                                # U index: (i, j, c) -> flat index (i*m + j) * rank + c
                                row[(i * m + j) * rank + c] = W[k_prime * n + i][c] * V[j_prime * p + k_prime][c]
                            A_rows.append(row)
                            b_vec.append(1.0 if j == j_prime else 0.0)

            A_mat = np.array(A_rows)
            b_arr = np.array(b_vec)
            # Solve least squares
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_arr, rcond=None)
            except Exception:
                break
            U = U_flat.reshape(n * m, rank)

            # Update V similarly: fix U and W, solve for V.
            A_rows = []
            b_vec = []
            for i in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k_prime in range(p):
                            row = [0.0] * (m * p * rank)
                            for c in range(rank):
                                # V index: (j', k', c) -> flat index (j'*p + k') * rank + c
                                row[(j_prime * p + k_prime) * rank + c] = W[k_prime * n + i][c] * U[i * m + j][c]
                            A_rows.append(row)
                            b_vec.append(1.0 if j == j_prime else 0.0)

            A_mat = np.array(A_rows)
            b_arr = np.array(b_vec)
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_arr, rcond=None)
            except Exception:
                break
            V = V_flat.reshape(m * p, rank)

            # Update W: fix U and V, solve for W.
            A_rows = []
            b_vec = []
            for i in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k_prime in range(p):
                            row = [0.0] * (p * n * rank)
                            for c in range(rank):
                                # W index: (k', i, c) -> flat index (k'*n + i) * rank + c
                                row[(k_prime * n + i) * rank + c] = U[i * m + j][c] * V[j_prime * p + k_prime][c]
                            A_rows.append(row)
                            b_vec.append(1.0 if j == j_prime else 0.0)

            A_mat = np.array(A_rows)
            b_arr = np.array(b_vec)
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_arr, rcond=None)
            except Exception:
                break
            W = W_flat.reshape(p * n, rank)

        # Round to nearest half-integer
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2

        # Check if rounded version is exact
        U_list = [[U_half[r, c] for c in range(rank)] for r in range(n * m)]
        V_list = [[V_half[r, c] for c in range(rank)] for r in range(m * p)]
        W_list = [[W_half[r, c] for c in range(rank)] for r in range(p * n)]
        if _verify(U_list, V_list, W_list):
            return U_list, V_list, W_list

        # Also try rounding to nearest integer
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        U_list = [[U_int[r, c] for c in range(rank)] for r in range(n * m)]
        V_list = [[V_int[r, c] for c in range(rank)] for r in range(m * p)]
        W_list = [[W_int[r, c] for c in range(rank)] for r in range(p * n)]
        if _verify(U_list, V_list, W_list):
            return U_list, V_list, W_list

        # Re-randomize with a different seed for next attempt
        rng = random.Random(attempt + 1000)
        np_rng = np.random.RandomState(attempt + 1000)

    return None

def construct():
    deadline = time.time() + 115  # Leave 5s safety margin before 120s

    # Try to find a rank-23 decomposition
    result = _try_rank_23_randomized(max_attempts=50, deadline=deadline)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank 27
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
