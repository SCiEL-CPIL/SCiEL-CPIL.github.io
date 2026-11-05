"""
Rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.

The matmul tensor T has indices (a, b, c) where:
  a = i*m + j  (i in [0,3), j in [0,4))  -> 12 indices
  b = j'*p + k (j' in [0,4), k in [0,5)) -> 20 indices
  c = k'*n + i'(k' in [0,5), i' in [0,3)) -> 15 indices

T[a, b, c] = 1 iff i == i', j == j', k == k', else 0.

A rank-r decomposition is U (12 x r), V (20 x r), W (15 x r) such that
  sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t] = T[a,b,c]  for all a,b,c.

The schoolbook rank is 60. We attempt to find a lower rank via structured search.
"""

import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 12, 20, 15

    # Build the matmul tensor T
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook decomposition (rank 60)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    # Strategy: For a given rank r, fix U and V, then solve for W via least squares.
    # The system for each c is: (U ⊙ V)_flat @ W[c,:] = T[:, :, c].flat
    # where (U ⊙ V)[a,b,t] = U[a,t]*V[b,t]

    # Try structured random searches with entries from {-1, 0, 1} and {-1, -0.5, 0, 0.5, 1}
    candidate_ranks = [55, 50, 48, 45, 42, 40, 38, 35, 32, 30, 28, 25]

    for r in candidate_ranks:
        if time.time() > deadline - 25:
            break

        # Multiple trial configurations
        max_trials = 80 if r >= 45 else 120
        found = False

        for trial in range(max_trials):
            if time.time() > deadline - 10:
                break

            # Random U, V with entries from a small set
            # Use a mix of dense and sparse patterns
            if trial % 3 == 0:
                # Dense {-1, 0, 1}
                U = np.random.choice([-1.0, 0.0, 1.0], size=(nm, r))
                V = np.random.choice([-1.0, 0.0, 1.0], size=(mp, r))
            elif trial % 3 == 1:
                # Half-integer entries
                U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(nm, r))
                V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(mp, r))
            else:
                # Mostly sparse with some structure
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                for t in range(r):
                    # Each column of U has ~3 nonzeros, V has ~4 nonzeros
                    nz_u = np.random.choice(nm, size=min(3, nm), replace=False)
                    nz_v = np.random.choice(mp, size=min(4, mp), replace=False)
                    U[nz_u, t] = np.random.choice([-1.0, 1.0], size=len(nz_u))
                    V[nz_v, t] = np.random.choice([-1.0, 1.0], size=len(nz_v))

            # Build M[a,b,t] = U[a,t] * V[b,t], flatten to (nm*mp, r)
            M = np.einsum('at,bt->abt', U, V)
            M_flat = M.reshape(nm * mp, r)

            # Check if M_flat has enough rank
            try:
                rank_M = np.linalg.matrix_rank(M_flat)
            except:
                continue
            if rank_M < r:
                # Still might work if the system is consistent, but unlikely
                pass

            # Solve for W: for each c, M_flat @ W[c,:] = T[:,:,c].flat
            W = np.zeros((pn, r))
            valid = True
            for c in range(pn):
                T_col = T[:, :, c].reshape(-1)
                try:
                    w, residuals, rank_sol, sv = np.linalg.lstsq(M_flat, T_col, rcond=None)
                    W[c, :] = w
                    residual = np.linalg.norm(M_flat @ w - T_col)
                    if residual > 1e-8:
                        valid = False
                        break
                except:
                    valid = False
                    break

            if not valid:
                continue

            # Full verification
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            if np.allclose(recon, T, atol=1e-8):
                best_rank = r
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                found = True
                break

        if found:
            break

    # If we found something better, great. Otherwise return schoolbook.
    # Convert to lists for the interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
