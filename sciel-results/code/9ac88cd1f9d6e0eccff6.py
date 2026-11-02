"""Matrix multiplication 2x3x6 tensor decomposition; DIRECT structural construction with rank reduction search."""
DIMENSIONS = [2, 3, 6]

import time
import numpy as np
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110.0  # safety margin before 120s
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 18, 12

    # Build the target tensor as a 3D array T[a,b,c]
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Trivial decomposition of rank 36
    rank_trivial = n * m * p
    U_trivial = np.zeros((nm, rank_trivial), dtype=float)
    V_trivial = np.zeros((mp, rank_trivial), dtype=float)
    W_trivial = np.zeros((pn, rank_trivial), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, col] = 1.0
                V_trivial[b, col] = 1.0
                W_trivial[c, col] = 1.0
                col += 1

    def verify(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        # Compute sum_l U[:,l] * V[:,l] * W[:,l]
        # This is a 3D tensor of shape (nm, mp, pn)
        # For efficiency, we can do it in chunks
        result = np.zeros((nm, mp, pn), dtype=float)
        for l in range(rank):
            result += np.outer(np.outer(U[:, l], V[:, l]), W[:, l])
        return np.max(np.abs(result - T)) < tol

    # Try to find a lower rank decomposition
    # Start from rank 35 down to 18
    best_rank = rank_trivial
    best_decomp = (U_trivial, V_trivial, W_trivial)

    # For each candidate rank, try a few random initializations
    for r in range(rank_trivial - 1, 17, -1):
        if time.time() > deadline - 10:
            break
        # Try a few random attempts for this rank
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            # Initialize with random sparse patterns
            U = np.zeros((nm, r), dtype=float)
            V = np.zeros((mp, r), dtype=float)
            W = np.zeros((pn, r), dtype=float)
            
            # Start from trivial and try to merge columns
            # Or use a random sparse initialization
            for l in range(r):
                # Each column gets a few non-zeros
                nz_u = random.randint(1, 3)
                nz_v = random.randint(1, 3)
                nz_w = random.randint(1, 3)
                u_idx = random.sample(range(nm), min(nz_u, nm))
                v_idx = random.sample(range(mp), min(nz_v, mp))
                w_idx = random.sample(range(pn), min(nz_w, pn))
                for idx in u_idx:
                    U[idx, l] = random.choice([1.0, -1.0, 0.5, -0.5])
                for idx in v_idx:
                    V[idx, l] = random.choice([1.0, -1.0, 0.5, -0.5])
                for idx in w_idx:
                    W[idx, l] = random.choice([1.0, -1.0, 0.5, -0.5])
            
            # Check if this random initialization works (unlikely, but fast check)
            if verify(U, V, W):
                best_rank = r
                best_decomp = (U, V, W)
                break
        
        if best_rank < rank_trivial:
            break

    # Return the best decomposition found
    U_best, V_best, W_best = best_decomp
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
