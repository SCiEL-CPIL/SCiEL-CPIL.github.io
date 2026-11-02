"""Matrix multiplication 3x3x5: tensor decomposition with deadline guard."""

import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all (i, j, k) entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # T[u,v,w] should be 1
                val = np.dot(U_arr[u_idx, :], np.multiply(V_arr[v_idx, :], W_arr[w_idx, :]))
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check that all other (u, v, w) combinations give 0
    # This is expensive but necessary for correctness
    for u in range(n * m):
        for v in range(m * p):
            for w in range(p * n):
                # Skip if this corresponds to a valid (i,j,k)
                i = u // m
                j = u % m
                jj = v // p
                k = v % p
                kk = w // n
                ii = w % n
                if i == ii and j == jj and k == kk:
                    continue  # Already checked above
                val = np.dot(U_arr[u, :], np.multiply(V_arr[v, :], W_arr[w, :]))
                if abs(val) > 1e-9:
                    return False
    return True

def _try_merge_terms(U, V, W, n, m, p):
    """Try to merge pairs of terms to reduce rank."""
    rank = len(U[0])
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    # Try all pairs of columns to see if they can be merged
    for r1 in range(rank):
        for r2 in range(r1 + 1, rank):
            # Check if columns r1 and r2 can be merged
            # Two terms can be merged if their supports are compatible
            # In our case, each term has exactly one 1 in each of U, V, W
            # We can try to replace two terms with one term that covers both
            
            # Get the indices where these columns have non-zero values
            u1_idx = [i for i in range(len(U)) if U[i][r1] != 0]
            u2_idx = [i for i in range(len(U)) if U[i][r2] != 0]
            v1_idx = [i for i in range(len(V)) if V[i][r1] != 0]
            v2_idx = [i for i in range(len(V)) if V[i][r2] != 0]
            w1_idx = [i for i in range(len(W)) if W[i][r1] != 0]
            w2_idx = [i for i in range(len(W)) if W[i][r2] != 0]
            
            # For schoolbook, each column has exactly one non-zero in each matrix
            # Try to see if we can create a new term that covers both
            # This requires the new term to have non-zero entries at all these positions
            
            # Simple heuristic: if the two terms share any index in U, V, or W,
            # we might be able to merge them with appropriate coefficients
            
            # For now, just try removing one term and see if the decomposition still works
            # This is a very crude approach but may find some improvements
            
            # Create a copy without column r2
            new_rank = rank - 1
            new_U = [[U[i][j] for j in range(rank) if j != r2] for i in range(len(U))]
            new_V = [[V[i][j] for j in range(rank) if j != r2] for i in range(len(V))]
            new_W = [[W[i][j] for j in range(rank) if j != r2] for i in range(len(W))]
            
            # Check if the decomposition still works
            # This is expensive, so only do it if we have time
            if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = new_rank
                break
        if best_rank < rank:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    """Return a rank-minimal exact tensor decomposition for 3x3 * 3x5."""
    start_time = time.time()
    deadline = start_time + 105  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    rank = n * m * p  # 45

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i  # W[k*n + i, r] convention
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    
    # Try to reduce rank by merging terms
    if time.time() < deadline - 30:
        U, V, W, rank = _try_merge_terms(U, V, W, n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
