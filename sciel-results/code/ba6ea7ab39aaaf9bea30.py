import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _time_ok(deadline):
    return time.time() < deadline

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Strassen components for 2x2x2
    # local_a: 0=A11, 1=A12, 2=A21, 3=A22
    # local_b: 0=B11, 1=B12, 2=B21, 3=B22
    # local_w: 0=C11, 1=C12, 2=C21, 3=C22
    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),           # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),           # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),           # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),           # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),           # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7
    ]
    
    # Blocks: (a_rows, a_cols, b_rows, b_cols)
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    rank = len(blocks) * len(strassen)  # 6 * 7 = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for a_rows, a_cols, b_rows, b_cols in blocks:
        if not _time_ok(deadline):
            break
        
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            if not _time_ok(deadline):
                break
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx], col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx], col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx], col] += coeff
            col += 1
    
    # Try to reduce rank by finding linear dependencies among the rank-1 terms
    # The tensor T is defined by T[i,j,k] = delta_{i+k, j}
    # A rank-1 term is u ⊗ v ⊗ w
    # Check if columns of U, V, W have linear dependencies
    
    if _time_ok(deadline):
        # Form the tensor and check for rank reduction
        # For each rank-1 term r, the contribution is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # We can check if any set of columns is linearly dependent
        # by checking the rank of the concatenated tensor
        
        # Build the full tensor T from the decomposition
        tensor = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            if not _time_ok(deadline):
                break
            tensor += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        
        # Verify correctness against the true multiplication tensor
        true_tensor = np.zeros((n*m, m*p, p*n))
        for i in range(n*m):
            for j in range(m*p):
                for k in range(p*n):
                    # i = a_row*m + a_col, j = b_row*p + b_col, k = c_col*n + c_row
                    a_row = i // m
                    a_col = i % m
                    b_row = j // p
                    b_col = j % p
                    c_col = k // n
                    c_row = k % n
                    # T[i,j,k] = 1 if a_row == c_row and b_col == c_col and a_col == b_row
                    if a_row == c_row and b_col == c_col and a_col == b_row:
                        true_tensor[i,j,k] = 1.0
        
        # Check if decomposition is correct
        if not np.allclose(tensor, true_tensor, atol=1e-10):
            # Fallback: return the Strassen decomposition as-is
            pass
        else:
            # Try to reduce rank by finding linearly dependent rank-1 terms
            # This is a hard problem, but we can try a simple approach:
            # Check if any column of U is zero (unlikely in Strassen)
            # Or use SVD to find low-rank structure
            
            # For now, try a greedy approach: remove terms that are linear combinations
            # of others. This is NP-hard in general, so we'll just check for exact zeros
            # or very small coefficients.
            
            # Actually, let's try a different approach: use the fact that the tensor
            # has a specific structure. The rank of the 2x4 x 4x6 multiplication tensor
            # is known to be at most 42 (from this construction). We can try to find
            # if it's lower.
            
            # One approach: try to express the tensor as a sum of fewer rank-1 terms
            # by solving a system of equations. This is computationally expensive.
            
            # For now, just return the current decomposition
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
