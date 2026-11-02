"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_rank28():
    """Build the rank-28 Strassen decomposition deterministically."""
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for b in range(2):  # B column block (which 2x2 block of B's columns)
        for a_blk in range(2):  # A column block (which 2x2 block of A's columns / B's rows)
            for s in range(7):
                c = col

                # U contributions: A[i, 2*a_blk + j]
                U_contrib = {
                    0: [(0, 2*a_blk, 1), (1, 2*a_blk+1, 1)],
                    1: [(1, 2*a_blk, 1), (1, 2*a_blk+1, 1)],
                    2: [(0, 2*a_blk, 1)],
                    3: [(1, 2*a_blk+1, 1)],
                    4: [(0, 2*a_blk, 1), (0, 2*a_blk+1, 1)],
                    5: [(1, 2*a_blk, 1), (0, 2*a_blk, -1)],
                    6: [(0, 2*a_blk+1, 1), (1, 2*a_blk+1, -1)],
                }
                for (i, j, coeff) in U_contrib[s]:
                    U[i * 4 + j, c] += coeff

                # V contributions: B[2*a_blk + j, 2*b + k]
                V_contrib = {
                    0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    1: [(2*a_blk, 2*b, 1)],
                    2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                    3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                    4: [(2*a_blk+1, 2*b+1, 1)],
                    5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                    6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                }
                for (j, k, coeff) in V_contrib[s]:
                    V[j * 4 + k, c] += coeff

                # W contributions: C[i, 2*b + k]
                # Convention: W[k*n + i, r] where i is row of C, k is col of C
                W_contrib = {
                    0: {(0, 0): 1, (1, 1): 1},
                    1: {(1, 0): 1, (1, 1): -1},
                    2: {(0, 1): 1, (1, 1): 1},
                    3: {(0, 0): 1, (1, 0): 1},
                    4: {(0, 0): -1, (0, 1): 1},
                    5: {(1, 1): 1},
                    6: {(0, 0): 1},
                }
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    W[(2 * b + k_loc) * 2 + i_loc, c] += coeff

                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W correctly decompose the matmul tensor.
    
    For matmul: C[i,j] = sum_k A[i,k] * B[k,j]
    Tensor form: T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    This should equal 1 for all i, k, j.
    """
    rank = U.shape[1]
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_rank27_by_null_space(n, m, p, U28, V28, W28, deadline):
    """Try to find a rank-27 decomposition by checking if one Strassen term
    can be absorbed into the others via a linear combination."""
    rank28 = U28.shape[1]
    
    # Build the tensor as a vector: T[i,k,j] indexed as i*m*m + k*p + j
    # Actually T has shape (n, m, p) = (2, 4, 4), so 32 entries
    # Flatten: index = i * (m*p) + k * p + j
    
    def tensor_from_factors(U, V, W):
        """Compute the full tensor from factors."""
        T = np.zeros((n, m, p), dtype=float)
        for r in range(U.shape[1]):
            for i in range(n):
                for k in range(m):
                    u_val = U[i*m + k, r]
                    if u_val == 0:
                        continue
                    for j in range(p):
                        T[i, k, j] += u_val * V[k*p + j, r] * W[j*n + i, r]
        return T
    
    target_tensor = tensor_from_factors(U28, V28, W28)
    
    # For each candidate term to remove, check if the remaining 27 terms
    # can be adjusted (by scaling) to still produce the target tensor.
    # 
    # Strategy: For term r, we want to find scalars alpha_s (s != r) such that
    # sum_{s != r} alpha_s * (U_s ⊗ V_s ⊗ W_s) = target_tensor
    #
    # This is a system of 32 equations in 27 unknowns, so it's overdetermined.
    # We can try least squares and check if the residual is near zero.
    
    for r in range(rank28):
        if time.time() > deadline - 5:
            break
        
        # Build the design matrix: each column is the flattened tensor for term s
        # We exclude term r
        cols = []
        indices = []
        for s in range(rank28):
            if s == r:
                continue
            # Compute tensor for term s
            T_s = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for k in range(m):
                    u_val = U28[i*m + k, s]
                    if u_val == 0:
                        continue
                    for j in range(p):
                        T_s[i, k, j] += u_val * V28[k*p + j, s] * W28[j*n + i, s]
            cols.append(T_s.flatten())
            indices.append(s)
        
        if len(cols) < 1:
            continue
        
        A = np.column_stack(cols)  # 32 x 27
        b = target_tensor.flatten()  # 32
        
        # Solve least squares
        try:
            alpha, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        except np.linalg.LinAlgError:
            continue
        
        # Check residual
        residual = np.linalg.norm(A @ alpha - b)
        if residual < 1e-8:
            # Found a valid rank-27 decomposition!
            U27 = np.zeros((n*m, rank28-1), dtype=float)
            V27 = np.zeros((m*p, rank28-1), dtype=float)
            W27 = np.zeros((p*n, rank28-1), dtype=float)
            
            idx = 0
            for s in indices:
                U27[:, idx] = alpha[idx] * U28[:, s]
                V27[:, idx] = V28[:, s]
                W27[:, idx] = W28[:, s]
                idx += 1
            
            if _verify_decomposition(U27, V27, W27, n, m, p):
                return U27, V27, W27
    
    return None


def construct():
    """
    Return a rank-28 exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    Uses Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 115  # 120s limit, keep margin

    # Build the base rank-28 Strassen decomposition
    U28, V28, W28 = _build_strassen_rank28()
    
    # Verify it's correct
    assert _verify_decomposition(U28, V28, W28, n, m, p), "Base rank-28 decomposition failed verification"
    
    # Try to find rank-27 via null space analysis
    result = _try_rank27_by_null_space(n, m, p, U28, V28, W28, deadline)
    if result is not None:
        U27, V27, W27 = result
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    # Return the rank-28 solution
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

# EVOLVE-BLOCK-END
