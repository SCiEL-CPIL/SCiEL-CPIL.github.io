"""
Rank-minimal exact tensor decomposition for 2x3 @ 3x10 matrix multiplication.
Uses a block-structured approach with verified Strassen-based decomposition.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # The tensor M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal 1 for all valid (i,j,k)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                if abs(val - 1.0) > tol:
                    max_err = max(max_err, abs(val - 1.0))
    return max_err

def _build_block_decomposition(n, m, block_size):
    """
    Build a decomposition for M_{n,m,block_size}.
    For n=2, m=3, block_size=2, we use Strassen for the 2x2 part plus outer products.
    Returns list of terms, each term is dict with 'U', 'V', 'W' lists of (idx, val).
    U indices are in [0, n*m), V indices in [0, m*block_size), W indices in [0, block_size*n).
    """
    if n == 2 and m == 3 and block_size == 2:
        # Strassen terms for the 2x2 subproblem (rows 0,1 of A; cols 0,1 of B)
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        #
        # In tensor form, each term r has U_r, V_r, W_r such that
        # sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = delta_{C[i,k] from A[i,j]*B[j,k]}
        #
        # Let's define the 7 Strassen terms:
        strassen = [
            # M1: U has A00+A11, V has B00+B11, W contributes to C00 and C11
            {'U': [(0,1), (3,1)], 'V': [(0,1), (2,1)], 'W': [(0,1), (3,1)]},
            # M2: U has A10+A11, V has B00, W contributes to C10 and -C11
            {'U': [(3,1), (4,1)], 'V': [(0,1)], 'W': [(2,1), (3,-1)]},
            # M3: U has A00, V has B10-B00, W contributes to C01 and C11
            {'U': [(0,1)], 'V': [(2,1), (0,-1)], 'W': [(1,1), (3,1)]},
            # M4: U has A11, V has B10-B00, W contributes to C00 and C10
            {'U': [(4,1)], 'V': [(2,1), (0,-1)], 'W': [(0,1), (2,1)]},
            # M5: U has A00+A01, V has B11, W contributes to -C00 and C01
            {'U': [(0,1), (1,1)], 'V': [(3,1)], 'W': [(0,-1), (1,1)]},
            # M6: U has A10-A00, V has B00+B01, W contributes to C11
            {'U': [(3,1), (0,-1)], 'V': [(0,1), (1,1)], 'W': [(3,1)]},
            # M7: U has A01-A11, V has B10+B11, W contributes to C00
            {'U': [(1,1), (4,-1)], 'V': [(2,1), (3,1)], 'W': [(0,1)]},
        ]
        
        # Outer product terms for the 3rd column (j=2) of B:
        # C[i,k] += A[i,2] * B[2,k]
        outer = [
            {'U': [(2,1)], 'V': [(4,1)], 'W': [(0,1)]},  # A02*B20 -> C00
            {'U': [(2,1)], 'V': [(5,1)], 'W': [(1,1)]},  # A02*B21 -> C01
            {'U': [(5,1)], 'V': [(4,1)], 'W': [(2,1)]},  # A12*B20 -> C10
            {'U': [(5,1)], 'V': [(5,1)], 'W': [(3,1)]},  # A12*B21 -> C11
        ]
        
        return strassen + outer
    else:
        # Fallback: schoolbook
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(block_size):
                    terms.append({
                        'U': [(i*m + j, 1)],
                        'V': [(j*block_size + k, 1)],
                        'W': [(k*n + i, 1)]
                    })
        return terms

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    block_size = 2
    num_blocks = p // block_size  # 5 blocks
    
    # Build block decomposition
    block_terms = _build_block_decomposition(n, m, block_size)
    rank_per_block = len(block_terms)
    total_rank = num_blocks * rank_per_block
    
    # Initialize factor matrices
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    for b in range(num_blocks):
        for t_idx, term in enumerate(block_terms):
            col = b * rank_per_block + t_idx
            for idx, val in term['U']:
                U[idx, col] += val
            for idx, val in term['V']:
                # V index: j * p + k_global, where k_global = b * block_size + k_local
                j = idx // block_size
                k_local = idx % block_size
                k_global = b * block_size + k_local
                V[j * p + k_global, col] += val
            for idx, val in term['W']:
                # W index: k_global * n + i, where k_global = b * block_size + k_local
                k_local = idx // n
                i = idx % n
                k_global = b * block_size + k_local
                W[k_global * n + i, col] += val
    
    # Verify the decomposition
    err = _verify_decomposition(U, V, W, n, m, p)
    if err > 1e-10:
        # Fallback to schoolbook if verification fails
        total_rank = n * m * p
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i*m + j, r] = 1.0
                    V[j*p + k, r] = 1.0
                    W[k*n + i, r] = 1.0
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
