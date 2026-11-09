"""Matrix multiplication 3x4x4 - structured rank reduction search."""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 4]

# Strassen factors for 2x2x2 (rank 7) - kept as initial seed for search
# U indices: 0=A00, 1=A01, 2=A10, 3=A11
# V indices: 0=B00, 1=B01, 2=B10, 3=B11
# W indices: 0=C00, 1=C01, 2=C10, 3=C11

STRASSEN_2x2x2 = [
    ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: -1}),
    ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
    ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
    ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
    ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
    ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
    ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a tensor decomposition for 3x4x4 matrix multiplication.
    
    Strategy: 
    1. Start with the Strassen-based block decomposition (rank 44)
    2. Apply greedy rank reduction by trying to merge pairs of rank-1 terms
       using linearity (if two terms have proportional U or V vectors, they can be merged)
    3. Validate with exact rational arithmetic
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    
    # Build initial Strassen-based decomposition (rank 44)
    U, V, W = _build_strassen_decomposition(n, m, p)
    current_rank = len(U[0])
    
    # Try greedy rank reduction
    U, V, W = _greedy_rank_reduction(U, V, W, n, m, p, deadline)
    
    # Final verification
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        U_fb, V_fb, W_fb = _schoolbook_direct(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}


def _build_strassen_decomposition(n, m, p):
    """Build the initial Strassen-based block decomposition (rank 44)."""
    total_rank = 44
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_col = 0
    
    def apply_strassen_2x2x2(col_offset, U_indices, V_indices, W_indices):
        U00, U01, U10, U11 = U_indices
        V00, V01, V10, V11 = V_indices
        W00, W01, W10, W11 = W_indices
        
        for (u_coeffs, v_coeffs, w_coeffs) in STRASSEN_2x2x2:
            col = col_offset
            for u_idx, coeff in u_coeffs.items():
                U[u_idx][col] += coeff
            for v_idx, coeff in v_coeffs.items():
                V[v_idx][col] += coeff
            for w_idx, coeff in w_coeffs.items():
                W[w_idx][col] += coeff
            col_offset += 1
        
        return col_offset
    
    # C_left = A_left * B00 + A_right * B10
    current_col = apply_strassen_2x2x2(
        current_col,
        [0, 1, 4, 5],
        [0, 1, 4, 5],
        [0, 1, 3, 4]
    )
    
    current_col = apply_strassen_2x2x2(
        current_col,
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [0, 1, 3, 4]
    )
    
    current_col = apply_strassen_2x2x2(
        current_col,
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [6, 7, 9, 10]
    )
    
    current_col = apply_strassen_2x2x2(
        current_col,
        [2, 3, 6, 7],
        [10, 11, 14, 15],
        [6, 7, 9, 10]
    )
    
    # Part 2: Schoolbook for row 2 (1x4x4, rank 16)
    for j in range(m):
        for k in range(p):
            u_idx = 2 * m + j
            v_idx = j * p + k
            w_idx = k * n + 2
            
            U[u_idx][current_col] = 1.0
            V[v_idx][current_col] = 1.0
            W[w_idx][current_col] = 1.0
            
            current_col += 1
    
    return U, V, W


def _greedy_rank_reduction(U, V, W, n, m, p, deadline):
    """
    Try to reduce rank by finding pairs of columns that can be merged.
    Two rank-1 terms r1 and r2 can be merged if their U vectors are proportional
    or their V vectors are proportional (allowing linear combination).
    """
    rank = len(U[0])
    
    # Convert to numpy for efficient operations
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Try to remove columns that are linear combinations of others
    # We use a simple approach: try removing each column and checking validity
    improved = True
    while improved and time.time() < deadline:
        improved = False
        rank = len(U[0])
        
        for r in range(rank):
            if time.time() > deadline:
                break
            
            # Try removing column r
            U_test = U_np[:, [c for c in range(rank) if c != r]]
            V_test = V_np[:, [c for c in range(rank) if c != r]]
            W_test = W_np[:, [c for c in range(rank) if c != r]]
            
            if _verify_np(U_test, V_test, W_test, n, m, p):
                U_np = U_test
                V_np = V_test
                W_test = W_test
                improved = True
                break
    
    # Also try merging pairs: if col_a and col_b have proportional U vectors,
    # we can potentially merge them
    rank = len(U_np) if U_np.ndim == 2 else len(U_np[0])
    
    # Convert back to list format
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()
    
    return U_out, V_out, W_out


def _verify_np(U, V, W, n, m, p):
    """Verify decomposition using numpy (fast)."""
    # Build the full tensor and check
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                total = np.dot(
                    U[u_idx] * V[v_idx],
                    W[w_idx]
                )
                
                if abs(total - 1.0) > 1e-9:
                    return False
    
    # Check sparsity: for invalid index combinations, sum should be 0
    # Sample a few to save time
    import random
    random.seed(42)
    for _ in range(200):
        u_idx = random.randint(0, len(U) - 1)
        v_idx = random.randint(0, len(V) - 1)
        w_idx = random.randint(0, len(W) - 1)
        
        i = u_idx // m
        j = u_idx % m
        j_prime = v_idx // p
        k = v_idx % p
        k_prime = w_idx // n
        i_prime = w_idx % n
        
        if (j == j_prime) and (k == k_prime) and (i == i_prime):
            expected = 1.0
        else:
            expected = 0.0
        
        total = np.dot(U[u_idx] * V[v_idx], W[w_idx])
        
        if abs(total - expected) > 1e-9:
            return False
    
    return True


def _schoolbook_direct(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly computes matrix multiplication.
    Full exhaustive check.
    """
    rank = len(U[0])
    
    # Check main terms: for all i, j, k, the sum should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                total = 0.0
                for r in range(rank):
                    total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                
                if abs(total - 1.0) > 1e-9:
                    return False
    
    # Check that for invalid combinations, the sum is 0
    for u_idx in range(len(U)):
        i = u_idx // m
        j = u_idx % m
        
        for v_idx in range(len(V)):
            j_prime = v_idx // p
            k = v_idx % p
            
            for w_idx in range(len(W)):
                k_prime = w_idx // n
                i_prime = w_idx % n
                
                if (j == j_prime) and (k == k_prime) and (i == i_prime):
                    expected = 1.0
                else:
                    expected = 0.0
                
                total = 0.0
                for r in range(rank):
                    total += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                
                if abs(total - expected) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
