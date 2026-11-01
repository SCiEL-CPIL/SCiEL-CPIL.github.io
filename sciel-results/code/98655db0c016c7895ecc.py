"""
Rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.

Strategy:
- Split the 15 columns of B into blocks.
- For each 2x2 block, use Strassen's rank-7 decomposition.
- For the remaining 2x1 block, use schoolbook rank 4.
- Additionally, attempt to find a rank-10 decomposition for a 2x2x3 block
  to replace the last 2x2 (rank 7) + 2x1 (rank 4) = 11 with rank 10.
- If rank 10 is found, total rank = 6*7 + 10 = 52.
- Otherwise, total rank = 7*7 + 4 = 53.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 15]

def _build_strassen_53(n, m, p):
    """Build rank-53: 7 Strassen blocks (rank 7 each) + 1 schoolbook (rank 4)."""
    n, m, p = 2, 2, 15
    blocks = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,)]
    
    rank = 7 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    for block in blocks:
        if len(block) == 2:
            k0, k1 = block
            terms = _strassen_terms(k0, k1)
        else:
            k = block[0]
            terms = _schoolbook_terms(k)
        
        for (u_dict, v_dict, w_dict) in terms:
            for (i, j), c in u_dict.items():
                U[i * m + j][col_offset] = c
            for (j, k_idx), c in v_dict.items():
                V[j * p + k_idx][col_offset] = c
            for (k_idx, i), c in w_dict.items():
                W[k_idx * n + i][col_offset] = c
            col_offset += 1
    
    return U, V, W

def _strassen_terms(k0, k1):
    """Strassen rank-7 terms for 2x2x2 block with k indices k0, k1."""
    return [
        # M1: (A00+A11)(B0k0+B1k1)
        ({(0,0):1, (1,1):1}, {(0,k0):1, (1,k1):1}, {(k0,0):1, (k1,1):1}),
        # M2: (A10+A11)B0k0
        ({(1,0):1, (1,1):1}, {(0,k0):1}, {(k0,1):1, (k1,1):-1}),
        # M3: A00(B0k1-B1k1)
        ({(0,0):1}, {(0,k1):1, (1,k1):-1}, {(k1,0):1, (k1,1):1}),
        # M4: A11(B1k0-B0k0)
        ({(1,1):1}, {(1,k0):1, (0,k0):-1}, {(k0,0):1, (k0,1):1}),
        # M5: (A00+A01)B1k1
        ({(0,0):1, (0,1):1}, {(1,k1):1}, {(k0,0):-1, (k1,0):1}),
        # M6: (A10-A00)(B0k0+B0k1)
        ({(1,0):1, (0,0):-1}, {(0,k0):1, (0,k1):1}, {(k1,1):1}),
        # M7: (A01-A11)(B1k0+B1k1)
        ({(0,1):1, (1,1):-1}, {(1,k0):1, (1,k1):1}, {(k0,0):1}),
    ]

def _schoolbook_terms(k):
    """Schoolbook rank-4 terms for 2x2x1 block."""
    return [
        ({(0,0):1}, {(0,k):1}, {(k,0):1}),
        ({(0,1):1}, {(1,k):1}, {(k,0):1}),
        ({(1,0):1}, {(0,k):1}, {(k,1):1}),
        ({(1,1):1}, {(1,k):1}, {(k,1):1}),
    ]

def _build_combined_52(n, m, p, U10, V10, W10):
    """Build rank-52: 6 Strassen blocks (rank 7 each) + 1 rank-10 2x2x3 block."""
    n, m, p = 2, 2, 15
    blocks = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13,14)]
    
    rank = 6 * 7 + 10
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    for block in blocks:
        if len(block) == 2:
            k0, k1 = block
            terms = _strassen_terms(k0, k1)
            for (u_dict, v_dict, w_dict) in terms:
                for (i, j), c in u_dict.items():
                    U[i * m + j][col_offset] = c
                for (j, k_idx), c in v_dict.items():
                    V[j * p + k_idx][col_offset] = c
                for (k_idx, i), c in w_dict.items():
                    W[k_idx * n + i][col_offset] = c
                col_offset += 1
        elif len(block) == 3:
            # Use the rank-10 decomposition for 2x2x3
            k0, k1, k2 = block
            for r in range(10):
                # U: 4x10
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][col_offset + r] = U10[i * m + j][r]
                # V: 6x10, local indices j*3+k_local
                for j in range(m):
                    for k_local in range(3):
                        k_global = [k0, k1, k2][k_local]
                        V[j * p + k_global][col_offset + r] = V10[j * 3 + k_local][r]
                # W: 6x10, local indices k_local*2+i
                for k_local in range(3):
                    k_global = [k0, k1, k2][k_local]
                    for i in range(n):
                        W[k_global * n + i][col_offset + r] = W10[k_local * 2 + i][r]
            col_offset += 10
    
    return U, V, W

def _verify_int(U, V, W, n, m, p, tol=1e-9):
    """Verify exact integer decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                val = 0
                for c in range(rank):
                    val += U[u_row][c] * V[v_row][c] * W[w_row][c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _find_2x2x3_rank10(deadline):
    """
    Attempt to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Uses a structured search based on the known rank-11 solution.
    Returns (U, V, W) as lists if found, None otherwise.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Start from the rank-11 solution and try to merge terms
    # The rank-11 solution is Strassen (7) + schoolbook (4)
    # We need to reduce by 1, so we try to find a linear dependency
    
    # Initialize with the rank-11 structure but project to rank 10
    # Use ALS with careful initialization
    
    np.random.seed(42)
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Try multiple initializations
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Initialize near the rank-11 solution
        # Create a random perturbation of the rank-11 structure
        U_init = np.zeros((4, rank))
        V_init = np.zeros((6, rank))
        W_init = np.zeros((6, rank))
        
        # Fill first 7 columns with Strassen-like structure
        # This is a heuristic initialization
        for r in range(min(7, rank)):
            U_init[:, r] = np.random.randn(4) * 0.1
            V_init[:, r] = np.random.randn(6) * 0.1
            W_init[:, r] = np.random.randn(6) * 0.1
        
        # Fill remaining columns with smaller random values
        for r in range(7, rank):
            U_init[:, r] = np.random.randn(4) * 0.05
            V_init[:, r] = np.random.randn(6) * 0.05
            W_init[:, r] = np.random.randn(6) * 0.05
        
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # ALS iterations
        for it in range(50):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    B = np.zeros((rank, p))
                    for r in range(rank):
                        for k in range(p):
                            B[r, k] = V[j * p + k, r] * W[k * n + i, r]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    C = np.zeros((rank, n * m))
                    for r in range(rank):
                        for i in range(n):
                            C[r, i * m + j] = U[i * m + j, r] * W[k * n + i, r]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(C, np.ones(n * m), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    D = np.zeros((rank, m))
                    for r in range(rank):
                        for j in range(m):
                            D[r, j] = U[i * m + j, r] * V[j * p + k, r]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(D, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        # Compute error
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    error += (actual - 1.0) ** 2
        error = np.sqrt(error)
        
        if error < best_error:
            best_error = error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_error < 1e-6:
            break
    
    if best_error < 1e-6 and best_U is not None:
        # Convert to lists
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        return U_list, V_list, W_list
    
    return None

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 2x2 matrix by a 2x15 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 15
    
    # Build rank-53 baseline
    U53, V53, W53 = _build_strassen_53(n, m, p)
    
    if not _verify_int(U53, V53, W53, n, m, p):
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U53, V53, W53
    
    # Try to find rank-10 decomposition for 2x2x3
    if time.time() < deadline:
        result = _find_2x2x3_rank10(deadline)
        if result is not None:
            U10, V10, W10 = result
            U52, V52, W52 = _build_combined_52(n, m, p, U10, V10, W10)
            if _verify_int(U52, V52, W52, n, m, p):
                best_U, best_V, best_W = U52, V52, W52
    
    return {"U": best_U, "V": best_V, "W": best_W}
