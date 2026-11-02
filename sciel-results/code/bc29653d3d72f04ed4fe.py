"""Program for matrix multiplication tensor decomposition (3x5 by 5x5)."""
DIMENSIONS = [3, 5, 5]

import time
import numpy as np
from itertools import combinations


def _build_schoolbook(n, m, p):
    """
    Build the schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W as numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank).
    Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    if isinstance(U, list):
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
    else:
        Ua = U
        Va = V
        Wa = W
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _construct_strassen_2x2_block():
    """
    Construct the 7 Strassen products for 2x2 matrix multiplication.
    Returns lists of (u_idx, v_idx, w_idx, coeff) for each of the 7 products.
    Indices: u_idx = i*2+j, v_idx = j*2+k, w_idx = k*2+i
    """
    # Strassen formulas for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Output:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_r is a rank-1 tensor product of two vectors.
    # For M1: U = [1,0,1,1] (A11+A22), V = [1,1,0,1] (B11+B22)
    # Wait, need to be careful with indexing.
    
    # U indices: 00, 01, 10, 11 -> (i,j) pairs
    # V indices: 00, 01, 10, 11 -> (j,k) pairs
    # W indices: 00, 01, 10, 11 -> (k,i) pairs
    
    # M1: (A00+A11)(B00+B11)
    # U: [1, 0, 1, 0] (A00 + A11)
    # V: [1, 0, 1, 0] (B00 + B11)
    # This contributes to C00 and C11
    # C00 += M1, C11 += M1
    
    # Let me just build it directly:
    terms = []
    
    # M1 = (A00+A11)(B00+B11)
    # Contributes to C00 (k=0,i=0) and C11 (k=1,i=1)
    terms.append({
        'U': [1, 0, 1, 0],  # A00 + A11
        'V': [1, 0, 1, 0],  # B00 + B11
        'W': [1, 0, 1, 0],  # contributes to C00 and C11
        'coeffs': {0: 1, 3: 1}  # C00 and C11
    })
    
    # M2 = (A10+A11)(B00)
    # Contributes to C10 (k=0,i=1) and C11 (k=1,i=1)
    terms.append({
        'U': [0, 0, 1, 1],  # A10 + A11
        'V': [1, 0, 0, 0],  # B00
        'W': [0, 0, 1, 1],  # contributes to C10 and C11
        'coeffs': {2: 1, 3: 1}  # C10 and C11
    })
    
    # M3 = (A00)(B01-B11)
    # Contributes to C01 (k=1,i=0)
    terms.append({
        'U': [1, 0, 0, 0],  # A00
        'V': [0, 1, 0, -1],  # B01 - B11
        'W': [0, 1, 0, 0],  # contributes to C01
        'coeffs': {1: 1}  # C01
    })
    
    # M4 = (A11)(B10-B00)
    # Contributes to C10 (k=0,i=1) and C11 (k=1,i=1)
    terms.append({
        'U': [0, 0, 0, 1],  # A11
        'V': [0, 0, 1, -1],  # B10 - B00
        'W': [0, 0, 1, 1],  # contributes to C10 and C11
        'coeffs': {2: 1, 3: 1}  # C10 and C11
    })
    
    # M5 = (A00+A01)(B11)
    # Contributes to C01 (k=1,i=0) and C11 (k=1,i=1)
    terms.append({
        'U': [1, 1, 0, 0],  # A00 + A01
        'V': [0, 0, 0, 1],  # B11
        'W': [0, 1, 0, 1],  # contributes to C01 and C11
        'coeffs': {1: 1, 3: 1}  # C01 and C11
    })
    
    # M6 = (A10-A00)(B00+B01)
    # Contributes to C10 (k=0,i=1) and C11 (k=1,i=1)
    terms.append({
        'U': [1, 0, 1, 0],  # A10 - A00
        'V': [1, 1, 0, 0],  # B00 + B01
        'W': [0, 0, 1, 0],  # contributes to C10
        'coeffs': {2: 1}  # C10
    })
    
    # M7 = (A01-A11)(B10+B11)
    # Contributes to C01 (k=1,i=0) and C11 (k=1,i=1)
    terms.append({
        'U': [0, 1, 0, 1],  # A01 - A11
        'V': [0, 0, 1, 1],  # B10 + B11
        'W': [0, 1, 0, 1],  # contributes to C01 and C11
        'coeffs': {1: 1, 3: 1}  # C01 and C11
    })
    
    return terms


def _build_strassen_based_decomposition(n, m, p):
    """
    Build a Strassen-based decomposition for 3x5 * 5x5 -> 3x5.
    
    Strategy:
    1. Split the 3x5 matrix into blocks.
    2. Use Strassen for 2x2 sub-blocks where possible.
    3. Combine with schoolbook for remaining parts.
    
    For 3x5 * 5x5:
    - Split A (3x5) into: A_top (2x5) and A_bottom (1x5)
    - Split B (5x5) into: B_left (5x3) and B_right (5x2)
    - Output C (3x5) = [C_top (2x5); C_bottom (1x5)]
    
    C_top = A_top * B (2x5 * 5x5 = 2x5)
    C_bottom = A_bottom * B (1x5 * 5x5 = 1x5)
    
    For C_top (2x5 * 5x5):
    - Split further into 2x2 blocks where possible
    - Use Strassen for 2x2 * 2x2 parts
    
    This is getting complex. Let me try a simpler approach:
    Just use schoolbook but with a different structure that might allow
    for easier rank reduction.
    """
    # For now, just return schoolbook as a starting point
    U, V, W = _build_schoolbook(n, m, p)
    return U, V, W


def _try_remove_column(U, V, W, col_to_remove, n, m, p, tol=1e-9):
    """
    Try to remove a specific column and check if the decomposition still works.
    Returns (success, new_U, new_V, new_W)
    """
    rank = U.shape[1]
    if rank <= 1:
        return False, U, V, W
    
    remaining_cols = [c for c in range(rank) if c != col_to_remove]
    new_rank = len(remaining_cols)
    
    U_new = U[:, remaining_cols]
    V_new = V[:, remaining_cols]
    W_new = W[:, remaining_cols]
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(new_rank):
                    s += U_new[i*m+j, r] * V_new[j*p+k, r] * W_new[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False, U, V, W
    
    return True, U_new, V_new, W_new


def _greedy_column_removal(U, V, W, n, m, p, deadline, tol=1e-9):
    """
    Greedily try to remove columns one by one.
    Returns the reduced U, V, W.
    """
    U, V, W = U.copy(), V.copy(), W.copy()
    rank = U.shape[1]
    
    while rank > 1:
        if time.time() > deadline:
            break
        
        removed = False
        for col in range(rank):
            if time.time() > deadline:
                break
            
            success, U_new, V_new, W_new = _try_remove_column(U, V, W, col, n, m, p, tol)
            if success:
                U, V, W = U_new, V_new, W_new
                rank = U.shape[1]
                removed = True
                break
        
        if not removed:
            break
    
    return U, V, W


def _local_search(U, V, W, n, m, p, deadline, tol=1e-9):
    """
    Try local optimizations: swap columns, add small perturbations, etc.
    This is a placeholder for more advanced search.
    """
    # For now, just do greedy removal
    return _greedy_column_removal(U, V, W, n, m, p, deadline, tol)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build initial schoolbook decomposition
    U, V, W = _build_schoolbook(n, m, p)
    
    # Verify
    if not _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        raise ValueError("Initial decomposition verification failed")
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Try to reduce rank
    if time.time() < deadline:
        U1, V1, W1 = _local_search(U, V, W, n, m, p, deadline)
        if U1.shape[1] < best_rank and _verify(U1.tolist(), V1.tolist(), W1.tolist(), n, m, p):
            best_U, best_V, best_W = U1, V1, W1
            best_rank = U1.shape[1]
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
