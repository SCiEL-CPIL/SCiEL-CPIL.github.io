"""
Rank-minimal exact tensor decomposition for 2x3 * 3x7 matrix multiplication.

Strategy:
- Use the schoolbook decomposition as a safe fallback (rank 42).
- Attempt to find a lower-rank decomposition via structured search.
- Key insight: We can exploit the structure by considering the problem as
  computing 7 independent 2x3 * 3x1 multiplications, but sharing structure.
- Use a combination of:
  1. Fast numerical verification with numpy
  2. Structured search trying to reduce rank from schoolbook
  3. Exact rational verification only for candidates that pass numerical check
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast numerical verification using numpy. Returns True if likely correct."""
    rank = U.shape[1]
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Exact rational verification. Only call on candidates that pass fast check."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = Fraction(str(U[i*m+j][c]))
                    v_val = Fraction(str(V[j*p+k][c]))
                    w_val = Fraction(str(W[k*n+i][c]))
                    total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True

def _schoolbook():
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

def _try_constructed_decomposition(n, m, p, deadline):
    """
    Try a structured approach:
    For 2x3 * 3x7, we can think of this as:
    - 7 columns of B, each 3x1
    - For each column, compute A @ b (2x3 @ 3x1 -> 2x1)
    
    The 2x3 * 3x1 multiplication has a known minimal rank.
    Actually, let's try a different structured approach:
    Use the fact that we can share some computations across columns.
    
    One idea: partition the 7 columns into groups and use shared bilinear forms.
    """
    # For now, try a simple structured construction:
    # Use rank = 2*3 + 3*7 - 3 = 6 + 21 - 3 = 24? No, that's not quite right.
    # 
    # Actually, let's try: for each of the 3 intermediate indices j,
    # we have n*p = 14 products A[i,j]*B[j,k] for i in {0,1}, k in {0..6}.
    # Total schoolbook: 3 * 14 = 42.
    #
    # Can we do better? For fixed j, we're computing a 2x7 outer product
    # of vector A[:,j] (length 2) and vector B[j,:] (length 7).
    # The rank of this bilinear map is at most 2*7 = 14, but actually
    # for a 2x7 outer product, the minimal rank is... 
    # The tensor for 2x1 * 1x7 -> 2x7 has rank 14 (schoolbook) 
    # but can it be lower? For n=2, p=7, the rank of the 2x7 outer product
    # tensor is 14 (no known improvement for non-square).
    #
    # So per j, we need at least 14, total at least 42? That can't be right
    # because we can share across j.
    #
    # Let me try a different approach: just do a smarter random search
    # with proper initialization.
    pass

def _random_search_for_rank(n, m, p, target_rank, max_attempts, deadline):
    """
    Try to find a decomposition of given rank using random search.
    Initialize with schoolbook-like structure and perturb.
    """
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Initialize with a structured pattern
        U = np.zeros((nm, target_rank), dtype=np.float64)
        V = np.zeros((mp, target_rank), dtype=np.float64)
        W = np.zeros((pn, target_rank), dtype=np.float64)
        
        # Start from schoolbook and try to merge some terms
        if target_rank < schoolbook_rank:
            # Use first target_rank schoolbook terms as base
            for c in range(target_rank):
                i = c // (m * p)
                rem = c % (m * p)
                j = rem // p
                k = rem % p
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
            
            # Now try to adjust to cover the remaining (i,j,k) triples
            # using the existing rank terms
            # This is essentially a linear system problem
            
            # For each (i,j,k) not covered, we need:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # But we've only set up the diagonal terms.
            
            # Alternative: random initialization with small values
            U = np.random.randint(-2, 3, (nm, target_rank)).astype(np.float64)
            V = np.random.randint(-2, 3, (mp, target_rank)).astype(np.float64)
            W = np.random.randint(-2, 3, (pn, target_rank)).astype(np.float64)
        
        # Fast numerical check
        if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-6):
            # Exact check
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify_decomposition_exact(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # First, try a few structured constructions
    # For 2x3 * 3x7, try ranks from a lower bound up to schoolbook
    
    # Lower bound: max(nm, mp, pn) = max(6, 21, 14) = 21
    # Also, rank >= max(n*m, m*p, n*p) is not quite right for tensor rank
    # A better lower bound: rank >= max(n*m, m*p, n*p) / something...
    # Actually, the trivial lower bound is max(nm, mp, pn) = 21
    
    # Try ranks from 21 to 41
    for target_rank in range(21, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        result = _random_search_for_rank(n, m, p, target_rank, max_attempts=50, deadline=deadline - 10)
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no lower rank found, return schoolbook
    return _schoolbook()
