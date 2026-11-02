import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Using the trace convention W[k*n+i, r]
    
    # We want to find rank-r decomposition:
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # where a = i*m+j (0..n*m-1), b = j*p+k (0..m*p-1), c = k*n+i (0..p*n-1)
    
    # Start with schoolbook decomposition (rank = n*m*p = 50)
    # Then try to reduce rank using various strategies
    
    # Strategy: Use a structured approach based on the observation that
    # for 2x5x5, we can potentially use rank < 50
    
    # Let's try a systematic approach:
    # 1. Start with schoolbook (rank 50)
    # 2. Try Strassen-like reductions on 2x2 blocks
    # 3. Try to merge terms where possible
    
    # First, let's build the schoolbook decomposition
    terms = []
    
    def add_schoolbook_term(i, j, k):
        u = {(i, j): 1}
        v = {(j, k): 1}
        w = {(k, i): 1}
        terms.append((u, v, w))
    
    # Full schoolbook: 2*5*5 = 50 terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_schoolbook_term(i, j, k)
    
    # Now try Strassen-like reductions
    # For a 2x2x2 block, Strassen uses 7 multiplications instead of 8
    # Let's try to apply this to overlapping 2x2 blocks
    
    # Reset terms and build a better decomposition
    terms = []
    
    def add_strassen_block(i0, j0, k0):
        """Add 7 terms for a 2x2x2 Strassen block starting at (i0, j0, k0)"""
        # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
        i1, i2 = i0, i0+1
        j1, j2 = j0, j0+1
        k1, k2 = k0, k0+1
        
        # M1 = (A11 + A22)(B11 + B22)
        u = {(i1, j1): 1, (i2, j2): 1}
        v = {(j1, k1): 1, (j2, k2): 1}
        w = {(k1, i1): 1, (k2, i2): 1}
        terms.append((u, v, w))
        
        # M2 = A22(B21 - B11)
        u = {(i2, j2): 1}
        v = {(j2, k1): 1, (j1, k1): -1}
        w = {(k1, i2): 1, (k2, i2): 1}
        terms.append((u, v, w))
        
        # M3 = (A11 - A21)B22
        u = {(i1, j1): 1, (i2, j1): -1}
        v = {(j2, k2): 1}
        w = {(k2, i1): 1, (k2, i2): 1}
        terms.append((u, v, w))
        
        # M4 = A11(B12 - B22)
        u = {(i1, j1): 1}
        v = {(j1, k2): 1, (j2, k2): -1}
        w = {(k1, i1): 1, (k2, i1): 1}
        terms.append((u, v, w))
        
        # M5 = (A21 + A11)(B11 + B12)
        u = {(i2, j1): 1, (i1, j1): 1}
        v = {(j1, k1): 1, (j1, k2): 1}
        w = {(k1, i2): 1, (k2, i1): 1}
        terms.append((u, v, w))
        
        # M6 = (A12 - A22)(B21 + B22)
        u = {(i1, j2): 1, (i2, j2): -1}
        v = {(j2, k1): 1, (j2, k2): 1}
        w = {(k1, i1): -1, (k2, i2): 1}
        terms.append((u, v, w))
        
        # M7 = (A11 - A12)(B12 + B22)
        u = {(i1, j1): 1, (i1, j2): -1}
        v = {(j1, k2): 1, (j2, k2): 1}
        w = {(k1, i1): 1, (k2, i2): 1}
        terms.append((u, v, w))
    
    def add_single_term(i, j, k, coeff=1):
        u = {(i, j): coeff}
        v = {(j, k): 1}
        w = {(k, i): 1}
        terms.append((u, v, w))
    
    # Plan: Use Strassen for 2x2 blocks where possible, schoolbook for the rest
    # i dimension: 2 elements (indices 0, 1)
    # j dimension: 5 elements (indices 0-4)
    # k dimension: 5 elements (indices 0-4)
    
    # For each pair of j values that form a 2-block, and each pair of k values:
    # We can use Strassen if both i values are used
    
    # Let's try: 
    # j blocks: (0,1), (2,3), and single j=4
    # k blocks: (0,1), (2,3), and single k=4
    
    # For j in {0,1} and k in {0,1}: Strassen (7 terms)
    add_strassen_block(0, 0, 0)
    
    # For j in {0,1} and k in {2,3}: Strassen (7 terms)
    add_strassen_block(0, 2, 0)
    
    # For j in {0,1} and k=4: schoolbook for i=0,1 (2 terms)
    add_single_term(0, 0, 4)
    add_single_term(1, 0, 4)
    add_single_term(0, 1, 4)
    add_single_term(1, 1, 4)
    
    # For j in {2,3} and k in {0,1}: Strassen (7 terms)
    add_strassen_block(0, 0, 2)
    
    # For j in {2,3} and k in {2,3}: Strassen (7 terms)
    add_strassen_block(0, 2, 2)
    
    # For j in {2,3} and k=4: schoolbook for i=0,1 (4 terms)
    add_single_term(0, 2, 4)
    add_single_term(1, 2, 4)
    add_single_term(0, 3, 4)
    add_single_term(1, 3, 4)
    
    # For j=4 and k in {0,1}: schoolbook for i=0,1 (4 terms)
    add_single_term(0, 4, 0)
    add_single_term(1, 4, 0)
    add_single_term(0, 4, 1)
    add_single_term(1, 4, 1)
    
    # For j=4 and k in {2,3}: schoolbook for i=0,1 (4 terms)
    add_single_term(0, 4, 2)
    add_single_term(1, 4, 2)
    add_single_term(0, 4, 3)
    add_single_term(1, 4, 3)
    
    # For j=4 and k=4: schoolbook for i=0,1 (2 terms)
    add_single_term(0, 4, 4)
    add_single_term(1, 4, 4)
    
    rank = len(terms)
    
    # Build matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
    
    # Verify the decomposition
    # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # which should be 1 if the indices are consistent, 0 otherwise
    
    # Let's verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check a few entries
    # For i=0, j=0, k=0: T[0, 0, 0] should be 1
    # sum_r U[0, r] * V[0, r] * W[0, r]
    val = np.sum(U_arr[0] * V_arr[0] * W_arr[0])
    if abs(val - 1) > 1e-9:
        # Fallback to schoolbook
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_single_term(i, j, k)
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c in range(rank):
            u_dict, v_dict, w_dict = terms[c]
            for (i, j), val in u_dict.items():
                U[i * m + j][c] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][c] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][c] = val
    
    return {"U": U, "V": V, "W": W}
