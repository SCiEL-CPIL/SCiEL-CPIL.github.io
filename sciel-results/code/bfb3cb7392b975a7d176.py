"""Program for matrix multiplication tensor rank minimization."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 7]
# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    # Try Kronecker-structured decomposition exploiting 4 = 2*2
    U, V, W = _kronecker_decomp(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _kronecker_decomp(n, m, p):
    """
    Exploit m = m1 * m2 where m1=2, m2=2.
    Write j = j1 + m1*j2 where j1 in [0,m1), j2 in [0,m2).
    
    The tensor T[i*m+j, j*p+k, k*n+i] = 1 becomes:
    T[i*m1*m2 + j1 + m1*j2, (j1 + m1*j2)*p + k, k*n + i] = 1
    
    We can factor this as a Kronecker product of:
    - A tensor for the "outer" j2 index: contributes to U and V
    - A tensor for the "inner" j1 index: contributes to U and V
    
    For the 2x2xp subproblem, rank is at most p (using identity structure).
    So total rank = m2 * p = 2 * 7 = 14? Let's verify.
    
    Actually, let's construct it directly.
    """
    n, m, p = 2, 4, 7
    m1, m2 = 2, 2  # m = m1 * m2
    
    # Rank will be m2 * p = 2 * 7 = 14
    r = m2 * p
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # For each j2 in [0, m2) and k in [0, p), we have one rank-1 term
    # The term contributes to:
    #   U[i*m + j1 + m1*j2, r_idx] = 1 for all i, j1
    #   V[(j1 + m1*j2)*p + k, r_idx] = 1 for all j1
    #   W[k*n + i, r_idx] = 1 for all i
    # 
    # Wait, this doesn't quite work because U and V need to depend on j1 too.
    # Let me think again.
    #
    # T[i*m + j, j*p + k, k*n + i] = 1 where j = j1 + m1*j2
    # = T[i*m1*m2 + j1 + m1*j2, (j1 + m1*j2)*p + k, k*n + i]
    #
    # For a fixed (j2, k), we want:
    #   sum over j1 of [something that gives 1 when indices match]
    #
    # Actually, the standard approach: for each (i, j, k), we need a term.
    # But we can share terms across j1 values.
    #
    # Let's use the structure: for each (j2, k), define a rank-1 term where:
    #   U[a, r(j2,k)] = 1 if a corresponds to (i, j1, j2) for any i, j1
    #   V[b, r(j2,k)] = 1 if b corresponds to (j1, j2, k) for any j1
    #   W[c, r(j2,k)] = 1 if c corresponds to (k, i) for any i
    #
    # But this would give T = sum over (j2,k) of [indicator that j2 matches] * [indicator that k matches] * [indicator that i matches]
    # which is not quite right.
    
    # Let me try a different approach: direct construction
    # For the 2x4x7 problem, use rank 28 (the lower bound)
    r = 28
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Use the structure: j = j1 + 2*j2
    # For each (i, j2, k), we have a term that handles both j1=0 and j1=1
    # Term index: i * m2 * p + j2 * p + k
    # But we need to handle j1 separately in U and V
    
    # Alternative: just use schoolbook but with a smarter ordering
    # Actually, let's just verify if rank 28 is achievable with a simple construction
    
    # Simple construction: for each (j, k), have p terms? No, that's 4*7=28 terms
    # For each (j, k), term r = j*p + k:
    #   U[i*m + j, r] = 1 for all i
    #   V[j*p + k, r] = 1
    #   W[k*n + i, r] = 1 for all i
    # This gives T[i*m+j, j*p+k, k*n+i] = sum over r where U,V,W all nonzero
    # = 1 if (i*m+j, j*p+k, k*n+i) matches, else 0
    # Wait, this is exactly schoolbook but with rank m*p = 28!
    
    col = 0
    for j in range(m):
        for k in range(p):
            r_idx = col
            # U[i*m + j, r_idx] = 1 for all i
            for i in range(n):
                U[i * m + j, r_idx] = 1.0
            # V[j*p + k, r_idx] = 1
            V[j * p + k, r_idx] = 1.0
            # W[k*n + i, r_idx] = 1 for all i
            for i in range(n):
                W[k * n + i, r_idx] = 1.0
            col += 1
    
    # Verify
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    decomposed = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(decomposed, T, atol=1e-10):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p, n*m*p)
    
    return U, V, W

def _schoolbook(n, m, p, rank):
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

# EVOLVE-BLOCK-END
