"""Matrix multiplication 2x3 by 3x4: search for low-rank exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 4]

# Deadline guard: kill search well before 150s wall-clock
DEADLINE_SECONDS = 140.0

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x4 matrix.
    
    The matrix multiplication tensor T has entries T[i,j,k] = 1 for all
    i in [0,n), j in [0,m), k in [0,p). We seek:
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    with minimal rank r.
    
    Strategy:
    1. Try known structured decompositions at lower ranks.
    2. Use algebraic search (rank-1 terms) to build up a decomposition.
    3. Fall back to schoolbook rank n*m*p = 24.
    """
    start_time = time.time()
    
    def time_remaining():
        return DEADLINE_SECONDS - (time.time() - start_time)
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    nm = n * m  # 6
    mp = m * p  # 12
    pn = p * n  # 8
    
    # Schoolbook decomposition (rank = n*m*p = 24)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-9):
        """Check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try a structured rank-12 decomposition
    # Idea: For each pair (i,k), we need sum_j over j of contributions = 1
    # But the tensor is all-ones, so we need:
    #   For all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    #
    # Let's try a decomposition where each rank-1 term covers specific (i,j,k) patterns.
    # 
    # A known approach: if we can find vectors u in R^(nm), v in R^(mp), w in R^(pn)
    # such that their outer product matches the tensor, we're done.
    # But the all-ones tensor of shape (2,3,4) can be written as:
    #   T = sum over c of (e_i_c ⊗ e_j_c ⊗ e_k_c) with appropriate weights
    #
    # Actually, the all-ones tensor [2,3,4] has a simple rank decomposition:
    # T[i,j,k] = sum_{a=0}^{1} sum_{b=0}^{2} sum_{c=0}^{3} ... no that's rank 24.
    #
    # But we can factor: T[i,j,k] = 1 = (1)(1)(1)
    # So T = 1_vector ⊗ 1_vector ⊗ 1_vector has rank 1!
    # Wait: if U[i*m+j, 0] = 1 for all, V[j*p+k, 0] = 1 for all, W[k*n+i, 0] = 1 for all,
    # then sum_c U*V*W = 1*1*1 = 1. That works! Rank 1!
    #
    # Let me verify: U has shape (6,1), all entries 1.
    # V has shape (12,1), all entries 1.
    # W has shape (8,1), all entries 1.
    # For any i,j,k: U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1. ✓
    
    # So the rank is 1!
    U = np.ones((nm, 1))
    V = np.ones((mp, 1))
    W = np.ones((pn, 1))
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    U, V, W = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
