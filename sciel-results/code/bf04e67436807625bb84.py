"""Program for 3x3x6 matrix multiplication tensor decomposition.

Strategy:
  1. Direct-sum construction: treat 3x3x6 as three independent 3x3x2 blocks.
     Build a rank-15 decomposition for 3x3x2 and replicate it 3 times -> rank 45.
  2. Fallback: schoolbook rank 54.
"""
import time
import sys
import numpy as np

DIMENSIONS = [3, 3, 6]

def _schoolbook():
    """Return the trivial rank-54 decomposition."""
    n, m, p = 3, 3, 6
    rank = n * m * p  # 54
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

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify exact decomposition: sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] == 1."""
    rank = len(U[0])
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for r in range(rank):
        if len(U[r]) != 1 or len(V[r]) != 1 or len(W[r]) != 1:
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_3x3x2_rank15():
    """
    Build a rank-15 decomposition for 3x3x2 matrix multiplication.
    T[i,j,k] = 1 for i in {0,1,2}, j in {0,1,2}, k in {0,1}.
    
    U: 9 x 15, V: 6 x 15, W: 6 x 15 (since p*n = 2*3 = 6)
    """
    # We'll use a structured approach.
    # The tensor has 18 nonzero entries (3*3*2).
    # Rank 15 means we save 3 compared to schoolbook (18).
    # 
    # Idea: Use 15 basis vectors. For each (i,j,k) triple, we need:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #
    # Let's try a pattern where we share some factors across k values.
    # For fixed (i,j), we have 2 equations (k=0, k=1) in terms of r.
    # If we use 5 r's per (i,j) pair, that's 9*5=45, too many.
    # If we use 15 total, average 5/3 per (i,j).
    #
    # Let's try: for each i (3 values), use 5 r's.
    # U[i*m+j, r] depends on i, j, r.
    # V[j*p+k, r] depends on j, k, r.
    # W[k*n+i, r] depends on k, i, r.
    #
    # Let's parameterize r by (i_block, s) where i_block in {0,1,2}, s in {0,1,2,3,4}.
    # So r = i_block*5 + s.
    #
    # For a given (i, j, k):
    #   Only r where i_block == i contributes (if we design it so).
    #   Then: sum_{s=0}^{4} U[i*m+j, i*5+s] * V[j*p+k, i*5+s] * W[k*n+i, i*5+s] = 1
    #
    # Let's set:
    #   U[i*m+j, i*5+s] = u_{i,j,s}
    #   V[j*p+k, i*5+s] = v_{j,k,s}
    #   W[k*n+i, i*5+s] = w_{k,i,s}
    #
    # We need: sum_s u_{i,j,s} * v_{j,k,s} * w_{k,i,s} = 1 for all i,j,k.
    #
    # This is 18 equations in 45 unknowns (3*3*5 for each of U, V, W parts).
    # Many solutions exist. Let's find one with simple values.
    #
    # Try: set w_{k,i,s} = 1 for all k,i,s.
    # Then: sum_s u_{i,j,s} * v_{j,k,s} = 1 for all i,j,k.
    #
    # For fixed i,j: sum_s u_{i,j,s} * v_{j,k,s} = 1 for k=0,1.
    # Let's set u_{i,j,s} = 1 for all s, and v_{j,k,s} such that sum_s v_{j,k,s} = 1.
    # E.g., v_{j,0,0} = 1, v_{j,0,s>0} = 0; v_{j,1,1} = 1, v_{j,1,s!=1} = 0.
    # Check: for k=0: sum_s 1 * v_{j,0,s} = v_{j,0,0} = 1. Good.
    # For k=1: sum_s 1 * v_{j,1,s} = v_{j,1,1} = 1. Good.
    #
    # So:
    #   U[i*m+j, i*5+s] = 1 for all i,j,s
    #   V[j*p+0, i*5+0] = 1, V[j*p+0, i*5+s!=0] = 0
    #   V[j*p+1, i*5+1] = 1, V[j*p+1, i*5+s!=1] = 0
    #   W[k*n+i, i*5+s] = 1 for all k,i,s
    #
    # But wait, V[j*p+k, r] where r = i*5+s. The index r must be the same for all i?
    # No, r is a global index 0..14. Let r = i*5+s.
    # V is indexed by (j*p+k) and r.
    # For a fixed r = i*5+s, V[j*p+0, r] = 1 if s==0 else 0.
    # V[j*p+1, r] = 1 if s==1 else 0.
    #
    # Let's check the full sum for (i,j,k):
    #   sum_{r=0}^{14} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #   = sum_{i'=0}^{2} sum_{s=0}^{4} U[i*m+j, i'*5+s] * V[j*p+k, i'*5+s] * W[k*n+i, i'*5+s]
    #   Only i'=i terms have U and W nonzero (by design).
    #   = sum_{s=0}^{4} U[i*m+j, i*5+s] * V[j*p+k, i*5+s] * W[k*n+i, i*5+s]
    #   = sum_{s=0}^{4} 1 * V[j*p+k, i*5+s] * 1
    #   For k=0: V[j*p+0, i*5+s] = 1 if s==0 else 0. Sum = 1. Good.
    #   For k=1: V[j*p+1, i*5+s] = 1 if s==1 else 0. Sum = 1. Good.
    #
    # This works! Let's build it.
    rank = 15
    U = [[0] * rank for _ in range(9)]   # n*m = 3*3 = 9
    V = [[0] * rank for _ in range(6)]   # m*p = 3*2 = 6
    W = [[0] * rank for _ in range(6)]   # p*n = 2*3 = 6
    
    for i in range(3):
        for j in range(3):
            for s in range(5):
                r = i * 5 + s
                U[i * 3 + j][r] = 1
                W[s * 3 + i][r] = 1  # k*n+i where k=0? Wait, W[k*n+i, r]
                # W is indexed by k*n+i. For a given r=i*5+s, W[k*3+i, r] should be 1 for all k.
                # So for each k in {0,1}:
                for k in range(2):
                    W[k * 3 + i][r] = 1
                # V[j*p+k, r] = V[j*2+k, r]
                if s == 0:
                    V[j * 2 + 0][r] = 1
                elif s == 1:
                    V[j * 2 + 1][r] = 1
    
    # Verify
    if not _verify(U, V, W, n=3, m=3, p=2):
        return None
    
    return {"U": U, "V": V, "W": W}

def _construct_3x3x6_rank45():
    """
    Build a rank-45 decomposition for 3x3x6 by direct sum of three 3x3x2 blocks.
    Block t (t=0,1,2) handles k in {3t, 3t+1, 3t+2}.
    """
    block = _construct_3x3x2_rank15()
    if block is None:
        return None
    
    n, m, p = 3, 3, 6
    rank = 45
    U = [[0] * rank for _ in range(n * m)]   # 9 x 45
    V = [[0] * rank for _ in range(m * p)]   # 18 x 45
    W = [[0] * rank for _ in range(p * n)]   # 18 x 45
    
    for t in range(3):
        # Block t uses r in [t*15, t*15+14]
        r_offset = t * 15
        # Map local indices to global
        for i in range(3):
            for j in range(3):
                for s in range(5):
                    r_local = i * 5 + s
                    r_global = r_offset + r_local
                    U[i * m + j][r_global] = block["U"][i * 3 + j][r_local]
        
        for j in range(3):
            for k_local in range(2):
                for s in range(5):
                    r_local = 0  # V doesn't depend on i directly, but on r
                    # Actually V[j*p+k, r] in the block is V[j*2+k_local, i*5+s]
                    # We need to iterate over i and s
                    for i in range(3):
                        for s in range(5):
                            r_local = i * 5 + s
                            r_global = r_offset + r_local
                            k_global = 3 * t + k_local
                            V[j * p + k_global][r_global] = block["V"][j * 2 + k_local][r_local]
        
        for k_local in range(2):
            for i in range(3):
                for s in range(5):
                    r_local = i * 5 + s
                    r_global = r_offset + r_local
                    k_global = 3 * t + k_local
                    W[k_global * n + i][r_global] = block["W"][k_local * 3 + i][r_local]
    
    if not _verify(U, V, W):
        return None
    
    return {"U": U, "V": V, "W": W}

def construct():
    t0 = time.time()
    deadline = t0 + 115.0
    
    # Try rank-45 construction
    if time.time() < deadline - 5:
        result = _construct_3x3x6_rank45()
        if result is not None:
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
