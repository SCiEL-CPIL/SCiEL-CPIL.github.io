import time
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 2, 3, 7
    NM, MP, PN = n*m, m*p, p*n  # 6, 21, 14
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    # Build schoolbook as fallback
    def schoolbook():
        rank = n*m*p  # 42
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    # Verify exact decomposition
    def verify(U, V, W):
        rank = U.shape[1]
        # Check required entries = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check zero entries
        for a in range(NM):
            i1, j1 = divmod(a, m)
            for b in range(MP):
                j2, k1 = divmod(b, p)
                for c in range(PN):
                    k2, i2 = divmod(c, n)
                    if not (i1==i2 and j1==j2 and k1==k2):
                        s = 0.0
                        for r in range(rank):
                            s += U[a, r] * V[b, r] * W[c, r]
                        if abs(s) > 1e-9:
                            return False
        return True
    
    # Fast verify (sample)
    def verify_fast(U, V, W, seed=42):
        rank = U.shape[1]
        rng = np.random.RandomState(seed)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        for _ in range(300):
            a = rng.randint(NM)
            b = rng.randint(MP)
            c = rng.randint(PN)
            i1, j1 = divmod(a, m)
            j2, k1 = divmod(b, p)
            k2, i2 = divmod(c, n)
            if not (i1==i2 and j1==j2 and k1==k2):
                s = 0.0
                for r in range(rank):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s) > 1e-8:
                    return False
        return True
    
    # Attempt to find a lower rank decomposition (e.g., rank 41) via random search
    # If found, return it. Otherwise, fallback to schoolbook.
    # Given time constraints, we try a few random merges.
    
    # Try rank 41: Merge two terms in the schoolbook.
    # We know merging same (i,j) different k is valid for the sub-problem if isolated.
    # But globally, it might interfere.
    # We'll just try to construct a rank 41 by merging two terms with same (i,j) and different k,
    # and verify. If it fails, we return schoolbook.
    
    def try_rank_41():
        # Pick two terms to merge: (i, j, k1) and (i, j, k2)
        # Let's pick i=0, j=0, k1=0, k2=1
        i, j, k1, k2 = 0, 0, 0, 1
        rank = 41
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        
        # Initialize with schoolbook
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    if (ii, jj, kk) == (i, j, k1) or (ii, jj, kk) == (i, j, k2):
                        continue
                    U[ii*m+jj, col] = 1.0
                    V[jj*p+kk, col] = 1.0
                    W[kk*n+ii, col] = 1.0
                    col += 1
        
        # Add merged term
        U[i*m+j, col] = 1.0
        V[j*p+k1, col] = 1.0
        V[j*p+k2, col] = 1.0
        W[k1*n+i, col] = 1.0
        W[k2*n+i, col] = 1.0
        col += 1
        
        if verify(U, V, W):
            return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
        return None

    res = try_rank_41()
    if res:
        return res

    # Fallback: schoolbook
    U, V, W = schoolbook()
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
# EVOLVE-BLOCK-END
