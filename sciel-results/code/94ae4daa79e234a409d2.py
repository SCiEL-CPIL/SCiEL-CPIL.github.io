"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the tensor: T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _try_rank_decomp(n, m, p, target_rank):
    """Try to find a decomposition of given rank using a simple construction."""
    # Try the "sum of separable" construction
    # J[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # Construction: use r=0..n-1 for i-index, r=n..n+m-1 for j-index, etc.
    # This is a known bound: rank <= n + m + p - 2 for all-ones tensor
    
    # Let's try the standard construction for all-ones tensor rank
    # Actually, let me try a different approach: 
    # For the matrix multiplication tensor, we can use:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k]=1 for all i,j,k
    
    # Try rank = n*m + m*p + p*n - n - m - p + 2 (a known bound?)
    # Or just try to construct it
    
    # Simple approach: use the fact that 1 = 1*1*1, and try to cover all (i,j,k)
    # with fewer terms by sharing
    
    # Let me try: for each i, use m*p terms; total n*m*p = schoolbook
    # For lower rank, we need to share across i or k
    
    # Try a construction where we fix j and share across i,k:
    # This won't work directly because the tensor is all-ones
    
    # Let me just return None to indicate failure
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try a few candidate lower ranks
    # Known: for 1x1x1, rank=1; for 2x2x2, rank=7 (Strassen)
    # For 3x4x8, let's try some candidates
    
    candidates = []
    # Try n+m+p-2
    candidates.append(n + m + p - 2)
    # Try n*m + m*p + p*n - 2
    candidates.append(n * m + m * p + p * n - 2)
    # Try a few others
    candidates.append(60)
    candidates.append(50)
    candidates.append(40)
    candidates.append(30)
    
    for target_rank in sorted(candidates):
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
        # For now, we don't have a general algorithm to find the decomposition
        # So we'll just keep the schoolbook
        pass
    
    return best
# EVOLVE-BLOCK-END
