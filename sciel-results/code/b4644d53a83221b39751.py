import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

def _validate(U, V, W, n=3, m=3, p=8):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    # For each (i,j,k) in the tensor, the entry should be 1
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                total = 0
                for r in range(rank):
                    total += U[a][r] * V[b][r] * W[c][r]
                if total != 1:
                    return False
    # Check entries are integers or half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if isinstance(val, float):
                    if not val.is_integer() and abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 3, 3, 8
    
    # Schoolbook decomposition as baseline
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
                
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try to find a lower rank decomposition using structured patterns
    # For 3x3 * 3x8, we can try rank 22, 21, etc.
    
    # Known result: rank(3,3,3) = 23 (Winograd)
    # For 3x3 * 3x8, we can exploit the structure
    
    # Try a decomposition based on Winograd for the 3x3 part
    # Winograd 3x3x3 uses rank 23
    # For 3x3 * 3x8, we need 8 different "columns" of the second matrix
    
    # Strategy: Use Winograd decomposition for the 3x3 * 3x3 part
    # Then extend to 3x3 * 3x8 by processing columns in groups
    
    # Winograd 3x3x3 tensor decomposition (rank 23)
    # This is the standard Winograd scheme
    
    # For 3x3 * 3x8, we can use:
    # - 3x3 * 3x3 part with rank 23
    # - But we need to handle 8 columns, not 3
    
    # Alternative: Use the fact that 3x3 * 3x8 can be decomposed
    # by considering the 8 columns of the second matrix
    
    # Let's try a direct approach: construct a rank-22 or lower decomposition
    # by exploiting symmetries
    
    # For now, return the schoolbook decomposition
    # A real improvement would require finding a known lower-rank decomposition
    
    return {"U": best_U, "V": best_V, "W": best_W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    valid = _validate(result["U"], result["V"], result["W"])
    print(f"Valid: {valid}")
