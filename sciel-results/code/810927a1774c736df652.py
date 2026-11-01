"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _validate_decomposition(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # For each output position (i, k), check that sum_c U[i*m+j,c] * V[j*p+k,c] = W[k*n+i,c] for all c
    # Actually, the decomposition requires: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c] = (1 if we're computing A*B else 0)
    # The tensor T[i,j,k] = 1 for all i,j,k (it's the full multiplication tensor)
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]... no wait.
    
    # Standard tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_pattern_decomposition(n, m, p, pattern_type, deadline):
    """Try to find a low-rank decomposition using a specific pattern type."""
    start_time = time.time()
    
    if pattern_type == "separable_rows":
        # Try: U[i*m+j, c] depends only on i and c (same for all j within row i)
        # V[j*p+k, c] depends only on j and k
        # W[k*n+i, c] depends only on k and i
        # This is essentially the standard form but with structure on U
        
        # Try rank = n*p + m*p + n*m - something (like Strassen-style)
        # For 2x5x5, try various ranks below 50
        
        best_rank = n * m * p
        best_result = None
        
        for rank in range(20, best_rank + 1):
            if time.time() > deadline:
                break
            
            # Try random initialization with structure
            for attempt in range(50):
                if time.time() > deadline:
                    break
                
                # Initialize U with structure: U[i*m+j, c] = a[i, c] for all j
                # V[j*p+k, c] = b[j, k, c]  
                # W[k*n+i, c] = d[k, i, c]
                
                U = [[0]*rank for _ in range(n*m)]
                V = [[0]*rank for _ in range(m*p)]
                W = [[0]*rank for _ in range(p*n)]
                
                # Random initialization
                for i in range(n):
                    for c in range(rank):
                        val = random.choice([-1, 0, 1])
                        for j in range(m):
                            U[i*m+j][c] = val
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            V[j*p+k][c] = random.choice([-1, 0, 1])
                for k in range(p):
                    for i in range(n):
                        for c in range(rank):
                            W[k*n+i][c] = random.choice([-1, 0, 1])
                
                if _validate_decomposition(U, V, W, n, m, p):
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
                    return best_result
                    
            if best_result is not None:
                break
        
        if best_result is not None:
            return best_result
            
    elif pattern_type == "block":
        # Try block-structured decomposition
        best_rank = n * m * p
        best_result = None
        
        for rank in range(25, best_rank + 1):
            if time.time() > deadline:
                break
            
            for attempt in range(30):
                if time.time() > deadline:
                    break
                
                U = [[0]*rank for _ in range(n*m)]
                V = [[0]*rank for _ in range(m*p)]
                W = [[0]*rank for _ in range(p*n)]
                
                # Sparse random initialization
                density = 0.3
                for row in range(n*m):
                    for c in range(rank):
                        if random.random() < density:
                            U[row][c] = random.choice([-1, 1])
                for row in range(m*p):
                    for c in range(rank):
                        if random.random() < density:
                            V[row][c] = random.choice([-1, 1])
                for row in range(p*n):
                    for c in range(rank):
                        if random.random() < density:
                            W[row][c] = random.choice([-1, 1])
                
                if _validate_decomposition(U, V, W, n, m, p):
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
                    return best_result
                    
            if best_result is not None:
                break
        
        if best_result is not None:
            return best_result
    
    return None

def construct():
    # Schoolbook decomposition of rank n*m*p as fallback
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
    fallback_result = {"U": U, "V": V, "W": W}
    
    # Set deadline with safety margin (120s limit, use 110s for safety)
    deadline = time.time() + 110
    
    # Try structured patterns
    for pattern in ["separable_rows", "block"]:
        if time.time() > deadline - 5:
            break
        result = _try_pattern_decomposition(n, m, p, pattern, deadline)
        if result is not None:
            return result
    
    return fallback_result
# EVOLVE-BLOCK-END
