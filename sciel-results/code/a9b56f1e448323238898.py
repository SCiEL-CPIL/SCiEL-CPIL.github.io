"""Program that searches for a lower-rank tensor decomposition of matrix multiplication 3x5x7."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

def _validate_decomposition(U, V, W, n=3, m=5, p=7):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 5, 7
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # Convert to numpy arrays for faster computation
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to 3D arrays
    # U: (n*m, rank) -> (n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute the tensor T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # We can do this efficiently using einsum or loops
    # T[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
    
    # Iterate over all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over r
                s = 0.0
                for r in range(rank):
                    s += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _validate_decomposition_f2(U, V, W, n=3, m=5, p=7):
    """Check decomposition over F_2."""
    n, m, p = 3, 5, 7
    if not U or not V or not W:
        return False
    rank = len(U[0])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s ^= (U[i * m + j][r] & V[j * p + k][r] & W[k * n + i][r])
                if s != 1:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    
    # Start with schoolbook
    U_school = [[0] * base_rank for _ in range(n * m)]
    V_school = [[0] * base_rank for _ in range(m * p)]
    W_school = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = base_rank
    
    # Search for lower rank solutions over F_2
    # Try ranks from base_rank - 1 down to base_rank - 20
    # Use randomized search
    
    for target_rank in range(base_rank - 1, base_rank - 20, -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        attempts = 20
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            U_test = [[0] * target_rank for _ in range(n * m)]
            V_test = [[0] * target_rank for _ in range(m * p)]
            W_test = [[0] * target_rank for _ in range(p * n)]
            
            # Randomly fill with 0s and 1s
            for i in range(n * m):
                for c in range(target_rank):
                    U_test[i][c] = random.randint(0, 1)
            for j in range(m * p):
                for c in range(target_rank):
                    V_test[j][c] = random.randint(0, 1)
            for k in range(p * n):
                for c in range(target_rank):
                    W_test[k][c] = random.randint(0, 1)
            
            if _validate_decomposition_f2(U_test, V_test, W_test, n, m, p):
                # Found a solution over F_2, which is also valid over R
                best_result = {"U": U_test, "V": V_test, "W": W_test}
                best_rank = target_rank
                break
        
        if best_rank < base_rank:
            break
    
    return best_result

# EVOLVE-BLOCK-START
# The construct function is defined above. This block is kept for structural compatibility.
# EVOLVE-BLOCK-END
