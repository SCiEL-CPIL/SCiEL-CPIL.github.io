"""Matrix multiplication 5,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # T[i,j,k] should be sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # The multiplication tensor has T[i,j,k] = 1 if i,j,k are the (row, col, result_col)
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor M[i,j,k] = 1 means A[i,j]*B[j,k] contributes to C[i,k]
    # So the tensor is M[i,j,k] = delta (always 1 for valid indices in the multiplication pattern)
    # Wait - the matrix multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k
    # because A[i,j] * B[j,k] is a term in C[i,k].
    # Actually the standard multiplication tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    # The decomposition requires: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Check a subset of entries for speed
    errors = 0
    n_checks = min(500, n*m*p)
    indices = list(product(range(n), range(m), range(p)))
    np.random.seed(42)
    if len(indices) > n_checks:
        indices = [indices[np.random.choice(len(indices), n_checks, replace=False)] for _ in range(n_checks)]
    
    for (i, j, k) in indices:
        s = 0.0
        for c in range(rank):
            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        if abs(s - 1.0) > tol:
            errors += 1
    
    return errors == 0

def _try_rank(r, n=5, m=5, p=5, max_time=120.0, seed=0):
    """Try to find a rank-r decomposition using random initialization + iterative refinement."""
    start_time = time.time()
    deadline = start_time + max_time
    
    np.random.seed(seed)
    
    # Initialize with random ±1 entries
    U = np.random.choice([-1, 1], size=(n*m, r))
    V = np.random.choice([-1, 1], size=(m*p, r))
    W = np.random.choice([-1, 1], size=(p*n, r))
    
    # Convert to float for optimization
    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Iterative: for each column, try to fix one factor
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        # Compute the "error tensor" E[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1
        # For each column c, we can try to optimize U[:,c], V[:,c], or W[:,c]
        
        # Optimize W[:, c] given U and V
        for c in range(r):
            if time.time() > deadline - 5:
                break
            
            # W[k*n+i, c] should satisfy: sum_c' U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] = 1
            # For fixed c, the contribution of W[k*n+i,c] is:
            # sum_{i,j} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should help bring the sum to 1
            
            # Actually, let's think differently. The constraint is:
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # This is a system of n*m*p = 125 equations in r*3 unknowns (but each column is independent)
            # Wait no - each column c contributes U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            # and the sum over c must equal 1 for each (i,j,k).
            
            # This is a non-linear system. Let me use a different approach.
            pass
        
        # Alternative: use the fact that for a valid decomposition, 
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
        
        # Let's use a coordinate descent approach
        # For each column c, given all other columns, solve for this column.
        
        pass
    
    # Compute final error
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                error += (s - 1.0)**2
    
    return error, U, V, W

def _strassen_block_decompose(block_size, n, m, p):
    """Apply Strassen-like block decomposition."""
    # This is a placeholder for more sophisticated approaches
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Start with schoolbook rank 125
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # Known: rank 48 for 5x5x5 is achievable
    # Let's try various approaches
    
    # Approach 1: Random search with ±1 entries for lower ranks
    for target_rank in [48, 52, 56, 60, 64, 72, 80, 90, 100, 112, 124]:
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + target_rank * 100 + attempt)
            
            U_t = np.random.choice([-1, 1], size=(n*m, target_rank)).astype(float)
            V_t = np.random.choice([-1, 1], size=(m*p, target_rank)).astype(float)
            W_t = np.random.choice([-1, 1], size=(p*n, target_rank)).astype(float)
            
            # Check if this is a valid decomposition
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = 0.0
                        for c in range(target_rank):
                            s += U_t[i*m+j, c] * V_t[j*p+k, c] * W_t[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U_t, V_t, W_t
                break
        
        if found:
            break
    
    # Approach 2: Try to use known structure
    # The multiplication tensor T[i,j,k] = 1 for all i,j,k in [5]
    # We need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let's try a structured approach using polynomial evaluation
    # Map (i,j) -> x^i * y^j, (j,k) -> y^j * z^k, result (i,k) -> x^i * z^k
    # Over a finite field, we can use polynomial multiplication
    
    # Use GF(31) with points
    field_size = 31
    points_x = list(range(5))  # x values
    points_y = list(range(5))  # y values  
    points_z = list(range(5))  # z values
    
    # Actually, let me try a direct approach with the constraint that entries are ±1 or 0
    
    # For rank 48, we need 48 columns where each entry is from {-1, 0, 1}
    # This is essentially a combinatorial search problem
    
    # Let's try a smarter random approach
    for target_rank in range(48, min(best_rank, 60)):
        if time.time() > deadline - 10:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(12345 + target_rank * 1000 + attempt)
            
            # Use sparse ±1 matrices
            density = 0.5
            U_t = np.zeros((n*m, target_rank))
            V_t = np.zeros((m*p, target_rank))
            W_t = np.zeros((p*n, target_rank))
            
            for c in range(target_rank):
                # Random pattern for each column
                for idx in range(n*m):
                    if np.random.random() < density:
                        U_t[idx, c] = np.random.choice([-1, 1])
                for idx in range(m*p):
                    if np.random.random() < density:
                        V_t[idx, c] = np.random.choice([-1, 1])
                for idx in range(p*n):
                    if np.random.random() < density:
                        W_t[idx, c] = np.random.choice([-1, 1])
            
            # Verify
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = sum(U_t[i*m+j, c] * V_t[j*p+k, c] * W_t[k*n+i, c] for c in range(target_rank))
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
            
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U_t, V_t, W_t
                break
    
    # Convert to list format
    U_list = [[float(best_U[i, c]) for c in range(best_rank)] for i in range(n*m)]
    V_list = [[float(best_V[i, c]) for c in range(best_rank)] for i in range(m*p)]
    W_list = [[float(best_W[i, c]) for c in range(best_rank)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
