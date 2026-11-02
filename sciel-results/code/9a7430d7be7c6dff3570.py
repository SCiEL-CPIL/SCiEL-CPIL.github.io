import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    # Try to find rank 12
    target_rank = 12
    if target_rank < n * m or target_rank < m * p or target_rank < p * n:
        target_rank = max(n * m, m * p, p * n)
        
    # Time limit guard
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin
    
    # Dimensions for matrices
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # We want to find U, V, W of shapes (dim_U, r), (dim_V, r), (dim_W, r)
    # such that sum_s U[i,j,s] V[j,k,s] W[k,i,s] = 1 for all i,j,k
    
    # We fix U and V, and solve for W.
    # The system is linear in W.
    # Let W be flattened to a vector of length dim_W * r.
    # The coefficient of W[k, i, s] in the equation for (i, j, k) is U[i, j, s] * V[j, k, s].
    # Note: W is indexed by (k, i) in the tensor, but the matrix W has rows (k*n+i).
    # So W_row = k * n + i.
    
    # Number of equations: n * m * p
    num_eq = n * m * p
    num_var = dim_W * target_rank
    
    # Build the matrix L such that L @ vec(W) = 1
    # L[row, col] where row corresponds to (i, j, k) and col to (W_row, s)
    # row index = i * m * p + j * p + k
    # col index = W_row * target_rank + s = (k * n + i) * target_rank + s
    
    L = np.zeros((num_eq, num_var))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = i * m * p + j * p + k
                for s in range(target_rank):
                    w_row = k * n + i
                    col = w_row * target_rank + s
                    L[row, col] = 1.0 # Placeholder, will be filled with U[i*m+j, s] * V[j*p+k, s]
                    
    # We need to iterate over random U, V
    # To make it efficient, we can precompute the structure.
    # But U, V change.
    
    found = False
    best_U = None
    best_V = None
    best_W = None
    
    # Try a few random configurations
    # We can use a heuristic: start with sparse U, V
    max_iterations = 5000
    iter_count = 0
    
    while not found and iter_count < max_iterations and (time.time() - start_time) < time_limit:
        # Generate random U, V with entries in {-1, 0, 1}
        # To ensure some non-zero structure, we can set some entries to 1
        U = np.random.randint(-1, 2, (dim_U, target_rank))
        V = np.random.randint(-1, 2, (dim_V, target_rank))
        
        # Fill L
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = i * m * p + j * p + k
                    for s in range(target_rank):
                        w_row = k * n + i
                        col = w_row * target_rank + s
                        L[row, col] = U[i * m + j, s] * V[j * p + k, s]
        
        # Solve L @ W_flat = 1
        try:
            # Use least squares if not square, but here num_var > num_eq usually
            # num_eq = 24, num_var = 8 * 12 = 96
            # So it is underdetermined. We want a specific solution.
            # We can use np.linalg.lstsq to find the minimum norm solution.
            W_flat, residuals, rank, sv = np.linalg.lstsq(L, np.ones(num_eq), rcond=None)
        except:
            iter_count += 1
            continue
            
        W = W_flat.reshape((dim_W, target_rank))
        
        # Check if W is close to half-integers
        # 2 * W should be close to integers
        W_2 = 2.0 * W
        W_int = np.round(W_2)
        diff = np.abs(W_2 - W_int)
        
        if np.max(diff) < 1e-6:
            # Found a solution
            W = W_int / 2.0
            # Verify
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for s in range(target_rank):
                            val += U[i * m + j, s] * V[j * p + k, s] * W[k * n + i, s]
                        error += (val - 1.0) ** 2
            if error < 1e-6:
                found = True
                best_U = U
                best_V = V
                best_W = W
        
        iter_count += 1
        
    if found:
        # Convert to lists of lists
        U_list = [[int(x) if x == int(x) else float(x) for x in row] for row in best_U]
        V_list = [[int(x) if x == int(x) else float(x) for x in row] for row in best_V]
        W_list = [[int(x) if x == int(x) else float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
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
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
