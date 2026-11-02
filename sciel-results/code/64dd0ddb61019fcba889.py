import time
import numpy as np
import random

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Schoolbook solution
    rank0 = n * m * p
    U0 = [[0] * rank0 for _ in range(n * m)]
    V0 = [[0] * rank0 for _ in range(m * p)]
    W0 = [[0] * rank0 for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j][col] = 1
                V0[j*p + k][col] = 1
                W0[k*n + i][col] = 1
                col += 1
    
    best_solution = {"U": U0, "V": V0, "W": W0}
    best_rank = rank0
    
    # Try to find rank 29 solution
    target_rank = 29
    
    def is_half_integer(x, tol=1e-6):
        return abs(x - round(2*x)/2.0) < tol

    def check_solution(U, V, W, r):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Search for rank 29
    # Strategy: Randomize U, V with entries in {-1, 0, 1}
    # For each (k, i) block, solve for W using the first few columns
    # We will try to use a sparse structure for W as well.
    
    # We iterate by randomly picking U and V, then solving for W.
    # To make W half-integer, we can solve the linear system and check.
    
    max_iters = 20000
    iters = 0
    
    while iters < max_iters and time.time() < deadline:
        iters += 1
        
        # Randomize U and V
        # Use sparse random matrices to increase chance of simple solutions
        # Probability of non-zero: 0.5
        U_rand = np.zeros((n * m, target_rank), dtype=int)
        V_rand = np.zeros((m * p, target_rank), dtype=int)
        
        # Fill with -1, 0, 1
        # We want to ensure the 3x3 submatrices for each block are likely invertible
        # So we should not be too sparse.
        for idx in range(n * m):
            for c in range(target_rank):
                if random.random() < 0.5:
                    U_rand[idx, c] = random.choice([-1, 1])
        
        for idx in range(m * p):
            for c in range(target_rank):
                if random.random() < 0.5:
                    V_rand[idx, c] = random.choice([-1, 1])
        
        # Check each block (k, i)
        W_candidate = np.zeros((p * n, target_rank), dtype=float)
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Extract 3x3 submatrix for columns 0, 1, 2
                # M[j, c] = U[i, m+j, c] * V[j, p+k, c]
                M = np.zeros((3, 3))
                for j in range(m):
                    for c in range(3):
                        M[j, c] = U_rand[i*m + j, c] * V_rand[j*p + k, c]
                
                # Solve M * w = [1, 1, 1]
                try:
                    w = np.linalg.solve(M, np.ones(3))
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check if w is half-integer
                if not (is_half_integer(w[0]) and is_half_integer(w[1]) and is_half_integer(w[2])):
                    valid = False
                    break
                
                # Store solution
                W_candidate[k*n + i, 0] = w[0]
                W_candidate[k*n + i, 1] = w[1]
                W_candidate[k*n + i, 2] = w[2]
                # Other columns are 0
        
        if not valid:
            continue
        
        # Convert to lists
        U_list = U_rand.tolist()
        V_list = V_rand.tolist()
        W_list = W_candidate.tolist()
        
        # Verify full solution
        if check_solution(U_list, V_list, W_list, target_rank):
            # Found a rank 29 solution!
            # Clean up W to be exact half-integers
            W_clean = [[round(x * 2) / 2.0 for x in row] for row in W_list]
            
            return {"U": U_list, "V": V_list, "W": W_clean}
        
        # If not found, continue searching
    
    # If no better solution found, return schoolbook
    return best_solution
# EVOLVE-BLOCK-END
