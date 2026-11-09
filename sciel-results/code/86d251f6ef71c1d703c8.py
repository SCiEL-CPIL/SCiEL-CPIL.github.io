"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_structured_decomposition(n, m, p, max_rank, deadline):
    """Try to find a low-rank decomposition using structured integer ansatz.
    
    Strategy: For 3x3x5, try to find decompositions where the rank-1 terms
    have a regular pattern. We parameterize U, V, W with small integer values
    and search for valid combinations.
    """
    # For each target rank, try to find integer solutions
    # The key insight: we can try to use the structure of matrix multiplication
    # to find a basis of rank-1 tensors that span the multiplication tensor
    
    # Approach: Start from schoolbook and try to reduce rank by finding
    # linear combinations that preserve the tensor equation
    
    # For 3x3x5, let's try a direct construction
    # The multiplication tensor T[i,j,k] = 1 if the entry (i,k) of A@B is computed
    # T[i,j,k] = delta(i from A row, j from inner, k from B col)
    
    # Actually, the tensor is: T[i,j,k] = 1 for all valid (i,j,k) where
    # i in [0,n), j in [0,m), k in [0,p)
    # And the decomposition is: sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1
    
    # Let's try a systematic search with small integers
    # For rank r, we need r columns in each of U, V, W
    # Each entry is from {-2, -1, 0, 1, 2} (small integers)
    
    best = None
    best_rank = max_rank + 1
    
    # Try ranks from low to high
    for rank in range(25, max_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # For this rank, try random integer initializations with structured patterns
        # Pattern 1: Block diagonal structure
        # Pattern 2: Cyclic structure
        
        # Let's try a more targeted approach:
        # For 3x3x5, we can think of it as 5 independent 3x3 multiplications
        # sharing the inner dimension. But that's not quite right.
        
        # Actually, let's try: for each k (output column), we need to compute
        # A @ B[:,k]. This is a 3x3 times 3x1 multiplication.
        # The rank of 3x3x1 multiplication is 3 (schoolbook).
        # But we can share across k values.
        
        # Try a construction where we use the same U and V patterns
        # but vary W to handle different output columns
        
        # Let's try: rank = 3*3 + 5*something
        # Or: use the fact that for each j (inner index), we have
        # contributions from all i and k
        
        # Direct construction attempt:
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        
        # Let's try a rank-30 construction:
        # For each of the 3 inner indices j, and 3 row indices i,
        # we have 3 terms for the 5 output columns k
        # But we can share: U[i*3+j, c] depends on i,j
        # V[j*5+k, c] depends on j,k  
        # W[k*3+i, c] depends on k,i
        
        # Try: for each (i,j) pair (9 pairs), and for each k (5 values),
        # we need U[i*3+j, c] * V[j*5+k, c] * W[k*3+i, c] to sum to 1
        # for each (i,j,k)
        
        # This is a system of 45 equations in 9*r + 15*r + 15*r = 39r unknowns
        # For r=12, we have 468 unknowns for 45 equations - very underdetermined
        
        # Let's try to find a solution with small integers
        # Using a greedy/structured approach
        
        # Construction idea: 
        # Let U[i*3+j, c] = a[i,j,c], V[j*5+k, c] = b[j,k,c], W[k*3+i, c] = d[k,i,c]
        # We need: sum_c a[i,j,c]*b[j,k,c]*d[k,i,c] = 1 for all i,j,k
        
        # Try: a[i,j,c] = 1 if c < 3 and c == i, else 0  (selects row i)
        #       b[j,k,c] = 1 if c < 3 and c == j, else 0  (selects inner j)  
        #       d[k,i,c] = 1 if c < 3 and c == k, else 0  (selects col k)
        # This gives rank 3, but only works for 3x3x3, not 3x3x5
        
        # For 3x3x5, we need more terms. Let's try:
        # Split the 5 output columns into groups and handle each group
        
        # Try rank = 3*5 = 15: for each (i,k) pair, have a dedicated term
        # U[i*3+j, c] where c = i*5+k: = 1 if j == j0 for some j0
        # This doesn't work directly
        
        # Let me try a different approach: use numpy to solve the system
        # for small ranks with integer constraints
        
        # For rank r, the equations are nonlinear (cubic in variables)
        # But if we fix two of the three matrices, the third is linear
        
        # Strategy: Fix U and V randomly with small integers, then solve for W
        # W[k*n+i, c] appears linearly in the equation for each (i,j,k)
        
        # For fixed U, V: 
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # This is linear in W for each (i,j,k)
        
        # But W[k*n+i, c] is shared across different j values for the same (k,i)
        # So we have a coupled system
        
        # Let's try: for each c, the term U[:,c] * V[:,c] * W[:,c] is a rank-1 tensor
        # in the (i,j,k) space
        
        # I'll try a stochastic approach with integer values
        np.random.seed(42)
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small integers
            U_int = np.random.randint(-2, 3, (n*m, rank))
            V_int = np.random.randint(-2, 3, (m*p, rank))
            
            # For each (k,i) pair, solve for W[k*n+i, :] 
            # The equation for (i,j,k) is:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # For fixed (i,k), this is: sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c] = 1
            # for j = 0, 1, 2
            # This is a system of 3 equations in r unknowns (W[k*n+i, 0..r-1])
            
            # Collect all equations
            # For each (i,k), we have 3 equations (one per j)
            # Total: n*p*3 = 45 equations
            # Unknowns: p*n*r = 15*r
            
            # Let's set up the linear system for W given U, V
            # Equation for (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # Group by (k,i): for fixed (k,i), equations are over j=0..m-1
            # Let A_jk = sum_c U[i*m+j, c] * V[j*p+k, c] for each j
            # Then: sum_c A_jk * W[k*n+i, c] = 1 for each j
            
            # This is: A_k * W[k*n+i, :] = 1 (vector of ones of length m)
            # where A_k is an m x r matrix with A_k[j, c] = U[i*m+j, c] * V[j*p+k, c]
            
            # Wait, that's not right. Let me re-derive.
            # For fixed (i,k), and variable j:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # Let w_c = W[k*n+i, c]
            # Then: sum_c (U[i*m+j, c] * V[j*p+k, c]) * w_c = 1 for j=0,1,2
            
            # So for each (i,k), we have a system: M_{i,k} * w = 1
            # where M_{i,k}[j, c] = U[i*m+j, c] * V[j*p+k, c]
            
            # This is a 3 x r system. If r >= 3, it's underdetermined.
            # We can find a solution if the system is consistent.
            
            W_int = np.zeros((p*n, rank), dtype=int)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the 3 x r matrix
                    M = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            M[j, c] = U_int[i*m+j, c] * V_int[j*p+k, c]
                    
                    # Solve M * w = 1 (where 1 is a vector of ones of length m)
                    # Use least squares
                    try:
                        w_sol, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        # Check if solution is close to integer
                        w_rounded = np.round(w_sol)
                        if np.allclose(M @ w_rounded, np.ones(m), atol=1e-6):
                            W_int[k*n+i, :] = w_rounded
                        else:
                            # Try to find an integer solution
                            # For small r, we can search
                            if rank <= 6:
                                found = False
                                for w_tuple in product(range(-3, 4), repeat=rank):
                                    w_test = np.array(w_tuple)
                                    if np.allclose(M @ w_test, np.ones(m), atol=1e-6):
                                        W_int[k*n+i, :] = w_test
                                        found = True
                                        break
                                if not found:
                                    valid = False
                                    break
                            else:
                                valid = False
                                break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }, rank
    
    return None, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Schoolbook decomposition of rank n*m*p
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    if time.time() < deadline - 20:
        result, rank = _try_structured_decomposition(n, m, p, rank_schoolbook - 1, deadline)
        if result is not None:
            best = result
            best_rank = rank
    
    return best
# EVOLVE-BLOCK-END
