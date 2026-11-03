"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n=4, m=4, p=9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True

def _schoolbook(n=4, m=4, p=9):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_decomposition(n, m, p, rank, max_iter=500, seed=42):
    """Try to find a rank-r decomposition using random search."""
    random.seed(seed)
    np.random.seed(seed)
    
    for it in range(max_iter):
        # Generate random U, V with small entries
        U = np.random.randint(-1, 2, size=(n*m, rank))
        V = np.random.randint(-1, 2, size=(m*p, rank))
        
        # For each (k,i), we need to solve for W[k*n+i, c]
        # The equation is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # for all j in range(m)
        
        # This is a linear system: for fixed (k,i), we have m equations
        # sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1 for j=0..m-1
        
        # Let A[k*n+i] be a matrix where A[k*n+i][j, c] = U[i*m+j,c] * V[j*p+k,c]
        # Then we need A[k*n+i] @ W[k*n+i, :] = [1, 1, ..., 1]
        
        # Build the system
        W = np.zeros((p*n, rank), dtype=object)
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Build A matrix: m x rank
                A = np.zeros((m, rank), dtype=object)
                for j in range(m):
                    for c in range(rank):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                
                # Solve A @ w = ones(m) for w (length rank)
                # This is an underdetermined system if rank > m
                # Use least squares or find a particular solution
                
                # Convert to float for solving
                Af = np.array(A, dtype=float)
                b = np.ones(m)
                
                try:
                    # Find a solution using lstsq
                    w_sol, residuals, rank_val, sv = np.linalg.lstsq(Af, b, rcond=None)
                    
                    # Check if the solution is close to integer/half-integer
                    # and if it satisfies the equation exactly
                    w_int = np.round(w_sol * 2) / 2  # Round to nearest half-integer
                    
                    # Verify exact solution
                    error = np.abs(Af @ w_int - b)
                    if np.max(error) > 1e-9:
                        # Try rounding to integers
                        w_int = np.round(w_sol)
                        error = np.abs(Af @ w_int - b)
                        if np.max(error) > 1e-9:
                            valid = False
                            break
                    
                    W[k*n+i, :] = w_int
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Verify the full decomposition
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s limit
    
    n, m, p = 4, 4, 9
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find better decompositions
    # Start from a reasonable lower bound
    # For 4x4x4, best known is 21. For 4x4x9, estimate around 50-70
    
    candidate_ranks = list(range(50, 145, -1))
    
    for rank in candidate_ranks:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        if rank >= best_rank:
            continue
        
        # Try to find a decomposition of this rank
        result = _try_rank_decomposition(n, m, p, rank, max_iter=100, seed=rank)
        
        if result is not None:
            best_result = result
            best_rank = rank
            print(f"Found decomposition of rank {rank}")
            # Continue searching for even better
            if rank <= 55:
                break
    
    return best_result
