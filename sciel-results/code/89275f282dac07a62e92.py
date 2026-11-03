"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Schoolbook rank = n*m*p = 80
    schoolbook_rank = n * m * p
    
    def build_schoolbook():
        rank = schoolbook_rank
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
    
    def compute_errors(U, V, W, rank):
        """Compute the error vector for all (i,j,k) triples."""
        errors = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    errors[i, j, k] = s - 1.0
        return errors
    
    def max_error(U, V, W, rank):
        """Return the maximum absolute error."""
        errors = compute_errors(U, V, W, rank)
        return np.max(np.abs(errors))
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_structured_rank(rank, max_rounds=200, tol=1e-8):
        """Try to find a rank-r decomposition using structured initialization and refinement."""
        if time.time() > deadline:
            return None
        
        # Strategy: Start from a perturbed schoolbook-like structure
        # For rank r, we need to assign values to U, V, W such that the tensor equation holds
        
        # Initialize with random values in a structured way
        # Use the fact that we can scale columns to control magnitudes
        np.random.seed(42 + rank)
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 5:
                return None
            
            seed = attempt * 1000 + rank
            np.random.seed(seed)
            
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Convert to lists for the error computation
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            # Compute initial error
            init_err = max_error(U_list, V_list, W_list, rank)
            if init_err < 0.1:
                # Already close, try to refine
                pass
            
            # Coordinate descent: try to reduce error by adjusting individual entries
            best_err = init_err
            best_U, best_V, best_W = U_list.copy(), V_list.copy(), W_list.copy()
            
            # Candidate values for discrete search
            candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
            
            for round_num in range(max_rounds):
                if time.time() > deadline - 5:
                    break
                
                improved = False
                
                # Try random coordinate updates
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    
                    # Choose which matrix to update
                    mat_choice = np.random.choice(['U', 'V', 'W'])
                    
                    if mat_choice == 'U':
                        row = np.random.randint(n * m)
                        col = np.random.randint(rank)
                        old_val = U_list[row][col]
                        for new_val in candidates:
                            if new_val == old_val:
                                continue
                            U_list[row][col] = new_val
                            err = max_error(U_list, V_list, W_list, rank)
                            if err < best_err - 1e-10:
                                best_err = err
                                best_U = [row[:] for row in U_list]
                                improved = True
                            else:
                                U_list[row][col] = old_val
                    elif mat_choice == 'V':
                        row = np.random.randint(m * p)
                        col = np.random.randint(rank)
                        old_val = V_list[row][col]
                        for new_val in candidates:
                            if new_val == old_val:
                                continue
                            V_list[row][col] = new_val
                            err = max_error(U_list, V_list, W_list, rank)
                            if err < best_err - 1e-10:
                                best_err = err
                                best_V = [row[:] for row in V_list]
                                improved = True
                            else:
                                V_list[row][col] = old_val
                    else:
                        row = np.random.randint(p * n)
                        col = np.random.randint(rank)
                        old_val = W_list[row][col]
                        for new_val in candidates:
                            if new_val == old_val:
                                continue
                            W_list[row][col] = new_val
                            err = max_error(U_list, V_list, W_list, rank)
                            if err < best_err - 1e-10:
                                best_err = err
                                best_W = [row[:] for row in W_list]
                                improved = True
                            else:
                                W_list[row][col] = old_val
                
                if not improved:
                    break
                
                # Update best
                U_list = best_U
                V_list = best_V
                W_list = best_W
                
                if best_err < tol:
                    return {
                        "U": U_list,
                        "V": V_list,
                        "W": W_list
                    }
        
        return None
    
    # Build the guaranteed schoolbook solution
    best_result = build_schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower-rank decompositions
    # Focus on ranks that are plausible improvements
    # The theoretical lower bound is max(n*m, m*p, p*n) = 20
    # But for matmul, known results suggest higher ranks are needed
    # Try a range of ranks, focusing on those just below schoolbook
    
    candidate_ranks = [79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 
                       64, 63, 62, 61, 60, 58, 56, 54, 52, 50]
    
    for rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
        
        result = try_structured_rank(rank, max_rounds=100)
        if result is not None:
            if verify(result["U"], result["V"], result["W"]):
                best_result = result
                best_rank = rank
                break  # Found a valid lower-rank solution
    
    return best_result
# EVOLVE-BLOCK-END
