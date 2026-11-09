"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition (rank = n*m*p = 54)
    base_rank = n * m * p
    best_U = [[0] * base_rank for _ in range(n * m)]
    best_V = [[0] * base_rank for _ in range(m * p)]
    best_W = [[0] * base_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    best_rank = base_rank
    best_result = (best_U, best_V, best_W)
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if isinstance(U, list) else U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(rank):
                        if isinstance(U, list):
                            u_val = U[i*m + j][r]
                            v_val = V[j*p + k][r]
                            w_val = W[k*n + i][r]
                        else:
                            u_val = U[i*m + j, r]
                            v_val = V[j*p + k, r]
                            w_val = W[k*n + i, r]
                        total += u_val * v_val * w_val
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank decompositions using random search
    np.random.seed(42)
    
    # Try ranks from 27 upward (theoretical lower bound)
    # Focus on reasonable range: 27 to 53
    for target_rank in range(27, base_rank):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        max_trials = 200
        for trial in range(max_trials):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random small integers/halves
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            # Use small integer values: -2, -1, 0, 1, 2
            for r in range(target_rank):
                U[:, r] = np.random.randint(-2, 3, size=n*m)
                V[:, r] = np.random.randint(-2, 3, size=m*p)
                W[:, r] = np.random.randint(-2, 3, size=p*n)
            
            # Verify
            if verify_decomposition(U, V, W):
                best_rank = target_rank
                best_result = (U.tolist(), V.tolist(), W.tolist())
                break
        
        if best_rank < base_rank:
            break
    
    # If no better found, try structured approaches
    if best_rank == base_rank and time.time() < deadline - 20:
        # Try a structured approach: for each (i,k), use a small number of terms
        # that capture the sum over j
        
        # Approach: use a decomposition where we group by output (i,k)
        # For each (i,k), we need sum_j A[i,j]*B[j,k]
        # We can use 3 terms per (i,k) if we can share across (i,k) pairs
        
        # Try rank = n*p*m / some_factor
        # Let's try a different structured approach
        
        # Use the fact that m=3 is small
        # For each j, we have n*p = 18 terms in the trivial decomposition
        # Total: 3*18 = 54
        
        # Can we share? If we use the same "pattern" for different j,
        # we might reduce the rank
        
        # Try a rank-27 decomposition (theoretical lower bound)
        # This would require each rank-1 term to be "efficient"
        
        # For now, stick with the trivial solution if no better found
        pass
    
    # Convert best result to list format
    if isinstance(best_result[0], np.ndarray):
        U_final = best_result[0].tolist()
        V_final = best_result[1].tolist()
        W_final = best_result[2].tolist()
    else:
        U_final = best_result[0]
        V_final = best_result[1]
        W_final = best_result[2]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
