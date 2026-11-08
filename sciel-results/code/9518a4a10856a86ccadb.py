import numpy as np
import random
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # A: n×m, B: m×p, C: n×p
    # Tensor T[i,j,k] = 1 if this term contributes to C[i,k] via A[i,j]*B[j,k]
    # U: (n*m) × rank, V: (m*p) × rank, W: (p*n) × rank
    # Convention: W[k*n + i, c]
    
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    # Schoolbook as baseline
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def verify_fast(U, V, W, rank):
        """Faster verification using numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Build tensor from decomposition
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # T[i,j,k] = sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
        # Use einsum
        T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        # Expected T is all ones
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-9)
    
    def verify_exact(U, V, W, rank):
        """Exact verification using integer arithmetic if possible."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0
                    for c in range(rank):
                        actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if actual != 1:
                        return False
        return True
    
    random.seed(42)
    
    # Strategy 1: Try structured decompositions at various ranks
    # Theoretical lower bound: max(n*m, m*p, n*p) = max(16, 24, 24) = 24
    
    for target_rank in range(24, best_rank):
        if time.time() > deadline - 20:
            break
        
        # Try several random initializations with small integer values
        for attempt in range(200):
            if time.time() > deadline - 20:
                break
            
            # Random sparse initialization with values from {-1, 0, 1}
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            for c in range(target_rank):
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.5:
                            U[i*m+j][c] = random.choice([1, -1])
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.5:
                            V[j*p+k][c] = random.choice([1, -1])
                for k in range(p):
                    for i in range(n):
                        if random.random() < 0.5:
                            W[k*n+i][c] = random.choice([1, -1])
            
            if verify_fast(U, V, W, target_rank):
                if verify_exact(U, V, W, target_rank):
                    best_rank = target_rank
                    best_U, best_V, best_W = U, V, W
                    break
    
    # Strategy 2: Local search - start from schoolbook and try to reduce rank
    if best_rank == n * m * p and time.time() < deadline - 30:
        # Try to find a lower rank by greedy elimination
        current_rank = best_rank
        U = [row[:] for row in best_U]
        V = [row[:] for row in best_V]
        W = [row[:] for row in best_W]
        
        # Try removing each column and see if we can fix it
        for col_to_remove in range(current_rank):
            if time.time() > deadline - 20:
                break
            
            # Temporarily remove this column
            U_temp = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
            V_temp = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
            W_temp = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
            new_rank = current_rank - 1
            
            # Check if the remaining decomposition still works
            if verify_fast(U_temp, V_temp, W_temp, new_rank):
                if verify_exact(U_temp, V_temp, W_temp, new_rank):
                    U, V, W = U_temp, V_temp, W_temp
                    current_rank = new_rank
                    best_rank = current_rank
                    best_U, best_V, best_W = U, V, W
                    # Restart the loop since indices shifted
                    col_to_remove = -1
    
    return {"U": best_U, "V": best_V, "W": best_W}
