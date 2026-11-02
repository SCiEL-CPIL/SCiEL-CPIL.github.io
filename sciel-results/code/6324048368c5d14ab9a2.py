"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 2x5 * 5x6 matrix multiplication.
    Falls back to schoolbook if no improvement is found within time budget.
    """
    deadline = time.time() + 120.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank = n*m*p = 60)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p  # 60
    
    # Try to find a lower-rank decomposition using random search with exact verification
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if and only if 
    # (i,j,k) corresponds to a valid multiplication term.
    # A rank-r decomposition gives U (nm x r), V (mp x r), W (pn x r) such that
    # for all (i,j,k): sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = T[i,j,k]
    
    # For exact integer/half-integer entries, we can try small integer values.
    
    # Strategy: try ranks below 60, starting from known lower bounds.
    # The Strassen algorithm gives rank 7 for 2x2, so for 2x5*5x6 we might do better than 60.
    # However, finding an exact decomposition is NP-hard in general.
    
    # Let's try a structured approach: use block decomposition ideas.
    # For 2x5 * 5x6, we can think of this as multiplying blocks.
    
    # Try random search for small ranks
    def verify_decomposition(U, V, W, T):
        """Check if U, V, W exactly decompose tensor T."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(r):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if T[i, j, k] == 0:
                        if s != 0:
                            return False
                    else:
                        if s != 1:
                            return False
        return True
    
    # Build the target tensor
    T = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1  # All entries are 1 for matrix multiplication
    
    # Try ranks from a lower bound up to best_rank - 1
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # But also, for matrix multiplication, there are known lower bounds.
    # Let's try ranks from 30 to best_rank - 1
    
    # Use random search with integer entries in {-1, 0, 1} or {-2, -1, 0, 1, 2}
    # This is a heuristic approach; may not find optimal but might find improvement.
    
    random.seed(42)
    
    for rank in range(30, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations for this rank
        max_attempts = 50 if rank > 40 else 20
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random U, V, W with small integer entries
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            
            # Fill with random values from {-1, 0, 1}
            for t in range(rank):
                # Each column of U, V, W gets random entries
                U[:, t] = np.random.randint(-1, 2, size=n * m)
                V[:, t] = np.random.randint(-1, 2, size=m * p)
                W[:, t] = np.random.randint(-1, 2, size=p * n)
            
            # Convert to lists
            U_list = [list(map(int, row)) for row in U]
            V_list = [list(map(int, row)) for row in V]
            W_list = [list(map(int, row)) for row in W]
            
            if verify_decomposition(U_list, V_list, W_list, T):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = rank
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
