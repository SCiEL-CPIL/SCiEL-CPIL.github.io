"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin before 150s kill

    # Schoolbook decomposition as fallback
    def schoolbook():
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

    def verify_fast(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor using NumPy."""
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
            
        # Convert to numpy arrays
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Reshape
        # U: (n*m, rank) -> (n, m, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        # V: (m*p, rank) -> (m, p, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        # W: (p*n, rank) -> (p, n, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # Compute sum over rank dimension
        # Indices: i, j, k
        # U[i, j, r], V[j, k, r], W[k, i, r]
        # We want result[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        
        try:
            # einsum: i j r, j k r, k i r -> i j k
            result = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            return False
            
        # Check if result is all 1s
        if not np.allclose(result, 1.0, atol=1e-9):
            return False
        return True

    def check_half_integers(U, V, W):
        """Check all entries are integers or half-integers."""
        for arr in [U, V, W]:
            for row in arr:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True

    best_result = schoolbook()
    best_rank = n * m * p

    # Try ranks from the lower bound upward
    lower_bound = max(n * m, m * p, p * n)
    
    # Use a fixed seed for reproducibility in the search process itself, 
    # but vary the initialization.
    rng = np.random.RandomState(42)
    
    # Expanded set of values to try
    # Includes 0, +/-1, +/-2, +/-0.5, +/-1.5
    choices = [0, 1, -1, 2, -2, 0.5, -0.5, 1.5, -1.5]
    
    # We will search ranks from lower_bound to best_rank - 1
    # To make it more effective, we can bias the search towards lower ranks
    # but also allow more attempts for lower ranks.
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Determine number of attempts based on remaining time and rank
        # Lower ranks are harder to find, so we might want more attempts there?
        # Or just a fixed number that is larger than before.
        # With fast verification, we can do many more attempts.
        attempts = 500 if target_rank < 50 else 100
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Generate random matrices
            # Using numpy random choice for speed
            U_flat = rng.choice(choices, size=(n * m, target_rank))
            V_flat = rng.choice(choices, size=(m * p, target_rank))
            W_flat = rng.choice(choices, size=(p * n, target_rank))
            
            # Convert to list of lists for the output format required, 
            # but keep numpy for verification
            U_list = U_flat.tolist()
            V_list = V_flat.tolist()
            W_list = W_flat.tolist()
            
            if verify_fast(U_list, V_list, W_list):
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                break
        
        # If we found a valid rank, we can break if we hit the lower bound
        if best_rank == lower_bound:
            break

    # Ensure the result is valid
    if not verify_fast(best_result["U"], best_result["V"], best_result["W"]):
        best_result = schoolbook()
        best_rank = n * m * p

    return best_result
# EVOLVE-BLOCK-END
