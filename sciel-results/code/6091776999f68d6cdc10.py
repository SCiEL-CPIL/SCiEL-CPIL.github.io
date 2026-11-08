"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _check_decomposition(U, V, W, n, m, p):
    """Check if decomposition is valid."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U = [[0] * rank_trivial for _ in range(n * m)]
    V = [[0] * rank_trivial for _ in range(m * p)]
    W = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_trivial
    
    # Search for lower rank decompositions
    # We try ranks from rank_trivial - 1 down to a lower bound
    # The minimum possible rank is bounded by max(n*m, m*p, p*n) = 16? No, rank is at least max(n*m, m*p, p*n) is false.
    # Rank is at least max(n*m, m*p, p*n) is for boolean rank? No.
    # For matrix multiplication, rank >= max(n*m, m*p, p*n) is not a general lower bound.
    # However, we know rank >= n+m+p-2 for square? No.
    # Let's just search down to 16.
    
    half_ints = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # We will iterate over ranks
    for rank in range(rank_trivial - 1, 15, -1):
        if time.time() > deadline:
            break
            
        # Number of random attempts for this rank
        # Reduce attempts as rank decreases to save time? Or keep constant?
        max_attempts = 200 if rank > 25 else 500
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + rank)
            
            # Generate random U and V with small integer/half-integer values
            # Using 0, 0.5, 1, -0.5, -1
            U_r = np.random.choice(half_ints, size=(n * m, rank)).tolist()
            V_r = np.random.choice(half_ints, size=(m * p, rank)).tolist()
            
            # Convert to numpy arrays for linear algebra
            U_np = np.array(U_r)
            V_np = np.array(V_r)
            
            W_np = np.zeros((p * n, rank))
            valid = True
            
            # For each (i, k), solve for W_{k, i, :}
            for i in range(n):
                for k in range(p):
                    # Construct the system A W = 1
                    # A has shape (m, rank)
                    # A[j, r] = U[i, j, r] * V[j, k, r]
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A[j, r] = U_np[i * m + j, r] * V_np[j * p + k, r]
                    
                    # Solve A W_col = 1
                    # W_col has shape (rank,)
                    try:
                        if rank == m:
                            # Square system
                            W_col = np.linalg.solve(A, np.ones(m))
                        else:
                            # Underdetermined (rank > m) or Overdetermined (rank < m)
                            # If rank < m, it might be inconsistent. 
                            # We use least squares.
                            W_col, residuals, rank_l, singular_values = np.linalg.lstsq(A, np.ones(m), rcond=None)
                            # Check if solution is close to exact
                            if rank < m:
                                if np.linalg.norm(A @ W_col - np.ones(m)) > 1e-6:
                                    valid = False
                                    break
                            # If rank > m, lstsq gives minimum norm solution.
                        # Check if W_col entries are close to half-integers
                        for r in range(rank):
                            val = W_col[r]
                            # Find closest half-integer
                            closest = np.round(val * 2) / 2
                            if abs(val - closest) > 1e-6:
                                # Try to adjust? Or just mark invalid?
                                # If it's not close, this random U,V doesn't yield a half-integer W.
                                valid = False
                                break
                            W_np[k * n + i, r] = closest
                        if not valid:
                            break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                    if not valid:
                        break
            
            if valid:
                # Verify the full decomposition
                if _check_decomposition(U_r, V_r, W_np.tolist(), n, m, p):
                    best_result = {"U": U_r, "V": V_r, "W": W_np.tolist()}
                    best_rank = rank
                    break # Found a valid decomposition for this rank, move to next lower rank
        
        if best_rank < rank_trivial:
            break
            
    return best_result
# EVOLVE-BLOCK-END
