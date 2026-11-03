"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if val != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions using numerical methods.
    # For each rank, we fix U and V, solve for W, and check if W is close to integers.
    
    lower_bound = max(n * m, m * p, p * n)  # 15
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 30:
            break
            
        found = False
        # Try multiple random initializations
        for seed in range(200):
            if time.time() > deadline - 30:
                break
                
            np.random.seed(seed + rank * 100000)
            
            # Try different initialization strategies
            strategy = seed % 3
            if strategy == 0:
                U = np.random.randint(-1, 2, (n * m, rank))
                V = np.random.randint(-1, 2, (m * p, rank))
            elif strategy == 1:
                U = np.random.choice([-1, 0, 1], (n * m, rank))
                V = np.random.choice([-1, 0, 1], (m * p, rank))
            else:
                U = np.random.randint(-2, 3, (n * m, rank))
                V = np.random.randint(-2, 3, (m * p, rank))
            
            # Build the linear system for W
            num_eqs = n * m * p  # 45
            num_unknowns = p * n * rank  # 15 * rank
            
            A_mat = np.zeros((num_eqs, num_unknowns))
            b_vec = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            coeff = U[i * m + j, r] * V[j * p + k, r]
                            if coeff != 0:
                                A_mat[eq_idx, (k * n + i) * rank + r] = coeff
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_arr = W_flat.reshape((p * n, rank))
                
                # Try rounding to integers and half-integers
                for round_type in [0, 0.5]:
                    W_rounded = np.round(W_arr - round_type) + round_type
                    W_rounded = np.where(np.abs(W_rounded) < 1e-10, 0, W_rounded)
                    
                    # Verify
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for r in range(rank):
                                    val += U[i * m + j, r] * V[j * p + k, r] * W_rounded[k * n + i, r]
                                if abs(val - 1) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W_rounded.tolist()}
                        best_rank = rank
                        found = True
                        break
                
                if found:
                    break
                    
            except Exception:
                continue
        
        if found:
            break
    
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
