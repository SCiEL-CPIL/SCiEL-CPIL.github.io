"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import itertools
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = start_time + 145.0  # 5s safety margin
    
    # Start with schoolbook as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Strategy: Use a smarter search that exploits the structure of matrix multiplication
    # We know that for matrix multiplication, there are known efficient algorithms
    # For 3x4 * 4x7, we can try to find a decomposition with rank < 84
    
    # Key insight: We can use the fact that the tensor T[i,j,k] = 1 for all i,j,k
    # This means we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try a more structured approach:
    # 1. Try to find decompositions using known fast matrix multiplication patterns
    # 2. Use linear algebra to solve for the coefficients given a sparsity pattern
    
    np.random.seed(42)
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But we need to be realistic - let's try ranks from 30 to 83
    
    found = False
    
    # Try to find a decomposition with rank as low as possible
    # Start from lower ranks and go up
    for target_rank in range(28, rank_school):
        if time.time() > DEADLINE - 30:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple random attempts for this rank
        num_attempts = 200 if target_rank < 40 else 100
        
        for attempt in range(num_attempts):
            if time.time() > DEADLINE - 20:
                break
                
            # Use a mix of strategies:
            # 1. Random sparse matrices
            # 2. Random dense matrices with small values
            
            if attempt % 2 == 0:
                # Sparse approach: each row has a few non-zeros
                U = np.zeros((n*m, target_rank), dtype=int)
                V = np.zeros((m*p, target_rank), dtype=int)
                W = np.zeros((p*n, target_rank), dtype=int)
                
                for c in range(target_rank):
                    # Choose which rows will have non-zero entries in column c
                    u_rows = np.random.choice(n*m, size=min(3, n*m), replace=False)
                    v_rows = np.random.choice(m*p, size=min(4, m*p), replace=False)
                    w_rows = np.random.choice(p*n, size=min(3, p*n), replace=False)
                    
                    for r in u_rows:
                        U[r, c] = np.random.choice([-1, 1])
                    for r in v_rows:
                        V[r, c] = np.random.choice([-1, 1])
                    for r in w_rows:
                        W[r, c] = np.random.choice([-1, 1])
            else:
                # Dense approach with small values
                max_val = 2
                U = np.random.randint(-max_val, max_val + 1, size=(n*m, target_rank))
                V = np.random.randint(-max_val, max_val + 1, size=(m*p, target_rank))
                W = np.random.randint(-max_val, max_val + 1, size=(p*n, target_rank))
            
            # Check if this decomposition works
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        total = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                        if total != 1:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            
            if success:
                found = True
                best_rank = target_rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        if found:
            break
    
    # If we found a good one, try to improve it further
    if found and time.time() < DEADLINE - 40:
        # Try to find an even better decomposition
        for target_rank in range(28, best_rank):
            if time.time() > DEADLINE - 25:
                break
            
            num_attempts = 50
            for attempt in range(num_attempts):
                if time.time() > DEADLINE - 20:
                    break
                
                max_val = 2
                U = np.random.randint(-max_val, max_val + 1, size=(n*m, target_rank))
                V = np.random.randint(-max_val, max_val + 1, size=(m*p, target_rank))
                W = np.random.randint(-max_val, max_val + 1, size=(p*n, target_rank))
                
                success = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i*m + j
                            v_idx = j*p + k
                            w_idx = k*n + i
                            total = np.sum(U[u_idx] * V[v_idx] * W[w_idx])
                            if total != 1:
                                success = False
                                break
                        if not success:
                            break
                    if not success:
                        break
                
                if success:
                    best_rank = target_rank
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break
            
            if best_rank == 28:
                break
    
    return best_result
# EVOLVE-BLOCK-END
