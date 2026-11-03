"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]
DEADLINE = 150.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 5
    nm, mp, pn = n * m, m * p, p * n  # 12, 20, 15
    
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    # We need U (nm x r), V (mp x r), W (pn x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    def verify(U, V, W):
        """Check if U, V, W correctly decompose the tensor."""
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(r):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Fast verification using numpy."""
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        # Reshape and use einsum
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # We need: for all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Let's check all 60 (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Schoolbook: rank 60
    rank_schoolbook = n * m * p
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = (U_school, V_school, W_school)
    
    if time.time() > DEADLINE - 5.0:
        return {"U": best_result[0].tolist(), "V": best_result[1].tolist(), "W": best_result[2].tolist()}
    
    # Try to find a lower rank decomposition using random search with structured initialization
    # Strategy: start from schoolbook and try to merge columns
    
    # Try ranks from 59 down to some minimum
    for target_rank in range(rank_schoolbook - 1, 19, -1):
        if time.time() > DEADLINE - 10.0:
            break
        
        # Try a few random attempts at this rank
        for attempt in range(5):
            if time.time() > DEADLINE - 10.0:
                break
            
            # Start from schoolbook and try to merge pairs of columns
            # If column c1 and c2 have the same pattern in one of U, V, W, we might merge
            # Actually, let's try a different approach: random sparse matrices
            
            # Use a structured approach: try to use the fact that some entries can be shared
            # For now, try random {0, ±1, ±1/2} matrices
            np.random.seed(attempt + target_rank)
            
            # Create random sparse matrices with entries from {0, 0.5, 1, -0.5, -1}
            U_rand = np.zeros((nm, target_rank))
            V_rand = np.zeros((mp, target_rank))
            W_rand = np.zeros((pn, target_rank))
            
            # Initialize with sparse random values
            for c in range(target_rank):
                # Pick random non-zero entries
                nz_U = np.random.choice(nm, size=min(nm, 3), replace=False)
                nz_V = np.random.choice(mp, size=min(mp, 3), replace=False)
                nz_W = np.random.choice(pn, size=min(pn, 3), replace=False)
                
                for idx in nz_U:
                    U_rand[idx, c] = np.random.choice([0.5, 1.0, -0.5, -1.0])
                for idx in nz_V:
                    V_rand[idx, c] = np.random.choice([0.5, 1.0, -0.5, -1.0])
                for idx in nz_W:
                    W_rand[idx, c] = np.random.choice([0.5, 1.0, -0.5, -1.0])
            
            if verify_fast(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_result = (U_rand.copy(), V_rand.copy(), W_rand.copy())
                break
        else:
            # If we couldn't find a decomposition at this rank, continue to lower ranks
            continue
        break
    
    if time.time() > DEADLINE - 3.0:
        pass
    
    return {"U": best_result[0].tolist(), "V": best_result[1].tolist(), "W": best_result[2].tolist()}
# EVOLVE-BLOCK-END

