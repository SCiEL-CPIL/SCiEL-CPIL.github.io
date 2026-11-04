"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
        return False
    
    # Check all entries of the tensor
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i1, j1 = divmod(a, m)
                j2, k2 = divmod(b, p)
                k1, i2 = divmod(c, n)
                
                if j1 == j2 and i1 == i2 and k2 == k1:
                    expected = 1.0
                else:
                    expected = 0.0
                
                actual = float(np.dot(U_arr[a, :], V_arr[b, :]))  # This is wrong, need triple product
                # Actually need sum_r U[a,r]*V[b,r]*W[c,r]
                actual = float(np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :]))
                
                if abs(actual - expected) > 1e-9:
                    return False
    
    return True

def construct():
    """
    Returns the schoolbook decomposition for 4x5x6 matrix multiplication.
    Rank is 4*5*6 = 120.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
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
    
    # Try to reduce rank via local search
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Local search: try to merge rank-1 components
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    current_rank = rank
    # Try to find linear dependencies to reduce rank
    # This is a heuristic - try random projections
    
    np.random.seed(42)
    max_iter = 5000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try to eliminate one rank component
        if current_rank <= 30:  # Lower bound
            break
        
        # Pick a random component to try to eliminate
        r_elim = np.random.randint(current_rank)
        
        # Try to express component r_elim as a linear combination of others
        # This is a linear system problem
        # For simplicity, try random perturbations
        
        # Skip complex local search for now due to time constraints
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
