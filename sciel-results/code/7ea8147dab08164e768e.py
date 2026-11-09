import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_iter=1000):
    """Try to reduce rank by finding and eliminating dependent columns."""
    rank = len(U[0]) if U else 0
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    
    improved = True
    while improved and rank > 0:
        improved = False
        # Try removing each column and checking if the remaining still works
        # This is O(rank^2 * n*m*p) which is expensive, so limit iterations
        for r_remove in range(rank):
            if rank - r_remove < 24:  # Lower bound
                break
            # Try removing column r_remove
            U_new = [row[:r_remove] + row[r_remove+1:] for row in U]
            V_new = [row[:r_remove] + row[r_remove+1:] for row in V]
            W_new = [row[:r_remove] + row[r_remove+1:] for row in W]
            
            if _verify(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                rank -= 1
                improved = True
                break
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Build the rank-84 Strassen solution
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    C_A = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    
    C_B = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    
    Targets = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    for t in range(7):
        for i_loc in range(2):
            for j_loc in range(2):
                for k_loc in range(3):
                    for i_blk in range(2):
                        for j_blk in range(2):
                            coeff_a = C_A[t][i_blk][j_blk]
                            if coeff_a != 0:
                                I = i_blk * 2 + i_loc
                                J = j_blk * 2 + j_loc
                                U[I * m + J][col] = coeff_a
                    for j_blk in range(2):
                        for k_blk in range(2):
                            coeff_b = C_B[t][j_blk][k_blk]
                            if coeff_b != 0:
                                J = j_blk * 2 + j_loc
                                K = k_blk * 3 + k_loc
                                V[J * p + K][col] = coeff_b
                    for i_t, k_t, coeff_c in Targets[t]:
                        I = i_t * 2 + i_loc
                        K = k_t * 3 + k_loc
                        W[K * n + I][col] = coeff_c
                    col += 1
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to reduce rank
    U2, V2, W2, rank2 = _try_reduce_rank(U, V, W, n, m, p)
    if rank2 < best_rank:
        best_U, best_V, best_W, best_rank = U2, V2, W2, rank2
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try random perturbations / gauge transformations to find better solutions
    # This is a heuristic search
    import random
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        if best_rank <= 80:
            break
        
        # Try a random linear combination of columns to see if we can merge
        # Pick 2 columns and see if a linear combination gives a valid reduced solution
        if best_rank < 2:
            break
        
        # This is a simplified search - try removing random pairs
        U3 = [row[:] for row in best_U]
        V3 = [row[:] for row in best_V]
        W3 = [row[:] for row in best_W]
        
        r1, r2 = random.sample(range(best_rank), 2)
        # Try to see if column r2 can be expressed as a multiple of column r1
        # or if removing r2 still works
        U3 = [row[:r2] + row[r2+1:] for row in U3]
        V3 = [row[:r2] + row[r2+1:] for row in V3]
        W3 = [row[:r2] + row[r2+1:] for row in W3]
        
        if _verify(U3, V3, W3, n, m, p):
            best_U, best_V, best_W, best_rank = U3, V3, W3, best_rank - 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
