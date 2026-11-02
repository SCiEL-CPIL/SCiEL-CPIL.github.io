"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 110  # Leave 40s safety margin
    
    # Trivial decomposition of rank n*m*p
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
    
    best = {'U': U, 'V': V, 'W': W}
    best_rank = rank_trivial
    
    # Build target tensor for verification
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    def verify(U_mat, V_mat, W_mat):
        """Verify exact decomposition using integer arithmetic."""
        U_arr = np.array(U_mat, dtype=np.int64)
        V_arr = np.array(V_mat, dtype=np.int64)
        W_arr = np.array(W_mat, dtype=np.int64)
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_comp = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.array_equal(T_comp, T_target)
    
    # Try block-diagonal row decomposition: each row i uses rank m*p terms
    # This is equivalent to the trivial decomposition, so skip.
    
    # Attempt structured search for rank reduction
    # Try ranks from best_rank-1 downward, with limited search per rank
    np.random.seed(42)
    
    for target_rank in range(best_rank - 1, max(best_rank - 20, 80), -1):
        if time.time() > deadline:
            break
        
        # Try a few random integer decompositions
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Generate random sparse integer matrices
            U_rand = np.zeros((n*m, target_rank), dtype=np.int64)
            V_rand = np.zeros((m*p, target_rank), dtype=np.int64)
            W_rand = np.zeros((p*n, target_rank), dtype=np.int64)
            
            # Each column gets a random pattern
            for r in range(target_rank):
                # Randomly choose which entries are nonzero
                # Use small integers: -1, 0, 1
                U_rand[:, r] = np.random.randint(-1, 2, n*m)
                V_rand[:, r] = np.random.randint(-1, 2, m*p)
                W_rand[:, r] = np.random.randint(-1, 2, p*n)
            
            if verify(U_rand.tolist(), V_rand.tolist(), W_rand.tolist()):
                best = {'U': U_rand.tolist(), 'V': V_rand.tolist(), 'W': W_rand.tolist()}
                best_rank = target_rank
                break  # Move to next lower rank
        
        if time.time() > deadline:
            break
    
    return best
# EVOLVE-BLOCK-END
