"""Matrix multiplication 2x5x5 - rank minimization via structured search."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    U_sb_np = np.array(U_sb, dtype=float)
    V_sb_np = np.array(V_sb, dtype=float)
    W_sb_np = np.array(W_sb, dtype=float)
    
    best_rank = rank_schoolbook
    best_U = U_sb_np
    best_V = V_sb_np
    best_W = W_sb_np
    
    def verify_fast(U, V, W, rank):
        """Faster verification using matrix operations."""
        # For each (i,j,k), we need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        # Let's check all entries efficiently
        for i in range(n):
            for k in range(p):
                w_row = W[k * n + i, :]
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    val = np.dot(U[a, :] * V[b, :], w_row)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    # Try to find rank 49 decomposition
    # Strategy: Random sparse initialization with local search
    def try_rank_49():
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(42)
        r = 49
        
        # Initialize with random sparse values
        # We want to find U, V, W such that the tensor is satisfied.
        # A good heuristic is to start with schoolbook and remove one term,
        # then try to adjust the remaining terms to compensate.
        
        # Schoolbook has 50 terms.
        # If we remove term 0 (i=0, j=0, k=0), we need to satisfy:
        # U[0,0]*V[0,0]*W[0,0] = 1 for (0,0,0) is missing.
        # Actually, let's just do random search.
        
        for attempt in range(100):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Set random non-zero entries
            # Ensure some structure
            for c in range(r):
                # Pick random rows
                u_row = rng.randint(n*m)
                v_row = rng.randint(m*p)
                w_row = rng.randint(p*n)
                
                U[u_row, c] = rng.choice([-1, 1])
                V[v_row, c] = rng.choice([-1, 1])
                W[w_row, c] = rng.choice([-1, 1])
            
            # Local search
            for it in range(500):
                if time.time() > deadline:
                    return None
                
                # Pick a random factor and entry
                factor_idx = rng.randint(3)
                if factor_idx == 0:
                    row = rng.randint(n * m)
                    col = rng.randint(r)
                    old_val = U[row, col]
                    for new_val in [-1, -0.5, 0, 0.5, 1]:
                        if new_val == old_val:
                            continue
                        U[row, col] = new_val
                        if verify_fast(U, V, W, r):
                            return (U, V, W, r)
                    U[row, col] = old_val
                elif factor_idx == 1:
                    row = rng.randint(m * p)
                    col = rng.randint(r)
                    old_val = V[row, col]
                    for new_val in [-1, -0.5, 0, 0.5, 1]:
                        if new_val == old_val:
                            continue
                        V[row, col] = new_val
                        if verify_fast(U, V, W, r):
                            return (U, V, W, r)
                    V[row, col] = old_val
                else:
                    row = rng.randint(p * n)
                    col = rng.randint(r)
                    old_val = W[row, col]
                    for new_val in [-1, -0.5, 0, 0.5, 1]:
                        if new_val == old_val:
                            continue
                        W[row, col] = new_val
                        if verify_fast(U, V, W, r):
                            return (U, V, W, r)
                    W[row, col] = old_val
        
        return None

    result = try_rank_49()
    if result is not None:
        U, V, W, rank = result
        if verify_fast(U, V, W, rank):
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W

    if best_rank == rank_schoolbook:
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
