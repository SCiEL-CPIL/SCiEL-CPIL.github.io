import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x5 matrix.
    Uses schoolbook decomposition (rank 100) as guaranteed fallback,
    then attempts rank reduction via random search and local refinement.
    """
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit

    # Schoolbook decomposition as guaranteed fallback
    base_rank = n * m * p
    U_sb = np.zeros((n * m, base_rank), dtype=np.int64)
    V_sb = np.zeros((m * p, base_rank), dtype=np.int64)
    W_sb = np.zeros((p * n, base_rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_rank = base_rank
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def validate(U, V, W, n, m, p, rank):
        """
        Validate that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for all i in [0,n), j in [0,m), k in [0,p).
        """
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        s += U[ui, r] * V[vj, r] * W[wk, r]
                    if s != 1:
                        return False
        return True

    def try_random_search(max_rank, time_budget):
        """
        Try to find a decomposition of rank max_rank using random initialization
        and local search. Returns (U, V, W) if found, else None.
        """
        nonlocal best_rank, best_U, best_V, best_W
        
        # Precompute index mappings for speed
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    indices.append((i * m + j, j * p + k, k * n + i))
        
        num_indices = len(indices)
        
        def check_decomp(U, V, W, rank):
            """Fast check: compute all triple products and verify sum == 1"""
            for ui, vj, wk in indices:
                s = 0
                for r in range(rank):
                    s += U[ui, r] * V[vj, r] * W[wk, r]
                if s != 1:
                    return False
            return True
        
        def local_search(U, V, W, rank, max_iters=100):
            """
            Local search: try flipping individual entries to fix errors.
            """
            U = U.copy()
            V = V.copy()
            W = W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if check_decomp(U, V, W, rank):
                    return U, V, W
                
                # Find an error
                error_found = False
                for ui, vj, wk in indices:
                    s = 0
                    for r in range(rank):
                        s += U[ui, r] * V[vj, r] * W[wk, r]
                    if s != 1:
                        error_found = True
                        # Try to fix by adjusting one entry
                        # This is a heuristic - try random column
                        r = np.random.randint(rank)
                        # Try flipping U[ui, r]
                        old_val = U[ui, r]
                        U[ui, r] = 1 - old_val if old_val in [0, 1] else 1
                        if check_decomp(U, V, W, rank):
                            return U, V, W
                        U[ui, r] = old_val
                        
                        # Try flipping V[vj, r]
                        old_val = V[vj, r]
                        V[vj, r] = 1 - old_val if old_val in [0, 1] else 1
                        if check_decomp(U, V, W, rank):
                            return U, V, W
                        V[vj, r] = old_val
                        
                        # Try flipping W[wk, r]
                        old_val = W[wk, r]
                        W[wk, r] = 1 - old_val if old_val in [0, 1] else 1
                        if check_decomp(U, V, W, rank):
                            return U, V, W
                        W[wk, r] = old_val
                        break
                if not error_found:
                    break
            
            if check_decomp(U, V, W, rank):
                return U, V, W
            return None
        
        # Try random binary initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U = np.random.randint(0, 2, (n * m, max_rank), dtype=np.int64)
            V = np.random.randint(0, 2, (m * p, max_rank), dtype=np.int64)
            W = np.random.randint(0, 2, (p * n, max_rank), dtype=np.int64)
            
            if check_decomp(U, V, W, max_rank):
                if max_rank < best_rank:
                    best_rank = max_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                return U, V, W
            
            # Try local search
            result = local_search(U, V, W, max_rank, max_iters=20)
            if result is not None:
                U, V, W = result
                if max_rank < best_rank:
                    best_rank = max_rank
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                return U, V, W
        
        return None

    # Try ranks from 99 down to 50 (reasonable lower bound)
    for rank in range(base_rank - 1, 49, -1):
        if time.time() > deadline:
            break
        
        result = try_random_search(rank, time_budget=2.0)
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break  # Found a valid lower rank, stop

    # Return the best decomposition found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
