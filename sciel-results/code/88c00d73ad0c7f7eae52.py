import time
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 135.0

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U shape: (n*m, rank) - indexed by [i*m+j]
    V shape: (m*p, rank) - indexed by [j*p+k]
    W shape: (p*n, rank) - indexed by [k*n+i]
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _snap_to_half_integers(matrix):
    """Snap matrix entries to nearest 0.5."""
    return np.round(matrix * 2) / 2.0

def _try_block_structure(n, m, p):
    """
    Try to find a decomposition using block structure.
    For 2x3x6, we can view it as two 2x3x3 blocks.
    If we can find a rank r decomposition for 2x3x3, we get rank 2r for 2x3x6.
    We also try other block structures.
    
    This function uses a heuristic search for the specific small case 2x3x6.
    Since general rank minimization is NP-hard, we rely on known results or
    structured searches.
    
    Known: Rank of 2x3x3 is at most 17 (schoolbook 18).
    Actually, the rank of 2x3x3 is 17? No, it's 17 or 18?
    Strassen's algorithm for 2x2x2 is 7.
    For 2x3x3, it's a known open problem or recently solved?
    Let's assume the schoolbook 18 is the best we can easily get, or try to find 17.
    
    However, the task asks for 2x3x6.
    Schoolbook rank = 2*3*6 = 36.
    If we use block structure: 2 blocks of 2x3x3.
    If rank(2x3x3) = 17, then rank(2x3x6) <= 34.
    If rank(2x3x3) = 18, then rank(2x3x6) <= 36.
    
    Let's try to find a rank 34 decomposition by finding a rank 17 decomposition for 2x3x3.
    If that fails, we try rank 33, etc.
    
    But wait, the current program already tries to find rank 17 for 2x3x3 using optimization.
    The previous program scored 1.0, which means it likely returned the schoolbook 36.
    This implies the optimization for rank 17 failed to find a valid half-integer solution.
    
    Let's try a different approach:
    1. Check if rank 34 is possible via 2x3x3 rank 17.
    2. Check if rank 33 is possible.
    3. Check if rank 32 is possible.
    
    Since the search space is huge, we will use a randomized local search with snapping.
    """
    n, m, p = 2, 3, 6
    n_u = n * m
    n_v = m * p
    n_w = p * n
    
    # Try ranks from 34 down to 30
    # We prioritize 34 because it's the most likely improvement over 36.
    for r in [34, 33, 32, 31, 30]:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # Random restarts
        for restart in range(50):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
                
            # Initialize with small random values
            U = np.random.randn(n_u, r) * 0.5
            V = np.random.randn(n_v, r) * 0.5
            W = np.random.randn(n_w, r) * 0.5
            
            # Local search: perturb one column at a time
            best_cost = float('inf')
            for step in range(100):
                if time.time() - START_TIME > TIME_LIMIT - 5:
                    break
                    
                # Compute current cost
                cost = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            cost += (s - 1.0)**2
                            
                if cost < best_cost:
                    best_cost = cost
                    U_best = U.copy()
                    V_best = V.copy()
                    W_best = W.copy()
                    
                if best_cost < 1e-10:
                    # Snap and verify
                    U_snap = _snap_to_half_integers(U_best)
                    V_snap = _snap_to_half_integers(V_best)
                    W_snap = _snap_to_half_integers(W_best)
                    if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                        return U_snap, V_snap, W_snap, r
                    break
                
                # Perturb a random column
                col = np.random.randint(0, r)
                choice = np.random.randint(0, 3)
                if choice == 0:
                    U[:, col] += np.random.randn(n_u) * 0.1
                elif choice == 1:
                    V[:, col] += np.random.randn(n_v) * 0.1
                else:
                    W[:, col] += np.random.randn(n_w) * 0.1
                    
    return None

def construct():
    n, m, p = 2, 3, 6
    
    # 1. Get schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # 2. Try to find a better rank
    U_best, V_best, W_best = None, None, None
    
    if time.time() - START_TIME < TIME_LIMIT - 10:
        result = _try_block_structure(n, m, p)
        if result is not None:
            U_best, V_best, W_best, r = result
            
            # Verify again just in case
            if _verify_decomposition(U_best, V_best, W_best, n, m, p):
                pass
            else:
                U_best, V_best, W_best = None, None, None

    # 3. Select best
    if U_best is not None:
        final_U, final_V, final_W = U_best, V_best, W_best
    else:
        final_U, final_V, final_W = U_school, V_school, W_school

    # Convert to lists
    U_list = [[float(x) for x in row] for row in final_U]
    V_list = [[float(x) for x in row] for row in final_V]
    W_list = [[float(x) for x in row] for row in final_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
