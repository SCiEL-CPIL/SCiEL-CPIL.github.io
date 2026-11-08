"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _construct_rank55(n, m, p):
    """Construct the known rank-55 decomposition."""
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        U[0 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[1 * 3 + 1][col] += 1
        V[0 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        W[k1 * 2 + 1][col] -= 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        V[0 * 10 + k1][col] += 1
        V[1 * 10 + k1][col] -= 1
        W[k1 * 2 + 0][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 1][col] += 1
        V[1 * 10 + k0][col] += 1
        V[0 * 10 + k0][col] -= 1
        W[k0 * 2 + 0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 0][col] += 1
        U[0 * 3 + 1][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] -= 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 0][col] += 1
        U[0 * 3 + 0][col] -= 1
        V[0 * 10 + k0][col] += 1
        V[0 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
        
        U[0 * 3 + 1][col] += 1
        U[1 * 3 + 1][col] -= 1
        V[1 * 10 + k0][col] += 1
        V[1 * 10 + k1][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
    
    return U, V, W

def _try_merge_columns(U, V, W, n, m, p, col_a, col_b, deadline):
    """Try to merge two columns by finding a linear combination."""
    # Columns a and b: try U_new = U_a + c*U_b, V_new = V_a + c*V_b, W_new = W_a + c*W_b
    # This works if the tensor contribution is linear in c
    # T_new = (U_a+c*U_b)(V_a+c*V_b)(W_a+c*W_b) = T_a + c*T_cross + c^2*T_b
    # We need T_cross + c^2*T_b = 0 for all (i,j,k) where T_a already contributes
    # This is complex; instead try: replace cols a,b with single col that's sum
    # Check if removing col_b and adjusting col_a works
    
    # Simple approach: try U_new = U_a + U_b, V_new = V_a + V_b, W_new = W_a + W_b
    # This gives extra cross terms, so only works in special cases
    
    # Better: try to find c such that the merged column works
    # For now, just try the sum and verify
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    rank = U_new[0].__len__()
    
    # Merge col_b into col_a: U_new[*,col_a] += U_new[*,col_b], then zero out col_b
    for i in range(n * m):
        U_new[i][col_a] += U_new[i][col_b]
        U_new[i][col_b] = 0
    for i in range(m * p):
        V_new[i][col_a] += V_new[i][col_b]
        V_new[i][col_b] = 0
    for i in range(p * n):
        W_new[i][col_a] += W_new[i][col_b]
        W_new[i][col_b] = 0
    
    # Remove col_b
    U_final = [row[:col_b] + row[col_b+1:] for row in U_new]
    V_final = [row[:col_b] + row[col_b+1:] for row in V_new]
    W_final = [row[:col_b] + row[col_b+1:] for row in W_new]
    
    if _verify(U_final, V_final, W_final, n, m, p):
        return U_final, V_final, W_final
    return None, None, None

def _try_rank_reduction(U, V, W, n, m, p, target_rank, deadline):
    """Try to reduce rank from current to target_rank by merging columns."""
    current_rank = len(U[0])
    
    if current_rank == target_rank:
        return U, V, W
    
    # Try all pairs of columns to merge
    best_result = None
    best_rank = current_rank
    
    U_cur = [row[:] for row in U]
    V_cur = [row[:] for row in V]
    W_cur = [row[:] for row in W]
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        if len(U_cur[0]) == target_rank:
            break
        
        # Pick two random columns to try merging
        cols = list(range(len(U_cur[0])))
        np.random.shuffle(cols)
        for _ in range(10):
            if time.time() > deadline:
                break
            if len(U_cur[0]) <= target_rank:
                break
            a, b = cols[0], cols[1]
            result_U, result_V, result_W = _try_merge_columns(
                U_cur, V_cur, W_cur, n, m, p, a, b, deadline)
            if result_U is not None:
                U_cur = result_U
                V_cur = result_V
                W_cur = result_W
                if len(U_cur[0]) == target_rank:
                    return U_cur, V_cur, W_cur
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin
    
    # Start with the known rank-55 solution
    U, V, W = _construct_rank55(n, m, p)
    best_rank = 55
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to reduce to rank 54 by merging columns
    np.random.seed(42)
    U54, V54, W54 = _try_rank_reduction(U, V, W, n, m, p, 54, deadline)
    if U54 is not None and _verify(U54, V54, W54, n, m, p):
        best_rank = 54
        best_result = {"U": U54, "V": V54, "W": W54}
    
    # Try to reduce further if time permits
    if best_rank > 50 and time.time() < deadline - 20:
        np.random.seed(123)
        U53, V53, W53 = _try_rank_reduction(U54 if U54 else U, V54 if V54 else V, W54 if W54 else W, n, m, p, 53, deadline)
        if U53 is not None and _verify(U53, V53, W53, n, m, p):
            best_rank = 53
            best_result = {"U": U53, "V": V53, "W": W53}
    
    return best_result
# EVOLVE-BLOCK-END
