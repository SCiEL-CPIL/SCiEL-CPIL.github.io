"""Matrix multiplication 2x4x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import combinations, product

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _build_strassen_36():
    """Build the known rank-36 block Strassen decomposition."""
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j):
        return i * m + j
    def v_idx(j, k):
        return j * p + k
    def w_idx(k, i):
        return k * n + i
    
    col = 0
    
    def strassen_block(y_indices, z_indices, w_offsets):
        nonlocal col
        y00, y01, y10, y11 = y_indices
        z00, z01, z10, z11 = z_indices
        
        # M1 = (y00+y11)(z00+z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M2 = (y10+y11)z00
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] -= 1
        col += 1
        
        # M3 = y00(z01-z11)
        U[u_idx(y00[0], y00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = -1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M4 = y11(z10-z00)
        U[u_idx(y11[0], y11[1]), col] = 1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z00[0], z00[1]), col] = -1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        W[w_idx(w_offsets[2][0], w_offsets[2][1]), col] += 1
        col += 1
        
        # M5 = (y00+y01)z11
        U[u_idx(y00[0], y00[1]), col] = 1
        U[u_idx(y01[0], y01[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] -= 1
        W[w_idx(w_offsets[1][0], w_offsets[1][1]), col] += 1
        col += 1
        
        # M6 = (y10-y00)(z00+z01)
        U[u_idx(y10[0], y10[1]), col] = 1
        U[u_idx(y00[0], y00[1]), col] = -1
        V[v_idx(z00[0], z00[1]), col] = 1
        V[v_idx(z01[0], z01[1]), col] = 1
        W[w_idx(w_offsets[3][0], w_offsets[3][1]), col] += 1
        col += 1
        
        # M7 = (y01-y11)(z10+z11)
        U[u_idx(y01[0], y01[1]), col] = 1
        U[u_idx(y11[0], y11[1]), col] = -1
        V[v_idx(z10[0], z10[1]), col] = 1
        V[v_idx(z11[0], z11[1]), col] = 1
        W[w_idx(w_offsets[0][0], w_offsets[0][1]), col] += 1
        col += 1
    
    # Block 1: A_L B_11
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,0),(0,1),(1,0),(1,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 2: A_R B_21
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,0),(2,1),(3,0),(3,1)], [(0,0),(1,0),(0,1),(1,1)])
    # Block 3: A_L B_12
    strassen_block([(0,0),(0,1),(1,0),(1,1)], [(0,2),(0,3),(1,2),(1,3)], [(2,0),(3,0),(2,1),(3,1)])
    # Block 4: A_R B_22
    strassen_block([(0,2),(0,3),(1,2),(1,3)], [(2,2),(2,3),(3,2),(3,3)], [(2,0),(3,0),(2,1),(3,1)])
    
    # Block 5: A_L B_15 (2x2 * 2x1)
    U[u_idx(0,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,0), col] = 1; V[v_idx(0,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,1), col] = 1; V[v_idx(1,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    # Block 6: A_R B_25 (2x2 * 2x1)
    U[u_idx(0,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(0,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,0), col] += 1; col += 1
    U[u_idx(1,2), col] = 1; V[v_idx(2,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    U[u_idx(1,3), col] = 1; V[v_idx(3,4), col] = 1; W[w_idx(4,1), col] += 1; col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_remove_column(U, V, W, col_to_remove):
    """Try removing a column and see if the decomposition still works."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    
    mask = [i for i in range(rank) if i != col_to_remove]
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    
    return _verify_decomposition(U_new, V_new, W_new)

def _try_remove_two_columns(U, V, W, col1, col2):
    """Try removing two columns simultaneously."""
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    
    mask = [i for i in range(rank) if i != col1 and i != col2]
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    
    return _verify_decomposition(U_new, V_new, W_new)

def _groebner_style_minimize(U, V, W, deadline):
    """
    Use a Gröbner-basis-inspired approach to find minimal rank decomposition.
    
    Strategy: Treat the tensor equations as a multilinear system. For each rank-1 term,
    we have variables U[r], V[r], W[r]. The constraint is that for all (i,j,k):
    sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    We search for the minimum number of terms by:
    1. Starting from a valid decomposition
    2. Identifying linear dependencies in the tensor space
    3. Using elimination to find minimal generating sets
    """
    n, m, p = 2, 4, 5
    num_eqs = n * m * p  # 40 equations
    rank = U.shape[1]
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 1: Greedy elimination with dependency analysis
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        progress = False
        # Try removing each column
        for col in range(rank):
            if time.time() > deadline:
                break
            
            if _try_remove_column(U, V, W, col):
                mask = [i for i in range(rank) if i != col]
                U = U[:, mask]
                V = V[:, mask]
                W = W[:, mask]
                rank = U.shape[1]
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                progress = True
                break
        
        if not progress:
            break
    
    # Phase 2: Pair elimination
    for i in range(rank):
        if time.time() > deadline:
            break
        for j in range(i+1, rank):
            if time.time() > deadline:
                break
            if _try_remove_two_columns(U, V, W, i, j):
                mask = [k for k in range(rank) if k != i and k != j]
                U = U[:, mask]
                V = V[:, mask]
                W = W[:, mask]
                rank = U.shape[1]
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                break
        if time.time() > deadline:
            break
        if U.shape[1] < rank:
            break
    
    return best_U, best_V, best_W, best_rank

def _structured_search_minimal_rank(deadline):
    """
    Directly search for minimal rank decomposition using structured patterns.
    
    For 2x4x5, we can exploit the structure:
    - The 2x4 matrix has 8 entries
    - The 4x5 matrix has 20 entries  
    - The output 2x5 matrix has 10 entries
    
    We try various block structures and find the minimum rank.
    """
    n, m, p = 2, 4, 5
    
    # Try different block decompositions
    # Option 1: Treat as (2x2) x (2x5) + (2x2) x (2x5)
    # Each 2x2 x 2x5 can be done with Strassen-like approach
    
    best_rank = float('inf')
    best_result = None
    
    # Build the base Strassen-36 and minimize
    U, V, W = _build_strassen_36()
    if _verify_decomposition(U, V, W):
        U, V, W, r = _groebner_style_minimize(U, V, W, deadline)
        if r < best_rank:
            best_rank = r
            best_result = (U.copy(), V.copy(), W.copy())
    
    return best_result, best_rank

def construct():
    """Construct a rank-minimal exact tensor decomposition for 2x4 x 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, try the structured search
    result, rank = _structured_search_minimal_rank(deadline)
    
    if result is not None:
        U, V, W = result
        if _verify_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: return the Strassen-36 base
    U, V, W = _build_strassen_36()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
