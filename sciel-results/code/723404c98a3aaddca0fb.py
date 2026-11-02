"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]
_START_TIME = time.time()
DEADLINE = 145.0  # 5s safety margin before 150s kill

# EVOLVE-BLOCK-START
def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_list, v_list, w_list) where:
    u_list: [(i, j, val), ...] for A[i,j]
    v_list: [(j, k, val), ...] for B[j,k]
    w_list: [(i, k, val), ...] for C[i,k]
    """
    return [
        # Term 1: (a+d)(e+h) -> contributes to C00 and C11
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # Term 2: (c+d)e -> contributes to C10 and C11
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # Term 3: a(f-h) -> contributes to C01 and C11
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
        # Term 4: d(g-e) -> contributes to C00 and C10
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
        # Term 5: (a+b)h -> contributes to C00 and C01
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # Term 6: (c-a)(e+f) -> contributes to C11
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # Term 7: (b-d)(g+h) -> contributes to C00
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
    ]

def _build_block_decomposition(n, m, p):
    """Build the block decomposition:
    - Part 1: j=0, schoolbook 2x1x6 = 12 terms
    - Part 2: j=1..4, split into two 2x2x6 blocks, each using 3 Strassen pairs = 21 terms each
    Total rank = 12 + 2*21 = 54
    """
    rank = 54
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    # Part 1: j = 0, schoolbook
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
    
    # Part 2: j in 1..4, two blocks of 2x2x6
    strassen_terms = _strassen_2x2_terms()
    
    for block_idx in range(2):
        j_base = 1 + block_idx * 2  # j=1,2 for block 0; j=3,4 for block 1
        
        for s in range(3):
            k_base = s * 2  # k pairs: (0,1), (2,3), (4,5)
            
            for term in strassen_terms:
                u_list, v_list, w_list = term
                
                for (li, lj, val) in u_list:
                    global_i = li
                    global_j = j_base + lj
                    U[global_i * m + global_j, col] += val
                
                for (lj, lk, val) in v_list:
                    global_j = j_base + lj
                    global_k = k_base + lk
                    V[global_j * p + global_k, col] += val
                
                for (li, lk, val) in w_list:
                    global_i = li
                    global_k = k_base + lk
                    W[global_k * n + global_i, col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                total += s
            if total != 1:
                return False
    return True

def _find_linear_dependency(stacked, col_to_remove, other_cols):
    """Try to express stacked[:, col_to_remove] as an integer linear combination
    of stacked[:, other_cols]. Returns coefficients or None."""
    # Build the system: stacked[:, other_cols] @ coeffs = stacked[:, col_to_remove]
    A = stacked[:, other_cols]  # (dim, num_other)
    b = stacked[:, col_to_remove]  # (dim,)
    
    # Solve using least squares first to check feasibility
    try:
        # Use exact rational arithmetic via fractions for small systems
        # First try with float to see if there's a solution
        sol, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
        
        # Check if the solution is close to integer
        if len(residuals) > 0 and residuals[0] > 1e-6:
            return None
        
        # Round to nearest integer
        int_sol = np.round(sol).astype(np.int64)
        
        # Verify exactness
        if np.allclose(A @ int_sol, b, atol=1e-10):
            # Check if all entries are small integers
            if np.all(np.abs(int_sol) <= 10):
                return int_sol
    except Exception:
        pass
    
    return None

def _try_rank_reduction(n, m, p):
    """Try to reduce rank by finding and eliminating linearly dependent columns."""
    if time.time() > DEADLINE:
        return None
    
    result = _build_block_decomposition(n, m, p)
    U, V, W = result["U"].copy(), result["V"].copy(), result["W"].copy()
    rank = U.shape[1]
    
    # Stack all matrices for dependency checking
    dim = n*m + m*p + p*n
    
    # Iteratively try to remove columns
    kept = list(range(rank))
    
    for attempt in range(rank):
        if time.time() > DEADLINE - 5:
            break
        
        removed_this_round = False
        for idx in range(len(kept)):
            col = kept[idx]
            others = [c for c in kept if c != col]
            
            if not others:
                break
            
            # Build stacked matrix for current kept columns
            stacked = np.vstack([U[:, kept], V[:, kept], W[:, kept]])
            
            # Try to remove this column
            coeffs = _find_linear_dependency(stacked, idx, list(range(len(kept)))[:idx] + list(range(len(kept)))[idx+1:])
            
            if coeffs is not None:
                # Remove this column
                kept.remove(col)
                removed_this_round = True
                break
        
        if not removed_this_round:
            break
    
    if len(kept) >= rank:
        return None
    
    # Build reduced matrices
    new_rank = len(kept)
    U_new = U[:, kept]
    V_new = V[:, kept]
    W_new = W[:, kept]
    
    return {"U": U_new, "V": V_new, "W": W_new}

def construct():
    n, m, p = DIMENSIONS
    
    # First, build the base rank-54 decomposition
    result = _build_block_decomposition(n, m, p)
    
    # Verify it's correct
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=np.int64)
        V = np.zeros((m * p, rank_school), dtype=np.int64)
        W = np.zeros((p * n, rank_school), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}
    
    # Try to find a lower-rank decomposition if time permits
    if time.time() < DEADLINE - 10:
        reduced = _try_rank_reduction(n, m, p)
        if reduced is not None:
            # Verify the reduced version
            if _verify_decomposition(reduced["U"], reduced["V"], reduced["W"], n, m, p):
                if reduced["U"].shape[1] < result["U"].shape[1]:
                    result = reduced
    
    # Convert to lists for the return interface
    return {
        "U": result["U"].tolist(),
        "V": result["V"].tolist(),
        "W": result["W"].tolist()
    }
# EVOLVE-BLOCK-END
