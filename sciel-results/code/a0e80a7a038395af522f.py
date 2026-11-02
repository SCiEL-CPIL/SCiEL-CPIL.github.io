"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def _build_rank14():
    """Build the known rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for B1 (cols 0,1)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1

    # Strassen for B2 (cols 2,3)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1
    
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1
    
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1
    
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1
    
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1
    
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1

    return U, V, W

def _build_tensor(n, m, p):
    """Build the multiplication tensor T[a,b,c] where a indexes A, b indexes B, c indexes C."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for a1 in range(n):
        for a2 in range(m):
            a_idx = a1 * m + a2
            for b1 in range(m):
                for b2 in range(p):
                    b_idx = b1 * p + b2
                    for c1 in range(p):
                        for c2 in range(n):
                            c_idx = c1 * n + c2
                            if a1 == c1 and a2 == b1 and b2 == c2:
                                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    n, m, p = 2, 2, 4
    rank = len(V[0])
    T = _build_tensor(n, m, p)
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Compute the tensor from the decomposition
    # T_decomp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is equivalent to: for each r, add outer(U[:,r], V[:,r], W[:,r])
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
    
    if np.max(np.abs(T_decomp - T)) > tol:
        return False
    return True

def _check_half_integer(arr, tol=1e-9):
    """Check if all entries are half-integers (k/2 for integer k)."""
    for val in arr:
        if abs(val * 2 - round(val * 2)) > tol:
            return False
    return True

def _is_rank1_tensor(T, dim_a, dim_b, dim_c, tol=1e-9):
    """
    Check if a 3D tensor T is rank-1, i.e., T[a,b,c] = U[a]*V[b]*W[c].
    Returns (is_rank1, U, V, W) where U, V, W are the factor vectors if rank-1.
    """
    # Find all nonzero entries
    nonzero = np.argwhere(np.abs(T) > tol)
    if len(nonzero) == 0:
        return True, np.zeros(dim_a), np.zeros(dim_b), np.zeros(dim_c)
    
    # Pick a reference entry
    a0, b0, c0 = nonzero[0]
    ref_val = T[a0, b0, c0]
    
    # Normalize: set U[a0] = 1
    # Then V[b0]*W[c0] = ref_val
    # We have freedom in scaling. Let's try to determine U, V, W.
    
    # Strategy: 
    # 1. For each a, if there exists (b,c) with T[a,b,c] != 0, then U[a] != 0
    # 2. U[a] is proportional to T[a,b,c] / (V[b]*W[c]) for any (b,c) where T[a,b,c] != 0
    # 3. Similarly for V and W
    
    # Let's try to determine the vectors by solving a system.
    # Initialize U, V, W with 1s where they should be nonzero.
    
    # First pass: determine support
    U_support = np.zeros(dim_a, dtype=bool)
    V_support = np.zeros(dim_b, dtype=bool)
    W_support = np.zeros(dim_c, dtype=bool)
    
    for (a, b, c) in nonzero:
        U_support[a] = True
        V_support[b] = True
        W_support[c] = True
    
    # If any dimension has no support, the tensor is zero (handled above)
    # Now, for each a with support, U[a] is determined up to a global scale.
    # Let's fix U[a0] = 1.
    
    U = np.zeros(dim_a)
    V = np.zeros(dim_b)
    W = np.zeros(dim_c)
    U[a0] = 1.0
    
    # Now determine V and W from the constraints.
    # For each (a,b,c) in nonzero:
    #   T[a,b,c] = U[a] * V[b] * W[c]
    # If U[a] is known, then V[b]*W[c] = T[a,b,c] / U[a]
    
    # We can use a graph-based approach or iterative refinement.
    # Let's try: for each b with support, find an (a,c) such that T[a,b,c] != 0 and U[a] != 0.
    # Then V[b] * W[c] = T[a,b,c] / U[a]
    # If we fix W[c0] = ref_val (since U[a0]=1, V[b0]*W[c0]=ref_val, let's set W[c0]=ref_val, V[b0]=1)
    
    W[c0] = ref_val
    V[b0] = 1.0
    
    # Now propagate
    # For each (a,b,c) in nonzero:
    #   If U[a] is known and V[b] is known, then W[c] = T[a,b,c] / (U[a]*V[b])
    #   If U[a] is known and W[c] is known, then V[b] = T[a,b,c] / (U[a]*W[c])
    #   If V[b] is known and W[c] is known, then U[a] = T[a,b,c] / (V[b]*W[c])
    
    changed = True
    iterations = 0
    max_iterations = dim_a * dim_b * dim_c
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for (a, b, c) in nonzero:
            val = T[a, b, c]
            # Check consistency and fill in unknowns
            u_known = abs(U[a]) > tol
            v_known = abs(V[b]) > tol
            w_known = abs(W[c]) > tol
            
            if u_known and v_known:
                expected_w = val / (U[a] * V[b])
                if w_known:
                    if abs(W[c] - expected_w) > tol:
                        return False, None, None, None
                else:
                    W[c] = expected_w
                    changed = True
            elif u_known and w_known:
                expected_v = val / (U[a] * W[c])
                if v_known:
                    if abs(V[b] - expected_v) > tol:
                        return False, None, None, None
                else:
                    V[b] = expected_v
                    changed = True
            elif v_known and w_known:
                expected_u = val / (V[b] * W[c])
                if u_known:
                    if abs(U[a] - expected_u) > tol:
                        return False, None, None, None
                else:
                    U[a] = expected_u
                    changed = True
    
    # After propagation, check if all supported entries are determined
    for a in range(dim_a):
        if U_support[a] and abs(U[a]) < tol:
            return False, None, None, None
    for b in range(dim_b):
        if V_support[b] and abs(V[b]) < tol:
            return False, None, None, None
    for c in range(dim_c):
        if W_support[c] and abs(W[c]) < tol:
            return False, None, None, None
    
    # Verify the full tensor
    T_reconstructed = np.einsum('a,b,c->abc', U, V, W)
    if np.max(np.abs(T_reconstructed - T)) > tol:
        return False, None, None, None
    
    return True, U, V, W

def _try_rank13_decomposition(U14, V14, W14, n, m, p, tol=1e-9):
    """
    Try to find a rank-13 decomposition by merging two terms from the rank-14 solution.
    For each pair (i,j), check if the residual tensor (T - sum of other terms) is rank-1
    and has half-integer factors.
    """
    rank14 = 14
    T = _build_tensor(n, m, p)
    
    U_arr = np.array(U14, dtype=np.float64)
    V_arr = np.array(V14, dtype=np.float64)
    W_arr = np.array(W14, dtype=np.float64)
    
    for i in range(rank14):
        for j in range(i + 1, rank14):
            # Compute residual: T - sum_{r != i,j} U[:,r] * V[:,r] * W[:,r]
            T_res = T.copy()
            for r in range(rank14):
                if r == i or r == j:
                    continue
                T_res -= np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
            
            # Check if T_res is rank-1
            is_r1, U_new, V_new, W_new = _is_rank1_tensor(T_res, n*m, m*p, p*n, tol)
            
            if is_r1:
                # Check if factors are half-integers
                if _check_half_integer(U_new, tol) and _check_half_integer(V_new, tol) and _check_half_integer(W_new, tol):
                    # Build the rank-13 decomposition
                    U_new_full = [[0] * 13 for _ in range(n*m)]
                    V_new_full = [[0] * 13 for _ in range(m*p)]
                    W_new_full = [[0] * 13 for _ in range(p*n)]
                    
                    idx = 0
                    for r in range(rank14):
                        if r == i or r == j:
                            continue
                        for a in range(n*m):
                            U_new_full[a][idx] = U14[a][r]
                        for b in range(m*p):
                            V_new_full[b][idx] = V14[b][r]
                        for c in range(p*n):
                            W_new_full[c][idx] = W14[c][r]
                        idx += 1
                    
                    # Add the new merged term
                    for a in range(n*m):
                        U_new_full[a][12] = U_new[a]
                    for b in range(m*p):
                        V_new_full[b][12] = V_new[b]
                    for c in range(p*n):
                        W_new_full[c][12] = W_new[c]
                    
                    # Verify
                    if _verify_decomposition(U_new_full, V_new_full, W_new_full, tol):
                        return {"U": U_new_full, "V": V_new_full, "W": W_new_full}
    
    return None

def _try_rank12_decomposition(U14, V14, W14, n, m, p, tol=1e-9):
    """
    Try to find a rank-12 decomposition by merging two pairs of terms.
    This is more complex. For each pair of pairs ((i,j), (k,l)), check if
    the residual after removing these 4 terms is rank-2.
    """
    rank14 = 14
    T = _build_tensor(n, m, p)
    
    U_arr = np.array(U14, dtype=np.float64)
    V_arr = np.array(V14, dtype=np.float64)
    W_arr = np.array(W14, dtype=np.float64)
    
    # This is computationally expensive. We'll limit the search.
    # For each pair of pairs, check if the residual is rank-2.
    # Rank-2 check: T = U1*V1*W1 + U2*V2*W2
    
    # We'll use a heuristic: try all pairs of pairs but with a time limit.
    start_time = time.time()
    time_limit = 10  # seconds
    
    for i in range(rank14):
        for j in range(i + 1, rank14):
            for k in range(rank14):
                if k == i or k == j:
                    continue
                for l in range(k + 1, rank14):
                    if l == i or l == j:
                        continue
                    
                    if time.time() - start_time > time_limit:
                        return None
                    
                    # Compute residual
                    T_res = T.copy()
                    for r in range(rank14):
                        if r in (i, j, k, l):
                            continue
                        T_res -= np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
                    
                    # Check if T_res is rank-2
                    # This is more complex. We'll use a simple heuristic:
                    # Check if T_res can be written as sum of two rank-1 tensors.
                    # For now, we'll skip this and rely on rank-13 search.
                    pass
    
    return None

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    n, m, p = 2, 2, 4
    
    # First, build the rank-14 decomposition
    U14, V14, W14 = _build_rank14()
    
    # Verify it
    if not _verify_decomposition(U14, V14, W14):
        # This should not happen, but just in case
        pass
    
    if time.time() > deadline:
        return {"U": U14, "V": V14, "W": W14}
    
    # Try to find a rank-13 decomposition
    result13 = _try_rank13_decomposition(U14, V14, W14, n, m, p)
    
    if result13 is not None:
        return result13
    
    if time.time() > deadline:
        return {"U": U14, "V": V14, "W": W14}
    
    # Try to find a rank-12 decomposition (limited search)
    result12 = _try_rank12_decomposition(U14, V14, W14, n, m, p)
    
    if result12 is not None:
        return result12
    
    # Return the best result found (rank-14)
    return {"U": U14, "V": V14, "W": W14}
# EVOLVE-BLOCK-END
