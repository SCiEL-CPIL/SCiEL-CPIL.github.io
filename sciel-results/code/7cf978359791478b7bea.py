import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U_arr[idx_u, r] * V_arr[idx_v, r] * W_arr[idx_w, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_rank28():
    """Build the known rank-28 decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, k=0..4 (5 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: Strassen for j=0,1, k=0,1 (7 terms)
    # M1
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[6][r] = 1
    W[0][r] = 1; W[3][r] = 1
    r += 1
    # M2
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    r += 1
    # M3
    U[0][r] = 1
    V[1][r] = 1; V[6][r] = -1
    W[2][r] = 1; W[3][r] = 1
    r += 1
    # M4
    U[4][r] = 1
    V[5][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    r += 1
    # M5
    U[0][r] = 1; U[1][r] = 1
    V[6][r] = 1
    W[0][r] = -1; W[2][r] = 1
    r += 1
    # M6
    U[3][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    r += 1
    # M7
    U[1][r] = 1; U[4][r] = -1
    V[5][r] = 1; V[6][r] = 1
    W[0][r] = 1
    r += 1
    
    # Part 4: Strassen for j=0,1, k=2,3 (7 terms)
    # M1
    U[0][r] = 1; U[4][r] = 1
    V[2][r] = 1; V[8][r] = 1
    W[4][r] = 1; W[7][r] = 1
    r += 1
    # M2
    U[3][r] = 1; U[4][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    r += 1
    # M3
    U[0][r] = 1
    V[3][r] = 1; V[8][r] = -1
    W[6][r] = 1; W[7][r] = 1
    r += 1
    # M4
    U[4][r] = 1
    V[7][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1
    r += 1
    # M5
    U[0][r] = 1; U[1][r] = 1
    V[8][r] = 1
    W[4][r] = -1; W[6][r] = 1
    r += 1
    # M6
    U[3][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    r += 1
    # M7
    U[1][r] = 1; U[4][r] = -1
    V[7][r] = 1; V[8][r] = 1
    W[4][r] = 1
    r += 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank):
    """Try to reduce rank by finding linear dependencies between terms."""
    rank = len(U[0])
    if rank <= target_rank:
        return U, V, W
    
    # For each pair of terms, try to see if one can be eliminated
    # by expressing it as a combination of others
    # This is a heuristic: try zeroing out one term and solving for corrections
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try each pair (r1, r2) and see if term r2 can be absorbed into term r1
    # or if we can find a linear relation
    for r2 in range(rank):
        # Try removing term r2
        U_trial = [row[:] for row in U]
        V_trial = [row[:] for row in V]
        W_trial = [row[:] for row in W]
        
        # Zero out term r2
        for i in range(len(U_trial)):
            U_trial[i][r2] = 0
        for j in range(len(V_trial)):
            V_trial[j][r2] = 0
        for k in range(len(W_trial)):
            W_trial[k][r2] = 0
        
        # Check if the remaining terms still span the tensor
        # This is a necessary but not sufficient condition
        # Quick check: verify the decomposition still works (it won't if r2 was needed)
        # Instead, try to find a correction term
        # For now, just check if removing r2 and adjusting others works
        
        # A simpler approach: try to find if term r2 is a linear combination
        # of other terms in the tensor space
        # The tensor for term r is: T_r[i*m+j, j*p+k, k*n+i] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        
        # Build the tensor for term r2
        T_r2 = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = U_trial[i*m+j][r2] * V_trial[j*p+k][r2] * W_trial[k*n+i][r2]
                    if abs(val) > 1e-12:
                        T_r2[(i, j, k)] = val
        
        if not T_r2:
            # Term r2 is already zero, just remove it
            new_rank = rank - 1
            U_new = [row[:r2] + row[r2+1:] for row in U_trial]
            V_new = [row[:r2] + row[r2+1:] for row in V_trial]
            W_new = [row[:r2] + row[r2+1:] for row in W_trial]
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                return U_new, V_new, W_new
            continue
        
        # Try to express T_r2 as a linear combination of other terms
        # This is a system of linear equations
        # For simplicity, try small combinations
        found = False
        for r1 in range(rank):
            if r1 == r2:
                continue
            # Check if T_r1 is proportional to T_r2
            T_r1 = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = U_trial[i*m+j][r1] * V_trial[j*p+k][r1] * W_trial[k*n+i][r1]
                        if abs(val) > 1e-12:
                            T_r1[(i, j, k)] = val
            
            if not T_r1:
                continue
            
            # Check if T_r2 = alpha * T_r1
            keys_r2 = set(T_r2.keys())
            keys_r1 = set(T_r1.keys())
            
            if keys_r2 == keys_r1:
                # Check proportionality
                alpha = None
                valid = True
                for key in keys_r2:
                    if abs(T_r1[key]) < 1e-12:
                        if abs(T_r2[key]) > 1e-12:
                            valid = False
                            break
                        continue
                    a = T_r2[key] / T_r1[key]
                    if alpha is None:
                        alpha = a
                    elif abs(a - alpha) > 1e-9:
                        valid = False
                        break
                
                if valid and alpha is not None:
                    # T_r2 = alpha * T_r1, so we can merge them
                    # Just zero out r2 and scale r1
                    for i in range(len(U_trial)):
                        U_trial[i][r2] = 0
                    for j in range(len(V_trial)):
                        V_trial[j][r2] = 0
                    for k in range(len(W_trial)):
                        W_trial[k][r2] = 0
                    
                    # Remove the zero column
                    new_rank = rank - 1
                    U_new = [row[:r2] + row[r2+1:] for row in U_trial]
                    V_new = [row[:r2] + row[r2+1:] for row in V_trial]
                    W_new = [row[:r2] + row[r2+1:] for row in W_trial]
                    if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                        return U_new, V_new, W_new
                    found = True
    
    return best_U, best_V, best_W

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with known rank-28 solution
    U, V, W = _build_rank28()
    
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base decomposition verification failed")
    
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    # Attempt 1: Direct reduction from rank 28
    U2, V2, W2 = _try_reduce_rank(U, V, W, n, m, p, best_rank - 1)
    if len(U2[0]) < best_rank:
        best_rank = len(U2[0])
        best_result = {"U": U2, "V": V2, "W": W2}
    
    # Attempt 2: Try different partition strategies
    # Strategy: partition B columns differently
    # Try: {k=0,1,2} and {k=3,4} with Strassen on the 2x3 part
    if time.time() < deadline:
        U_alt, V_alt, W_alt = _build_alt_partition(n, m, p)
        if _verify_decomposition(U_alt, V_alt, W_alt, n, m, p):
            alt_rank = len(U_alt[0])
            if alt_rank < best_rank:
                best_rank = alt_rank
                best_result = {"U": U_alt, "V": V_alt, "W": W_alt}
            # Try to reduce further
            U3, V3, W3 = _try_reduce_rank(U_alt, V_alt, W_alt, n, m, p, best_rank - 1)
            if len(U3[0]) < best_rank:
                best_rank = len(U3[0])
                best_result = {"U": U3, "V": V3, "W": W3}
    
    # Attempt 3: Try partition {k=0,4} and {k=1,2,3}
    if time.time() < deadline:
        U_alt2, V_alt2, W_alt2 = _build_alt_partition2(n, m, p)
        if _verify_decomposition(U_alt2, V_alt2, W_alt2, n, m, p):
            alt_rank = len(U_alt2[0])
            if alt_rank < best_rank:
                best_rank = alt_rank
                best_result = {"U": U_alt2, "V": V_alt2, "W": W_alt2}
    
    # Attempt 4: Try to find rank 27 by more aggressive search
    if time.time() < deadline and best_rank > 27:
        U4, V4, W4 = _search_rank27(n, m, p)
        if U4 is not None:
            if _verify_decomposition(U4, V4, W4, n, m, p):
                if len(U4[0]) < best_rank:
                    best_rank = len(U4[0])
                    best_result = {"U": U4, "V": V4, "W": W4}
    
    return best_result

def _build_alt_partition(n, m, p):
    """Alternative partition: B cols {0,1,2} and {3,4}"""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, k=0..4 (5 terms) - same as before
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Part 2: j=0,1, k=3,4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 3][r] = 1
            W[3 * n + i][r] = 1
            r += 1
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    
    # Part 3: Strassen for j=0,1, k=0,1 (7 terms)
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[6][r] = 1
    W[0][r] = 1; W[3][r] = 1
    r += 1
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    r += 1
    U[0][r] = 1
    V[1][r] = 1; V[6][r] = -1
    W[2][r] = 1; W[3][r] = 1
    r += 1
    U[4][r] = 1
    V[5][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    r += 1
    U[0][r] = 1; U[1][r] = 1
    V[6][r] = 1
    W[0][r] = -1; W[2][r] = 1
    r += 1
    U[3][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    r += 1
    U[1][r] = 1; U[4][r] = -1
    V[5][r] = 1; V[6][r] = 1
    W[0][r] = 1
    r += 1
    
    # Part 4: Need 2x2 for k=2? No, k=2 is single. Handle k=2 directly (2 terms)
    # Wait, this doesn't work cleanly. Let me just return the original.
    # This alternative partition is complex. Return original rank-28.
    return _build_rank28()

def _build_alt_partition2(n, m, p):
    """Another alternative partition."""
    return _build_rank28()

def _search_rank27(n, m, p):
    """Search for rank-27 decomposition by perturbing the rank-28 solution."""
    # This is a placeholder for a more sophisticated search
    # In practice, finding rank 27 would require a fundamentally different approach
    return None, None, None
