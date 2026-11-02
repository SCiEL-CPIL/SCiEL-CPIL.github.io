import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 8]
START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check exact decomposition: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1"""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _validate_sub(U_sub, V_sub, W_sub, n, m, p_sub, tol=1e-6):
    """Validate a sub-decomposition for n x m by m x p_sub."""
    return _validate(U_sub, V_sub, W_sub, n, m, p_sub, tol)

def _try_rank_sub(r, n, m, p_sub, iters=1000, seed=0):
    """
    Try to find a rank-r decomposition for n x m by m x p_sub multiplication.
    Returns (U, V, W) as lists or None.
    """
    rng = np.random.default_rng(seed)
    for it in range(iters):
        if time.time() > DEADLINE:
            return None
        
        U = rng.normal(size=(n * m, r))
        V = rng.normal(size=(m * p_sub, r))
        W = rng.normal(size=(p_sub * n, r))
        
        # Try integer rounding
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        
        if not (np.all(U_r == 0) or np.all(V_r == 0) or np.all(W_r == 0)):
            if _validate(U_r, V_r, W_r, n, m, p_sub):
                def to_list(M):
                    return [[int(v) for v in row] for row in M]
                return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
        
        # Try half-integer rounding
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if not (np.all(U_r == 0) or np.all(V_r == 0) or np.all(W_r == 0)):
            if _validate(U_r, V_r, W_r, n, m, p_sub):
                def to_list(M):
                    return [[float(v) for v in row] for row in M]
                return {"U": to_list(U_r), "V": to_list(V_r), "W": to_list(W_r)}
    return None

def _try_structured_sub(n, m, p_sub):
    """
    Try structured decompositions for small cases.
    For 3x3 by 3x4, try rank 12 (the lower bound).
    Use a known decomposition pattern if available.
    """
    # For 3x3 by 3x4, the lower bound is max(3*4, 3*4, 3*3) = 12
    # Try to find a rank-12 decomposition using a structured approach
    
    # One approach: use the fact that we can decompose the 4 columns
    # into two pairs, and use rank-6 decompositions for each 3x3 by 3x2
    # But 3x3 by 3x2 has lower bound max(6, 6, 9) = 9, so rank 9 each, total 18
    
    # Alternative: try rank 12 directly with a structured pattern
    # Use the "bit decomposition" idea: represent 4 as 2 bits
    # But this is complex. For now, rely on random search.
    
    return None

def _combine_sub_decompositions(sub1, sub2, n, m, p1, p2):
    """
    Combine two sub-decompositions:
    sub1: for n x m by m x p1
    sub2: for n x m by m x p2
    Result: for n x m by m x (p1+p2)
    
    The combined decomposition has rank = rank1 + rank2.
    U_combined = [U1 | U2] (horizontal concat)
    V_combined = [V1 | V2] (horizontal concat)
    W_combined = [W1; W2] (vertical concat, but need to align indices)
    
    Actually, W for the combined problem has shape (p1+p2)*n x rank.
    W_combined[k*n+i, c] for k in [0, p1) comes from W1
    W_combined[k*n+i, c] for k in [p1, p1+p2) comes from W2
    """
    U1 = np.asarray(sub1["U"], dtype=float)
    V1 = np.asarray(sub1["V"], dtype=float)
    W1 = np.asarray(sub1["W"], dtype=float)
    
    U2 = np.asarray(sub2["U"], dtype=float)
    V2 = np.asarray(sub2["V"], dtype=float)
    W2 = np.asarray(sub2["W"], dtype=float)
    
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    r = r1 + r2
    
    U_comb = np.hstack([U1, U2])
    V_comb = np.hstack([V1, V2])
    
    W_comb = np.zeros(((p1 + p2) * n, r))
    W_comb[:p1 * n, :r1] = W1
    W_comb[p1 * n:, r1:] = W2
    
    def to_list(M):
        return [[float(v) for v in row] for row in M]
    
    return {"U": to_list(U_comb), "V": to_list(V_comb), "W": to_list(W_comb)}

def construct():
    n, m, p = DIMENSIONS
    best = _build_schoolbook()
    best_rank = n * m * p  # 72
    
    # Strategy: decompose p=8 into p1=4 and p2=4
    # Find rank-r1 decomposition for 3x3 by 3x4
    # Find rank-r2 decomposition for 3x3 by 3x4
    # Total rank = r1 + r2
    
    p1, p2 = 4, 4
    
    # Lower bound for 3x3 by 3x4 is max(3*4, 3*4, 3*3) = 12
    # Try to find rank-12 decompositions
    # If we find two rank-12 decompositions, total rank = 24
    
    # Try different rank values for the sub-problems
    # Start from the lower bound and go up
    sub_ranks_to_try = [12, 13, 14, 15, 16, 18, 20, 24]
    
    found_sub = []
    seed_counter = 0
    
    for r_sub in sub_ranks_to_try:
        if time.time() > DEADLINE - 10:
            break
        
        # Try to find two independent sub-decompositions of rank r_sub
        for attempt in range(2):
            if time.time() > DEADLINE - 10:
                break
            
            iters = 2000 if r_sub <= 14 else 1000
            res = _try_rank_sub(r_sub, n, m, p1, iters=iters, seed=seed_counter)
            seed_counter += 1000
            
            if res is not None:
                found_sub.append(res)
                if len(found_sub) == 2:
                    break
        
        if len(found_sub) == 2:
            # Combine the two sub-decompositions
            combined = _combine_sub_decompositions(found_sub[0], found_sub[1], n, m, p1, p2)
            
            # Validate the combined result
            if _validate(combined["U"], combined["V"], combined["W"], n, m, p):
                total_rank = r_sub * 2
                if total_rank < best_rank:
                    best = combined
                    best_rank = total_rank
                    break  # Found a good result, stop searching
    
    # If we didn't find a sub-decomposition, try direct random search for the full problem
    # But this is very unlikely to succeed for rank < 72
    # So we fall back to the best found (which might be schoolbook)
    
    return best

if __name__ == "__main__":
    res = construct()
    print("rank:", len(res["U"][0]))
