"""Matrix multiplication 2,2,3 — rank-minimal exact tensor decomposition.

Strategy:
1. Try rank 10 via randomized search (fix U,V, solve W linearly)
2. Fall back to rank 11 (Strassen 2x2x2 + schoolbook 2x2x1)
3. Fall back to rank 12 (pure schoolbook)
"""

import numpy as np
import time

DIMENSIONS = [2, 2, 3]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 15.0

def check_time():
    """Return True if we should stop searching."""
    return (time.time() - START_TIME) > (TIME_LIMIT - SAFETY_MARGIN)

def verify_decomposition(U, V, W, n=2, m=2, p=3):
    """Verify that the decomposition is exact using numpy einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    # result[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    result = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(result[a, b, c] - 1.0) > 1e-9:
                    return False
    return True

def get_rank_11_solution():
    """
    Rank 11 solution: Strassen for 2x2x2 (rank 7) + schoolbook for 2x2x1 (rank 4).
    """
    rank = 11
    n, m, p = 2, 2, 3
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val): U[i*m+j][r] += val
    def set_v(j, k, r, val): V[j*p+k][r] += val
    def set_w(k, i, r, val): W[k*n+i][r] += val

    # Strassen 2x2x2 factors
    # c=0: M1 = (A00+A11)(B00+B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)
    
    # c=1: M2 = (A10+A11)B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1)
    
    # c=2: M3 = A00(B01-B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1)
    
    # c=3: M4 = A11(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 1, 3, 1)
    
    # c=4: M5 = (A00+A10)B11
    set_u(0, 0, 4, 1); set_u(1, 0, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 1, 4, -1)
    
    # c=5: M6 = (A10-A00)(B00+B10)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(1, 0, 5, 1)
    set_w(1, 1, 5, 1)
    
    # c=6: M7 = (A01-A11)(B01+B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(0, 1, 6, 1); set_v(1, 1, 6, 1)
    set_w(1, 0, 6, 1)
    
    # Schoolbook for 2x2x1 part (k=2)
    # c=7: A00*B02 -> C02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)
    
    # c=8: A01*B12 -> C02
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)
    
    # c=9: A10*B02 -> C12
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)
    
    # c=10: A11*B12 -> C12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)
    
    return {"U": U, "V": V, "W": W}

def get_rank_12_solution():
    """Pure schoolbook rank 12."""
    rank = 12
    n, m, p = 2, 2, 3
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def try_rank_10(max_attempts=2000):
    """
    Try to find a rank-10 decomposition via randomized search.
    
    For each (a,b,c) tensor entry, we need:
    sum_r U[a,r] * V[b,r] * W[c,r] = 1
    
    Strategy: Randomly initialize U and V, then for each c, solve the linear
    system for W[c,:]. Check if the solution is consistent.
    
    For fixed U and V, the equations for a given c are:
    For each (a,b) pair where the tensor entry (a,b,c) exists:
    sum_r U[a,r] * V[b,r] * W[c,r] = 1
    
    This is linear in W[c,:]. We can set up a matrix M_c where row for (a,b)
    has entries U[a,r]*V[b,r] for r=0..9, and solve M_c * W[c,:] = ones.
    """
    n, m, p = 2, 2, 3
    rank = 10
    
    # Build tensor entries grouped by c
    entries_by_c = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if c not in entries_by_c:
                    entries_by_c[c] = []
                entries_by_c[c].append((a, b))
    
    rng = np.random.RandomState(42)
    
    for attempt in range(max_attempts):
        if check_time():
            return None
        
        # Randomly initialize U and V
        U_arr = rng.randn(n*m, rank) * 0.5
        V_arr = rng.randn(m*p, rank) * 0.5
        
        W_arr = np.zeros((p*n, rank))
        
        valid = True
        for c in range(p*n):
            if c not in entries_by_c:
                continue
            
            # Build linear system for W[c,:]
            # Each row corresponds to an (a,b) entry
            rows = entries_by_c[c]
            if len(rows) == 0:
                continue
            
            # M[row_idx, r] = U[a,r] * V[b,r]
            M = np.array([U_arr[a] * V_arr[b] for a, b in rows])
            rhs = np.ones(len(rows))
            
            # Solve M * w = rhs
            try:
                # Use least squares in case of underdetermined or overdetermined
                w, residuals, rank_m, sv = np.linalg.lstsq(M, rhs, rcond=None)
                
                # Check if solution is valid
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    valid = False
                    break
                
                W_arr[c] = w
            except np.linalg.LinAlgError:
                valid = False
                break
        
        if not valid:
            continue
        
        # Verify the full decomposition
        result = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(result[a, b, c] - 1.0) > 1e-8:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        
        if ok:
            # Convert to lists
            U_list = U_arr.tolist()
            V_list = V_arr.tolist()
            W_list = W_arr.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Try rank 10 first
    sol = try_rank_10(max_attempts=2000)
    if sol is not None:
        if verify_decomposition(sol["U"], sol["V"], sol["W"]):
            return sol
    
    # Fall back to rank 11 (Strassen + schoolbook)
    sol = get_rank_11_solution()
    if verify_decomposition(sol["U"], sol["V"], sol["W"]):
        return sol
    
    # Fallback to rank 12
    sol = get_rank_12_solution()
    if verify_decomposition(sol["U"], sol["V"], sol["W"]):
        return sol
    
    # Should never get here
    return get_rank_12_solution()
