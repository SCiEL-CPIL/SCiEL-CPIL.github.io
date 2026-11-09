import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin: 115s < 120s limit
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=7
    
    def make_schoolbook():
        """Return the standard rank n*m*p decomposition."""
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
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 56
    
    def verify_exact(U, V, W):
        """Check if decomposition is valid using exact arithmetic."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Check if decomposition is valid using floating point."""
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1) > tol:
                        return False
        return True
        
    def compute_error(U, V, W):
        """Compute the maximum error of the decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err = abs(s - 1)
                    if err > max_err:
                        max_err = err
        return max_err

    def to_clean(arr):
        """Convert floating point values to clean integers or half-integers."""
        clean = []
        for row in arr:
            clean_row = []
            for val in row:
                # Try to snap to nearest half-integer
                snapped = round(val * 2) / 2
                if abs(val - snapped) < 1e-6:
                    clean_row.append(snapped)
                else:
                    clean_row.append(val)
            clean.append(clean_row)
        return clean

    def alternating_minimize(U, V, W, max_iter=200, tol=1e-12):
        """Alternating minimization to refine a rank-r decomposition.
        U: (nm, r), V: (mp, r), W: (pn, r)
        """
        nm, mp, pn = n*m, m*p, p*n
        r = U.shape[1]
        
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            prev_err = compute_error(U, V, W)
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # We need: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
                    # A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    # A @ U[i*m+j,:] = ones(p) -> U[i*m+j,:] @ A.T = ones(p)
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        # Solve A.T @ x = ones(p)
                        U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # We need: sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
                    # B[i,c] = U[i*m+j,c] * W[k*n+i,c]
                    # B @ V[j*p+k,:] = ones(n) -> V[j*p+k,:] @ B.T = ones(n)
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # We need: sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
                    # C[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    # C @ W[k*n+i,:] = ones(m) -> W[k*n+i,:] @ C.T = ones(m)
                    C = np.zeros((m, r))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            curr_err = compute_error(U, V, W)
            if abs(curr_err - prev_err) < tol:
                break
        
        return U, V, W, curr_err

    def try_exact_polish(U_float, V_float, W_float, r):
        """Attempt to find an exact integer/half-integer solution near the floating point one."""
        # Round to nearest half-integers
        U_clean = to_clean(U_float.tolist())
        V_clean = to_clean(V_float.tolist())
        W_clean = to_clean(W_float.tolist())
        
        if verify_exact(U_clean, V_clean, W_clean):
            return U_clean, V_clean, W_clean
        
        # Try fixing two and solving for the third exactly
        # Case 1: Fix U, V, solve for W
        U_fix = np.array(U_clean, dtype=float)
        V_fix = np.array(V_clean, dtype=float)
        W_sol = np.zeros((p*n, r))
        valid = True
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, r))
                for j in range(m):
                    C[j, :] = U_fix[i*m+j, :] * V_fix[j*p+k, :]
                try:
                    W_sol[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                except:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            W_clean_sol = to_clean(W_sol.tolist())
            if verify_exact(U_clean, V_clean, W_clean_sol):
                return U_clean, V_clean, W_clean_sol
        
        # Case 2: Fix U, W, solve for V
        U_fix = np.array(U_clean, dtype=float)
        W_fix = np.array(W_clean, dtype=float)
        V_sol = np.zeros((m*p, r))
        valid = True
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, r))
                for i in range(n):
                    B[i, :] = U_fix[i*m+j, :] * W_fix[k*n+i, :]
                try:
                    V_sol[j*p+k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                except:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            V_clean_sol = to_clean(V_sol.tolist())
            if verify_exact(U_clean, V_clean_sol, W_clean):
                return U_clean, V_clean_sol, W_clean
        
        # Case 3: Fix V, W, solve for U
        V_fix = np.array(V_clean, dtype=float)
        W_fix = np.array(W_clean, dtype=float)
        U_sol = np.zeros((n*m, r))
        valid = True
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, r))
                for k in range(p):
                    A[k, :] = V_fix[j*p+k, :] * W_fix[k*n+i, :]
                try:
                    U_sol[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                except:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            U_clean_sol = to_clean(U_sol.tolist())
            if verify_exact(U_clean_sol, V_clean, W_clean):
                return U_clean_sol, V_clean, W_clean

        return None

    # Try ranks from low to high
    # Lower bound is max(nm, mp, pn) = 28
    lower_bound = max(n*m, m*p, p*n)
    ranks_to_try = list(range(lower_bound, min(45, best_rank)))
    
    for r in ranks_to_try:
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            break
        
        n_trials = 30  # Number of random restarts per rank
        for trial in range(n_trials):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            U_refined, V_refined, W_refined, err = alternating_minimize(U, V, W, max_iter=100)
            
            if err < 1e-8:
                result = try_exact_polish(U_refined, V_refined, W_refined, r)
                if result:
                    U_clean, V_clean, W_clean = result
                    best_rank = r
                    best_result = {"U": U_clean, "V": V_clean, "W": W_clean}
                    break
    
    return best_result
