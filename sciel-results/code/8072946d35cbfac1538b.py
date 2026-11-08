"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def check_exact(U, V, W):
        """Verify exact decomposition using Fraction arithmetic."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(R):
                        u = Fraction(U[i*m+j][r])
                        v = Fraction(V[j*p+k][r])
                        w = Fraction(W[k*n+i][r])
                        s += u * v * w
                    if s != Fraction(1):
                        return False
        return True

    def check_exact_float(U, V, W):
        """Fast float check, then exact if close."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True

    def snap_to_half(arr):
        """Snap values to nearest half-integer."""
        return [[round(x * 2) / 2 for x in row] for row in arr]

    def try_fixed_W(W, R, max_attempts=20):
        """Given fixed W, try to find U and V."""
        # For fixed W, the equation is:
        # sum_r W[k,i,r] * U[i,j,r] * V[j,k,r] = 1
        # This is bilinear in U and V.
        # Strategy: Fix U randomly, solve for V linearly? No, it's bilinear.
        # Alternative: Fix U, then for each (j,k): sum_r W[k,i,r]*U[i,j,r]*V[j,k,r] = 1 for all i
        # This gives n equations in V[j,k,0..R-1] for each (j,k).
        # But U is also unknown.
        
        # Better: Fix both U and V randomly, check if there's a W solution.
        # Or: Use alternating minimization.
        
        # Let's try: fix U, solve for V given W.
        # For fixed U and W, the equation for V[j,k,r] is:
        # sum_r W[k,i,r] * U[i,j,r] * V[j,k,r] = 1 for all i
        # This is linear in V[j,k,:] for each (j,k).
        # For each (j,k), we have n equations:
        # sum_r (W[k,i,r] * U[i,j,r]) * V[j,k,r] = 1, for i=0..n-1
        # This is a system: M_{i,r} * V[j,k,r] = 1 where M[i,r] = W[k,i,r]*U[i,j,r]
        # n equations, R unknowns. If n <= R, we can solve (least squares or exact).
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Random U with entries from {-1, 0, 1}
            U = [[0]*R for _ in range(n*m)]
            for idx in range(n*m):
                for r in range(R):
                    U[idx][r] = np.random.randint(-1, 2)
            
            # Solve for V
            V = [[0]*R for _ in range(m*p)]
            valid = True
            for j in range(m):
                for k in range(p):
                    # Build system: for i=0..n-1: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
                    A = np.zeros((n, R))
                    for i in range(n):
                        for r in range(R):
                            A[i, r] = W[k*n+i][r] * U[i*m+j][r]
                    b = np.ones(n)
                    
                    # Solve A V_col = b
                    try:
                        V_col, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                        if len(residuals) > 0 and residuals[0] > 1e-10:
                            valid = False
                            break
                        # Snap to half-integers
                        V_col_snapped = np.round(V_col * 2) / 2
                        # Verify
                        for i in range(n):
                            s_val = sum(A[i, r] * V_col_snapped[r] for r in range(R))
                            if abs(s_val - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                        V[j*p+k] = V_col_snapped.tolist()
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify full decomposition
                if check_exact_float(U, V, W):
                    if check_exact(U, V, W):
                        return (U, V, W)
        
        return None

    # Strategy 1: Try structured W patterns
    # W has shape (p*n, R) = (16, R)
    # Try W as identity-like: W[k*n+i, r] = 1 if r == k*n+i, else 0
    # Then sum_r U[i,j,r] * V[j,k,r] * delta_{r,k*n+i} = U[i,j,k*n+i] * V[j,k,k*n+i] = 1
    # So we need U[i,j,k*n+i] * V[j,k,k*n+i] = 1 for all i,j,k
    # This means U[i,j,k*n+i] = 1 and V[j,k,k*n+i] = 1 (or both -1)
    # But then R = p*n = 16, and we need U and V to have specific entries.
    # Let's check: if W is the identity (16x16), then R=16.
    # U[i*m+j, k*n+i] must be 1 and V[j*p+k, k*n+i] must be 1.
    # But U has only R=16 columns, indexed 0..15.
    # For U[i*m+j, r] where r = k*n+i: this is a specific column.
    # For each (i,j), we need U[i*m+j, k*n+i] = 1 for all k.
    # But k*n+i ranges over {0,2,4,6} for i=0 and {1,3,5,7} for i=1 as k varies.
    # So U[i*m+j, 0] = U[i*m+j, 2] = U[i*m+j, 4] = U[i*m+j, 6] = 1 for i=0
    # And U[i*m+j, 1] = U[i*m+j, 3] = U[i*m+j, 5] = U[i*m+j, 7] = 1 for i=1
    # Similarly for V.
    # This gives a valid decomposition of rank 16!
    
    # Let's verify this:
    R_test = p * n  # 16
    W_test = [[0]*R_test for _ in range(p*n)]
    for r in range(R_test):
        W_test[r][r] = 1
    
    U_test = [[0]*R_test for _ in range(n*m)]
    V_test = [[0]*R_test for _ in range(m*p)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = k*n + i
                U_test[i*m+j][r] = 1
                V_test[j*p+k][r] = 1
    
    if check_exact(U_test, V_test, W_test):
        if R_test < best_rank:
            best_rank = R_test
            best_result = {"U": U_test, "V": V_test, "W": W_test}
    
    # Now try to do better than 16.
    # Lower bound considerations:
    # Rank >= max(n*m, m*p, p*n) is not right.
    # For matrix multiplication, rank >= max(n, m, p) is weak.
    # Better lower bound: rank >= n*m*p / max(n,m,p) ? Not standard.
    # Let's just try R from 8 to 15.
    
    # Strategy: For each R from 8 to 15, try random W patterns and solve.
    for R in range(8, min(16, best_rank)):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random W initializations
        max_W_attempts = 30
        if R <= 10:
            max_W_attempts = 50
        
        for w_attempt in range(max_W_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random W with sparse entries from {-1, 0, 1}
            W = [[0]*R for _ in range(p*n)]
            for idx in range(p*n):
                for r in range(R):
                    if np.random.rand() < 0.3:  # Sparse
                        W[idx][r] = np.random.randint(-1, 2)
            
            # Ensure W is not all zeros
            if all(all(x == 0 for x in row) for row in W):
                continue
            
            result = try_fixed_W(W, R, max_attempts=5)
            if result is not None:
                U_found, V_found, W_found = result
                if R < best_rank:
                    best_rank = R
                    best_result = {"U": U_found, "V": V_found, "W": W_found}
                    return best_result  # Found a lower rank, return immediately
    
    # Strategy 2: Try specific structured decompositions
    # For 2x3x8, try to exploit the small n=2.
    # W has shape (16, R). With n=2, W[k*2+i, r] for i in {0,1}.
    # Let's try W with a block structure.
    
    # Try R from lower values with more systematic search
    for R in range(8, min(16, best_rank)):
        if time.time() > deadline - 10:
            break
        
        # Try W as random but with more structure
        for w_attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # W with entries from {-1, 0, 1}, denser
            W = [[0]*R for _ in range(p*n)]
            for idx in range(p*n):
                for r in range(R):
                    W[idx][r] = np.random.randint(-1, 2)
            
            if all(all(x == 0 for x in row) for row in W):
                continue
            
            result = try_fixed_W(W, R, max_attempts=3)
            if result is not None:
                U_found, V_found, W_found = result
                if R < best_rank:
                    best_rank = R
                    best_result = {"U": U_found, "V": V_found, "W": W_found}
                    return best_result

    return best_result

# EVOLVE-BLOCK-END
