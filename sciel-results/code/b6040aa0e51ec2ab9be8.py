"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _validate(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the target tensor T[i,j,k] = 1
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    # We can verify by checking that for a random set of matrices, the product matches.
    # But for exactness with integer/half-integer entries, we can check the tensor directly.
    # Let's use a more efficient check: contract U and V, then check against W.
    
    # Actually, the cleanest way: the tensor T_{ijk} = sum_c U_{im+j,c} V_{jp+k,c} W_{kn+i,c}
    # We need T_{ijk} = 1 for all i,j,k.
    
    # Let's do it with numpy for speed, but we need exactness.
    # Since entries are integer or half-integer, we can multiply by 2 to get integers.
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is O(n*m*p*rank) which is 4*4*9*rank. For rank ~ 60-100, that's manageable.
    
    # More efficient: for fixed i,j, we have a vector u = U[i*m+j, :] of length rank
    # for fixed j,k, we have v = V[j*p+k, :]
    # for fixed k,i, we have w = W[k*n+i, :]
    # We need u dot v dot w (element-wise product then sum) = 1
    
    # Let's just iterate.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s to be safe
    
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_school
    
    # Try to find lower rank decompositions.
    # The known lower bound for 4x4x4 is 49 (Strassen).
    # For 4x4x9, we can think of it as 4x4 times 4x9.
    # One approach: use a 4x4x4 decomposition of rank r4, and extend to 4x4x9.
    # But that's complex. Let's try a direct search.
    
    # Actually, let's think about this differently.
    # The tensor is T_{ijk} = 1 for i in [0,4), j in [0,4), k in [0,9).
    # We want to find the minimum rank.
    
    # A simple approach: try to construct a decomposition of rank r by solving a system of equations.
    # For a given rank r, we have r columns in U, V, W.
    # The constraints are: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
    # That's 4*4*9 = 144 equations.
    # Unknowns: r * (n*m + m*p + p*n) = r * (16 + 36 + 36) = 88r.
    # So for r >= 2, we have more unknowns than equations, so solutions likely exist.
    # But we need integer or half-integer entries.
    
    # Let's try a randomized approach:
    # 1. Pick a rank r.
    # 2. Randomly initialize U, V, W with integer entries.
    # 3. Use some optimization to adjust them to satisfy the constraints.
    
    # Actually, a simpler idea: 
    # We can use the fact that the tensor is "simple" (all ones).
    # Consider a decomposition where each column c corresponds to a "pattern".
    
    # Let's try rank 48, 49, etc.
    # For rank 48: 88*48 = 4224 unknowns, 144 equations. Very underdetermined.
    
    # Strategy: 
    # Fix U and V randomly (integer entries), then solve for W.
    # W[k*n+i, c] appears in equations for fixed (k,i).
    # For fixed (k,i), the equation is: sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # Let A_{c,j} = U[i*m+j,c]*V[j*p+k,c]. Then sum_c A_{c, :} dot (W[k*n+i,c] * V[j*p+k,c]... 
    # This is getting complicated.
    
    # Alternative: use numpy to solve.
    # For fixed i,k, we have m equations (one for each j):
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for j=0..m-1
    # This is a linear system in the variables W[k*n+i,c] for c=0..r-1.
    # But it's not linear because W[k*n+i,c] is multiplied by U and V entries which depend on c.
    # Actually, for fixed i,k, and variable w_c = W[k*n+i,c], the equation for each j is:
    # sum_c w_c * U[i*m+j,c] * V[j*p+k,c] = 1
    # This IS linear in w_c! The coefficients are U[i*m+j,c]*V[j*p+k,c].
    
    # So the plan:
    # 1. Choose rank r.
    # 2. Randomly initialize U and V with small integer entries.
    # 3. For each (i,k), solve the linear system for W[k*n+i, :].
    # 4. Check if W entries are (approximately) integer or half-integer.
    # 5. If not, try different U, V.
    
    # But solving for W requires that the matrix M_{j,c} = U[i*m+j,c]*V[j*p+k,c] is invertible (or at least has full row rank m).
    # Since m=4 and r >= 4, this is possible.
    
    # Let's implement this.
    
    def try_rank(r, max_attempts=1000):
        nonlocal best_result, best_rank
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return
            # Randomly initialize U and V with entries from {-1, 0, 1}
            U_try = np.random.randint(-1, 2, size=(n*m, r)).astype(np.float64)
            V_try = np.random.randint(-1, 2, size=(m*p, r)).astype(np.float64)
            
            W_try = np.zeros((p*n, r), dtype=np.float64)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build the m x r matrix M where M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    M = np.zeros((m, r), dtype=np.float64)
                    for j in range(m):
                        for c in range(r):
                            M[j, c] = U_try[i*m+j, c] * V_try[j*p+k, c]
                    
                    # Solve M @ w = 1 for w (length r)
                    # Use least squares
                    try:
                        w, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    except:
                        valid = False
                        break
                    
                    W_try[k*n+i, :] = w
            
            if not valid:
                continue
            
            # Check if W entries are close to integer or half-integer
            # Multiply by 2 and check if close to integer
            W2 = W_try * 2
            W2_rounded = np.round(W2)
            if np.max(np.abs(W2 - W2_rounded)) > 1e-6:
                continue
            
            # Also check U and V are integer (they are, by construction)
            # Now verify the full decomposition
            U_list = U_try.tolist()
            V_list = V_try.tolist()
            W_list = (W2_rounded / 2).tolist()
            
            # Verify
            if _validate(U_list, V_list, W_list, n, m, p):
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    return True
        
        return False
    
    # Try ranks from low to high
    # Theoretical lower bound: at least max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    # But for matrix multiplication, the lower bound is higher.
    # For 4x4x4, it's 49. For 4x4x9, let's try from 40 upwards.
    
    for r in range(36, 145):
        if time.time() > deadline:
            break
        success = try_rank(r, max_attempts=500)
        if success:
            break
    
    return best_result
# EVOLVE-BLOCK-END
