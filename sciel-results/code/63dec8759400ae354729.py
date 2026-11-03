import time
import numpy as np

DIMENSIONS = [3, 3, 8]
n, m, p = DIMENSIONS

def _time_left(limit=145):
    return limit - (time.time() - _start_time)

_start_time = time.time()

def _schoolbook():
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1) > tol:
                    return False
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            jp = b // p
            k = b % p
            for c in range(p * n):
                kp = c // n
                ii = c % n
                if j == jp and k == kp and i == ii:
                    continue
                if abs(T[a, b, c]) > tol:
                    return False
    return True

def construct():
    global _start_time
    _start_time = time.time()
    TIME_LIMIT = 145
    
    result = _schoolbook()
    best_rank = n * m * p
    best_result = result
    
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    rank = n * m * p
    
    # Try to reduce rank by removing one term at a time
    # For each term to remove, fix U and V (without that term) and solve for W
    for remove_col in range(rank):
        if _time_left(TIME_LIMIT) < 5:
            break
        
        # Remove column remove_col
        U_new = np.delete(U, remove_col, axis=1)
        V_new = np.delete(V, remove_col, axis=1)
        r_new = rank - 1
        
        # Solve for W_new: for each (a,b,c), sum_r U_new[a,r] * V_new[b,r] * W_new[c,r] = T[a,b,c]
        # This is linear in W_new[c,r] for each c
        # For each c, we have equations: sum_r U_new[a,r] * V_new[b,r] * W_new[c,r] = T[a,b,c]
        # This is a linear system in W_new[c,:]
        
        W_new = np.zeros((p * n, r_new))
        valid = True
        
        for c in range(p * n):
            # Build the linear system for W_new[c,:]
            # For each (a,b), we have: sum_r U_new[a,r] * V_new[b,r] * W_new[c,r] = T[a,b,c]
            # This is: (U_new[a,:] * V_new[b,:]) dot W_new[c,:] = T[a,b,c]
            # But this is only for valid (a,b) pairs where T[a,b,c] is defined
            
            # Actually, the equation is: for all a,b, sum_r U_new[a,r] * V_new[b,r] * W_new[c,r] = T[a,b,c]
            # This is a linear system in W_new[c,:]
            # The coefficient matrix is: M[a,b,r] = U_new[a,r] * V_new[b,r]
            # The RHS is: T[a,b,c]
            
            # This is a linear system with (n*m * m*p) equations and r_new unknowns
            # That's 9*24 = 216 equations and r_new unknowns
            # For r_new = 71, this is overdetermined, so use least squares
            
            # Build the matrix M of shape (216, r_new)
            M = np.zeros((n * m * m * p, r_new))
            rhs = np.zeros(n * m * m * p)
            eq_idx = 0
            for a in range(n * m):
                i = a // m
                j = a % m
                for b in range(m * p):
                    jp = b // p
                    k = b % p
                    # T[a,b,c] = 1 if j==jp and k==k_c and i==i_c, else 0
                    # where c = k_c * n + i_c
                    k_c = c // n
                    i_c = c % n
                    if j == jp and k == k_c and i == i_c:
                        rhs[eq_idx] = 1
                    else:
                        rhs[eq_idx] = 0
                    for r in range(r_new):
                        M[eq_idx, r] = U_new[a, r] * V_new[b, r]
                    eq_idx += 1
            
            # Solve M @ W_new[c,:] = rhs
            try:
                sol, residuals, rank_sol, sv = np.linalg.lstsq(M, rhs, rcond=None)
                W_new[c, :] = sol
                # Check if the solution is exact
                if residuals.size > 0 and residuals[0] > 1e-7:
                    valid = False
                    break
            except:
                valid = False
                break
        
        if valid:
            # Verify the solution
            if _verify(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                best_rank = r_new
                best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                U = U_new
                V = V_new
                W = W_new
                rank = r_new
                # Continue trying to reduce further
    
    return best_result
