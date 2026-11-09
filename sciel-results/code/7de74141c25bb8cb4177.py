import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 140.0
    
    n, m, p = 3, 3, 5
    size_U = n * m  # 9
    size_V = m * p  # 15
    size_W = p * n  # 15
    
    # Trivial rank-45 decomposition
    R0 = n * m * p  # 45
    U0 = np.zeros((size_U, R0))
    V0 = np.zeros((size_V, R0))
    W0 = np.zeros((size_W, R0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1.0
                V0[j*p+k, col] = 1.0
                W0[k*n+i, col] = 1.0
                col += 1
    
    best_result = {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
    best_rank = R0
    
    def compute_T(U, V, W, R):
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return T
    
    def verify_exact(U, V, W, R):
        """Verify exact decomposition using rational arithmetic"""
        try:
            U_frac = np.array([[Fraction(int(round(u * 2))) / 2 for u in row] for row in U])
            V_frac = np.array([[Fraction(int(round(v * 2))) / 2 for v in row] for row in V])
            W_frac = np.array([[Fraction(int(round(w * 2))) / 2 for w in row] for row in W])
            
            U_r = U_frac.reshape(n, m, R)
            V_r = V_frac.reshape(m, p, R)
            W_r = W_frac.reshape(p, n, R)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = Fraction(0)
                        for r in range(R):
                            s += U_r[i, j, r] * V_r[j, k, r] * W_r[k, i, r]
                        if s != Fraction(1):
                            return False
            return True
        except:
            return False
    
    def try_reduce_by_merging(U, V, W, R):
        """Try to reduce rank by merging pairs of columns"""
        if time.time() - start_time > timeout - 20:
            return None, None, None, R
        
        # Try merging pairs of columns
        for r1 in range(R):
            if time.time() - start_time > timeout - 20:
                return None, None, None, R
            for r2 in range(r1 + 1, R):
                if time.time() - start_time > timeout - 20:
                    return None, None, None, R
                
                # Create a new column as sum of columns r1 and r2
                # and remove both, replacing with one
                # This is a heuristic - try different combinations
                
                # Try: new_col = col_r1 + col_r2
                U_new = np.delete(U, [r1, r2], axis=1)
                V_new = np.delete(V, [r1, r2], axis=1)
                W_new = np.delete(W, [r1, r2], axis=1)
                
                # Insert new column at position r1
                U_col = U[:, r1] + U[:, r2]
                V_col = V[:, r1] + V[:, r2]
                W_col = W[:, r1] + W[:, r2]
                
                U_new = np.column_stack([U_new[:, :r1], U_col, U_new[:, r1:]])
                V_new = np.column_stack([V_new[:, :r1], V_col, V_new[:, r1:]])
                W_new = np.column_stack([W_new[:, :r1], W_col, W_new[:, r1:]])
                
                R_new = R - 1
                T_new = compute_T(U_new, V_new, W_new, R_new)
                
                if np.allclose(T_new, 1.0, atol=1e-8):
                    if verify_exact(U_new, V_new, W_new, R_new):
                        return U_new, V_new, W_new, R_new
        
        return U, V, W, R
    
    def try_local_search(U, V, W, R, max_iters=100):
        """Local search to reduce rank by small perturbations"""
        if time.time() - start_time > timeout - 20:
            return None, None, None, R
        
        U_cur = U.copy()
        V_cur = V.copy()
        W_cur = W.copy()
        R_cur = R
        
        for iteration in range(max_iters):
            if time.time() - start_time > timeout - 20:
                break
            
            # Try removing a random column
            if R_cur <= 1:
                break
            
            col_to_remove = np.random.randint(R_cur)
            
            U_test = np.delete(U_cur, col_to_remove, axis=1)
            V_test = np.delete(V_cur, col_to_remove, axis=1)
            W_test = np.delete(W_cur, col_to_remove, axis=1)
            
            R_test = R_cur - 1
            T_test = compute_T(U_test, V_test, W_test, R_test)
            
            if np.allclose(T_test, 1.0, atol=1e-8):
                if verify_exact(U_test, V_test, W_test, R_test):
                    return U_test, V_test, W_test, R_test
            else:
                # Try to fix by adjusting remaining columns
                # This is a simplified approach
                pass
        
        return U, V, W, R
    
    # Start with trivial decomposition
    U_cur = U0.copy()
    V_cur = V0.copy()
    W_cur = W0.copy()
    R_cur = R0
    
    # Try iterative rank reduction
    for iteration in range(20):
        if time.time() - start_time > timeout - 25:
            break
        
        # Try merging columns
        U_new, V_new, W_new, R_new = try_reduce_by_merging(U_cur, V_cur, W_cur, R_cur)
        
        if R_new < R_cur:
            U_cur, V_cur, W_cur, R_cur = U_new, V_new, W_new, R_new
            best_result = {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
            best_rank = R_cur
        else:
            # Try local search
            U_new, V_new, W_new, R_new = try_local_search(U_cur, V_cur, W_cur, R_cur)
            if R_new < R_cur:
                U_cur, V_cur, W_cur, R_cur = U_new, V_new, W_new, R_new
                best_result = {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
                best_rank = R_cur
            else:
                break
    
    # Try direct search for lower ranks
    for R_target in range(min(R_cur - 1, 40), 30, -1):
        if time.time() - start_time > timeout - 15:
            break
        
        found = False
        for attempt in range(30):
            if time.time() - start_time > timeout - 10:
                break
            
            np.random.seed(100 + attempt * 13)
            
            # Initialize with structured random values
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(size_U, R_target))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(size_V, R_target))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(size_W, R_target))
            
            T = compute_T(U, V, W, R_target)
            
            if np.allclose(T, 1.0, atol=1e-8):
                if verify_exact(U, V, W, R_target):
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = R_target
                    found = True
                    break
        
        if found:
            break
    
    return best_result
