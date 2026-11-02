"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m) x rank, V: (m*p) x rank, W: (p*n) x rank
    # Constraint: for all a in range(n), b in range(m), c in range(p):
    #   sum_r U[a*m+b, r] * V[b*p+c, r] * W[c*n+a, r] = 1
    # And for all "invalid" (x,y,z) triples, sum should be 0.
    
    def check_solution(U, V, W, tol=1e-8):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        
        # Check all required 1s
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    x = a * m + b
                    y = b * p + c
                    z = c * n + a
                    val = np.dot(U[x, :], V[y, :] * W[z, :])
                    if abs(val - 1.0) > tol:
                        return False
                    if time.time() > DEADLINE:
                        return None
        
        # Check a sample of 0-entries for performance
        import random
        random.seed(42)
        num_checks = min(50000, int((DEADLINE - time.time()) * 10000))
        if num_checks < 1000:
            return True  # Assume valid if too close to deadline
        
        for _ in range(num_checks):
            x = random.randint(0, n*m - 1)
            y = random.randint(0, m*p - 1)
            z = random.randint(0, p*n - 1)
            # Skip if this is a required 1-entry
            ax, bx = divmod(x, m)
            by, cy = divmod(y, p)
            cz, az = divmod(z, n)
            if ax == az and bx == by and cy == cz:
                continue
            val = np.dot(U[x, :], V[y, :] * W[z, :])
            if abs(val) > tol:
                return False
            if time.time() > DEADLINE:
                return None
        
        return True
    
    def schoolbook_decomposition():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    U, V, W = schoolbook_decomposition()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def try_reduce_zero_terms(U, V, W):
        """Remove zero terms."""
        rank = U.shape[1]
        U_work = U.copy()
        V_work = V.copy()
        W_work = W.copy()
        
        while True:
            if time.time() > DEADLINE - 5:
                break
            removed = False
            for r1 in range(U_work.shape[1]):
                if (np.abs(U_work[:, r1]).max() < 1e-10 or 
                    np.abs(V_work[:, r1]).max() < 1e-10 or 
                    np.abs(W_work[:, r1]).max() < 1e-10):
                    U_work = np.delete(U_work, r1, axis=1)
                    V_work = np.delete(V_work, r1, axis=1)
                    W_work = np.delete(W_work, r1, axis=1)
                    removed = True
                    break
            if not removed:
                break
        
        result = check_solution(U_work, V_work, W_work)
        if result is True:
            return U_work, V_work, W_work, U_work.shape[1]
        return U, V, W, rank
    
    U, V, W, rank = try_reduce_zero_terms(U, V, W)
    if rank < best_rank:
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try numerical optimization for lower ranks
    # Use alternating least squares approach
    for target_rank in range(70, best_rank):
        if time.time() > DEADLINE - 10:
            break
        
        best_for_rank = None
        for trial in range(3):
            if time.time() > DEADLINE - 10:
                break
            
            np.random.seed(42 + trial * 100)
            
            # Initialize with structured random matrices
            U_init = np.random.randn(n*m, target_rank) * 0.1
            V_init = np.random.randn(m*p, target_rank) * 0.1
            W_init = np.random.randn(p*n, target_rank) * 0.1
            
            # Add some structure: ensure some entries are non-zero
            for r in range(target_rank):
                idx_u = np.random.randint(0, n*m)
                idx_v = np.random.randint(0, m*p)
                idx_w = np.random.randint(0, p*n)
                U_init[idx_u, r] += 1.0
                V_init[idx_v, r] += 1.0
                W_init[idx_w, r] += 1.0
            
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            # Alternating least squares
            for it in range(200):
                if time.time() > DEADLINE - 10:
                    break
                
                # Update U: for each x, solve least squares
                for x in range(n*m):
                    ax, bx = divmod(x, m)
                    A_rows = []
                    b_vals = []
                    for y in range(m*p):
                        by, cy = divmod(y, p)
                        for z in range(p*n):
                            cz, az = divmod(z, n)
                            prod = V_cur[y, :] * W_cur[z, :]
                            if ax == az and bx == by and cy == cz:
                                tval = 1.0
                            else:
                                tval = 0.0
                            A_rows.append(prod)
                            b_vals.append(tval)
                    
                    if A_rows:
                        A_arr = np.array(A_rows)
                        b_arr = np.array(b_vals)
                        # Use SVD for stability
                        try:
                            U_cur[x, :], _, _, _ = np.linalg.lstsq(A_arr, b_arr, rcond=1e-10)
                        except:
                            pass
                
                # Update V: for each y, solve least squares
                for y in range(m*p):
                    by, cy = divmod(y, p)
                    A_rows = []
                    b_vals = []
                    for x in range(n*m):
                        ax, bx = divmod(x, m)
                        for z in range(p*n):
                            cz, az = divmod(z, n)
                            prod = U_cur[x, :] * W_cur[z, :]
                            if ax == az and bx == by and cy == cz:
                                tval = 1.0
                            else:
                                tval = 0.0
                            A_rows.append(prod)
                            b_vals.append(tval)
                    
                    if A_rows:
                        A_arr = np.array(A_rows)
                        b_arr = np.array(b_vals)
                        try:
                            V_cur[y, :], _, _, _ = np.linalg.lstsq(A_arr, b_arr, rcond=1e-10)
                        except:
                            pass
                
                # Update W: for each z, solve least squares
                for z in range(p*n):
                    cz, az = divmod(z, n)
                    A_rows = []
                    b_vals = []
                    for x in range(n*m):
                        ax, bx = divmod(x, m)
                        for y in range(m*p):
                            by, cy = divmod(y, p)
                            prod = U_cur[x, :] * V_cur[y, :]
                            if ax == az and bx == by and cy == cz:
                                tval = 1.0
                            else:
                                tval = 0.0
                            A_rows.append(prod)
                            b_vals.append(tval)
                    
                    if A_rows:
                        A_arr = np.array(A_rows)
                        b_arr = np.array(b_vals)
                        try:
                            W_cur[z, :], _, _, _ = np.linalg.lstsq(A_arr, b_arr, rcond=1e-10)
                        except:
                            pass
                
                if time.time() > DEADLINE - 10:
                    break
                
                # Check convergence
                loss = 0.0
                for a in range(n):
                    for b in range(m):
                        for c in range(p):
                            x = a * m + b
                            y = b * p + c
                            z = c * n + a
                            val = np.dot(U_cur[x, :], V_cur[y, :] * W_cur[z, :])
                            loss += (val - 1.0) ** 2
                
                if loss < 1e-12:
                    break
                
                # Early exit if loss is increasing
                if it > 0 and loss > 1e6:
                    break
            
            # Final check
            result = check_solution(U_cur, V_cur, W_cur)
            if result is True:
                # Try to reduce zero terms
                U_red, V_red, W_red, rank_red = try_reduce_zero_terms(U_cur, V_cur, W_cur)
                if rank_red < target_rank:
                    best_for_rank = (U_red.copy(), V_red.copy(), W_red.copy())
                else:
                    best_for_rank = (U_cur.copy(), V_cur.copy(), W_cur.copy())
                break
        
        if best_for_rank:
            U, V, W = best_for_rank
            best_rank = U.shape[1]
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break
    
    # Final polish: tighten the solution
    if time.time() < DEADLINE - 5:
        # Do a few more ALS iterations with tighter tolerance
        for it in range(50):
            if time.time() > DEADLINE - 5:
                break
            
            # Update U
            for x in range(n*m):
                ax, bx = divmod(x, m)
                A_rows = []
                b_vals = []
                for y in range(m*p):
                    by, cy = divmod(y, p)
                    for z in range(p*n):
                        cz, az = divmod(z, n)
                        prod = best_V[y, :] * best_W[z, :]
                        if ax == az and bx == by and cy == cz:
                            tval = 1.0
                        else:
                            tval = 0.0
                        A_rows.append(prod)
                        b_vals.append(tval)
                
                if A_rows:
                    A_arr = np.array(A_rows)
                    b_arr = np.array(b_vals)
                    try:
                        best_U[x, :], _, _, _ = np.linalg.lstsq(A_arr, b_arr, rcond=1e-12)
                    except:
                        pass
            
            # Update V
            for y in range(m*p):
                by, cy = divmod(y, p)
                A_rows = []
                b_vals = []
                for x in range(n*m):
                    ax, bx = divmod(x, m)
                    for z in range(p*n):
                        cz, az = divmod(z, n)
                        prod = best_U[x, :] * best_W[z, :]
                        if ax == az and bx == by and cy == cz:
                            tval = 1.0
                        else:
                            tval = 0.0
                        A_rows.append(prod)
                        b_vals.append(tval)
                
                if A_rows:
                    A_arr = np.array(A_rows)
                    b_arr = np.array(b_vals)
                    try:
                        best_V[y, :], _, _, _ = np.linalg.lstsq(A_arr, b_arr, rcond=1e-12)
                    except:
                        pass
            
            # Update W
            for z in range(p*n):
                cz, az = divmod(z, n)
                A_rows = []
                b_vals = []
                for x in range(n*m):
                    ax, bx = divmod(x, m)
                    for y in range(m*p):
                        by, cy = divmod(y, p)
                        prod = best_U[x, :] * best_V[y, :]
                        if ax == az and bx == by and cy == cz:
                            tval = 1.0
                        else:
                            tval = 0.0
                        A_rows.append(prod)
                        b_vals.append(tval)
                
                if A_rows:
                    A_arr = np.array(A_rows)
                    b_arr = np.array(b_vals)
                    try:
                        best_W[z, :], _, _, _ = np.linalg.lstsq(A_arr, b_arr, rcond=1e-12)
                    except:
                        pass
    
    def to_list(mat):
        return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
    
    # Final verification
    final_check = check_solution(best_U, best_V, best_W, tol=1e-6)
    if final_check is True:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # Fallback: schoolbook
    U_fb, V_fb, W_fb = schoolbook_decomposition()
    return {"U": to_list(U_fb), "V": to_list(V_fb), "W": to_list(W_fb)}
# EVOLVE-BLOCK-END
