import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook solution
    R_school = n * m * p
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = R_school

    def check_decomposition(U, V, W, n, m, p):
        r = U.shape[1]
        # Vectorized check is faster for small dimensions
        # T_target[a, b, c, d, i, k] = 1 if a==i, c==k, b==d else 0
        # Sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l]
        
        # Pre-compute indices
        a_idx = np.repeat(np.arange(n), m * p * n * p)
        b_idx = np.tile(np.repeat(np.arange(m), p * n * p), n)
        c_idx = np.tile(np.repeat(np.arange(m), p * n * p), n * m)
        d_idx = np.tile(np.arange(p * n * p), n * m) % p # This logic is getting complex, simple loop is safer for correctness and small size
        # Total checks: 2*3*3*6*2*6 = 1296. 
        # Loop is fine.
        
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                target = 1.0 if (a == i and c == k and b == d) else 0.0
                                s = 0.0
                                # Use dot product for speed
                                # U[a*m+b, :] * V[c*p+d, :] * W[k*n+i, :]
                                # This is element-wise product then sum
                                val = np.dot(U[a * m + b] * V[c * p + d], W[k * n + i])
                                if not np.isclose(val, target, atol=1e-9):
                                    return False
        return True

    def als_solve_U(V, W, n, m, p, r):
        # Solve for U given V, W
        U_new = np.zeros((n * m, r))
        # Equations: for each a, b
        # Sum_l U[a,b,l] * V[c,d,l] * W[k,i,l] = delta(a,i) * delta(c,k) * delta(b,d)
        # Variables: U[a,b, 0..r-1]
        
        # Precompute the matrix A for each (a,b)
        # A has rows for each (c,d,i,k)
        # Row idx: c * p * n * p + d * n * p + i * p + k ? 
        # Actually, we just need to iterate and build the system.
        
        for a in range(n):
            for b in range(m):
                # Build A (size: num_eq x r) and b_vec (size: num_eq)
                # num_eq = m * p * n * p
                A = np.zeros((m * p * n * p, r))
                b_vec = np.zeros(m * p * n * p)
                row = 0
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                target = 1.0 if (a == i and c == k and b == d) else 0.0
                                b_vec[row] = target
                                for l in range(r):
                                    A[row, l] = V[c * p + d, l] * W[k * n + i, l]
                                row += 1
                
                # Solve least squares
                # Use np.linalg.lstsq
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                    U_new[a * m + b] = sol
                except:
                    U_new[a * m + b] = 0
        return U_new

    def als_solve_V(U, W, n, m, p, r):
        V_new = np.zeros((m * p, r))
        for c in range(m):
            for d in range(p):
                A = np.zeros((n * m * p * n, r)) # a,b,k,i
                b_vec = np.zeros(n * m * p * n)
                row = 0
                for a in range(n):
                    for b in range(m):
                        for k in range(p):
                            for i in range(n):
                                target = 1.0 if (a == i and c == k and b == d) else 0.0
                                b_vec[row] = target
                                for l in range(r):
                                    A[row, l] = U[a * m + b, l] * W[k * n + i, l]
                                row += 1
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                    V_new[c * p + d] = sol
                except:
                    V_new[c * p + d] = 0
        return V_new

    def als_solve_W(U, V, n, m, p, r):
        W_new = np.zeros((p * n, r))
        for k in range(p):
            for i in range(n):
                A = np.zeros((n * m * m * p, r)) # a,b,c,d
                b_vec = np.zeros(n * m * m * p)
                row = 0
                for a in range(n):
                    for b in range(m):
                        for c in range(m):
                            for d in range(p):
                                target = 1.0 if (a == i and c == k and b == d) else 0.0
                                b_vec[row] = target
                                for l in range(r):
                                    A[row, l] = U[a * m + b, l] * V[c * p + d, l]
                                row += 1
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
                    W_new[k * n + i] = sol
                except:
                    W_new[k * n + i] = 0
        return W_new

    def find_decomposition_233(target_rank, time_limit=10):
        n2, m2, p2 = 2, 3, 3
        start = time.time()
        r = target_rank
        U_dim = n2 * m2
        V_dim = m2 * p2
        W_dim = p2 * n2
        
        for restart in range(20):
            if time.time() - start > time_limit:
                break
            
            np.random.seed(restart * 100 + r)
            U = np.random.randn(U_dim, r)
            V = np.random.randn(V_dim, r)
            W = np.random.randn(W_dim, r)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(50):
                if time.time() - start > time_limit:
                    break
                
                U = als_solve_U(V, W, n2, m2, p2, r)
                V = als_solve_V(U, W, n2, m2, p2, r)
                W = als_solve_W(U, V, n2, m2, p2, r)
                
                # Check convergence
                # Compute residual
                res = 0
                for a in range(n2):
                    for b in range(m2):
                        for c in range(m2):
                            for d in range(p2):
                                for i in range(n2):
                                    for k in range(p2):
                                        target = 1.0 if (a == i and c == k and b == d) else 0.0
                                        val = np.dot(U[a * m2 + b] * V[c * p2 + d], W[k * n2 + i])
                                        res += (val - target) ** 2
                                        
                if res < 1e-10:
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if check_decomposition(U_snap, V_snap, W_snap, n2, m2, p2):
                        return U_snap, V_snap, W_snap
                    # Try integer snap
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if check_decomposition(U_int, V_int, W_int, n2, m2, p2):
                        return U_int, V_int, W_int
                    # Return real if valid (though task asks for int/half-int)
                    # If real is valid but snap fails, we might need more iterations or different snap
                    # But usually if res is 1e-10, snap works if the solution is close to rational.
        
        return None

    # Search for 2x3x3 decomposition
    # Try ranks 17, 16, 15
    for r233 in [17, 16, 15]:
        if time.time() > deadline - 20:
            break
            
        res = find_decomposition_233(r233, time_limit=15)
        if res is not None:
            U233, V233, W233 = res
            r_comb = 2 * r233
            
            # Combine to 2x3x6
            U_comb = np.zeros((2 * 3, 2 * r233))
            U_comb[:, :r233] = U233
            U_comb[:, r233:] = U233
            
            V_comb = np.zeros((3 * 6, 2 * r233))
            for j in range(3):
                for k in range(3):
                    V_comb[j * 6 + k, :r233] = V233[j * 3 + k]
                    V_comb[j * 6 + 3 + k, r233:] = V233[j * 3 + k]
                    
            W_comb = np.zeros((6 * 2, 2 * r233))
            for k in range(3):
                for i in range(2):
                    W_comb[k * 2 + i, :r233] = W233[k * 2 + i]
                    W_comb[(k + 3) * 2 + i, r233:] = W233[k * 2 + i]
            
            if check_decomposition(U_comb, V_comb, W_comb, 2, 3, 6):
                if r_comb < best_rank:
                    best_rank = r_comb
                    best_U = U_comb
                    best_V = V_comb
                    best_W = W_comb
                break # Found a valid one, stop searching lower ranks? 
                # Actually, lower r233 is better. But 17 is likely the first one found.
                # If we found 17, rank is 34. 
                # Should we try 16? Yes, if time permits.
                # But 17 is a good improvement. 
                # Let's break to save time for safety, or continue?
                # "rank-minimal" is the goal. 
                # If we find 34, we should try for 32.
                # But 16 is much harder. 
                # Let's just return the first one found to be safe?
                # No, let's try to find the best one within time.
                # But if 17 works, 16 might not. 
                # Let's break to ensure we return a valid improvement.
                pass 

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
