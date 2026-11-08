import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # 150s limit, keep 10s margin

    # Target tensor equations
    # We need to satisfy:
    # sum_t U[i,j,t] V[j,k,t] W[k',i',t] = 1 if i==i' and k==k' else 0
    
    # We will try to find rank r
    # Start with schoolbook rank 40
    max_rank = n * m * p
    
    # Try ranks from max_rank-1 down to some lower bound
    # Lower bound for 2x4x5? 
    # T(n,m,p) >= n*m*p / max(n,m,p) ? No.
    # T(2,4,5) >= 2*4*5 / 5 = 8? No.
    # Lower bound is at least max(n*m, m*p, p*n) ? No.
    # T(2,4,5) >= 2*4 = 8, 4*5=20, 5*2=10. So >= 20.
    
    # We will try to find a solution for r < 40.
    # If we fail, we return schoolbook.
    
    def get_error(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Indices: U[i*m+j, t], V[j*p+k, t], W[k'*n+i', t]
        # Note: W index in task is k*n+i. 
        # So W[k'*n+i', t].
        # Target: 1 if i==i' and k==k' else 0.
        # Note: k in V is the column of B (row of C? No, B is m x p, so k is 0..p-1).
        # k' in W is the column of C.
        # So k==k' means the column index matches.
        
        err = 0.0
        r = U.shape[1]
        # Precompute sums for efficiency?
        # sum_t U[i,j,t] V[j,k,t] is a matrix M_j of size n x p?
        # Let A_t = U_t (n x m), B_t = V_t (m x p), C_t = W_t (p x n).
        # Sum_t A_t[i,j] B_t[j,k] = (A_t B_t)[i,k].
        # Then sum_t (A_t B_t)[i,k] C_t[k',i'] = sum_t Tr( (A_t B_t) C_t^T )?
        # No, C_t is p x n. C_t[k', i'].
        # So sum_t (A_t B_t)[i,k] C_t[k', i'].
        
        # We can compute M = sum_t (A_t B_t) C_t^T ?
        # M is n x p x p x n?
        # M[i, k, k', i'] = sum_t (A_t B_t)[i,k] C_t[k', i'].
        
        # Let's just loop for clarity, it's small.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            target = 1.0 if (i == ip and k == kp) else 0.0
                            val = 0.0
                            for t in range(r):
                                val += U[i*m+j, t] * V[j*p+k, t] * W[kp*n+ip, t]
                            err += (val - target) ** 2
        return err

    def refine(U, V, W):
        # Simple gradient descent? 
        # Or just check if current is close to integer/half-integer and snap.
        pass

    # Strategy:
    # 1. Try to find a solution for r = 39, 38, ...
    # 2. Use random initialization and local search (e.g., coordinate descent or gradient).
    # 3. If a solution is found (error < 1e-6), snap to half-integers and verify.
    
    # Since implementing a robust non-linear solver for this specific bilinear system is complex,
    # and the space is small, we can try a "merge" heuristic on the schoolbook solution.
    # Or, we can use the fact that we can set some variables to 0.
    
    # Actually, for small dimensions, we can use a SAT solver or ILP?
    # But we only have NumPy/SciPy.
    
    # Let's try a simple iterative improvement.
    # Start with schoolbook.
    # Randomly pick two terms t1, t2.
    # Try to merge them into one term t_new by solving for U_new, V_new, W_new
    # such that U1 V1 W1 + U2 V2 W2 = U_new V_new W_new.
    # This is generally impossible for rank reduction unless there's structure.
    
    # Alternative: Random restarts with gradient descent.
    
    def solve_for_rank(r):
        if time.time() > deadline - 5:
            return None
        
        best_err = float('inf')
        best_sol = None
        
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            # Use random sparse initialization
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 5:
                    return best_sol
                
                # Compute gradient
                # err = sum (sum_t U V W - target)^2
                # dErr/dU[i,j,t] = 2 * (val - target) * V[j,k,t] * W[k',i',t]
                # We need to sum over all i,j,k,k',i' where target is defined.
                
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                
                # Efficiently compute residuals
                # Res[i, j, k, kp, ip] = sum_t U[i,j,t] V[j,k,t] W[kp,ip,t] - target
                
                # We can compute this by iterating t?
                # Or using tensor contractions.
                
                # Let's use a simple loop for gradient calculation
                # It might be slow, but n,m,p are small.
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    target = 1.0 if (i == ip and k == kp) else 0.0
                                    val = 0.0
                                    for t in range(r):
                                        u = U[i*m+j, t]
                                        v = V[j*p+k, t]
                                        w = W[kp*n+ip, t]
                                        val += u * v * w
                                        
                                        # Gradient contributions
                                        # d(val)/dU = v*w
                                        # d(val)/dV = u*w
                                        # d(val)/dW = u*v
                                        
                                        # We accumulate gradients
                                        # This inner loop is O(r).
                                        # Total ops: n*m*p*p*n * r = 2*4*5*5*2 * r = 400 * r.
                                        # For r=40, 16000 ops per step.
                                        # 200 steps -> 3.2M ops.
                                        # 50 restarts -> 160M ops.
                                        # This is fine in Python if vectorized, but this is loops.
                                        # We should vectorize.
                                    
                                    diff = val - target
                                    for t in range(r):
                                        u = U[i*m+j, t]
                                        v = V[j*p+k, t]
                                        w = W[kp*n+ip, t]
                                        gU[i*m+j, t] += 2 * diff * v * w
                                        gV[j*p+k, t] += 2 * diff * u * w
                                        gW[kp*n+ip, t] += 2 * diff * u * v
                                    
                # Update
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                
                # Check error
                if step % 10 == 0:
                    err = get_error(U, V, W)
                    if err < best_err:
                        best_err = err
                        best_sol = (U.copy(), V.copy(), W.copy())
                    
                    if err < 1e-6:
                        # Try to snap to half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        if get_error(U_snap, V_snap, W_snap) < 1e-6:
                            return (U_snap, V_snap, W_snap)
            
            # After restart, check best
            if best_sol:
                U, V, W = best_sol
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if get_error(U_snap, V_snap, W_snap) < 1e-6:
                    return (U_snap, V_snap, W_snap)
        
        return best_sol

    # Try ranks
    for r in range(n*m*p - 1, 19, -1):
        if time.time() > deadline - 10:
            break
        sol = solve_for_rank(r)
        if sol:
            U, V, W = sol
            # Convert to lists of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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

# EVOLVE-BLOCK-END
