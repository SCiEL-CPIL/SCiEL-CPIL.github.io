import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Tensor T
    # T[u, v, w] = 1 if u = i*m + j, v = j*p + k, w = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    # Lower bound for rank
    # Rank >= max(n*p, m*p, n*m) = max(20, 25, 20) = 25
    min_rank = max(n * p, m * p, n * m)
    
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # Try ranks from min_rank upwards
    # We might not find a solution for small ranks, so we iterate
    # But we must finish in time.
    # If we don't find a solution, return schoolbook.
    
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # ALS Solver
    def solve_als(r, max_iters=50, restarts=5):
        for restart in range(restarts):
            # Initialize
            # Use small random integers or structured
            U = np.random.randint(-1, 2, (n * m, r))
            V = np.random.randint(-1, 2, (m * p, r))
            W = np.random.randint(-1, 2, (p * n, r))
            
            # If all zeros, add noise
            if np.all(U == 0): U += np.random.rand(n * m, r)
            if np.all(V == 0): V += np.random.rand(m * p, r)
            if np.all(W == 0): W += np.random.rand(p * n, r)

            for it in range(max_iters):
                if time.time() > deadline:
                    return None

                # Update W
                # For each w, solve M_w W_w = T_w
                # M_w has shape (m*p, r)
                # T_w has shape (m*p,)
                for w in range(p * n):
                    M_w = np.zeros((m * p, r))
                    T_w = np.zeros(m * p)
                    for v in range(m * p):
                        # Sum over u? No, equation is for each u, v.
                        # Wait, the equation is sum_t U[u,t] V[v,t] W[w,t] = T[u,v,w]
                        # For fixed w, we want to find W[w,t].
                        # The equations are for all u, v.
                        # But T[u,v,w] is non-zero only for specific u, v.
                        # Actually, we can form the system for W[w, :] using all u, v.
                        # M_w[v, t] = sum_u U[u,t] V[v,t]? No.
                        # The equation is linear in W[w,t].
                        # sum_t (sum_u U[u,t] V[v,t] * delta(w, w(u,v))) W[w,t] ... No.
                        
                        # Let's re-evaluate the linear system for W.
                        # We have equations:
                        # sum_t U[u,t] V[v,t] W[w,t] = T[u,v,w]
                        # This is one equation for each (u,v,w).
                        # But W[w,t] appears in equations for all u, v.
                        # So for a fixed w, we have n*m equations?
                        # No, for a fixed w, the variable is W[w, :].
                        # The equation involves W[w,t] for t=0..r-1.
                        # The coefficient of W[w,t] in the equation for (u,v) is U[u,t] V[v,t].
                        # So the system for W[w, :] is:
                        # sum_t (U[u,t] V[v,t]) W[w,t] = T[u,v,w]
                        # This must hold for ALL u, v.
                        # So we have n*m equations for each w.
                        # M_w has shape (n*m, r).
                        # M_w[u, t] = U[u,t] V[v,t]? No, v is also an index.
                        # Wait, the equation is for a specific u, v, w.
                        # So for fixed w, we have equations for all (u,v).
                        # M_w[u, t] = ?
                        # The term is U[u,t] V[v,t] W[w,t].
                        # This depends on v.
                        # So we cannot form a single matrix M_w for all u,v unless we treat (u,v) as a single index.
                        # Let idx = u * (m*p) + v.
                        # Then M_w[idx, t] = U[u,t] V[v,t].
                        # And T_w[idx] = T[u,v,w].
                        # So M_w has shape (n*m * m*p, r).
                        # This is (20 * 25, r) = (500, r).
                        # This is a large system.
                        
                        pass # Placeholder
                    
                    # Efficiently construct M_w
                    # M_w[idx, t] = U[u,t] V[v,t]
                    # We can compute this using broadcasting or loops.
                    # Since n*m=20 and m*p=25, 500 equations.
                    # r is small (~25-50).
                    # We can build M_w as a 500 x r matrix.
                    
                    M_w = np.zeros((n * m * m * p, r))
                    T_w = np.zeros(n * m * m * p)
                    idx = 0
                    for u in range(n * m):
                        for v in range(m * p):
                            for t in range(r):
                                M_w[idx, t] = U[u, t] * V[v, t]
                            T_w[idx] = T[u, v, w]
                            idx += 1
                            
                    # Solve M_w W_w = T_w
                    # Least squares
                    W[w, :], residuals, rank, sv = np.linalg.lstsq(M_w, T_w, rcond=None)
                
                # Update U
                for u in range(n * m):
                    M_u = np.zeros((m * p * p * n, r))
                    T_u = np.zeros(m * p * p * n)
                    idx = 0
                    for v in range(m * p):
                        for w in range(p * n):
                            for t in range(r):
                                M_u[idx, t] = V[v, t] * W[w, t]
                            T_u[idx] = T[u, v, w]
                            idx += 1
                    U[u, :], residuals, rank, sv = np.linalg.lstsq(M_u, T_u, rcond=None)

                # Update V
                for v in range(m * p):
                    M_v = np.zeros((n * m * p * n, r))
                    T_v = np.zeros(n * m * p * n)
                    idx = 0
                    for u in range(n * m):
                        for w in range(p * n):
                            for t in range(r):
                                M_v[idx, t] = U[u, t] * W[w, t]
                            T_v[idx] = T[u, v, w]
                            idx += 1
                    V[v, :], residuals, rank, sv = np.linalg.lstsq(M_v, T_v, rcond=None)
                
                # Check error
                # Compute tensor from U, V, W
                # T_calc[u, v, w] = sum_t U[u,t] V[v,t] W[w,t]
                # This is expensive.
                # We can check a few random points or the full tensor if small.
                # n*m=20, m*p=25, p*n=20. Total 10000 entries.
                # r ~ 30. 10000 * 30 = 300k ops. Fast.
                
                error = 0.0
                for u in range(n * m):
                    for v in range(m * p):
                        for w in range(p * n):
                            val = np.dot(U[u, :], V[v, :] * W[w, :])
                            error += (val - T[u, v, w]) ** 2
                
                if error < 1e-6:
                    # Try to round
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    
                    # Verify
                    error_int = 0.0
                    for u in range(n * m):
                        for v in range(m * p):
                            for w in range(p * n):
                                val = np.dot(U_int[u, :], V_int[v, :] * W_int[w, :])
                                error_int += (val - T[u, v, w]) ** 2
                    
                    if error_int < 1e-9:
                        return U_int, V_int, W_int

        return None

    # Try ranks
    for r in range(min_rank, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = solve_als(r, max_iters=10, restarts=5)
        if result:
            U_int, V_int, W_int = result
            # Convert to list of lists
            U_list = U_int.tolist()
            V_list = V_int.tolist()
            W_list = W_int.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return {"U": best_U, "V": best_V, "W": best_W}
