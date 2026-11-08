import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Target tensor is all 1s of shape (n, m, p)
    # T[i, j, k] = 1
    
    # Try ranks from 19 to 22
    best_result = None
    best_rank = n * m * p  # 36
    
    # Schoolbook fallback
    U_school = np.zeros((n*m, best_rank))
    V_school = np.zeros((m*p, best_rank))
    W_school = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # ALS Solver
    def solve_als(rank, max_iters=500, tol=1e-6):
        # Initialize
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None, float('inf')
            
            # Update U
            # For each i, j, solve M[i,j] M[i,j]^T u = M[i,j] 1
            new_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # V slice: V[j*p : (j+1)*p, :] shape (p, rank)
                    # W slice: W[:, i] ? No, W is (p*n, rank). Index is k*n+i.
                    # W slice: W.reshape(p, n, rank)[:, i, :] shape (p, rank)
                    V_slice = V[j*p:(j+1)*p, :]
                    W_slice = W.reshape(p, n, rank)[:, i, :]
                    M = V_slice * W_slice # (p, rank)
                    # Solve M^T M u = M^T 1
                    # Actually minimize || M^T u - 1 ||^2
                    # M M^T u = M 1
                    A_mat = M @ M.T # (p, p)
                    b_vec = M @ np.ones(p) # (p,)
                    try:
                        u_sol = np.linalg.solve(A_mat, b_vec)
                    except:
                        u_sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    new_U[i*m+j, :] = u_sol
            
            # Update V
            new_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    # U slice: U[:, j] ? No, U is (n*m, rank). Index is i*m+j.
                    # U slice: U.reshape(n, m, rank)[:, j, :] shape (n, rank)
                    # W slice: W.reshape(p, n, rank)[k, :, :] shape (n, rank)
                    U_slice = U.reshape(n, m, rank)[:, j, :]
                    W_slice = W.reshape(p, n, rank)[k, :, :]
                    M = U_slice * W_slice # (n, rank)
                    A_mat = M @ M.T # (n, n)
                    b_vec = M @ np.ones(n) # (n,)
                    try:
                        v_sol = np.linalg.solve(A_mat, b_vec)
                    except:
                        v_sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    new_V[j*p+k, :] = v_sol
            
            # Update W
            new_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # U slice: U.reshape(n, m, rank)[i, :, :] shape (m, rank)
                    # V slice: V.reshape(m, p, rank)[:, k, :] shape (m, rank)
                    U_slice = U.reshape(n, m, rank)[i, :, :]
                    V_slice = V.reshape(m, p, rank)[:, k, :]
                    M = U_slice * V_slice # (m, rank)
                    A_mat = M @ M.T # (m, m)
                    b_vec = M @ np.ones(m) # (m,)
                    try:
                        w_sol = np.linalg.solve(A_mat, b_vec)
                    except:
                        w_sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    new_W[k*n+i, :] = w_sol
            
            U, V, W = new_U, new_V, new_W
            
            # Check convergence
            # Compute residual
            # Res = 1 - sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # We can compute one random sample or full sum
            if it % 10 == 0:
                res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            term = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            res += (1.0 - term)**2
                if res < tol:
                    return (U, V, W), res
        
        # Final residual check
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    term = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    res += (1.0 - term)**2
        return (U, V, W), res

    # Try ranks
    for rank in [19, 20, 21, 22]:
        if time.time() > deadline:
            break
        result, res = solve_als(rank)
        if result is not None and res < 1e-4:
            # Verify
            U, V, W = result
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        term = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if abs(term - 1.0) > 1e-4:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if valid:
                best_rank = rank
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break # Found minimal rank in this set
    
    if best_result is None:
        # Fallback to schoolbook
        best_result = {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
        
    return best_result
