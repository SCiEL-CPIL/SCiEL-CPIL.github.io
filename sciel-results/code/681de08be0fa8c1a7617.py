import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Fallback: Rank 42 decomposition
    def get_rank_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(i_start, j_start, k_start, offset):
            # M1
            col = offset
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            # M2
            col = offset + 1
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] -= 1
            # M3
            col = offset + 2
            U[(i_start+0)*m + (j_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            # M4
            col = offset + 3
            U[(i_start+1)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+0)][col] -= 1
            W[(k_start+0)*n + (i_start+0)][col] += 1
            W[(k_start+0)*n + (i_start+1)][col] += 1
            # M5
            col = offset + 4
            U[(i_start+0)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+1)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] -= 1
            W[(k_start+1)*n + (i_start+0)][col] += 1
            # M6
            col = offset + 5
            U[(i_start+1)*m + (j_start+0)][col] += 1
            U[(i_start+0)*m + (j_start+0)][col] -= 1
            V[(j_start+0)*p + (k_start+0)][col] += 1
            V[(j_start+0)*p + (k_start+1)][col] += 1
            W[(k_start+1)*n + (i_start+1)][col] += 1
            # M7
            col = offset + 6
            U[(i_start+0)*m + (j_start+1)][col] += 1
            U[(i_start+1)*m + (j_start+1)][col] -= 1
            V[(j_start+1)*p + (k_start+0)][col] += 1
            V[(j_start+1)*p + (k_start+1)][col] += 1
            W[(k_start+0)*n + (i_start+0)][col] += 1

        add_strassen_block(0, 0, 0, 0)
        add_strassen_block(0, 0, 2, 7)
        add_strassen_block(0, 0, 4, 14)
        add_strassen_block(0, 2, 0, 21)
        add_strassen_block(0, 2, 2, 28)
        add_strassen_block(0, 2, 4, 35)
        
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(rank):
        if time.time() > deadline - 10:
            return None
        
        U_size = n * m
        V_size = m * p
        W_size = p * n
        total_vars = (U_size + V_size + W_size) * rank
        
        # Residual function
        def residuals(params):
            U = params[:U_size*rank].reshape(U_size, rank)
            V = params[U_size*rank:(U_size+V_size)*rank].reshape(V_size, rank)
            W = params[(U_size+V_size)*rank:].reshape(W_size, rank)
            
            # Compute tensor sum
            # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # We need T[i,j,k] == 1
            
            res = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                        res.append(s - 1.0)
            return np.array(res)

        best_solution = None
        best_cost = np.inf
        
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            # Scale matters. Let's use small random values.
            x0 = np.random.uniform(-1, 1, size=total_vars)
            
            try:
                # Use 'trf' method for robustness
                sol = least_squares(residuals, x0, method='trf', max_nfev=1000)
                if sol.cost < best_cost:
                    best_cost = sol.cost
                    best_solution = sol.x
                    
                    # If cost is very low, check for integer/half-integer snap
                    if best_cost < 1e-8:
                        U = best_solution[:U_size*rank].reshape(U_size, rank)
                        V = best_solution[U_size*rank:(U_size+V_size)*rank].reshape(V_size, rank)
                        W = best_solution[(U_size+V_size)*rank:].reshape(W_size, rank)
                        
                        # Snap to half-integers
                        U_snap = np.round(U * 2) / 2
                        V_snap = np.round(V * 2) / 2
                        W_snap = np.round(W * 2) / 2
                        
                        # Verify snap
                        if verify(U_snap.tolist(), V_snap.tolist(), W_snap.tolist()):
                            return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                            
            except Exception:
                continue
                
        return None

    # Try lower ranks
    for rank in [40, 38, 36]:
        result = try_rank(rank)
        if result is not None:
            return result
            
    # Fallback
    return get_rank_42()
