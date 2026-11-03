"""Program for rank-minimal tensor decomposition of 3x3 by 3x6 matrix multiplication."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    # Schoolbook decomposition for the full 3x3x6 problem (Rank 54)
    def schoolbook_3x3x6():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Function to verify a decomposition for 3x3x3
    def verify_333(U, V, W, tol=1e-6):
        max_err = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u_idx = i*3 + j
                    v_idx = j*3 + k
                    w_idx = k*3 + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    err = abs(1.0 - val)
                    if err > max_err:
                        max_err = err
        return max_err

    # Function to try to find a low rank decomposition for 3x3x3
    def find_333_decomposition(max_rank=25, min_rank=23):
        best_U = None
        best_V = None
        best_W = None
        best_rank = 27 # Fallback
        
        for target_rank in range(min_rank, max_rank + 1):
            if time.time() > deadline - 20:
                break
            
            success = False
            for attempt in range(50):
                if time.time() > deadline - 20:
                    break
                
                # Random initialization
                U = np.random.randn(9, target_rank)
                V = np.random.randn(9, target_rank)
                W = np.random.randn(9, target_rank)
                
                def params_to_matrices(params):
                    U = params[:9*target_rank].reshape(9, target_rank)
                    V = params[9*target_rank:18*target_rank].reshape(9, target_rank)
                    W = params[18*target_rank:].reshape(9, target_rank)
                    return U, V, W

                def matrices_to_params(U, V, W):
                    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

                def objective(params):
                    U, V, W = params_to_matrices(params)
                    # Compute error
                    err = 0.0
                    for i in range(3):
                        for j in range(3):
                            for k in range(3):
                                u_idx = i*3 + j
                                v_idx = j*3 + k
                                w_idx = k*3 + i
                                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                                err += (1.0 - val)**2
                    return np.sqrt(err) # Use L2 norm for least_squares

                x0 = matrices_to_params(U, V, W)
                
                try:
                    res = least_squares(objective, x0, method='trf', max_nfev=10000)
                    if res.cost < 1e-8:
                        U_opt, V_opt, W_opt = params_to_matrices(res.x)
                        # Try to snap to half-integers
                        U_snap = np.round(U_opt * 2) / 2
                        V_snap = np.round(V_opt * 2) / 2
                        W_snap = np.round(W_opt * 2) / 2
                        
                        # Verify snapped
                        err = verify_333(U_snap, V_snap, W_snap)
                        if err < 1e-9:
                            best_U, best_V, best_W = U_snap, V_snap, W_snap
                            best_rank = target_rank
                            success = True
                            break
                except Exception:
                    continue
            
            if success:
                break
        
        return best_U, best_V, best_W, best_rank

    # Lift 3x3x3 decomposition to 3x3x6
    def lift_to_336(U3, V3, W3, rank3):
        rank6 = 2 * rank3
        U = np.zeros((9, rank6))
        V = np.zeros((18, rank6))
        W = np.zeros((18, rank6))
        
        for s in range(rank3):
            # Term 2s (k in 0..2)
            idx1 = 2*s
            for i in range(3):
                for j in range(3):
                    U[i*3+j, idx1] = U3[i*3+j, s]
            for j in range(3):
                for k in range(3):
                    V[j*6+k, idx1] = V3[j*3+k, s]
            for k in range(3):
                for i in range(3):
                    W[k*3+i, idx1] = W3[k*3+i, s]
            
            # Term 2s+1 (k in 3..5)
            idx2 = 2*s + 1
            for i in range(3):
                for j in range(3):
                    U[i*3+j, idx2] = U3[i*3+j, s]
            for j in range(3):
                for k in range(3):
                    V[j*6+k+3, idx2] = V3[j*3+k, s]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3+i, idx2] = W3[k*3+i, s]
        
        return U, V, W

    # Verify full 3x3x6 decomposition
    def verify_336(U, V, W, tol=1e-9):
        rank = U.shape[1]
        max_err = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(6):
                    u_idx = i*3 + j
                    v_idx = j*6 + k
                    w_idx = k*3 + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    err = abs(1.0 - val)
                    if err > max_err:
                        max_err = err
        return max_err

    # Main logic
    try:
        U3, V3, W3, rank3 = find_333_decomposition(max_rank=25, min_rank=23)
        
        if rank3 < 27:
            U, V, W = lift_to_336(U3, V3, W3, rank3)
            err = verify_336(U, V, W)
            if err < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    except Exception as e:
        pass

    # Fallback to schoolbook
    U, V, W = schoolbook_3x3x6()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
