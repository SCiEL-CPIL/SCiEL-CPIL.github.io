import time
import random
import numpy as np
import scipy.optimize

DIMENSIONS = [2, 4, 8]

def _compute_error(U, V, W, A_list, B_list, n, m, p):
    """
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    A_list, B_list: lists of (n,m) and (m,p) matrices
    """
    total_err = 0.0
    R = U.shape[1]
    for A, B in zip(A_list, B_list):
        C_true = A @ B
        C_dec = np.zeros((n, p))
        for c in range(R):
            # Extract vectors
            u_vec = U[:, c].reshape(n, m)
            v_vec = V[:, c].reshape(m, p)
            w_vec = W[:, c].reshape(p, n)
            
            # Term: (A @ u_vec) * (v_vec @ B) * w_vec
            # A @ u_vec is (n, 1) if u_vec is (n, m) -> wait.
            # u_vec is (n, m). A is (n, m).
            # The formula is sum_j A[i,j] u[i,j,c]. This is element-wise product sum.
            # Let's check the index mapping again.
            # U[i*m+j, c] is u[i,j,c].
            # So term for i is sum_j A[i,j] * U[i*m+j, c].
            # This is NOT A @ U_c. U_c is a matrix.
            # Let U_c be the matrix where U_c[i,j] = U[i*m+j, c].
            # Then sum_j A[i,j] U_c[i,j] is the dot product of row i of A and row i of U_c?
            # No, it's sum_j A[i,j] U_c[i,j]. This is the element-wise product of A and U_c summed over j.
            # This is equivalent to (A * U_c).sum(axis=1) for each i.
            
            # Let's re-read the verify function in the prompt to be sure.
            # factor_A[i] = sum(A[i, j] * U[i * m + j][c] for j in range(m))
            # Yes, it is element-wise product summed over j.
            
            # So, factor_A[i] = np.sum(A * U_c, axis=1)[i]
            # Similarly, factor_B[k] = sum_j V[j*p+k, c] * B[j,k].
            # V[j*p+k, c] is V_c[j,k].
            # factor_B[k] = np.sum(V_c * B, axis=0)[k]
            
            # C_dec[i,k] += factor_A[i] * factor_B[k] * W[k*n+i, c]
            
        # To speed up, we can vectorize the c-loop?
        # R is small (~60). Looping in Python is okay for 20 trials.
        
        # Optimization:
        # A_flat = A.reshape(-1)
        # But let's stick to the loop for clarity and correctness first.
        
        for c in range(R):
            U_c = U[:, c].reshape(n, m)
            V_c = V[:, c].reshape(m, p)
            W_c = W[:, c].reshape(p, n)
            
            # f_A[i] = sum_j A[i,j] U_c[i,j]
            f_A = np.sum(A * U_c, axis=1) # Shape (n,)
            
            # f_B[k] = sum_j V_c[j,k] B[j,k]
            f_B = np.sum(V_c * B, axis=0) # Shape (p,)
            
            # Add to C_dec
            # C_dec[i,k] += f_A[i] * f_B[k] * W_c[k,i]
            # Outer product of f_A and f_B, multiplied element-wise by W_c
            C_dec += np.outer(f_A, f_B) * W_c
        
        total_err += np.sum((C_dec - C_true) ** 2)
        
    return total_err / len(A_list)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Pre-generate test matrices
    rng = np.random.default_rng(42)
    A_tests = [rng.random((n, m)) for _ in range(20)]
    B_tests = [rng.random((m, p)) for _ in range(20)]
    
    # Initialize Schoolbook
    def get_schoolbook(R):
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = get_schoolbook(schoolbook_rank)
    best_rank = schoolbook_rank
    best_err = 0.0

    # Try lower ranks
    for target_rank in range(schoolbook_rank - 1, 50, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try a few random restarts for this rank
        for restart in range(5):
            if time.time() > deadline:
                break
                
            # Initialize with noise around 0 or random
            U_init = np.random.randn(n * m, target_rank) * 0.1
            V_init = np.random.randn(m * p, target_rank) * 0.1
            W_init = np.random.randn(p * n, target_rank) * 0.1
            
            # Flatten for optimizer
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def unpack(x):
                size_U = n * m * target_rank
                size_V = m * p * target_rank
                size_W = p * n * target_rank
                U_f = x[:size_U].reshape(n * m, target_rank)
                V_f = x[size_U:size_U+size_V].reshape(m * p, target_rank)
                W_f = x[size_U+size_V:].reshape(p * n, target_rank)
                return U_f, V_f, W_f
            
            def objective(x):
                U_f, V_f, W_f = unpack(x)
                return _compute_error(U_f, V_f, W_f, A_tests, B_tests, n, m, p)
            
            # Optimize
            try:
                res = scipy.optimize.minimize(objective, x0, method='Nelder-Mead', 
                                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < 1e-8:
                    U_sol, V_sol, W_sol = unpack(res.x)
                    # Verify with more trials
                    A_ver = [rng.random((n, m)) for _ in range(50)]
                    B_ver = [rng.random((m, p)) for _ in range(50)]
                    err = _compute_error(U_sol, V_sol, W_sol, A_ver, B_ver, n, m, p)
                    if err < 1e-9:
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        best_rank = target_rank
                        best_err = err
                        found = True
                        print(f"Found valid rank {target_rank}")
                        break
            except Exception as e:
                pass
                
        if found:
            break

    # If no lower rank found, return schoolbook
    if best_rank == schoolbook_rank:
        U, V, W = get_schoolbook(schoolbook_rank)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
