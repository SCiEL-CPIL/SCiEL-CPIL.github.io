import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Schoolbook solution
    schoolbook_rank = n * m * p
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def to_list(mat):
        return mat.tolist()

    # The correct condition for matrix multiplication decomposition:
    # For all i, k, j, j2:
    #   sum_r U[i*m+j, r] * V[j2*p+k, r] * W[k*n+i, r] = delta_{j,j2}
    
    # This is equivalent to: for each (i,k), the matrix
    #   M_{i,k} = sum_r W[k*n+i, r] * (U_i[:, r] * V_k[:, r]^T) = I_m
    # where U_i is the block of U for row i (m x r), V_k is the block of V for column k (m x r)

    def check_solution(U, V, W, r):
        """Check if the decomposition satisfies the identity condition."""
        err = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for j2 in range(m):
                        s = 0.0
                        for r_idx in range(r):
                            s += U[i*m+j, r_idx] * V[j2*p+k, r_idx] * W[k*n+i, r_idx]
                        target = 1.0 if j == j2 else 0.0
                        err += (s - target) ** 2
        return err

    def check_solution_fast(U, V, W, r):
        """Faster check using numpy."""
        err = 0.0
        for i in range(n):
            for k in range(p):
                # U_i: (m, r), V_k: (m, r), W_ik: (r,)
                U_i = U[i*m:(i+1)*m, :]
                V_k = V[:, k].reshape(m, -1).T if False else V[np.arange(m)*p+k].reshape(m, r)
                W_ik = W[k*n+i, :]
                
                # M_ik[j, j2] = sum_r U_i[j, r] * V_k[j2, r] * W_ik[r]
                # = sum_r W_ik[r] * U_i[:, r] * V_k[:, r]^T
                # This is a weighted sum of outer products
                M_ik = np.zeros((m, m))
                for r_idx in range(r):
                    if W_ik[r_idx] != 0:
                        M_ik += W_ik[r_idx] * np.outer(U_i[:, r_idx], V_k[:, r_idx])
                
                # Compare with identity
                err += np.sum((M_ik - np.eye(m)) ** 2)
        return err

    # Try to find a better rank using gradient descent
    # We'll try ranks from low to high, with a few restarts each
    
    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = schoolbook_rank
    
    # Known lower bounds and potential ranks to try
    # For 2x3x7, schoolbook is 42.
    # Let's try to find rank < 42.
    
    # We'll use a smooth optimization approach.
    # Variables: U (nm x r), V (mp x r), W (pn x r)
    # Objective: sum_{i,k,j,j2} (sum_r U[i*m+j,r] V[j2*p+k,r] W[k*n+i,r] - delta_{j,j2})^2
    
    # For efficiency, we'll use numpy and try to converge.
    
    for r in range(1, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        # Try a few restarts for this rank
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            np.random.seed(42 + r * 100 + restart)
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Gradient descent
            best_err = check_solution_fast(U, V, W, r)
            no_improve = 0
            lr = 0.01
            
            for step in range(200):
                if time.time() > deadline - 2:
                    break
                
                # Compute gradients approximately
                # For each (i,k,j,j2), the residual is:
                # res = sum_r U[i*m+j,r] V[j2*p+k,r] W[k*n+i,r] - delta_{j,j2}
                # Gradient w.r.t U[i*m+j, r] = res * V[j2*p+k, r] * W[k*n+i, r]
                # Gradient w.r.t V[j2*p+k, r] = res * U[i*m+j, r] * W[k*n+i, r]
                # Gradient w.r.t W[k*n+i, r] = res * U[i*m+j, r] * V[j2*p+k, r]
                
                # We'll accumulate gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for k in range(p):
                        U_i = U[i*m:(i+1)*m, :]
                        V_k = V[np.arange(m)*p+k].reshape(m, r)
                        W_ik = W[k*n+i, :]
                        
                        # M_ik[j, j2] = sum_r U_i[j,r] V_k[j2,r] W_ik[r]
                        M_ik = np.zeros((m, m))
                        for r_idx in range(r):
                            M_ik += W_ik[r_idx] * np.outer(U_i[:, r_idx], V_k[:, r_idx])
                        
                        # Residual
                        R_ik = M_ik - np.eye(m)
                        
                        # Gradients
                        for j in range(m):
                            for j2 in range(m):
                                res = R_ik[j, j2]
                                if abs(res) > 1e-10:
                                    for r_idx in range(r):
                                        grad_U[i*m+j, r_idx] += res * V_k[j2, r_idx] * W_ik[r_idx]
                                        grad_V[j2*p+k, r_idx] += res * U_i[j, r_idx] * W_ik[r_idx]
                                        grad_W[k*n+i, r_idx] += res * U_i[j, r_idx] * V_k[j2, r_idx]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                err = check_solution_fast(U, V, W, r)
                if err < best_err - 1e-8:
                    best_err = err
                    no_improve = 0
                    # Adaptive learning rate
                    if step % 50 == 0 and step > 0:
                        lr = max(0.001, lr * 0.5)
                else:
                    no_improve += 1
                    if no_improve > 30:
                        break
                
                if best_err < 1e-10:
                    break
            
            # Check if we found a good solution
            if best_err < 1e-6:
                # Try rounding to half-integers
                for scale in [2, 4, 8, 16]:
                    U_round = np.round(U * scale) / scale
                    V_round = np.round(V * scale) / scale
                    W_round = np.round(W * scale) / scale
                    
                    err_round = check_solution_fast(U_round, V_round, W_round, r)
                    if err_round < 1e-9:
                        best_U, best_V, best_W = U_round, V_round, W_round
                        best_rank = r
                        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    
    # If we didn't find a better solution, return schoolbook
    return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
