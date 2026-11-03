import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # 120s limit + safety margin, but prompt says 150s kill. 145 is safe.
    
    n, m, p = DIMENSIONS
    
    # 1. Baseline Rank 55 Solution
    def get_rank_55():
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        # Part 1: Strassen for A[:, 0:2] * B[0:2, :]
        # A is 2x2, B is 2x10.
        # Split B into 5 blocks of 2 columns each.
        
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            
            # M1 = (A00 + A11) * (B00 + B11)
            U[0*3 + 0][col] += 1
            U[1*3 + 1][col] += 1
            V[0*10 + k0][col] += 1
            V[1*10 + k1][col] += 1
            W[k0*2 + 0][col] += 1
            W[k1*2 + 1][col] += 1
            col += 1
            
            # M2 = (A10 + A11) * B00
            U[1*3 + 0][col] += 1
            U[1*3 + 1][col] += 1
            V[0*10 + k0][col] += 1
            W[k0*2 + 1][col] += 1
            W[k1*2 + 1][col] += -1
            col += 1
            
            # M3 = A00 * (B01 - B11)
            U[0*3 + 0][col] += 1
            V[0*10 + k1][col] += 1
            V[1*10 + k1][col] += -1
            W[k1*2 + 0][col] += 1
            W[k1*2 + 1][col] += 1
            col += 1
            
            # M4 = A11 * (B10 - B00)
            U[1*3 + 1][col] += 1
            V[1*10 + k0][col] += 1
            V[0*10 + k0][col] += -1
            W[k0*2 + 0][col] += 1
            W[k0*2 + 1][col] += 1
            col += 1
            
            # M5 = (A00 + A01) * B11
            U[0*3 + 0][col] += 1
            U[0*3 + 1][col] += 1
            V[1*10 + k1][col] += 1
            W[k0*2 + 0][col] += -1
            W[k1*2 + 0][col] += 1
            col += 1
            
            # M6 = (A10 - A00) * (B00 + B01)
            U[1*3 + 0][col] += 1
            U[0*3 + 0][col] += -1
            V[0*10 + k0][col] += 1
            V[0*10 + k1][col] += 1
            W[k1*2 + 1][col] += 1
            col += 1
            
            # M7 = (A01 - A11) * (B10 + B11)
            U[0*3 + 1][col] += 1
            U[1*3 + 1][col] += -1
            V[1*10 + k0][col] += 1
            V[1*10 + k1][col] += 1
            W[k0*2 + 0][col] += 1
            col += 1

        # Part 2: Naive for A[:, 2] * B[2, :]
        for i in range(2):
            for k in range(10):
                U[i*3 + 2][col] += 1
                V[2*10 + k][col] += 1
                W[k*2 + i][col] += 1
                col += 1
                
        return {"U": U, "V": V, "W": W}

    baseline = get_rank_55()
    
    # 2. Search for Lower Rank using ALS
    def check_residual(U, V, W):
        R = U.shape[1]
        U_mat = U.reshape(n, m, R)
        V_mat = V.reshape(m, p, R)
        W_mat = W.reshape(p, n, R)
        
        err = 0.0
        for i in range(n):
            for k in range(p):
                M_ik = np.zeros((m, m))
                W_k_i = W_mat[k, i, :]
                U_i = U_mat[i, :, :]
                V_k = V_mat[:, k, :]
                
                A = U_i * W_k_i[None, :]
                M_ik = A.T @ V_k
                
                diff = M_ik - np.eye(m)
                err += np.sum(diff**2)
        return err

    def try_als(rank, max_iters=50):
        # Initialize with small random values
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        for it in range(max_iters):
            # Update W
            W_mat = W.reshape(p, n, rank)
            U_mat = U.reshape(n, m, rank)
            V_mat = V.reshape(m, p, rank)
            
            for k in range(p):
                for i in range(n):
                    # Solve A w = b
                    # A[j,l, r] = U[i,j,r] * V[l,k,r]
                    # b[j,l] = delta(j,l)
                    A = np.zeros((m*m, rank))
                    b = np.zeros(m*m)
                    for j in range(m):
                        for l in range(m):
                            idx = j*m + l
                            A[idx, :] = U_mat[i, j, :] * V_mat[l, k, :]
                            if j == l:
                                b[idx] = 1.0
                    # Least squares
                    w_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_mat[k, i, :] = w_sol
                    
            W = W_mat.reshape(p*n, rank)
            
            # Update U
            U_mat = U.reshape(n, m, rank)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p*m, rank))
                    b = np.zeros(p*m)
                    for k in range(p):
                        for l in range(m):
                            idx = k*m + l
                            # Equation: sum_r U[i,j,r] V[l,k,r] W[k,i,r] = delta(j,l)
                            # For fixed i,j, we solve for U[i,j,:].
                            # Coefficient of U[i,j,r] is V[l,k,r] * W[k,i,r]
                            A[idx, :] = V_mat[l, k, :] * W_mat[k, i, :]
                            if j == l:
                                b[idx] = 1.0
                    u_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_mat[i, j, :] = u_sol
            U = U_mat.reshape(n*m, rank)
            
            # Update V
            V_mat = V.reshape(m, p, rank)
            for k in range(p):
                for l in range(m):
                    A = np.zeros((n*m, rank))
                    b = np.zeros(n*m)
                    for i in range(n):
                        for j in range(m):
                            idx = i*m + j
                            A[idx, :] = U_mat[i, j, :] * W_mat[k, i, :]
                            if j == l:
                                b[idx] = 1.0
                    v_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_mat[l, k, :] = v_sol
            V = V_mat.reshape(m*p, rank)
            
            # Check residual
            err = check_residual(U, V, W)
            if err < 1e-8:
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                err_r = check_residual(U_r, V_r, W_r)
                if err_r < 1e-6:
                    return {"U": U_r, "V": V_r, "W": W_r}
                else:
                    # If rounding fails, maybe return floats? 
                    # Task says "integer/half-integer". 
                    # If we can't satisfy that, we should probably not return this.
                    pass
            if time.time() - start_time > DEADLINE - 5:
                return None
                
        return None

    # Try ranks 54 down to 50
    for rank in range(54, 49, -1):
        if time.time() - start_time > DEADLINE:
            break
        sol = try_als(rank)
        if sol is not None:
            return sol
            
    return baseline

# EVOLVE-BLOCK-END
