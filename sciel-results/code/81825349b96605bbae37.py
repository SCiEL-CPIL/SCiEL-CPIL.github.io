import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety

    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school

    def verify(U, V, W):
        # Check integrality (half-integers)
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2)):
                return False
            if np.any(np.abs(mat) > 100):
                return False
        
        # Check decomposition
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        try:
            T_calc = np.einsum('ijr,jkr,kri->ijk', U_r, V_r, W_r)
            if not np.allclose(T_calc, 1.0, atol=1e-6):
                return False
        except Exception:
            return False
        return True

    def solve_rank(R, max_restarts=20):
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Initialize randomly
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Alternating minimization
            for iter in range(1000):
                if time.time() > deadline:
                    return None
                
                # Update U
                # For each i, j: U[i,j,:] . sum_k (V[j,k,:] * W[k,i,:]) = 1
                for i in range(n):
                    for j in range(m):
                        S = np.zeros(R)
                        for k in range(p):
                            S += V[j*p+k, :] * W[k*n+i, :]
                        if np.linalg.norm(S) > 1e-12:
                            U[i*m+j, :] = S / np.dot(S, S)
                        else:
                            U[i*m+j, :] = 0 # Fallback, shouldn't happen often
                
                # Update V
                # For each j, k: V[j,k,:] . sum_i (U[i,j,:] * W[k,i,:]) = 1
                for j in range(m):
                    for k in range(p):
                        S = np.zeros(R)
                        for i in range(n):
                            S += U[i*m+j, :] * W[k*n+i, :]
                        if np.linalg.norm(S) > 1e-12:
                            V[j*p+k, :] = S / np.dot(S, S)
                        else:
                            V[j*p+k, :] = 0
                
                # Update W
                # For each k, i: W[k,i,:] . sum_j (U[i,j,:] * V[j,k,:]) = 1
                for k in range(p):
                    for i in range(n):
                        S = np.zeros(R)
                        for j in range(m):
                            S += U[i*m+j, :] * V[j*p+k, :]
                        if np.linalg.norm(S) > 1e-12:
                            W[k*n+i, :] = S / np.dot(S, S)
                        else:
                            W[k*n+i, :] = 0
                
                # Check convergence or error
                # Calculate error
                U_r = U.reshape(n, m, R)
                V_r = V.reshape(m, p, R)
                W_r = W.reshape(p, n, R)
                T_calc = np.einsum('ijr,jkr,kri->ijk', U_r, V_r, W_r)
                err = np.max(np.abs(T_calc - 1.0))
                
                if err < 1e-9:
                    # Round to half-integers
                    U_int = np.round(U * 2) / 2.0
                    V_int = np.round(V * 2) / 2.0
                    W_int = np.round(W * 2) / 2.0
                    
                    if verify(U_int, V_int, W_int):
                        return U_int, V_int, W_int
            
            # If we didn't get a verified solution, try local search on the last iteration's rounded values?
            # Or just restart.
            pass
        
        return None

    # Try ranks from 19 upwards
    # The rank of 2x3x4 is 19.
    for R in range(19, 25):
        if time.time() > deadline:
            break
        
        # Increase restarts for lower ranks if time permits, but keep it bounded
        result = solve_rank(R)
        if result:
            U, V, W = result
            best_rank = R
            best_U, best_V, best_W = U, V, W
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
