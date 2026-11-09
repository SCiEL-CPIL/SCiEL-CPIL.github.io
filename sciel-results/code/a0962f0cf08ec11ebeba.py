import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    # Prepare T tensors
    # T is a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0
                
    T1 = T.reshape(n * m, m * p * p * n)
    T2 = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T3 = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)

    def try_rank(R, max_iters=200, restarts=5):
        for _ in range(restarts):
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            for it in range(max_iters):
                # Solve for U: U @ M_VW = T1
                # M_VW[:, r] = vec(V[:, r] * W[:, r])
                # V[:, r] is m*p x 1, W[:, r] is p*n x 1
                # Outer product is (m*p) x (p*n)
                # vec is (m*p * p*n) x 1
                M_VW = np.zeros((m * p * p * n, R))
                for r in range(R):
                    M_VW[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                U, res, rank, sv = np.linalg.lstsq(M_VW.T, T1.T, rcond=None)
                U = U.T # U is R x n*m, we need n*m x R
                
                # Solve for V: V @ M_UW = T2
                M_UW = np.zeros((n * m * p * n, R))
                for r in range(R):
                    M_UW[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                V, res, rank, sv = np.linalg.lstsq(M_UW.T, T2.T, rcond=None)
                V = V.T
                
                # Solve for W: W @ M_UV = T3
                M_UV = np.zeros((n * m * m * p, R))
                for r in range(R):
                    M_UV[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                W, res, rank, sv = np.linalg.lstsq(M_UV.T, T3.T, rcond=None)
                W = W.T
                
                # Check residual
                # Reconstruct T and compare
                # T_recon[alpha, beta, gamma] = sum_r U[alpha, r] V[beta, r] W[gamma, r]
                # This is slow to do explicitly for every iteration.
                # Instead, compute norm of T1 - U @ M_VW
                recon1 = U @ M_VW
                err1 = np.linalg.norm(recon1.T - T1.T)
                if err1 < 1e-6:
                    # Check if we can round to half-integers
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    
                    # Verify rounded solution
                    M_VW_r = np.zeros((m * p * p * n, R))
                    for r in range(R):
                        M_VW_r[:, r] = np.outer(V_round[:, r], W_round[:, r]).flatten()
                    recon1_r = U_round @ M_VW_r
                    err_r = np.linalg.norm(recon1_r.T - T1.T)
                    
                    if err_r < 1e-9:
                        return U_round, V_round, W_round
                    else:
                        # If rounding fails, maybe keep floats? 
                        # Task requires integer/half-integer. 
                        # If we can't round, this rank might not have a rational solution 
                        # or we need to find a better local minimum.
                        # For now, if exact float works but round doesn't, 
                        # we might have to reject this solution.
                        pass
            
        return None

    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    # Try ranks lower than schoolbook
    for R in range(rank_school - 1, 27, -1):
        if time.time() > deadline:
            break
        res = try_rank(R)
        if res is not None:
            U, V, W = res
            # Convert to lists of lists as per interface? 
            # Interface: {'U', 'V', 'W'}: factor matrices
            # The seed program returns lists of lists.
            # Let's return numpy arrays or lists? 
            # "factor matrices of shapes ..."
            # The seed returns lists. I will return lists to be safe.
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
