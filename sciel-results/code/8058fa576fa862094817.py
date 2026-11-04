import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
        start_time = time.time()
        timeout = 120.0
        
        n, m, p = DIMENSIONS
        
        def schoolbook():
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
            return {"U": U, "V": V, "W": W}

        def verify(U, V, W):
            # Check if U, V, W form a valid decomposition
            # U: (nm, r), V: (mp, r), W: (pn, r)
            # Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
            # for all i, j, k
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(len(U[0])):
                            val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                        if abs(val - 1.0) > 1e-9:
                            return False
            return True

        def try_rank_optimized(r):
            # Fast search for rank r
            # Use random initialization and gradient descent or ALS
            # Since we need half-integers, we solve for reals then round/check.
            
            # Time budget for this rank
            time_limit = 10.0
            start = time.time()
            
            for seed in range(5):
                if time.time() - start > time_limit:
                    break
                np.random.seed(100 + seed)
                
                U = np.random.randn(n * m, r)
                V = np.random.randn(m * p, r)
                W = np.random.randn(p * n, r)
                
                # Normalize to prevent divergence
                U /= np.linalg.norm(U)
                V /= np.linalg.norm(V)
                W /= np.linalg.norm(W)
                
                # ALS with early stopping
                for step in range(50):
                    # Solve W
                    for k in range(p):
                        for i in range(n):
                            # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                            A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                            # We want A @ W_col == 1
                            try:
                                W_col, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                                W[k*n+i, :] = W_col
                            except:
                                return None
                                
                    # Solve U
                    for i in range(n):
                        for j in range(m):
                            A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                            try:
                                U_col, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                                U[i*m+j, :] = U_col
                            except:
                                return None
                                
                    # Solve V
                    for j in range(m):
                        for k in range(p):
                            A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                            try:
                                V_col, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                                V[j*p+k, :] = V_col
                            except:
                                return None
                                
                    # Check loss
                    loss = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                                loss += (val - 1.0)**2
                    if loss < 1e-12:
                        # Try to round to half-integers
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        if verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                            return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                        
                        # If rounding fails, maybe scale?
                        # The task requires integer/half-integer.
                        # If the real solution is valid but not half-integer, 
                        # we might need to find an equivalent half-integer solution.
                        # This is a hard problem. 
                        # However, for small ranks, if a solution exists, 
                        # it's often simple.
                        pass

            return None

        # Try rank 71
        if time.time() - start_time < 10:
            res = try_rank_optimized(71)
            if res:
                return res

        # Try rank 70
        if time.time() - start_time < 20:
            res = try_rank_optimized(70)
            if res:
                return res

        # Fall back to schoolbook
        return schoolbook()
