import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Precompute the target tensor unfoldings
    T1 = np.zeros((n*m, m*p * p*n))
    T2 = np.zeros((m*p, n*m * p*n))
    T3 = np.zeros((p*n, n*m * m*p))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                T1[a, b * (p*n) + c] = 1.0
                T2[b, a * (p*n) + c] = 1.0
                T3[c, a * (m*p) + b] = 1.0
                
    T1_T = T1.T
    T2_T = T2.T
    T3_T = T3.T

    def khatri_rao(A, B):
        dim1, R = A.shape
        dim2, _ = B.shape
        prod = A.T[:, :, np.newaxis] * B.T[:, np.newaxis, :]
        return prod.reshape(R, -1).T

    def check_exact(U, V, W):
        MWV = khatri_rao(W, V)
        T1_recon = U @ MWV.T
        if not np.allclose(T1_recon, T1, atol=1e-9):
            return False
        MUW = khatri_rao(U, W)
        T2_recon = V @ MUW.T
        if not np.allclose(T2_recon, T2, atol=1e-9):
            return False
        MVU = khatri_rao(V, U)
        T3_recon = W @ MVU.T
        if not np.allclose(T3_recon, T3, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        if check_exact(U_s, V_s, W_s):
            return U_s, V_s, W_s
        return None

    def snap_and_verify_int(U, V, W):
        U_s = np.round(U)
        V_s = np.round(V)
        W_s = np.round(W)
        if check_exact(U_s, V_s, W_s):
            return U_s, V_s, W_s
        return None

    def solve_als(U, V, W, max_iter=100, tol=1e-7):
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Update U
            MWV = khatri_rao(W, V)
            U = np.linalg.lstsq(MWV, T1_T, rcond=None)[0].T
            
            # Update V
            MUW = khatri_rao(U, W)
            V = np.linalg.lstsq(MUW, T2_T, rcond=None)[0].T
            
            # Update W
            MVU = khatri_rao(V, U)
            W = np.linalg.lstsq(MVU, T3_T, rcond=None)[0].T
            
            # Calculate error
            MWV = khatri_rao(W, V)
            T1_recon = U @ MWV.T
            err = np.linalg.norm(T1_recon - T1)
            
            if err < tol:
                res = snap_and_verify(U, V, W)
                if res:
                    return res, True
                res = snap_and_verify_int(U, V, W)
                if res:
                    return res, True
            
            if it > 5 and err > prev_err * 1.5:
                break
            prev_err = err
            
        return (U, V, W), False

    def polish_solution(U, V, W, max_iter=50):
        """High-precision polish pass with tighter tolerance"""
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            MWV = khatri_rao(W, V)
            U = np.linalg.lstsq(MWV, T1_T, rcond=None)[0].T
            
            MUW = khatri_rao(U, W)
            V = np.linalg.lstsq(MUW, T2_T, rcond=None)[0].T
            
            MVU = khatri_rao(V, U)
            W = np.linalg.lstsq(MVU, T3_T, rcond=None)[0].T
            
            MWV = khatri_rao(W, V)
            T1_recon = U @ MWV.T
            err = np.linalg.norm(T1_recon - T1)
            
            if err < 1e-10:
                res = snap_and_verify(U, V, W)
                if res:
                    return res, True
                res = snap_and_verify_int(U, V, W)
                if res:
                    return res, True
            
            if it > 10 and err > prev_err * 1.1:
                break
            prev_err = err
            
        return (U, V, W), False

    def try_rank(R, n_restarts=30):
        """Try to find a decomposition of rank R"""
        for r in range(n_restarts):
            if time.time() > deadline:
                return None
            
            # Different initialization strategies
            if r % 3 == 0:
                # Pure random
                U = np.random.randn(n*m, R)
                V = np.random.randn(m*p, R)
                W = np.random.randn(p*n, R)
            elif r % 3 == 1:
                # Structured: initialize with some pattern
                U = np.random.randn(n*m, R) * 0.5 + 0.5
                V = np.random.randn(m*p, R) * 0.5 + 0.5
                W = np.random.randn(p*n, R) * 0.5 + 0.5
            else:
                # Sparse-like initialization
                U = np.random.randn(n*m, R)
                V = np.random.randn(m*p, R)
                W = np.random.randn(p*n, R)
                # Zero out some entries to encourage sparsity
                mask_u = np.random.rand(n*m, R) > 0.5
                mask_v = np.random.rand(m*p, R) > 0.5
                mask_w = np.random.rand(p*n, R) > 0.5
                U = U * mask_u
                V = V * mask_v
                W = W * mask_w
            
            sol, success = solve_als(U, V, W, max_iter=50, tol=1e-6)
            if success:
                return sol
            
            # If close but not exact, try polish
            if not success:
                U_final, V_final, W_final = sol
                MWV = khatri_rao(W_final, V_final)
                T1_recon = U_final @ MWV.T
                err = np.linalg.norm(T1_recon - T1)
                if err < 0.1:
                    sol2, success2 = polish_solution(U_final, V_final, W_final)
                    if success2:
                        return sol2
        
        return None

    # Try ranks from low to high
    # Theoretical lower bound for matrix multiplication tensor rank
    # For n x m x p, lower bound is max(n*m, m*p, p*n) but that's not tight
    # For 2x5x6, known constructions exist below 60
    # Try from 25 to 55
    
    found_solution = None
    best_rank = None
    
    # First try a range of ranks
    for R in range(25, 56):
        if time.time() > deadline:
            break
        
        # More restarts for lower ranks (harder to find)
        restarts = 40 if R < 40 else 20
        sol = try_rank(R, n_restarts=restarts)
        if sol:
            found_solution = sol
            best_rank = R
            break
    
    if found_solution:
        U, V, W = found_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to Schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
