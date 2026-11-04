"""Matrix multiplication 3,4,8 rank-minimal decomposition search."""
DIMENSIONS = [3, 4, 8]
import time
import numpy as np

def _verify_float(U, V, W, n, m, p, tol=1e-8):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Verify with exact arithmetic."""
    from fractions import Fraction
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    U, V, W, rank = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    np.random.seed(42)
    
    # Search for lower rank decompositions
    # For each target rank, try to find U, V, W such that
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    
    # Approach: fix U and V randomly, solve for W, verify.
    # Iterate with different random U, V.
    
    for target_rank in range(35, 97):
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
        
        max_trials = 200
        found = False
        
        for trial in range(max_trials):
            if time.time() > deadline - 15:
                break
            
            # Random U, V
            U_r = np.random.randn(n*m, target_rank)
            V_r = np.random.randn(m*p, target_rank)
            
            # Solve for W: for each (i,k), solve A[i,k] * W[k*n+i,:] = ones(m)
            # where A[i,k,j,c] = U_r[i*m+j,c] * V_r[j*p+k,c]
            W_sol = np.zeros((p*n, target_rank))
            feasible = True
            
            for i in range(n):
                for k in range(p):
                    A_mat = np.zeros((m, target_rank))
                    for j in range(m):
                        A_mat[j, :] = U_r[i*m+j, :] * V_r[j*p+k, :]
                    
                    # Solve A_mat * w = ones(m)
                    try:
                        # Use lstsq for robustness
                        sol, residuals, rnk, sv = np.linalg.lstsq(A_mat, np.ones(m), rcond=1e-12)
                        if np.linalg.norm(A_mat @ sol - np.ones(m)) > 1e-8:
                            feasible = False
                            break
                        W_sol[k*n+i, :] = sol
                    except:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if not feasible:
                continue
            
            # Verify
            if _verify_float(U_r, V_r, W_sol, n, m, p, tol=1e-7):
                # Round to half-integers
                U_round = np.round(2 * U_r) / 2
                V_round = np.round(2 * V_r) / 2
                W_round = np.round(2 * W_sol) / 2
                
                # Check if rounded version works
                if _verify_float(U_round, V_round, W_round, n, m, p, tol=1e-9):
                    U_list = [[float(x) for x in row] for row in U_round]
                    V_list = [[float(x) for x in row] for row in V_round]
                    W_list = [[float(x) for x in row] for row in W_round]
                    best_U, best_V, best_W, best_rank = U_list, V_list, W_list, target_rank
                    found = True
                    break
        
        if found:
            continue
    
    if best_rank >= 96:
        U, V, W, rank = _schoolbook(n, m, p)
        best_U, best_V, best_W, best_rank = U, V, W, rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
