import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # Leave 20s safety margin before 150s kill

    def build_trivial():
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

    def check_decomposition(U, V, W, rank, tol=1e-6):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        expected = 1.0 if j == jp else 0.0
                        actual = 0.0
                        for s in range(rank):
                            actual += W[k * n + i][s] * U[i * m + j][s] * V[jp * p + k][s]
                        if abs(actual - expected) > tol:
                            return False
        return True

    def try_find_decomposition(r, max_iters=2000, seed=42):
        """Try to find a rank-r decomposition using alternating minimization."""
        rng = np.random.RandomState(seed)
        
        # Initialize
        U_np = rng.randn(n * m, r) * 0.5
        V_np = rng.randn(m * p, r) * 0.5
        W_np = rng.randn(p * n, r) * 0.5
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Fix U and V, solve for W
            # For each (i,k): sum_s W[k*n+i,s] * U[i*m+j,s] * V[j'*p+k,s] = delta_{j,j'}
            # This is a linear system in W[k*n+i, :]
            W_new = np.zeros((p * n, r))
            for i in range(n):
                for k in range(p):
                    # Build the system: for each (j,j'), sum_s W[k*n+i,s] * A_s = delta_{j,j'}
                    # where A_s = U[i*m+j,s] * V[j'*p+k,s]
                    # We have m*m equations and r unknowns (W[k*n+i,0..r-1])
                    # Use least squares
                    A_mat = np.zeros((m * m, r))
                    b_vec = np.zeros(m * m)
                    idx = 0
                    for j in range(m):
                        for jp in range(m):
                            for s in range(r):
                                A_mat[idx, s] = U_np[i * m + j, s] * V_np[jp * p + k, s]
                            b_vec[idx] = 1.0 if j == jp else 0.0
                            idx += 1
                    # Solve least squares
                    W_row, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_new[k * n + i, :] = W_row
            
            # Fix V and W, solve for U
            U_new = np.zeros((n * m, r))
            for i in range(n):
                for j in range(m):
                    # For each k: sum_s W[k*n+i,s] * U[i*m+j,s] * V[j'*p+k,s] = delta_{j,j'} for all j'
                    # This gives p equations (one for each k) and r unknowns
                    A_mat = np.zeros((p, r))
                    b_vec = np.zeros(p)
                    for k in range(p):
                        for s in range(r):
                            A_mat[k, s] = W_new[k * n + i, s] * V_np[j * p + k, s]
                        b_vec[k] = 1.0  # delta_{j,j} = 1
                    # Wait, this isn't right. The constraint is for all (i,k,j,j'):
                    # sum_s W[k*n+i,s] * U[i*m+j,s] * V[j'*p+k,s] = delta_{j,j'}
                    # For fixed i,j, we need this to hold for all k and j'.
                    # So we have p*m equations.
                    A_mat = np.zeros((p * m, r))
                    b_vec = np.zeros(p * m)
                    idx = 0
                    for k in range(p):
                        for jp in range(m):
                            for s in range(r):
                                A_mat[idx, s] = W_new[k * n + i, s] * V_np[jp * p + k, s]
                            b_vec[idx] = 1.0 if j == jp else 0.0
                            idx += 1
                    U_row, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U_new[i * m + j, :] = U_row
            
            # Fix U and W, solve for V
            V_new = np.zeros((m * p, r))
            for jp in range(m):
                for k in range(p):
                    # For each i: sum_s W[k*n+i,s] * U[i*m+jp,s] * V[jp*p+k,s] = delta_{jp,j'} for all j'
                    # This gives n*m equations.
                    A_mat = np.zeros((n * m, r))
                    b_vec = np.zeros(n * m)
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for s in range(r):
                                A_mat[idx, s] = W_new[k * n + i, s] * U_new[i * m + j, s]
                            b_vec[idx] = 1.0 if jp == j else 0.0
                            idx += 1
                    V_row, residuals, rank_ls, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V_new[jp * p + k, :] = V_row
            
            # Check convergence
            diff_U = np.max(np.abs(U_new - U_np))
            diff_V = np.max(np.abs(V_new - V_np))
            diff_W = np.max(np.abs(W_new - W_np))
            U_np, V_np, W_np = U_new, V_new, W_new
            
            if diff_U < 1e-8 and diff_V < 1e-8 and diff_W < 1e-8:
                break
        
        # Check if the solution is valid
        U_list = [[float(U_np[i, s]) for s in range(r)] for i in range(n * m)]
        V_list = [[float(V_np[i, s]) for s in range(r)] for i in range(m * p)]
        W_list = [[float(W_np[i, s]) for s in range(r)] for i in range(p * n)]
        
        if check_decomposition(U_list, V_list, W_list, r, tol=1e-4):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try to find a decomposition with rank < 72
    best_result = build_trivial()
    best_rank = n * m * p
    
    # Try ranks from 71 down, but be careful about time
    for r in range(best_rank - 1, 35, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_find_decomposition(r, max_iters=1000, seed=r)
        if result is not None:
            best_result = result
            best_rank = r
            break  # Found a lower rank, stop
    
    return best_result
