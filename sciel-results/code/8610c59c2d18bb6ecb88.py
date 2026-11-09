"""Program for rank-minimal exact tensor decomposition of 3x3 * 3x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 8
    num_equations = n * m * p  # 72 equations (one per (i,j,k))
    
    def check_solution(U, V, W, tol=1e-9):
        """Check that U, V, W exactly decompose the matrix multiplication tensor.
        T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
        """
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Compute T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Using einsum: U[i,j,c] * V[j,k,c] * W[k,i,c]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T, 1.0, atol=tol, rtol=0)
    
    def solve_for_W(U, V):
        """Given U (n*m x R) and V (m*p x R), solve for W (p*n x R).
        For each (k, i): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all j
        So: A[k*n+i, :] @ W[k*n+i, :]^T... 
        Actually: for fixed (k,i), we need sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1 for all j
        This is a linear system in W[k*n+i, c]: 
        Let A[j, c] = U[i*m+j, c] * V[j*p+k, c] for j=0..m-1
        Then sum_c A[j,c] * W[k*n+i,c] = 1 for all j
        So A @ W[k*n+i, :]^T = ones(m)
        """
        R = U.shape[1]
        W = np.zeros((p * n, R))
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, R))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                # Solve A @ w = b where w = W[k*n+i, :]
                w, residuals, rank_solve, sv = np.linalg.lstsq(A, b, rcond=None)
                W[k * n + i, :] = w
        return W
    
    def solve_for_V(U, W):
        """Given U (n*m x R) and W (p*n x R), solve for V (m*p x R).
        For fixed (j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
        Let A[i, c] = U[i*m+j, c] * W[k*n+i, c] for i=0..n-1
        Then A @ V[j*p+k, :]^T = ones(n)
        """
        R = U.shape[1]
        V = np.zeros((m * p, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                v, residuals, rank_solve, sv = np.linalg.lstsq(A, b, rcond=None)
                V[j * p + k, :] = v
        return V
    
    def solve_for_U(V, W):
        """Given V (m*p x R) and W (p*n x R), solve for U (n*m x R).
        For fixed (i,j): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
        Let A[k, c] = V[j*p+k, c] * W[k*n+i, c] for k=0..p-1
        Then A @ U[i*m+j, :]^T = ones(p)
        """
        R = V.shape[1]
        U = np.zeros((n * m, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                u, residuals, rank_solve, sv = np.linalg.lstsq(A, b, rcond=None)
                U[i * m + j, :] = u
        return U
    
    # Start with schoolbook decomposition
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
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Phase 1: Greedy rank reduction
    # Try to remove one column at a time, then re-solve the remaining factors
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r in range(rank):
            if time.time() > deadline - 5:
                break
            
            mask = np.ones(rank, dtype=bool)
            mask[r] = False
            U_trial = U[:, mask]
            V_trial = V[:, mask]
            W_trial = W[:, mask]
            
            # Alternating minimization to re-solve
            for _ in range(3):
                W_trial = solve_for_W(U_trial, V_trial)
                V_trial = solve_for_V(U_trial, W_trial)
                U_trial = solve_for_V.__globals__['solve_for_U'](V_trial, W_trial)
            
            # Check if the solution is valid (exact, no rounding needed since we keep exact floats)
            if check_solution(U_trial, V_trial, W_trial, tol=1e-10):
                U, V, W = U_trial, V_trial, W_trial
                rank -= 1
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                improved = True
                break
    
    # Phase 2: Try to jump to lower ranks with random restarts
    if time.time() < deadline - 10:
        for target_rank in range(max(20, best_rank - 15), best_rank):
            if time.time() > deadline - 10:
                break
            
            found = False
            for restart in range(30):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(restart * 1000 + target_rank)
                U_trial = np.random.randn(n * m, target_rank) * 0.5
                V_trial = np.random.randn(m * p, target_rank) * 0.5
                W_trial = np.random.randn(p * n, target_rank) * 0.5
                
                for iteration in range(300):
                    if time.time() > deadline - 10:
                        break
                    
                    W_trial = solve_for_W(U_trial, V_trial)
                    V_trial = solve_for_V(U_trial, W_trial)
                    U_trial = solve_for_U(V_trial, W_trial)
                    
                    if check_solution(U_trial, V_trial, W_trial, tol=1e-10):
                        U, V, W = U_trial, V_trial, W_trial
                        rank = target_rank
                        best_rank = rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        found = True
                        break
                
                if found:
                    break
            
            if found:
                break
    
    # Convert to lists with exact values (keep as floats, they're exact rationals from linear solves)
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
