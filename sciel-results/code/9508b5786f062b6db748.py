"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The multiplication tensor T has indices (i, j, k) where:
    - i in [0, n) = [0, 3) : row of first matrix
    - j in [0, m) = [0, 3) : column of first matrix / row of second matrix  
    - k in [0, p) = [0, 7) : column of second matrix
    
    T[i,j,k] = 1 if the product C[i,k] = sum_j A[i,j]*B[j,k]
    
    The decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Interface: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    With W following the trace convention W[k*n+i, r]
    """
    start_time = time.time()
    deadline = 145.0  # Leave safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as baseline
    rank_schoolbook = n * m * p  # 63
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = [row[:] for row in U_school]
    best_V = [row[:] for row in V_school]
    best_W = [row[:] for row in W_school]
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        r = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U_mat[i, c] * V_mat[j, c] * W_mat[k, c] for c in range(r))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def convert_to_interface(U_mat, V_mat, W_mat):
        """Convert from (n x r), (m x r), (p x r) to interface format."""
        n, m, p = DIMENSIONS
        r = U_mat.shape[1]
        U_list = [[0] * r for _ in range(n * m)]
        V_list = [[0] * r for _ in range(m * p)]
        W_list = [[0] * r for _ in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    U_list[i * m + j][c] = U_mat[i, c]
        
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    V_list[j * p + k][c] = V_mat[j, c]
        
        for k in range(p):
            for i in range(n):
                for c in range(r):
                    W_list[k * n + i][c] = W_mat[k, c]
        
        return U_list, V_list, W_list

    def try_rounding(U_sol, V_sol, W_sol, tol=1e-9):
        """
        Try multiple rounding strategies to find an exact half-integer solution.
        Returns (U, V, W) if found, else None.
        """
        n, m, p = DIMENSIONS
        r = U_sol.shape[1]
        
        # Strategy 1: Round to nearest half-integer
        candidates = []
        
        # Try rounding each matrix independently to {0, ±1/2, ±1, ±3/2, ±2}
        def round_to_set(mat, step=0.5):
            return np.round(mat / step) * step
        
        # Try a few different rounding approaches
        for u_step in [0.5, 1.0]:
            for v_step in [0.5, 1.0]:
                for w_step in [0.5, 1.0]:
                    U_r = np.round(U_sol / u_step) * u_step
                    V_r = np.round(V_sol / v_step) * v_step
                    W_r = np.round(W_sol / w_step) * w_step
                    
                    # Verify
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_r[i], V_r[j] * W_r[k])
                                if abs(val - 1.0) > tol:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        return U_r, V_r, W_r
        
        # Strategy 2: Local search - perturb individual entries
        U_r = np.round(U_sol * 2) / 2.0
        V_r = np.round(V_sol * 2) / 2.0
        W_r = np.round(W_sol * 2) / 2.0
        
        # Check if already good
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_r[i], V_r[j] * W_r[k])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < tol:
            return U_r, V_r, W_r
        
        # Try local search: adjust individual entries by ±0.5 or ±1
        for attempt in range(100):
            if time.time() > deadline - 2:
                break
            
            # Pick a random entry to adjust
            mat_choice = np.random.choice(3)  # 0=U, 1=V, 2=W
            if mat_choice == 0:
                idx = (np.random.randint(n), np.random.randint(r))
                old_val = U_r[idx]
                delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                U_r[idx] = old_val + delta
            elif mat_choice == 1:
                idx = (np.random.randint(m), np.random.randint(r))
                old_val = V_r[idx]
                delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                V_r[idx] = old_val + delta
            else:
                idx = (np.random.randint(p), np.random.randint(r))
                old_val = W_r[idx]
                delta = np.random.choice([-1.0, -0.5, 0.5, 1.0])
                W_r[idx] = old_val + delta
            
            # Check if improved
            max_err_new = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_r[i], V_r[j] * W_r[k])
                        max_err_new = max(max_err_new, abs(val - 1.0))
            
            if max_err_new >= max_err and attempt > 10:
                # Revert
                if mat_choice == 0:
                    U_r[idx] = old_val
                elif mat_choice == 1:
                    V_r[idx] = old_val
                else:
                    W_r[idx] = old_val
            else:
                max_err = max_err_new
            
            if max_err < tol:
                return U_r, V_r, W_r
        
        return None

    def als_solve(U, V, W, r, iterations=30):
        """
        Alternating Least Squares to minimize sum (1 - sum_c U[i,c]V[j,c]W[k,c])^2
        """
        n, m, p = DIMENSIONS
        for it in range(iterations):
            if time.time() > deadline - 5:
                break
                
            # Solve for U
            A_U = np.zeros((m * p, r))
            for j in range(m):
                for k in range(p):
                    A_U[j * p + k, :] = V[j, :] * W[k, :]
            
            for i in range(n):
                U[i, :] = np.linalg.lstsq(A_U, np.ones(m * p), rcond=None)[0]
            
            # Solve for V
            A_V = np.zeros((n * p, r))
            for i in range(n):
                for k in range(p):
                    A_V[i * p + k, :] = U[i, :] * W[k, :]
            
            for j in range(m):
                V[j, :] = np.linalg.lstsq(A_V, np.ones(n * p), rcond=None)[0]
                
            # Solve for W
            A_W = np.zeros((n * m, r))
            for i in range(n):
                for j in range(m):
                    A_W[i * m + j, :] = U[i, :] * V[j, :]
            
            for k in range(p):
                W[k, :] = np.linalg.lstsq(A_W, np.ones(n * m), rcond=None)[0]
                
        return U, V, W

    np.random.seed(42)
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Try a focused range
    
    for r in range(21, 63):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations
        n_init = 5
        for init in range(n_init):
            if time.time() > deadline - 10:
                break
            
            # Structured initialization: use the schoolbook pattern as a guide
            U_init = np.zeros((n, r))
            V_init = np.zeros((m, r))
            W_init = np.zeros((p, r))
            
            # Fill with small random values plus some structure
            for c in range(r):
                # Each rank-1 component should contribute to some (i,j,k) pattern
                i_c = c % n
                j_c = (c // n) % m
                k_c = (c // (n * m)) % p
                
                U_init[i_c, c] = np.random.uniform(0.5, 1.5)
                V_init[j_c, c] = np.random.uniform(0.5, 1.5)
                W_init[k_c, c] = np.random.uniform(0.5, 1.5)
                
                # Add some cross-terms
                if c < n * m * p:
                    i_extra = c % n
                    j_extra = (c // n) % m
                    U_init[i_extra, c] += np.random.uniform(0, 0.3)
                    V_init[j_extra, c] += np.random.uniform(0, 0.3)
            
            # Also try pure random
            if init >= 3:
                U_init = np.random.randn(n, r) * 0.3 + 0.7
                V_init = np.random.randn(m, r) * 0.3 + 0.7
                W_init = np.random.randn(p, r) * 0.3 + 0.7
            
            U_sol, V_sol, W_sol = als_solve(U_init.copy(), V_init.copy(), W_init.copy(), r, iterations=25)
            
            # Check error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U_sol[i], V_sol[j] * W_sol[k])
                        err += (val - 1.0) ** 2
                        
            if err < 1e-4:
                # Try rounding
                result = try_rounding(U_sol, V_sol, W_sol)
                if result is not None:
                    U_rounded, V_rounded, W_rounded = result
                    # Final verification
                    if verify_decomposition(U_rounded, V_rounded, W_rounded):
                        U_list, V_list, W_list = convert_to_interface(U_rounded, V_rounded, W_rounded)
                        best_rank = r
                        best_U = U_list
                        best_V = V_list
                        best_W = W_list
                        break
        
        if best_rank <= r:
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
