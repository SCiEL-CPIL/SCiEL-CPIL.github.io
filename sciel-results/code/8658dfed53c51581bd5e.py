"""Matrix multiplication 2x5x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_exact(U, V, W, tol=1e-8):
        """Check if the decomposition is exact with integer/half-integer entries."""
        rank = len(U[0])
        # Check entries are integer or half-integer
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > tol:
                        return False
        # Check tensor equation
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def tensor_residual(U, V, W):
        """Compute max absolute residual of tensor equation."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err = abs(total - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def round_to_half_integer(mat):
        """Round a numpy matrix to nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    def try_rank(r, max_outer=5, max_inner=300):
        """Try to find a decomposition of rank r using alternating least squares."""
        if time.time() > deadline - 10:
            return None
        
        best_result = None
        best_residual = float('inf')
        
        for restart in range(max_outer):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with different scales
            scale = 1.0 + 0.5 * restart
            U = np.random.randn(n*m, r) * scale
            V = np.random.randn(m*p, r) * scale
            W = np.random.randn(p*n, r) * scale
            
            for iteration in range(max_inner):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), solve least squares
                for i in range(n):
                    for j in range(m):
                        # M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        # Want: min_x ||M @ x - 1||^2
                        M = V[j*p:(j+1)*p, :] * W[::p, i]  # shape (p, r)
                        # W[::p, i] gives W[k*n+i, :] for k=0..p-1
                        M = np.array([[V[j*p+k, c] * W[k*n+i, c] for c in range(r)] for k in range(p)])
                        b = np.ones(p)
                        try:
                            x, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                            U[i*m+j, :] = x
                        except:
                            pass
                
                # Update V: for each (j,k), solve least squares
                for j in range(m):
                    for k in range(p):
                        # M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        M = np.array([[U[i*m+j, c] * W[k*n+i, c] for c in range(r)] for i in range(n)])
                        b = np.ones(n)
                        try:
                            y, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                            V[j*p+k, :] = y
                        except:
                            pass
                
                # Update W: for each (k,i), solve least squares
                for k in range(p):
                    for i in range(n):
                        # M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        M = np.array([[U[i*m+j, c] * V[j*p+k, c] for c in range(r)] for j in range(m)])
                        b = np.ones(m)
                        try:
                            z, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                            W[k*n+i, :] = z
                        except:
                            pass
                
                # Check residual
                residual = tensor_residual(U, V, W)
                if residual < best_residual:
                    best_residual = residual
                    U_best = U.copy()
                    V_best = V.copy()
                    W_best = W.copy()
                
                if residual < 1e-10:
                    # Try rounding to half-integers
                    for U_round in [round_to_half_integer(U), 
                                     np.round(U * 4) / 4.0,
                                     np.round(U * 2) / 2.0]:
                        for V_round in [round_to_half_integer(V),
                                         np.round(V * 4) / 4.0,
                                         np.round(V * 2) / 2.0]:
                            for W_round in [round_to_half_integer(W),
                                             np.round(W * 4) / 4.0,
                                             np.round(W * 2) / 2.0]:
                                U_list = U_round.tolist()
                                V_list = V_round.tolist()
                                W_list = W_round.tolist()
                                if verify_exact(U_list, V_list, W_list):
                                    return {"U": U_list, "V": V_list, "W": W_list}
                    # If close, do a few more iterations to refine
                    continue
            
            # After inner loop, check best so far
            if best_residual < 1e-8:
                U_round = round_to_half_integer(U_best)
                V_round = round_to_half_integer(V_best)
                W_round = round_to_half_integer(W_best)
                U_list = U_round.tolist()
                V_list = V_round.tolist()
                W_list = W_round.tolist()
                if verify_exact(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from 25 upward
    best_result = None
    for r in range(25, 50):
        if time.time() > deadline - 15:
            break
        result = try_rank(r)
        if result is not None:
            best_result = result
            break
    
    if best_result is None:
        # Try a higher rank with more restarts
        for r in range(50, 65):
            if time.time() > deadline - 10:
                break
            result = try_rank(r, max_outer=3, max_inner=200)
            if result is not None:
                best_result = result
                break
    
    if best_result is None:
        return schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
