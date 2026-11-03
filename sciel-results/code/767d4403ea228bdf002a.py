"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Schoolbook decomposition (baseline)
    rank_school = n * m * p  # 4*4*7 = 112
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # The tensor for matrix multiplication C = A * B where A is n×m, B is m×p, C is p×n
    # T[i, j, k] = 1 if the (i,k) entry of C depends on A[i,j] * B[j,k]
    # In our indexing: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition satisfies: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Wait, that's not right. Let me reconsider.
    
    # Actually, the standard tensor decomposition for matrix multiplication:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
    # where T_{i,j,k} = 1 (since C_{i,k} = sum_j A_{i,j} B_{j,k})
    
    # So we need: for each (i, j, k) in [0,n)×[0,m)×[0,p),
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's verify schoolbook:
    # For term (i,j,k) at column col: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # So the product is 1, and only this term contributes to T[i,j,k]. Correct.
    
    # Strategy: Try to find a lower-rank decomposition using optimization.
    # We'll use a randomized approach with local search.
    
    # For small ranks, we can try to solve the system of equations.
    # The number of constraints is n*m*p = 112.
    # Unknowns: rank * (n*m + m*p + p*n) = rank * (16 + 28 + 28) = rank * 72
    
    # For rank r, we have 72r unknowns and 112 constraints (one per (i,j,k) triple).
    # This is a system of polynomial equations (cubic in the unknowns).
    
    # Let's try a gradient-based approach with random restarts for various ranks.
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is valid."""
        # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] ≈ 1
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False, val, (i, j, k)
        return True, None, None
    
    # Try ranks from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) / something... 
    # Actually, the rank must be at least max(n*m, m*p, p*n) / min(n,m,p) roughly
    # For 4×4×7: lower bound is at least max(16, 28, 28) = 28, but realistically higher.
    # Known results: rank(4,4,4) ≤ 22, rank(4,4,7) is unknown but likely > 28.
    
    # Let's try to find a decomposition with rank < 112 using optimization.
    
    def try_rank(r, max_iter=500, lr=0.01, n_restarts=3):
        """Try to find a rank-r decomposition using gradient descent."""
        global best_rank, best_U, best_V, best_W
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                return False
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # For small r, we might need larger initial values
            if r < 40:
                U = np.random.randn(n*m, r) * 1.0
                V = np.random.randn(m*p, r) * 1.0
                W = np.random.randn(p*n, r) * 1.0
            
            for it in range(max_iter):
                if time.time() > deadline - 5:
                    break
                
                # Compute gradients
                # Loss = sum_{i,j,k} (sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1)^2
                # 
                # dL/dU[i*m+j, r] = 2 * sum_{k} (val[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]
                # dL/dV[j*p+k, r] = 2 * sum_{i} (val[i,j,k] - 1) * U[i*m+j, r] * W[k*n+i, r]
                # dL/dW[k*n+i, r] = 2 * sum_{j} (val[i,j,k] - 1) * U[i*m+j, r] * V[j*p+k, r]
                
                # Compute val[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                val = np.zeros((n, m, p))
                for r_idx in range(r):
                    # U[:, r_idx] is (n*m,), V[:, r_idx] is (m*p,), W[:, r_idx] is (p*n,)
                    u = U[:, r_idx].reshape(n, m)
                    v = V[:, r_idx].reshape(m, p)
                    w = W[:, r_idx].reshape(p, n)
                    # val[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val[i,j,k] += u[i,j] * v[j,k] * w[k,i]
                
                error = val - 1.0
                loss = np.sum(error**2)
                
                # If loss is very small, check validity
                if loss < 1e-10:
                    valid, _, _ = check_decomposition(U, V, W, tol=1e-5)
                    if valid:
                        if r < best_rank:
                            best_rank = r
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W.copy()
                        return True
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            e = error[i,j,k]
                            for r_idx in range(r):
                                grad_U[i*m+j, r_idx] += 2 * e * V[j*p+k, r_idx] * W[k*n+i, r_idx]
                                grad_V[j*p+k, r_idx] += 2 * e * U[i*m+j, r_idx] * W[k*n+i, r_idx]
                                grad_W[k*n+i, r_idx] += 2 * e * U[i*m+j, r_idx] * V[j*p+k, r_idx]
                
                # Gradient descent step
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if it % 50 == 0 and it > 0:
                    new_loss = np.sum((val - 1.0)**2)
                    if new_loss > loss:
                        lr *= 0.5
                    else:
                        lr = min(lr * 1.1, 0.1)
            
            # Final check
            if time.time() < deadline - 5:
                valid, _, _ = check_decomposition(U, V, W, tol=1e-5)
                if valid:
                    if r < best_rank:
                        best_rank = r
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                    return True
        
        return False
    
    # Try ranks from low to high, but start with a reasonable guess
    # For 4×4×7, let's try ranks 30 to 80
    for r in range(30, 80, 2):
        if time.time() > deadline - 10:
            break
        if try_rank(r, max_iter=300, lr=0.005, n_restarts=2):
            # Found a valid decomposition at this rank
            # Try to go lower
            for r2 in range(28, r, -1):
                if time.time() > deadline - 10:
                    break
                if try_rank(r2, max_iter=300, lr=0.005, n_restarts=2):
                    pass  # already updated best
            break
    
    # Try some specific known-good ranks with more iterations
    for r in [32, 36, 40, 44, 48, 52, 56, 60]:
        if time.time() > deadline - 15:
            break
        if r < best_rank:
            try_rank(r, max_iter=500, lr=0.003, n_restarts=3)
    
    # Convert to list format for output
    result_U = best_U.tolist()
    result_V = best_V.tolist()
    result_W = best_W.tolist()
    
    # Round to nearest half-integer if close
    def round_half(x):
        return round(x * 2) / 2
    
    # Only round if the result is still valid after rounding
    for attempt in range(3):
        new_U = [[round_half(x) for x in row] for row in result_U]
        new_V = [[round_half(x) for x in row] for row in result_V]
        new_W = [[round_half(x) for x in row] for row in result_W]
        
        # Check if rounded version is still valid
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(new_U[i*m+j][r] * new_V[j*p+k][r] * new_W[k*n+i][r] 
                             for r in range(best_rank))
                    if abs(val - 1) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            result_U = new_U
            result_V = new_V
            result_W = new_W
            break
    
    return {"U": result_U, "V": result_V, "W": result_W}
# EVOLVE-BLOCK-END
