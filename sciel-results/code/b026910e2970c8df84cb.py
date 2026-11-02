import numpy as np
from scipy.optimize import nnls
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Target tensor: T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want rank-r decomposition: T = sum_{r=1}^R U_r * V_r * W_r
    # where U_r is in R^{n*m}, V_r in R^{m*p}, W_r in R^{p*n}
    #
    # The multiplication tensor for C = A * B (A: n x m, B: m x p, C: n x p)
    # has T[i, j, k] = 1 if we define C[i,k] = sum_j A[i,j] B[j,k]
    #
    # Index convention:
    # U has shape (n*m, rank) = (8, rank), indexed by (i*m + j)
    # V has shape (m*p, rank) = (32, rank), indexed by (j*p + k)
    # W has shape (p*n, rank) = (16, rank), indexed by (k*n + i)
    #
    # The constraint is: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Try to find a rank-32 decomposition using nonlinear least squares
    # with random initialization and iterative refinement
    
    def random_init(rank, seed=42):
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        return U, V, W
    
    def compute_residual(U, V, W):
        """Compute the residual: sum_r U[:,r] * V[:,r] * W[:,r] - T"""
        n, m, p = DIMENSIONS
        T = np.ones((n, m, p))
        
        # Compute the tensor reconstruction
        rec = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            # U[:, r] is (n*m,), reshape to (n, m)
            u_reshaped = U[:, r].reshape(n, m)
            # V[:, r] is (m*p,), reshape to (m, p)
            v_reshaped = V[:, r].reshape(m, p)
            # W[:, r] is (p*n,), reshape to (p, n)
            w_reshaped = W[:, r].reshape(p, n)
            
            # rec[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            for i in range(n):
                for j in range(m):
                    u_val = U[i*m + j, r]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[j*p + k, r]
                        w_val = W[k*n + i, r]
                        rec[i, j, k] += u_val * v_val * w_val
        
        return rec - T
    
    def verify(U, V, W, tol=1e-6):
        """Check if the decomposition is valid"""
        n, m, p = DIMENSIONS
        T = np.ones((n, m, p))
        rec = np.zeros((n, m, p))
        
        for r in range(U.shape[1]):
            for i in range(n):
                for j in range(m):
                    u_val = U[i*m + j, r]
                    if abs(u_val) < 1e-10:
                        continue
                    for k in range(p):
                        v_val = V[j*p + k, r]
                        w_val = W[k*n + i, r]
                        rec[i, j, k] += u_val * v_val * w_val
        
        max_err = np.max(np.abs(rec - T))
        return max_err < tol, max_err
    
    def refine(U, V, W, max_iter=100, tol=1e-10):
        """Iteratively refine the decomposition using alternating least squares"""
        n, m, p = DIMENSIONS
        T = np.ones((n, m, p))
        rank = U.shape[1]
        
        for it in range(max_iter):
            if time.time() - start_time > 100:
                break
            
            # Update U: for each (i,j) and each r, minimize sum_k (U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - T[i,j,k])^2
            for r in range(rank):
                # Fix V[:, r] and W[:, r], solve for U[:, r]
                # This is a linear problem in U[:, r]
                # sum_r' U[i*m+j,r'] * V[j*p+k,r'] * W[k*n+i,r'] = T[i,j,k]
                # For fixed r, the contribution from term r is U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # But we need all terms to sum to T.
                
                # Actually, let's use a different approach: project out the current estimate
                # and solve the residual as a linear system.
                
                pass  # This is getting complex; let's use a simpler approach
        
        return U, V, W
    
    # Strategy: Use the known rank-56 Strassen solution as a starting point,
    # then try to reduce the rank by projecting onto a lower-dimensional space.
    
    # First, build the rank-56 Strassen solution
    U_strassen = np.zeros((n * m, 56), dtype=float)
    V_strassen = np.zeros((m * p, 56), dtype=float)
    W_strassen = np.zeros((p * n, 56), dtype=float)
    
    # Strassen components for 2x2 matrix multiplication
    U_local = np.zeros((4, 7), dtype=float)
    V_local = np.zeros((4, 7), dtype=float)
    W_local = np.zeros((4, 7), dtype=float)
    
    # r=0 (M1)
    U_local[0][0] = 1; U_local[3][0] = 1
    V_local[0][0] = 1; V_local[3][0] = 1
    W_local[0][0] = 1
    W_local[3][0] = 1
    
    # r=1 (M2)
    U_local[2][1] = 1; U_local[3][1] = 1
    V_local[0][1] = 1
    W_local[2][1] = 1
    W_local[3][1] = -1
    
    # r=2 (M3)
    U_local[0][2] = 1
    V_local[1][2] = 1; V_local[3][2] = -1
    W_local[1][2] = 1
    W_local[3][2] = 1
    
    # r=3 (M4)
    U_local[3][3] = 1
    V_local[2][3] = 1; V_local[0][3] = -1
    W_local[0][3] = 1
    W_local[2][3] = 1
    
    # r=4 (M5)
    U_local[0][4] = 1; U_local[1][4] = 1
    V_local[3][4] = 1
    W_local[0][4] = -1
    W_local[1][4] = 1
    
    # r=5 (M6)
    U_local[2][5] = 1; U_local[0][5] = -1
    V_local[0][5] = 1; V_local[1][5] = 1
    W_local[3][5] = 1
    
    # r=6 (M7)
    U_local[1][6] = 1; U_local[3][6] = -1
    V_local[2][6] = 1; V_local[3][6] = 1
    W_local[0][6] = 1
    
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7
            
            a_map = [0]*4
            if s == 0:
                a_map[0] = 0*4 + 0
                a_map[1] = 0*4 + 1
                a_map[2] = 1*4 + 0
                a_map[3] = 1*4 + 1
            else:
                a_map[0] = 0*4 + 2
                a_map[1] = 0*4 + 3
                a_map[2] = 1*4 + 2
                a_map[3] = 1*4 + 3
                
            b_map = [0]*4
            if s == 0:
                b_map[0] = 0*8 + (2*j)
                b_map[1] = 0*8 + (2*j+1)
                b_map[2] = 1*8 + (2*j)
                b_map[3] = 1*8 + (2*j+1)
            else:
                b_map[0] = 2*8 + (2*j)
                b_map[1] = 2*8 + (2*j+1)
                b_map[2] = 3*8 + (2*j)
                b_map[3] = 3*8 + (2*j+1)
                
            c_map = [0]*4
            c_map[0] = (2*j) * 2 + 0
            c_map[1] = (2*j+1) * 2 + 0
            c_map[2] = (2*j) * 2 + 1
            c_map[3] = (2*j+1) * 2 + 1

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la][l]
                    if val != 0:
                        U_strassen[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb][l]
                    if val != 0:
                        V_strassen[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc][l]
                    if val != 0:
                        W_strassen[c_map[lc], r] += val
    
    # Now try to find a rank-32 decomposition
    # Use random initialization and gradient descent on the residual
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for trial in range(5):
        if time.time() - start_time > 80:
            break
            
        seed = 42 + trial
        U, V, W = random_init(32, seed=seed)
        
        # Gradient descent
        lr = 0.001
        for it in range(200):
            if time.time() - start_time > 100:
                break
            
            # Compute gradient
            n, m, p = DIMENSIONS
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # For each (i,j,k), the residual is:
            # err[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1
            # d(err[i,j,k])/dU[i*m+j,r] = V[j*p+k,r] * W[k*n+i,r]
            # d(err[i,j,k])/dV[j*p+k,r] = U[i*m+j,r] * W[k*n+i,r]
            # d(err[i,j,k])/dW[k*n+i,r] = U[i*m+j,r] * V[j*p+k,r]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        T_val = 1.0
                        
                        err = 0.0
                        for r in range(32):
                            err += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                        err -= T_val
                        
                        for r in range(32):
                            dU[u_idx, r] += err * V[v_idx, r] * W[w_idx, r]
                            dV[v_idx, r] += err * U[u_idx, r] * W[w_idx, r]
                            dW[w_idx, r] += err * U[u_idx, r] * V[v_idx, r]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check error
            if it % 50 == 0 or it == 199:
                valid, err = verify(U, V, W, tol=0.1)
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-6:
                    break
        
        # Check final
        valid, err = verify(U, V, W, tol=1e-6)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # If we found a valid rank-32 solution, use it
    if best_err < 1e-6:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to rank-56 Strassen solution
    return {"U": U_strassen.tolist(), "V": V_strassen.tolist(), "W": W_strassen.tolist()}
# EVOLVE-BLOCK-END
