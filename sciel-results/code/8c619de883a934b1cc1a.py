import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]
DEADLINE = 145.0  # 5s margin from 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    
    # Schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((N, rank_sb))
    V_sb = np.zeros((M, rank_sb))
    W_sb = np.zeros((P, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    def reconstruct(U, V, W):
        """Compute the tensor from U, V, W using the trace convention."""
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return T
    
    def check_error(U, V, W):
        """Check if the decomposition exactly matches the matrix multiplication tensor."""
        T = reconstruct(U, V, W)
        T_target = np.ones((n, m, p))
        err = np.sum((T - T_target)**2)
        return err
    
    def snap_to_half_integers(U, V, W):
        """Try to snap values to half-integers."""
        U_snap = np.round(U * 2) / 2.0
        V_snap = np.round(V * 2) / 2.0
        W_snap = np.round(W * 2) / 2.0
        return U_snap, V_snap, W_snap
    
    def snap_to_integers(U, V, W):
        """Try to snap values to integers."""
        U_snap = np.round(U)
        V_snap = np.round(V)
        W_snap = np.round(W)
        return U_snap, V_snap, W_snap
    
    def verify_exact(U, V, W, tol=1e-10):
        """Verify that the decomposition is exact."""
        err = check_error(U, V, W)
        return err < tol
    
    def try_rank_with_alternating_least_squares(R, max_iters=500, tol=1e-12, seed=42):
        """
        Use alternating least squares to find a rank-R decomposition.
        The key insight: for matrix multiplication, we can fix one factor
        and solve for the others linearly.
        """
        np.random.seed(seed)
        
        # Initialize randomly
        U = np.random.randn(N, R) * 0.5
        V = np.random.randn(M, R) * 0.5
        W = np.random.randn(P, R) * 0.5
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > DEADLINE - 30.0:
                break
                
            # Update U: for each row i*m+j, we need
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is a linear system in U[i*m+j, :]
            # Let A[i*m+j] be the (p x R) matrix where A[i*m+j][k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Then we need A[i*m+j] @ U[i*m+j, :] = ones(p)
            
            # Update V: for each row j*p+k, we need
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
            # Let B[j*p+k] be the (n x R) matrix where B[j*p+k][i, r] = U[i*m+j, r] * W[k*n+i, r]
            # Then we need B[j*p+k] @ V[j*p+k, :] = ones(n)
            
            # Update W: for each row k*n+i, we need
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
            # Let C[k*n+i] be the (m x R) matrix where C[k*n+i][j, r] = U[i*m+j, r] * V[j*p+k, r]
            # Then we need C[k*n+i] @ W[k*n+i, :] = ones(m)
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Build A[i*m+j]: (p x R)
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    # Solve A @ U[i*m+j, :] = ones(p)
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # Build B[j*p+k]: (n x R)
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    # Solve B @ V[j*p+k, :] = ones(n)
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # Build C[k*n+i]: (m x R)
                    C = np.zeros((m, R))
                    for j in range(m):
                        C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    # Solve C @ W[k*n+i, :] = ones(m)
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            err = check_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
        
        # Try to snap to half-integers or integers
        if best_err < 1e-8:
            U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
            if verify_exact(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap, best_err
            
            U_snap, V_snap, W_snap = snap_to_integers(best_U, best_V, best_W)
            if verify_exact(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap, best_err
        
        return None
    
    def try_rank_with_gradient_descent(R, max_iters=1000, lr=0.01, seed=42):
        """
        Use gradient descent to find a rank-R decomposition.
        """
        np.random.seed(seed)
        
        U = np.random.randn(N, R) * 0.1
        V = np.random.randn(M, R) * 0.1
        W = np.random.randn(P, R) * 0.1
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > DEADLINE - 30.0:
                break
                
            # Compute gradients using automatic differentiation via finite differences
            # or use the chain rule
            T = reconstruct(U, V, W)
            T_target = np.ones((n, m, p))
            err = np.sum((T - T_target)**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-12:
                break
            
            # Compute gradients
            # d(err)/d(U[i*m+j, r]) = 2 * sum_{i',j',k'} (T[i',j',k'] - 1) * d(T[i',j',k'])/d(U[i*m+j, r])
            # d(T[i',j',k'])/d(U[i*m+j, r]) = V[j'*p+k', r] * W[k'*n+i', r] if i==i', j==j' else 0
            # So d(err)/d(U[i*m+j, r]) = 2 * (T[i,j,:] - 1) dot (V[j*p+0:r, r] * W[0*n+i:r, r] * ... )
            
            # More efficiently:
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            # Gradient w.r.t. U
            # d(err)/d(U[i,j,r]) = 2 * sum_k (T[i,j,k] - 1) * V[j,k,r] * W[k,i,r]
            grad_U = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    for r in range(R):
                        grad_U[i * m + j, r] = 2.0 * np.sum(
                            (T[i, j, :] - 1.0) * V_3d[j, :, r] * W_3d[:, i, r]
                        )
            
            # Gradient w.r.t. V
            # d(err)/d(V[j,k,r]) = 2 * sum_i (T[i,j,k] - 1) * U[i,j,r] * W[k,i,r]
            grad_V = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    for r in range(R):
                        grad_V[j * p + k, r] = 2.0 * np.sum(
                            (T[:, j, k] - 1.0) * U_3d[:, j, r] * W_3d[k, :, r]
                        )
            
            # Gradient w.r.t. W
            # d(err)/d(W[k,i,r]) = 2 * sum_j (T[i,j,k] - 1) * U[i,j,r] * V[j,k,r]
            grad_W = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    for r in range(R):
                        grad_W[k * n + i, r] = 2.0 * np.sum(
                            (T[i, :, k] - 1.0) * U_3d[i, :, r] * V_3d[:, k, r]
                        )
            
            # Update with learning rate
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Reduce learning rate
            lr *= 0.999
        
        # Try to snap to half-integers or integers
        if best_err < 1e-8:
            U_snap, V_snap, W_snap = snap_to_half_integers(best_U, best_V, best_W)
            if verify_exact(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap, best_err
            
            U_snap, V_snap, W_snap = snap_to_integers(best_U, best_V, best_W)
            if verify_exact(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap, best_err
        
        return None
    
    # Try ranks from 49 down to 40
    # For 3x4x5, the minimum rank is known to be at most 49 (from Strassen-type results)
    # but could be lower.
    for R in range(49, 39, -1):
        if time.time() > DEADLINE - 40.0:
            break
        
        # Try alternating least squares with a few seeds
        found = False
        for seed in range(3):
            if time.time() > DEADLINE - 30.0:
                break
            
            sol = try_rank_with_alternating_least_squares(R, max_iters=300, seed=seed)
            if sol:
                U, V, W, err = sol
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try gradient descent with a few seeds
        for seed in range(2):
            if time.time() > DEADLINE - 30.0:
                break
            
            sol = try_rank_with_gradient_descent(R, max_iters=500, seed=seed)
            if sol:
                U, V, W, err = sol
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
