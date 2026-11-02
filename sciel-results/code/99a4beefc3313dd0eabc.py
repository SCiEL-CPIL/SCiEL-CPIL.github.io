"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_nm = n * m  # 9
    rank_mp = m * p  # 9
    rank_pn = p * n  # 9
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin before 150s kill
    
    def verify_decomposition(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        n, m, p = DIMENSIONS
        rank_mat = U_mat.shape[1]
        
        # Generate random test matrices
        np.random.seed(12345)
        for trial in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            # Compute C using the decomposition
            C_comp = np.zeros((n, p))
            for r in range(rank_mat):
                # Extract linear forms
                u_vec = U_mat[:, r]  # length n*m
                v_vec = V_mat[:, r]  # length m*p
                w_vec = W_mat[:, r]  # length p*n
                
                # The contribution of rank-r term:
                # For each output (i,k), the coefficient is w_vec[k*n+i]
                # The scalar from A is sum_{i,j} A[i,j] * u_vec[i*m+j]
                # The scalar from B is sum_{j,k} B[j,k] * v_vec[j*p+k]
                # But this isn't quite right for matrix mult tensor...
                
                # Correct interpretation:
                # The tensor T[a,b,c] where a=(i,j), b=(j',k), c=(i',k')
                # T[a,b,c] = 1 if i=i', j=j', k=k' (schoolbook)
                # Decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                
                # So for specific A, B:
                # C[i,k] = sum_{j} A[i,j] * B[j,k]
                # = sum_{j} sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # Let's compute this properly
                for i in range(n):
                    for k in range(p):
                        val = 0.0
                        for j in range(m):
                            for r in range(rank_mat):
                                val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                        C_comp[i, k] = val
                        if abs(C_comp[i, k] - C_true[i, k]) > tol:
                            return False
        return True
    
    def verify_decomposition_fast(U_mat, V_mat, W_mat, tol=1e-10):
        """Faster verification using tensor contraction."""
        n, m, p = DIMENSIONS
        rank_mat = U_mat.shape[1]
        
        # Build the tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # and check it equals the matrix multiplication tensor
        
        # The matrix multiplication tensor: T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k)
        # i.e., T[i*m+j, j*p+k, k*n+i] = 1, all else 0
        
        np.random.seed(12345)
        for trial in range(3):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            # C_comp[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * V[j*p+k,r])
            
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for r in range(rank_mat):
                        # sum_j U[i*m+j, r] * V[j*p+k, r]
                        s = 0.0
                        for j in range(m):
                            s += U_mat[i*m+j, r] * V_mat[j*p+k, r]
                        C_comp[i, k] += W_mat[k*n+i, r] * s
            
            if np.max(np.abs(C_comp - C_true)) > tol:
                return False
        return True
    
    # Start with schoolbook decomposition (rank 27)
    rank_schoolbook = n * m * p  # 27
    U_sb = np.zeros((rank_nm, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((rank_mp, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((rank_pn, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try to find lower rank decompositions using optimization
    # The key insight: we can try rank 26, 25, etc.
    # Use a smooth optimization approach to minimize the tensor reconstruction error
    
    def try_rank(rank_target, max_iters=500, tol=1e-10):
        """Try to find a rank-target decomposition using gradient descent."""
        if rank_target >= best_rank:
            return None
        
        # Initialize with random values
        np.random.seed(rank_target * 1000 + 42)
        U = np.random.randn(rank_nm, rank_target) * 0.1
        V = np.random.randn(rank_mp, rank_target) * 0.1
        W = np.random.randn(rank_pn, rank_target) * 0.1
        
        # Use Adam optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute gradient of reconstruction error
            # Loss = ||C_comp - C_true||^2 for random A, B
            # C_comp[i,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            
            np.random.seed(it * 7 + 123)
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_true = A @ B
            
            # Forward pass
            C_comp = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(rank_target):
                            C_comp[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for k in range(p):
                    err = C_comp[i, k] - C_true[i, k]
                    for j in range(m):
                        for r in range(rank_target):
                            dU[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                            dV[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                            dW[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
            
            # Adam update
            for name, param, grad, m_arr, v_arr in [
                ('U', U, dU, m_U, v_U),
                ('V', V, dV, m_V, v_V),
                ('W', W, dW, m_W, v_W)
            ]:
                m_arr[:] = beta1 * m_arr + (1 - beta1) * grad
                v_arr[:] = beta2 * v_arr + (1 - beta2) * grad**2
                m_hat = m_arr / (1 - beta1**(it+1))
                v_hat = v_arr / (1 - beta2**(it+1))
                param[:] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Check convergence
            if np.max(np.abs(C_comp - C_true)) < tol:
                # Verify with multiple random matrices
                if verify_decomposition_fast(U, V, W, tol):
                    return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from 26 down to 20
    for rank_target in range(best_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(rank_target, max_iters=200)
        if result is not None:
            best_rank = rank_target
            best_result = {
                "U": result["U"].tolist(),
                "V": result["V"].tolist(),
                "W": result["W"].tolist()
            }
    
    # Also try a few more random initializations for the best found rank
    if time.time() < deadline - 10:
        for attempt in range(3):
            if time.time() > deadline:
                break
            result = try_rank(best_rank, max_iters=300)
            if result is not None:
                best_result = {
                    "U": result["U"].tolist(),
                    "V": result["V"].tolist(),
                    "W": result["W"].tolist()
                }
    
    return best_result
# EVOLVE-BLOCK-END
