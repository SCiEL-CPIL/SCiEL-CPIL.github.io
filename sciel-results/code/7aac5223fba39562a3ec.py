"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    deadline = time.time() + 140  # 10s safety margin for 150s limit

    # Build the matrix multiplication tensor
    # T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This is equivalent to: for all i,j,k: sum_c U_mat[i,c] * V_mat[j,c] * W_mat[k,c] = 1
    # where U_mat is (n, rank), V_mat is (m, rank), W_mat is (p, rank)

    def verify_full(U_full, V_full, W_full, tol=1e-10):
        """Verify the full decomposition"""
        # U_full: (n*m, rank), V_full: (m*p, rank), W_full: (p*n, rank)
        # C[i,k] = sum_j sum_c U_full[i*m+j, c] * V_full[j*p+k, c] * W_full[k*n+i, c]
        # Should equal A[i,:] @ B[:,k] for arbitrary A, B
        # Check: for each i, k, the coefficient of A[i,j]*B[j,k] should be delta_{i,j,k}
        # Actually, the tensor T[i,j,k] = 1 means every term A[i,j]*B[j,k] contributes to C[i,k]
        # So we need: sum_c U_full[i*m+j, c] * V_full[j*p+k, c] * W_full[k*n+i, c] = 1 for all i,j,k
        
        U_mat = U_full.reshape(n, m, -1)  # (n, m, rank)
        V_mat = V_full.reshape(m, p, -1)  # (m, p, rank)
        W_mat = W_full.reshape(p, n, -1)  # (p, n, rank)
        
        # T[i,j,k] = sum_c U_mat[i,j,c] * V_mat[j,k,c] * W_mat[k,i,c]
        # Note: W_mat[k,i,c] = W_full[k*n+i, c]
        T = np.einsum('ijc,jkc,kic->ijk', U_mat, V_mat, W_mat)
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)

    def verify_small(U_mat, V_mat, W_mat, tol=1e-10):
        """Verify using the small matrices U_mat (n,rank), V_mat (m,rank), W_mat (p,rank)"""
        # T[i,j,k] = sum_c U_mat[i,c] * V_mat[j,c] * W_mat[k,c]
        T = np.einsum('ic,jc,kc->ijk', U_mat, V_mat, W_mat)
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=tol)

    def small_to_full(U_mat, V_mat, W_mat, rank):
        """Convert small matrices to full format"""
        U_full = np.zeros((n * m, rank))
        for i in range(n):
            for j in range(m):
                U_full[i * m + j] = U_mat[i]
        
        V_full = np.zeros((m * p, rank))
        for j in range(m):
            for k in range(p):
                V_full[j * p + k] = V_mat[j]
        
        W_full = np.zeros((p * n, rank))
        for k in range(p):
            for i in range(n):
                W_full[k * n + i] = W_mat[k]
        
        return U_full, V_full, W_full

    def try_random_decomposition(rank, n_trials=200, seed=42):
        """Try to find a rank-r decomposition using random initialization"""
        rng = np.random.RandomState(seed)
        for trial in range(n_trials):
            # Random initialization
            U_mat = rng.randn(n, rank)
            V_mat = rng.randn(m, rank)
            W_mat = rng.randn(p, rank)
            
            if verify_small(U_mat, V_mat, W_mat, tol=1e-8):
                return U_mat, V_mat, W_mat
        return None

    def try_grad_descent(rank, max_iter=5000, lr=0.01, seed=42):
        """Try gradient descent to find a rank-r decomposition"""
        rng = np.random.RandomState(seed)
        U_mat = rng.randn(n, rank)
        V_mat = rng.randn(m, rank)
        W_mat = rng.randn(p, rank)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute T[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
            T = np.einsum('ic,jc,kc->ijk', U_mat, V_mat, W_mat)
            target = np.ones((n, m, p))
            err = T - target
            
            # Gradient: dL/dU[i,c] = sum_{j,k} err[i,j,k] * V[j,c] * W[k,c]
            # dL/dU[i,c] = sum_{j,k} err[i,j,k] * V[j,c] * W[k,c]
            # = sum_j (sum_k err[i,j,k] * W[k,c]) * V[j,c]
            # = (err @ W[:,c]) @ V[:,c] ... let me think more carefully
            
            # dL/dU[i,c] = sum_{j,k} err[i,j,k] * V[j,c] * W[k,c]
            # Let E[i,j,k] = err[i,j,k]
            # dU[i,c] = sum_{j,k} E[i,j,k] * V[j,c] * W[k,c]
            # = sum_j V[j,c] * (sum_k E[i,j,k] * W[k,c])
            # = sum_j V[j,c] * (E[i,j,:] @ W[:,c])
            
            # More efficiently:
            # dU[i,c] = sum_{j,k} E[i,j,k] * V[j,c] * W[k,c]
            # Let's compute: for each c, dU[:,c] = sum_{j,k} E[:,:,k] * V[j,c] * W[k,c]
            # = sum_k W[k,c] * (E @ V[:,c]) ... no
            
            # dU[i,c] = sum_{j,k} E[i,j,k] * V[j,c] * W[k,c]
            # = sum_j V[j,c] * sum_k E[i,j,k] * W[k,c]
            
            # Let's use einsum:
            # dU[i,c] = einsum('ijk,jc,kc->ic', err, V_mat, W_mat)
            # dV[j,c] = einsum('ijk,ic,kc->jc', err, U_mat, W_mat)
            # dW[k,c] = einsum('ijk,ic,jc->kc', err, U_mat, V_mat)
            
            dU = np.einsum('ijk,jc,kc->ic', err, V_mat, W_mat)
            dV = np.einsum('ijk,ic,kc->jc', err, U_mat, W_mat)
            dW = np.einsum('ijk,ic,jc->kc', err, U_mat, V_mat)
            
            # Gradient step
            U_mat -= lr * dU
            V_mat -= lr * dV
            W_mat -= lr * dW
            
            # Check convergence
            if np.max(np.abs(err)) < 1e-10:
                if verify_small(U_mat, V_mat, W_mat, tol=1e-10):
                    return U_mat, V_mat, W_mat
        
        if verify_small(U_mat, V_mat, W_mat, tol=1e-10):
            return U_mat, V_mat, W_mat
        return None

    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(6, 30, 20) = 30
    lower_bound = max(n * m, m * p, p * n)
    
    best_result = None
    
    # Try random search for ranks below schoolbook
    for rank in range(lower_bound, rank_schoolbook):
        if time.time() > deadline:
            break
        
        result = try_random_decomposition(rank, n_trials=50, seed=rank)
        if result is not None:
            best_result = (rank, result)
            break
    
    # Try gradient descent for ranks below schoolbook
    if best_result is None:
        for rank in range(lower_bound, min(rank_schoolbook, lower_bound + 10)):
            if time.time() > deadline:
                break
            
            result = try_grad_descent(rank, max_iter=2000, lr=0.005, seed=rank)
            if result is not None:
                best_result = (rank, result)
                break
    
    if best_result is not None:
        rank, (U_mat, V_mat, W_mat) = best_result
        U_full, V_full, W_full = small_to_full(U_mat, V_mat, W_mat, rank)
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback to schoolbook
    rank = rank_schoolbook
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
# EVOLVE-BLOCK-END
