"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    
    def schoolbook():
        rank = schoolbook_rank
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
    
    # Convert to numpy for computation
    def to_numpy(U, V, W):
        return np.array(U, dtype=np.float64), np.array(V, dtype=np.float64), np.array(W, dtype=np.float64)
    
    def verify(U, V, W):
        """Check if decomposition is exact: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] == matmul tensor"""
        n, m, p = DIMENSIONS
        U, V, W = to_numpy(U, V, W)
        rank = U.shape[1]
        # Build the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # W is indexed as W[k*n+i, r]
        tensor = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_col = U[:, r]  # shape (n*m,)
            v_col = V[:, r]  # shape (m*p,)
            w_col = W[:, r]  # shape (p*n,)
            # Reshape u_col to (n, m), v_col to (m, p), w_col to (p, n)
            u_mat = u_col.reshape(n, m)
            v_mat = v_col.reshape(m, p)
            w_mat = w_col.reshape(p, n)
            # For each (i, j, k): u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
        # The matmul tensor should be: T[i,j,k] = 1 if this is a valid multiplication path, else 0
        # Actually for matrix multiplication C = A*B where A is n×m, B is m×p, C is n×p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor is T[i,j,k] = 1 for all valid (i,j,k)
        expected = np.ones((n, m, p), dtype=np.float64)
        return np.allclose(tensor, expected, atol=1e-8)
    
    # Try to find lower rank decompositions
    # Start from known bounds and try to improve
    best_result = None
    best_rank = schoolbook_rank
    
    # Try various ranks from low to high
    # Known: rank(2,4,6) <= 2*4*6 = 48
    # Try to find minimal rank
    for target_rank in range(12, min(schoolbook_rank, 30)):
        if time.time() > deadline - 30:
            break
        if target_rank >= best_rank:
            continue
            
        # Try multiple random initializations
        n_tries = 5
        for trial in range(n_tries):
            if time.time() > deadline - 30:
                break
                
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Simple iterative refinement
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                    
                # Compute current tensor
                tensor = np.zeros((n, m, p), dtype=np.float64)
                for r in range(target_rank):
                    u_mat = U[:, r].reshape(n, m)
                    v_mat = V[:, r].reshape(m, p)
                    w_mat = W[:, r].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                tensor[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                
                expected = np.ones((n, m, p), dtype=np.float64)
                error = np.linalg.norm(tensor - expected)
                
                if error < 1e-6:
                    # Found a valid decomposition
                    if target_rank < best_rank:
                        best_rank = target_rank
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
                
                # Gradient step (simplified)
                lr = 0.01 / (1 + it * 0.1)
                for r in range(target_rank):
                    u_mat = U[:, r].reshape(n, m)
                    v_mat = V[:, r].reshape(m, p)
                    w_mat = W[:, r].reshape(p, n)
                    
                    # Gradient for U
                    grad_U = np.zeros_like(U[:, r])
                    grad_V = np.zeros_like(V[:, r])
                    grad_W = np.zeros_like(W[:, r])
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                diff = tensor[i, j, k] - expected[i, j, k]
                                grad_U[i * m + j] += 2 * diff * v_mat[j, k] * w_mat[k, i]
                                grad_V[j * p + k] += 2 * diff * u_mat[i, j] * w_mat[k, i]
                                grad_W[k * n + i] += 2 * diff * u_mat[i, j] * v_mat[j, k]
                    
                    U[:, r] -= lr * grad_U
                    V[:, r] -= lr * grad_V
                    W[:, r] -= lr * grad_W
            
            # Check final error
            tensor = np.zeros((n, m, p), dtype=np.float64)
            for r in range(target_rank):
                u_mat = U[:, r].reshape(n, m)
                v_mat = V[:, r].reshape(m, p)
                w_mat = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            tensor[i, j, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
            expected = np.ones((n, m, p), dtype=np.float64)
            if np.linalg.norm(tensor - expected) < 1e-6:
                if target_rank < best_rank:
                    best_rank = target_rank
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
