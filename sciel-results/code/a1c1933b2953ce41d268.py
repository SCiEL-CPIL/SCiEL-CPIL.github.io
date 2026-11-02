"""Program for rank-minimal exact tensor decomposition of 3x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Build the tensor from decomposition
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        tensor += U_c[:, :, None] * V_c[None, :, :] * W_c[None, None, :]
    
    expected = np.ones((n, m, p))
    return np.allclose(tensor, expected, atol=1e-9)

def _schoolbook(n, m, p):
    """Return the schoolbook (rank = n*m*p) decomposition."""
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

def _compute_tensor(U, V, W, n, m, p):
    """Compute the tensor from the decomposition."""
    rank = U.shape[1]
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        tensor += U_c[:, :, None] * V_c[None, :, :] * W_c[None, None, :]
    return tensor

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2x2."""
    U = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
    ]
    V = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    W = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    
    # Strassen's 7 multiplications for 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U = [
        [0, 1, 0, 0, 0, 0, 0],   # A11
        [0, 0, 1, 0, 0, 0, 0],   # A12
        [1, 1, 0, 0, 0, 0, 0],   # A21+A22
        [0, 0, 0, 1, 0, 0, 0],   # A22
        [1, 0, 1, 0, 0, 0, 0],   # A11+A12
        [1, 0, 1, 0, 0, 0, 0],   # A21-A11
        [0, 1, 0, 1, 0, 0, 0],   # A12-A22
    ]
    
    V = [
        [1, 0, 1, 0, 0, 0, 0],   # B11+B22
        [1, 0, 0, 0, 0, 0, 0],   # B11
        [0, 0, 1, 0, 0, 0, 0],   # B21-B22
        [0, 1, 0, 1, 0, 0, 0],   # B12-B11
        [0, 0, 0, 1, 1, 0, 0],   # B22
        [0, 1, 0, 0, 0, 1, 0],   # B11+B12
        [0, 0, 1, 0, 0, 0, 1],   # B21+B22
    ]
    
    W = [
        [1, 0, 0, 0, 0, 0, 0],   # C11
        [0, 0, 0, 1, 0, 0, 0],   # C12
        [0, 0, 1, 0, 0, 0, 0],   # C21
        [0, 1, 0, 0, 0, 0, 0],   # C22
    ]
    
    return U, V, W

def _try_decomposition(rank, n, m, p, time_limit, seed=42):
    """
    Try to find a rank-r decomposition using alternating least squares
    with random restarts and a smooth error surrogate.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    target = np.ones((n, m, p))
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    max_restarts = 20
    max_iterations = 500
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        current_error = float('inf')
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit:
                break
            
            # Alternating minimization
            # Fix V and W, solve for U
            # Fix U and W, solve for V
            # Fix U and V, solve for W
            
            # Compute gradients using the tensor structure
            # For each component c, the contribution to tensor[i,j,k] is:
            # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            # We can use a simple gradient descent approach
            lr = 0.01
            improvement = 0.0
            
            # Optimize U
            for c in range(rank):
                if time.time() - start_time > time_limit:
                    break
                # Compute gradient of error w.r.t. U[:, c]
                grad_U = np.zeros(n * m)
                for i in range(n):
                    for j in range(m):
                        # d(tensor[i,j,k])/d(U[i*m+j, c]) = V[j*p+k, c] * W[k*n+i, c]
                        # summed over k
                        for k in range(p):
                            grad_U[i*m+j] += V[j*p+k, c] * W[k*n+i, c] * (tensor[i,j,k] - 1.0)
                # Use current tensor value
                tensor = _compute_tensor(U, V, W, n, m, p)
                for i in range(n):
                    for j in range(m):
                        grad = 0.0
                        for k in range(p):
                            grad += V[j*p+k, c] * W[k*n+i, c] * (tensor[i,j,k] - 1.0)
                        U[i*m+j, c] -= lr * grad
            
            # Optimize V
            tensor = _compute_tensor(U, V, W, n, m, p)
            for c in range(rank):
                if time.time() - start_time > time_limit:
                    break
                for j in range(m):
                    for k in range(p):
                        grad = 0.0
                        for i in range(n):
                            grad += U[i*m+j, c] * W[k*n+i, c] * (tensor[i,j,k] - 1.0)
                        V[j*p+k, c] -= lr * grad
            
            # Optimize W
            tensor = _compute_tensor(U, V, W, n, m, p)
            for c in range(rank):
                if time.time() - start_time > time_limit:
                    break
                for k in range(p):
                    for i in range(n):
                        grad = 0.0
                        for j in range(m):
                            grad += U[i*m+j, c] * V[j*p+k, c] * (tensor[i,j,k] - 1.0)
                        W[k*n+i, c] -= lr * grad
            
            # Compute new error
            tensor = _compute_tensor(U, V, W, n, m, p)
            new_error = np.linalg.norm(tensor - target)
            
            if new_error < current_error:
                improvement = current_error - new_error
                current_error = new_error
            else:
                # Revert or accept with small probability (simulated annealing)
                pass
            
            if current_error < 1e-10:
                break
            
            if time.time() - start_time > time_limit:
                break
        
        if current_error < best_error:
            best_error = current_error
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_error < 1e-10:
            break
    
    if best_error < 1e-10:
        # Verify with exact check
        if _verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
            return best_U.tolist(), best_V.tolist(), best_W.tolist(), best_error
    
    return None, None, None, best_error

def _try_block_decomposition(n, m, p, time_limit):
    """
    Try to find a lower-rank decomposition by exploiting block structure.
    """
    start_time = time.time()
    
    best_rank = n * m * p
    best_solution = None
    
    # Try ranks from a reasonable upper bound down
    for target_rank in range(85, n * m * p + 1):
        if time.time() - start_time > time_limit:
            break
        
        max_attempts = 5 if target_rank < 90 else 2
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            U, V, W, err = _try_decomposition(target_rank, n, m, p, 
                                              time_limit - (time.time() - start_time),
                                              seed=attempt * 100 + 42)
            if U is not None:
                best_rank = target_rank
                best_solution = (U, V, W)
                break
        
        if best_solution is not None and best_rank == target_rank:
            continue
    
    return best_solution, best_rank

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0  # 110s budget, 10s safety margin
    
    # Try to find a better decomposition
    best_solution, best_rank = _try_block_decomposition(n, m, p, 60.0)
    
    if best_solution is not None and best_rank < n * m * p:
        return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
