"""Matrix multiplication 3x3 by 3x8 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    """
    Attempt to find a low-rank decomposition of the 3x3 * 3x8 matrix multiplication tensor.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 72
    
    def make_schoolbook():
        rank = schoolbook_rank
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
    
    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        u_val = U[i * m + j][r]
                        v_val = V[j * p + k][r]
                        w_val = W[k * n + i][r]
                        actual += u_val * v_val * w_val
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    def to_half_integer_matrix(mat):
        """Convert a matrix of floats to half-integers."""
        result = []
        for row in mat:
            r = []
            for val in row:
                v2 = round(val * 2)
                if v2 % 2 == 0:
                    r.append(v2 // 2)
                else:
                    r.append(v2 / 2.0)
            result.append(r)
        return result
    
    def try_gradient_descent(rank, max_iters, lr, time_limit):
        """Try to find a rank-r decomposition using gradient descent."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        target = np.ones((n, m, p))
        
        def compute_and_grad(U, V, W):
            """Compute tensor and gradients."""
            T = np.zeros((n, m, p))
            dU = np.zeros((n * m, rank))
            dV = np.zeros((m * p, rank))
            dW = np.zeros((p * n, rank))
            
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
            
            # Compute gradients
            for r in range(rank):
                u = U[:, r].reshape(n, m)
                v = V[:, r].reshape(m, p)
                w = W[:, r].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            err = T[i, j, k] - 1.0
                            dU[i * m + j, r] += 2 * err * v[j, k] * w[k, i]
                            dV[j * p + k, r] += 2 * err * u[i, j] * w[k, i]
                            dW[k * n + i, r] += 2 * err * u[i, j] * v[j, k]
            
            error = np.sum((T - target) ** 2)
            return error, dU, dV, dW
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            error, dU, dV, dW = compute_and_grad(U, V, W)
            
            if error < 1e-14:
                break
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        error, _, _, _ = compute_and_grad(U, V, W)
        return U, V, W, error
    
    def try_to_rationalize(U, V, W, max_search=50):
        """Try to convert float solution to half-integers."""
        # Scale to make entries close to half-integers
        for scale in [1, 2, 0.5, 4, 0.25, 8, 0.125]:
            Us = U * scale
            Vs = V * scale
            Ws = W * scale
            
            # Check if close to half-integers
            Us2 = np.round(Us * 2)
            Vs2 = np.round(Vs * 2)
            Ws2 = np.round(Ws * 2)
            
            max_dev = max(np.max(np.abs(Us2 - Us * 2)), 
                         np.max(np.abs(Vs2 - Vs * 2)),
                         np.max(np.abs(Ws2 - Ws * 2)))
            
            if max_dev < 0.1:
                # Convert to half-integer lists
                U_list = to_half_integer_matrix(Us)
                V_list = to_half_integer_matrix(Vs)
                W_list = to_half_integer_matrix(Ws)
                
                # Verify
                if verify(U_list, V_list, W_list):
                    return U_list, V_list, W_list
        
        return None, None, None
    
    # Try gradient descent for various ranks
    best_result = make_schoolbook()
    best_rank = schoolbook_rank
    
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: at least max(n*m, m*p, n*p) = 24
    for rank in range(24, min(schoolbook_rank, 50)):
        if time.time() > deadline - 30:
            break
        
        found = False
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            U, V, W, error = try_gradient_descent(rank, max_iters=2000, lr=0.005, time_limit=deadline - 30)
            
            if error < 1e-10:
                # Found a solution, try to rationalize
                U_list, V_list, W_list = try_to_rationalize(U, V, W)
                if U_list is not None:
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                    found = True
                    break
        
        if found:
            break
    
    return best_result
