import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank is 20 (2 * 10)
    # We will try to find a rank-20 decomposition.
    # If it fails, we fall back to schoolbook.
    
    deadline = time.time() + 110  # 110s limit to be safe
    
    # Initialize with schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    # Try to find a lower rank decomposition
    # We know R(2,3,4) = 10, so R(2,3,8) = 20 is a good target.
    target_rank = 20
    
    # We will use a randomized search with local optimization.
    # The system is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # We can fix U and V and solve for W? No, it's cubic.
    # We can use gradient descent on the error.
    
    def error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(len(U[0])):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    err += (val - 1.0) ** 2
        return err

    # Randomized search
    np.random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
            
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Local optimization using gradient descent
        # We will use a simple step size
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline - 5:
                break
                
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) # This is not correct, it's a sum over r of product
                        # Actually, val = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        val = 0.0
                        for r in range(target_rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        diff = val - 1.0
                        for r in range(target_rank):
                            grad_U[i*m+j, r] += 2 * diff * V[j*p+k, r] * W[k*n+i, r]
                            grad_V[j*p+k, r] += 2 * diff * U[i*m+j, r] * W[k*n+i, r]
                            grad_W[k*n+i, r] += 2 * diff * U[i*m+j, r] * V[j*p+k, r]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if np.linalg.norm(grad_U) < 1e-6 and np.linalg.norm(grad_V) < 1e-6 and np.linalg.norm(grad_W) < 1e-6:
                break
        
        # Check if the solution is valid
        err = error(U.tolist(), V.tolist(), W.tolist())
        if err < 1e-6:
            # Round to half-integers if possible?
            # The prompt says integer/half-integer entries.
            # We can try to round and check.
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            err_rounded = error(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
            if err_rounded < 1e-6:
                best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                best_rank = target_rank
                break
            else:
                # Use the unrounded solution if it's good enough?
                # The prompt requires integer/half-integer.
                # So we must round.
                pass

    return best_result
# EVOLVE-BLOCK-END
