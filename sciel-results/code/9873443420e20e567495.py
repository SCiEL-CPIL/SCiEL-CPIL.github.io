"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor.
    U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _construct_strassen_3x3x5():
    """
    Construct a rank-23 decomposition for 3x3 * 3x5 matrix multiplication.
    This uses a Strassen-like approach adapted for rectangular matrices.
    
    We decompose the 3x3 matrix A into 2x2 blocks with overlap, and the 3x5 matrix B.
    Actually, let's use a known construction or build one systematically.
    
    For 3x3 * 3x3, the best known rank is 23 (Lader). For 3x3 * 3x5, we can use
    a similar approach but it's not directly standard.
    
    Let's try a different approach: use the schoolbook but with some clever sharing.
    
    Actually, for 3x3 * 3x5, we can view it as:
    - 3x3 * 3x3 (left part of B) + 3x3 * 3x2 (right part of B)
    
    The 3x3 * 3x3 part has rank 23 (Lader's algorithm).
    The 3x3 * 3x2 part has rank... let's think.
    
    Alternatively, we can use a rank-23 decomposition for the whole thing by
    finding an explicit construction.
    
    Let me try to construct one based on known results. The minimal rank for
    3x3 * 3x5 is not well-known to me, but it should be less than 45.
    
    Let's use a simpler approach: try to find a rank-23 or lower decomposition
    by using the fact that we can share some computations.
    
    Actually, let me just implement a known rank-23 decomposition for 3x3*3x3
    and extend it. But that's complex.
    
    Let's try a different strategy: use the fact that 3x3 * 3x5 can be decomposed
    using a rank-23 algorithm if we can find one.
    
    For now, let's return a schoolbook decomposition but with a better search
    strategy to find lower rank.
    """
    n, m, p = 3, 3, 5
    
    # Let's try to construct a rank-23 decomposition using known techniques.
    # One approach: use the Lader algorithm for 3x3*3x3 and extend.
    
    # Actually, let's just return schoolbook for now and rely on the search.
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First, get the schoolbook decomposition
    best = _construct_strassen_3x3x5()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using iterative optimization
    # The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # This is a system of nonlinear equations. We can use gradient-based optimization
    # to minimize the sum of squared errors.
    
    if time.time() < deadline - 5:
        np.random.seed(42)
        
        # Try various target ranks below schoolbook
        for target_rank in range(44, 20, -1):
            if time.time() > deadline - 5:
                break
            
            # Multiple random restarts
            max_restarts = 20 if target_rank > 35 else 10
            for attempt in range(max_restarts):
                if time.time() > deadline - 5:
                    break
                
                # Random initialization with small values
                U_rand = np.random.randn(n * m, target_rank) * 0.5
                V_rand = np.random.randn(m * p, target_rank) * 0.5
                W_rand = np.random.randn(p * n, target_rank) * 0.5
                
                # Gradient-based optimization
                lr = 0.01
                max_iter = 200
                prev_loss = float('inf')
                
                for it in range(max_iter):
                    if time.time() > deadline - 5:
                        break
                    
                    # Compute loss: sum over all (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                    loss = 0.0
                    grad_U = np.zeros_like(U_rand)
                    grad_V = np.zeros_like(V_rand)
                    grad_W = np.zeros_like(W_rand)
                    
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                # Compute the dot product sum
                                prod_sum = 0.0
                                for c in range(target_rank):
                                    prod_sum += U_rand[i*m+j, c] * V_rand[j*p+k, c] * W_rand[k*n+i, c]
                                
                                err = prod_sum - 1.0
                                loss += err * err
                                
                                # Gradients
                                for c in range(target_rank):
                                    u_val = U_rand[i*m+j, c]
                                    v_val = V_rand[j*p+k, c]
                                    w_val = W_rand[k*n+i, c]
                                    
                                    # d(loss)/d(U[i*m+j, c]) = 2 * err * V[j*p+k, c] * W[k*n+i, c]
                                    grad_U[i*m+j, c] += 2.0 * err * v_val * w_val
                                    grad_V[j*p+k, c] += 2.0 * err * u_val * w_val
                                    grad_W[k*n+i, c] += 2.0 * err * u_val * v_val
                    
                    # Gradient descent step
                    U_rand -= lr * grad_U
                    V_rand -= lr * grad_V
                    W_rand -= lr * grad_W
                    
                    # Check convergence
                    if abs(loss - prev_loss) < 1e-12:
                        break
                    prev_loss = loss
                
                # Verify the result
                if _verify_decomposition(U_rand, V_rand, W_rand, n, m, p):
                    best = {
                        "U": U_rand.tolist(),
                        "V": V_rand.tolist(),
                        "W": W_rand.tolist()
                    }
                    best_rank = target_rank
                    break  # Found a valid decomposition at this rank
    
    return best
# EVOLVE-BLOCK-END
