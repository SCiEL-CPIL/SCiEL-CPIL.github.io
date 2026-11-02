import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def _get_target_tensor():
    """
    Construct the target matrix multiplication tensor T_mm.
    T_mm[i*m + j, j*p + r, r*n + i] = 1
    """
    n, m, p = DIMENSIONS
    T_mm = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for r in range(p):
                T_mm[i*m + j, j*p + r, r*n + i] = 1.0
    return T_mm

def _verify(U, V, W, T_target, tol=1e-8):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Construct the tensor T from U, V, W
    # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    T_calc = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
    return np.allclose(T_calc, T_target, atol=tol)

def _rank11_solution():
    """
    Returns the known rank 11 decomposition.
    """
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen part for columns 0 and 1 of B
    # M1 = (a+d)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[4][0] = 1
    
    # M2 = (c+d)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    
    # M3 = a(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[4][2] = -1
    
    # M4 = d(b10-b00)
    U[3][3] = 1
    V[3][3] = 1; V[0][3] = -1
    
    # M5 = (a+b)b11
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    
    # M6 = (c-a)(b00+b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    
    # M7 = (b-d)(b10+b11)
    U[1][6] = 1; U[3][6] = -1
    V[3][6] = 1; V[4][6] = 1
    
    # W for Strassen outputs
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1
    # C01 = M3 + M5
    W[2][2] = 1; W[2][4] = 1
    # C10 = M2 + M4
    W[1][1] = 1; W[1][3] = 1
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1
    
    # Schoolbook part for column 2 of B
    # M8 = a * b02
    U[0][7] = 1
    V[2][7] = 1
    
    # M9 = b * b12
    U[1][8] = 1
    V[5][8] = 1
    
    # M10 = c * b02
    U[2][9] = 1
    V[2][9] = 1
    
    # M11 = d * b12
    U[3][10] = 1
    V[5][10] = 1
    
    # W for Schoolbook outputs
    # C02 = M8 + M9
    W[4][7] = 1; W[4][8] = 1
    # C12 = M10 + M11
    W[5][9] = 1; W[5][10] = 1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_10_search(deadline):
    """
    Attempt to find a rank 10 decomposition using numerical optimization.
    This is a structural change: active search for a potentially better rank.
    """
    n, m, p = DIMENSIONS
    rank = 10
    T_target = _get_target_tensor()
    
    # Use a simple gradient-free or gradient-based approach.
    # Given the non-convexity, random restarts with limited steps are best.
    
    best_error = float('inf')
    best_solution = None
    
    # Number of restarts to try within time limit
    # Each step involves calculating the tensor product difference.
    # T_calc = sum_r U[:,r] V[:,r] W[:,r]
    # We minimize || T_calc - T_target ||_F^2
    
    num_restarts = 50
    steps_per_restart = 100
    learning_rate = 0.01
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to prevent explosion
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Calculate current tensor
            T_calc = np.zeros((n*m, m*p, p*n))
            for r in range(rank):
                T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            error = np.linalg.norm(T_calc - T_target)
            if error < best_error:
                best_error = error
                best_solution = (U.copy(), V.copy(), W.copy())
                
            if error < 1e-8:
                return {"U": best_solution[0].tolist(), "V": best_solution[1].tolist(), "W": best_solution[2].tolist()}
            
            # Calculate gradients
            # dL/dU = sum_r (V[:,r] W[:,r])^T (T_calc - T_target) ... complicated
            # For simplicity and speed, we might skip true gradient descent
            # and just do a random perturbation search (Hill Climbing)
            # which is less efficient but easier to implement correctly in short time.
            
            # Let's stick to a very simple perturbation if gradient is too slow to code safely.
            # Actually, gradient calculation is:
            # grad_U[i, r] = sum_{j,k} (T_calc[i,j,k] - T_target[i,j,k]) * V[j,r] * W[k,r]
            
            dT = T_calc - T_target
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for r in range(rank):
                # grad_U[:, r] = sum_{j,k} dT[:, j, k] * V[j, r] * W[k, r]
                # This is a tensor contraction.
                # We can use einsum or loops.
                # For small dims, loops are fine.
                for i in range(n*m):
                    for j in range(m*p):
                        for k in range(p*n):
                            val = dT[i, j, k]
                            if val != 0:
                                grad_U[i, r] += val * V[j, r] * W[k, r]
                                grad_V[j, r] += val * U[i, r] * W[k, r]
                                grad_W[k, r] += val * U[i, r] * V[j, r]
                                
            # Update
            U -= learning_rate * grad_U
            V -= learning_rate * grad_V
            W -= learning_rate * grad_W
            
            # Re-normalize to keep scale stable
            U /= np.linalg.norm(U) + 1e-9
            V /= np.linalg.norm(V) + 1e-9
            W /= np.linalg.norm(W) + 1e-9

    # Check if best solution is valid
    if best_solution is not None:
        U, V, W = best_solution
        if _verify(U, V, W, T_target):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    # Reserve 10s for fallback and return
    deadline = start_time + 140
    
    # 1. Try to find Rank 10 (structural change: active search)
    # This is unlikely to succeed if rank 10 doesn't exist, but it's a valid attempt.
    result_rank_10 = _try_rank_10_search(deadline)
    
    if result_rank_10:
        return result_rank_10
        
    # 2. Fallback to known Rank 11
    return _rank11_solution()
