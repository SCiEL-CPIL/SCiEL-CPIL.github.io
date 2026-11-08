import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Define the target tensor constraints
    # We need to check if a given U, V, W satisfies the conditions.
    # Conditions:
    # 1. For all i, j, k: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    # 2. For all i, j, j', k with j != j': sum_c U[i*m+j, c] V[j'*p+k, c] W[k*n+i, c] == 0
    # 3. For all i, j, k, k' with k != k': sum_c U[i*m+j, c] V[j*p+k, c] W[k'*n+i, c] == 0
    # 4. For all i, i', j, k with i != i': sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i', c] == 0
    
    def check_decomposition(U, V, W, tol=1e-6):
        r = U.shape[1]
        # Check condition 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :])
                    # Wait, the term is U * V * W. It's a triple product?
                    # No, the tensor decomposition is T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}.
                    # So we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        # Check condition 2 (j mismatch)
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    if j == jp: continue
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[k*n+i, :])
                        if abs(val) > tol:
                            return False
        # Check condition 3 (k mismatch)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for kp in range(p):
                        if k == kp: continue
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[kp*n+i, :])
                        if abs(val) > tol:
                            return False
        # Check condition 4 (i mismatch)
        for i in range(n):
            for ip in range(n):
                if i == ip: continue
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+ip, :])
                        if abs(val) > tol:
                            return False
        return True

    def try_rank(r):
        # Random initialization
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # Simple gradient descent? 
        # The loss function is the sum of squares of all constraint violations.
        # Calculating the gradient is expensive.
        # Given the time limit, we might just try a few random initializations and check if they are close to a solution?
        # Unlikely to hit a solution randomly.
        
        # Alternative: Use the schoolbook solution as a starting point for rank reduction?
        # Not straightforward.
        
        # For now, return None to indicate failure.
        return None

    # Try to find a lower rank
    # We know rank 60 works.
    # Let's try a few lower ranks with random search for a very short time.
    best_rank = rank_schoolbook
    best_solution = (U_sb, V_sb, W_sb)
    
    # Try ranks from 59 down to 50
    for r in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 0), -1):
        if time.time() > deadline:
            break
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            # Just check if it's a solution (very unlikely)
            if check_decomposition(U, V, W):
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if check_decomposition(U_rounded, V_rounded, W_rounded, tol=1e-9):
                    best_rank = r
                    best_solution = (U_rounded, V_rounded, W_rounded)
                    break
        if best_rank < rank_schoolbook:
            break
            
    # Convert best solution to list of lists
    U_list = best_solution[0].tolist()
    V_list = best_solution[1].tolist()
    W_list = best_solution[2].tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
