import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    # Target tensor T
    # T[a, b, c] = 1 if a=5i+j, b=5j+k, c=5k+i
    # Dimensions: 20 x 25 x 20
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Try to find a lower rank
    # Start from rank_school - 1 downwards
    # We will use a simple random restart + gradient descent
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Helper to check and round
    def check_and_round(U, V, W, r):
        # U, V, W are numpy arrays of shape (dim, r)
        # Compute tensor
        # T_calc[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        # This is a tensor product sum.
        # We can compute it efficiently using einsum or loops.
        # Given small size, loops are fine.
        T_calc = np.zeros((n*m, m*p, p*n))
        for k in range(r):
            uk = U[:, k]
            vk = V[:, k]
            wk = W[:, k]
            # Outer product
            T_calc += np.einsum('a,b,c->abc', uk, vk, wk)
        
        # Check error
        error = np.linalg.norm(T - T_calc)
        if error < 1e-5:
            # Round to nearest half-integer
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            # Re-check with rounded values
            T_calc_r = np.zeros((n*m, m*p, p*n))
            for k in range(r):
                uk = U_r[:, k]
                vk = V_r[:, k]
                wk = W_r[:, k]
                T_calc_r += np.einsum('a,b,c->abc', uk, vk, wk)
            
            error_r = np.linalg.norm(T - T_calc_r)
            if error_r < 1e-9:
                # Convert to list of lists
                U_list = [[float(val) for val in row] for row in U_r]
                V_list = [[float(val) for val in row] for row in V_r]
                W_list = [[float(val) for val in row] for row in W_r]
                return True, U_list, V_list, W_list
        return False, None, None, None

    # Search loop
    for r in range(rank_school - 1, 40, -1): # Try down to 40
        if time.time() > deadline:
            break
        
        # Try a few random restarts for this rank
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize
            # Use small random values
            np.random.seed(42 + restart)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Gradient Descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Compute gradient
                # L = || T - sum U_k V_k W_k ||^2
                # dL/dU[a, k] = -2 * sum_{b,c} (T[a,b,c] - T_calc[a,b,c]) V[b,k] W[c,k]
                
                T_calc = np.zeros((n*m, m*p, p*n))
                for k in range(r):
                    uk = U[:, k]
                    vk = V[:, k]
                    wk = W[:, k]
                    T_calc += np.einsum('a,b,c->abc', uk, vk, wk)
                
                diff = T - T_calc
                
                # Check convergence
                err = np.linalg.norm(diff)
                if err < 1e-5:
                    break
                
                # Gradient
                # grad_U[a, k] = -2 * sum_{b,c} diff[a,b,c] V[b,k] W[c,k]
                # We can compute this efficiently
                # For each k, grad_U[:, k] = -2 * (diff * V[:, k] * W[:, k]) summed over b, c
                # diff is (a,b,c). V is (b,k). W is (c,k).
                # For fixed k, term is sum_b,c diff[a,b,c] V[b,k] W[c,k]
                # This is (diff * V_k * W_k).sum(axis=(1,2))
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for k in range(r):
                    vk = V[:, k]
                    wk = W[:, k]
                    # diff * vk * wk
                    # diff shape (a,b,c), vk shape (b,), wk shape (c,)
                    # We want sum over b,c
                    # This is np.einsum('abc,b,c->a', diff, vk, wk)
                    grad_U[:, k] = -2 * np.einsum('abc,b,c->a', diff, vk, wk)
                    grad_V[:, k] = -2 * np.einsum('abc,a,c->b', diff, U[:, k], wk)
                    grad_W[:, k] = -2 * np.einsum('abc,a,b->c', diff, U[:, k], vk)
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Optional: project to positive? No, signs matter.
            
            # Check result
            success, U_list, V_list, W_list = check_and_round(U, V, W, r)
            if success:
                return {"U": U_list, "V": V_list, "W": W_list}

    # If no better rank found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
