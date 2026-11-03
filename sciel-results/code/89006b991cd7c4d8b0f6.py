import numpy as np
import time
import random

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if indices match, else 0
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def compute_error(U, V, W):
        """
        Compute the squared error between the target tensor and the decomposition.
        U: (n*m, R), V: (m*p, R), W: (p*n, R)
        """
        R = U.shape[1]
        # Decomposed tensor D
        # D[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        # We can compute this efficiently using einsum or loops if R is small.
        # For R=28, a loop over r is fine.
        D = np.zeros_like(T)
        for r in range(R):
            # Outer product of U[:, r], V[:, r], W[:, r]
            D += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        return np.sum((T - D) ** 2)

    def verify(U, V, W, tol=1e-9):
        """
        Check if the decomposition is exact and entries are half-integers.
        """
        R = U.shape[1]
        D = np.zeros_like(T)
        for r in range(R):
            D += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        # Check exactness
        if np.max(np.abs(T - D)) > tol:
            return False
        
        # Check half-integer constraint
        # Entries should be close to k/2
        for mat in [U, V, W]:
            for val in mat.flatten():
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
        return True

    def local_search(U_init, V_init, W_init, target_rank, max_iters=20000):
        """
        Perform local search to minimize error.
        """
        U_cur = U_init.copy()
        V_cur = V_init.copy()
        W_cur = W_init.copy()
        
        current_error = compute_error(U_cur, V_cur, W_cur)
        
        rows = [n * m, m * p, p * n]
        
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Pick a random matrix and element
            mat_idx = random.choice([0, 1, 2])
            row = random.randint(0, rows[mat_idx] - 1)
            col = random.randint(0, target_rank - 1)
            
            # Store old value
            if mat_idx == 0:
                old_val = U_cur[row, col]
                U_cur[row, col] += random.choice([-0.5, 0.5])
            elif mat_idx == 1:
                old_val = V_cur[row, col]
                V_cur[row, col] += random.choice([-0.5, 0.5])
            else:
                old_val = W_cur[row, col]
                W_cur[row, col] += random.choice([-0.5, 0.5])
            
            new_error = compute_error(U_cur, V_cur, W_cur)
            
            if new_error < current_error:
                current_error = new_error
                if current_error < 1e-9:
                    if verify(U_cur, V_cur, W_cur):
                        return U_cur, V_cur, W_cur, current_error
            else:
                # Revert
                if mat_idx == 0:
                    U_cur[row, col] = old_val
                elif mat_idx == 1:
                    V_cur[row, col] = old_val
                else:
                    W_cur[row, col] = old_val
        
        return U_cur, V_cur, W_cur, current_error

    # Try ranks from 28 to 35
    best_U = None
    best_V = None
    best_W = None
    best_rank = 36
    
    # Schoolbook fallback
    U_school = np.zeros((n * m, 36))
    V_school = np.zeros((m * p, 36))
    W_school = np.zeros((p * n, 36))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    for target_rank in [28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline - 20:
            break
        
        found = False
        num_restarts = 0
        while time.time() < deadline - 5 and not found:
            num_restarts += 1
            
            # Initialize with random half-integers
            # Using a wider range might help escape local minima, 
            # but -1 to 1 is standard for these decompositions.
            U_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, target_rank))
            V_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, target_rank))
            W_rand = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, target_rank))
            
            U_res, V_res, W_res, err = local_search(U_rand, V_rand, W_rand, target_rank)
            
            if err < 1e-9:
                if verify(U_res, V_res, W_res):
                    best_U = U_res.copy()
                    best_V = V_res.copy()
                    best_W = W_res.copy()
                    best_rank = target_rank
                    found = True
                    break

    if best_U is None:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
