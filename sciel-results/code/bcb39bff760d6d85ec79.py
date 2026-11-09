import numpy as np
import time
import random

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Target Tensor T
    # T[u, v, w] where u=i*m+j, v=j*p+k, w=k*n+i
    # Non-zero when: u=3i+j, v=3j+k, w=3k+i
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    def check_solution(U, V, W, r):
        """
        U: (9, r)
        V: (9, r)
        W: (9, r)
        Returns True if decomposition is exact.
        """
        # T_rec[u, v, w] = sum_k U[u, k] * V[v, k] * W[w, k]
        # Use einsum for speed
        # T_rec = np.einsum('uk,vk,wk->uvw', U, V, W)
        # But T is sparse, so checking non-zeros is faster
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    s = np.dot(U[u, :], V[v, :] * W[w, :])
                    if s != 1:
                        return False
        # Check zeros
        # This is expensive, but we can skip if we trust the linear solve residuals
        # For safety, we check a few random zeros
        for _ in range(50):
            u = random.randint(0, n*m-1)
            v = random.randint(0, m*p-1)
            w = random.randint(0, p*n-1)
            # Ensure (u,v,w) is not a non-zero entry
            # Non-zero entries are (3i+j, 3j+k, 3k+i)
            # We can just check if T[u,v,w] == 0
            if T[u, v, w] == 0:
                s = np.dot(U[u, :], V[v, :] * W[w, :])
                if s != 0:
                    return False
        return True

    def solve_for_W(U, V, r):
        """
        Solves for W given U, V.
        Returns W if a clean solution exists, else None.
        """
        # Construct M: (81, r) where M[idx, k] = U[u, k] * V[v, k]
        # idx corresponds to (u, v) pair
        M = np.zeros((n*m * m*p, r))
        idx = 0
        for u in range(n*m):
            for v in range(m*p):
                M[idx, :] = U[u, :] * V[v, :]
                idx += 1
        
        # Construct B: (81, 9) where B[idx, w] = T[u, v, w]
        B = np.zeros((n*m * m*p, p*n))
        idx = 0
        for u in range(n*m):
            for v in range(m*p):
                B[idx, :] = T[u, v, :]
                idx += 1

        # Solve M X = B for X (r x 9)
        try:
            X, residuals, rank, s = np.linalg.lstsq(M, B, rcond=None)
        except np.linalg.LinAlgError:
            return None
            
        # Check residuals
        if residuals is not None and len(residuals) > 0 and residuals[0] > 1e-9:
            return None
        if residuals is None: 
             if not np.allclose(M @ X, B):
                 return None
        
        # X is (r, 9). W is (9, r). So W = X.T
        W_candidate = X.T
        
        # Check if W_candidate entries are integers or half-integers
        W_scaled = 2.0 * W_candidate
        W_rounded = np.round(W_scaled)
        if np.max(np.abs(W_scaled - W_rounded)) > 1e-9:
            return None
        
        W = W_rounded / 2.0
        
        return W

    # Search strategy
    # Try ranks from low to high
    random.seed(42)
    
    max_tries_per_rank = 500  # Increased due to speedup
    ranks_to_try = [23, 24, 25, 26] # Focus on best known rank 23

    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        for _ in range(max_tries_per_rank):
            if time.time() > deadline:
                break
                
            # Generate random U, V
            # Sparse random with {-1, 0, 1}
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            
            # Use vectorized random generation
            U_mask = np.random.rand(n*m, r) < 0.3
            V_mask = np.random.rand(m*p, r) < 0.3
            
            U[U_mask] = np.random.choice([-1, 1], size=np.sum(U_mask))
            V[V_mask] = np.random.choice([-1, 1], size=np.sum(V_mask))
            
            # Solve for W
            W = solve_for_W(U, V, r)
            
            if W is not None:
                # Verify
                if check_solution(U, V, W, r):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }

    # If no better rank found, return schoolbook
    rank = n * m * p
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
