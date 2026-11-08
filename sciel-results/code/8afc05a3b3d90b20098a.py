import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes:
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    # Target Tensor T
    # T[u, v, w] where u=i*m+j, v=j'*p+k', w=k''*n+i''
    # T[u, v, w] = 1 if j==j' and k'==k'' and i''==i, else 0.
    
    # We will try ranks from 27 upwards.
    # Known: R(2,3,6) is likely 27.
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute target tensor T
    # u in [0, n*m), v in [0, m*p), w in [0, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0

    def compute_tensor(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Result: (nm, mp, pn)
        # Result[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        # Using einsum or loops?
        # np.einsum('ur, vr, wr -> uvw', U, V, W)
        return np.einsum('ur, vr, wr -> uvw', U, V, W)

    def verify(U, V, W):
        # Check if sum_r U[:,r] V[:,r] W[:,r] == T exactly (with tolerance for float, then round)
        # Actually, we round U, V, W to nearest 0.5 first.
        Ur = np.round(U * 2) / 2
        Vr = np.round(V * 2) / 2
        Wr = np.round(W * 2) / 2
        
        # Check if entries are close to 0 or 1? No, entries can be anything.
        # The sum must equal T.
        Res = compute_tensor(Ur, Vr, Wr)
        if np.allclose(Res, T, atol=1e-9):
            return Ur, Vr, Wr
        return None

    def solve_rank(R):
        # Initialize
        np.random.seed(42) # Deterministic for reproducibility in this context? 
        # Maybe random seed per attempt is better to escape local minima.
        
        U = np.random.randn(n*m, R) * 0.5
        V = np.random.randn(m*p, R) * 0.5
        W = np.random.randn(p*n, R) * 0.5
        
        # Gradient Descent
        lr = 0.01
        # Adaptive learning rate?
        
        for step in range(2000):
            if time.time() > deadline:
                break
            
            Res = compute_tensor(U, V, W)
            Err = Res - T
            Loss = np.sum(Err**2)
            
            if Loss < 1e-12:
                break
            
            # Gradients
            # dL/dU[u, r] = 2 * sum_{v,w} Err[u, v, w] * V[v, r] * W[w, r]
            # dL/dV[v, r] = 2 * sum_{u,w} Err[u, v, w] * U[u, r] * W[w, r]
            # dL/dW[w, r] = 2 * sum_{u,v} Err[u, v, w] * U[u, r] * V[v, r]
            
            # Efficient computation using einsum
            # grad_U: (nm, R)
            # grad_U[u, r] = 2 * einsum('uvw, vr, wr -> ur', Err, V, W)
            grad_U = 2.0 * np.einsum('uvw, vr, wr -> ur', Err, V, W)
            grad_V = 2.0 * np.einsum('uvw, ur, wr -> vr', Err, U, W)
            grad_W = 2.0 * np.einsum('uvw, ur, vr -> wr', Err, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Periodic rounding check?
            if step % 100 == 0:
                sol = verify(U, V, W)
                if sol is not None:
                    return sol
        
        sol = verify(U, V, W)
        if sol is not None:
            return sol
        return None

    # Try ranks
    for R in range(27, 37):
        if time.time() > deadline:
            break
        # Try a few restarts for each rank
        for attempt in range(5):
            if time.time() > deadline:
                break
            np.random.seed(R * 100 + attempt)
            sol = solve_rank(R)
            if sol is not None:
                U, V, W = sol
                # Convert to lists for return
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        
        # If we found a solution with R, we can stop if we are confident it's minimal?
        # The task asks for rank-minimal. 
        # If we find one for 27, it's likely minimal.
        # If we only find 32, we might try harder for 27.
        
    # Fallback to schoolbook (rank 36) if nothing found
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
