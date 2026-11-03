import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook solution
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_schoolbook
    best_solution = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds search time, leaving 10s margin
    
    def validate(U, V, W):
        """Check if U, V, W form a valid exact decomposition."""
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Target: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k
        # We can vectorize the check.
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # Compute the tensor product sum
        # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is a trace over the r dimension of the outer products?
        # Let's use einsum or explicit loops for clarity and speed in this small case.
        # n=3, m=3, p=5. R is up to 45.
        # Loops are fine.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    def optimize_rank(R, deadline, seed=None):
        """Try to find a decomposition of rank R using gradient descent."""
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize U, V, W with small random values
        # We want to avoid zero initialization which gets stuck.
        U = np.random.uniform(-1, 1, (n * m, R))
        V = np.random.uniform(-1, 1, (m * p, R))
        W = np.random.uniform(-1, 1, (p * n, R))
        
        # Learning rate
        lr = 0.01
        # Pre-compute indices for speed?
        # The loss is L = sum_{i,j,k} ( sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1 )^2
        
        # We will do a few iterations of gradient descent
        # To make it faster, we can batch the gradient computation.
        # But n*m*p is small (45). R is moderate.
        
        # Let's use a simple iterative method.
        # We'll run for a fixed number of steps or until deadline.
        steps = 0
        max_steps = 5000
        min_loss = float('inf')
        
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            # Compute gradients
            # grad_U[i*m+j, r] = 2 * sum_{k} (sum_{r'} U[i,j,r'] V[j,k,r'] W[k,i,r'] - 1) * V[j,k,r] * W[k,i,r]
            # This is expensive to compute naively.
            # Let's compute the error tensor E[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1
            # Then grad_U[i,j,r] = 2 * sum_k E[i,j,k] * V[j,k,r] * W[k,i,r]
            
            # Efficient computation using matrix operations?
            # U (n,m,R), V (m,p,R), W (p,n,R)
            # E[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            
            # Let's stick to loops for correctness and simplicity, given small sizes.
            # n=3, m=3, p=5, R~30.
            # Inner loop: 45 * 30 = 1350 ops for forward.
            # Gradient: 45 * 30 * 3 (for i,j,k loops) ~ 4000 ops.
            # Total ~ 5000 ops per step. 5000 steps = 25M ops. Very fast.
            
            # Compute E
            E = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(R):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        E[i, j, k] = s - 1.0
            
            loss = np.sum(E**2)
            if loss < min_loss:
                min_loss = loss
                if loss < 1e-8:
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    if validate(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = E[i, j, k]
                        if abs(e) < 1e-10:
                            continue
                        for r in range(R):
                            # dE/dr_U[i,j,r] = V[j,k,r] * W[k,i,r]
                            # dE/dr_V[j,k,r] = U[i,j,r] * W[k,i,r]
                            # dE/dr_W[k,i,r] = U[i,j,r] * V[j,k,r]
                            grad_U[i * m + j, r] += 2 * e * V[j * p + k, r] * W[k * n + i, r]
                            grad_V[j * p + k, r] += 2 * e * U[i * m + j, r] * W[k * n + i, r]
                            grad_W[k * n + i, r] += 2 * e * U[i * m + j, r] * V[j * p + k, r]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Optional: decay learning rate
            if steps % 500 == 0:
                lr *= 0.95
                
            steps += 1
            
        # Final check
        if min_loss < 1e-6:
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            if validate(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap
                
        return None

    # Try ranks from 44 down to 25
    # We allocate a fraction of the time to each rank
    # Let's try 5 ranks: 44, 42, 40, 38, 35
    ranks_to_try = [44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]
    
    # We will use a time slice for each rank
    # Total time available: 140s
    # Let's give 5s to each of the first 10 ranks, and remaining to the rest?
    # Or just loop and break on deadline.
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        # Limit trials to keep time bounded
        max_trials = 5
        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            result = optimize_rank(rank, deadline, seed=np.random.randint(10000))
            if result is not None:
                U, V, W = result
                best_rank = rank
                best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                # Break out of trial loop, continue to next (lower) rank
                break
        
    return best_solution
