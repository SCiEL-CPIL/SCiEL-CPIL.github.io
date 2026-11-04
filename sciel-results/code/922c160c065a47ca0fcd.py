"""Program that attempts to find a low-rank decomposition for 3x5x6 matrix multiplication.

Strategy: Use vectorized gradient-based tensor decomposition search.
The matrix multiplication tensor for (n x m) * (m x p) has the structure:
T[i, j, k] = 1 if (i, j, k) is a valid triple (i.e., always, since the tensor
is defined as the sum over j of A[i,j]*B[j,k])

Actually, the correct tensor structure is:
The multiplication tensor T has T[i, j, k] = 1 for the "diagonal" entries where
the (i,j,k) indices form a valid multiplication pattern.

For matrix multiplication C = A * B where A is n x m and B is m x p:
C[i, k] = sum_j A[i, j] * B[j, k]

The tensor T[i, j, k] = 1 for all (i, j, k) in the range, representing:
C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]

So T[i,j,k] = 1 for ALL (i,j,k) triples! The schoolbook uses rank n*m*p.

We want to find a decomposition with fewer rank-1 terms:
T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

where each term is a rank-1 tensor.
"""

import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct the tensor decomposition for 3x5x6 matrix multiplication.
    
    Returns:
        dict with keys 'U', 'V', 'W' where:
        - U has shape (n*m, rank)
        - V has shape (m*p, rank)  
        - W has shape (p*n, rank)
        with W using the trace convention W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
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
        return U, V, W, rank
    
    # Build the target tensor T[i, j, k] = 1 for all valid (i, j, k)
    # This represents: C[i,k] = sum_j A[i,j] * B[j,k]
    T = np.ones((n, m, p), dtype=np.float64)
    
    # Precompute index mappings for vectorized operations
    # For each (i, j, k) triple, we need:
    #   U index: i*m + j
    #   V index: j*p + k
    #   W index: k*n + i
    
    def compute_residual(U, V, W, rank):
        """Compute the residual tensor: R[i,j,k] = T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]"""
        R = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(n):
                for j in range(m):
                    u_val = uc[i * m + j]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        R[i, j, k] += u_val * vc[j * p + k] * wc[k * n + i]
        R -= T
        return R
    
    def compute_loss(U, V, W, rank):
        """Compute the squared loss."""
        R = compute_residual(U, V, W, rank)
        return np.sum(R * R)
    
    def gradient_descent(U, V, W, rank, lr=0.01, max_steps=2000):
        """Run gradient descent to minimize the tensor decomposition loss."""
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Compute gradients
            # dL/dU[i*m+j, c] = -2 * sum_{i',j',k'} R[i',j',k'] * V[j'*p+k',c] * W[k'*n+i',c] * delta(i*m+j == i'*m+j')
            # More precisely:
            # L = sum_{i,j,k} (T[i,j,k] - sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])^2
            # dL/dU[a,c] = -2 * sum_{i,j,k: i*m+j=a} R[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
            
            dU = np.zeros((n * m, rank), dtype=np.float64)
            dV = np.zeros((m * p, rank), dtype=np.float64)
            dW = np.zeros((p * n, rank), dtype=np.float64)
            
            R = compute_residual(U, V, W, rank)
            
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    for k in range(p):
                        r_val = R[i, j, k]
                        if abs(r_val) < 1e-15:
                            continue
                        v_idx = j * p + k
                        w_idx = k * n + i
                        for c in range(rank):
                            u_val = U[u_idx, c]
                            v_val = V[v_idx, c]
                            w_val = W[w_idx, c]
                            dU[u_idx, c] += -2.0 * r_val * v_val * w_val
                            dV[v_idx, c] += -2.0 * r_val * u_val * w_val
                            dW[w_idx, c] += -2.0 * r_val * u_val * v_val
            
            # Update with learning rate
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        return U, V, W
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact (within tolerance)."""
        R = compute_residual(U, V, W, rank)
        return np.max(np.abs(R)) < 1e-9
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try to find lower rank decompositions
    # Known lower bound: max(n*m, m*p, n*p) = max(15, 30, 18) = 30
    # Try ranks from 30 up to 89
    for target_rank in range(30, 90):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Random initialization with small values
            U = np.random.uniform(-1, 1, (n * m, target_rank))
            V = np.random.uniform(-1, 1, (m * p, target_rank))
            W = np.random.uniform(-1, 1, (p * n, target_rank))
            
            # Run gradient descent with adaptive learning rate
            lr = 0.05
            for epoch in range(5):
                if time.time() > deadline:
                    break
                
                U, V, W = gradient_descent(U, V, W, target_rank, lr=lr, max_steps=300)
                
                # Check if we found a valid decomposition
                if verify(U, V, W, target_rank):
                    best_U, best_V, best_W, best_rank = U, V, W, target_rank
                    # Convert to lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # Decay learning rate
                lr *= 0.5
            
            # Check if the final result is close enough
            R = compute_residual(U, V, W, target_rank)
            max_err = np.max(np.abs(R))
            if max_err < 1e-6:
                # Try to polish with smaller learning rate
                for polish_epoch in range(3):
                    if time.time() > deadline:
                        break
                    U, V, W = gradient_descent(U, V, W, target_rank, lr=0.001, max_steps=500)
                    R = compute_residual(U, V, W, target_rank)
                    if np.max(np.abs(R)) < 1e-9:
                        best_U, best_V, best_W, best_rank = U, V, W, target_rank
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return the best result found (schoolbook if nothing better)
    U_list = best_U.tolist() if isinstance(best_U, np.ndarray) else best_U
    V_list = best_V.tolist() if isinstance(best_V, np.ndarray) else best_V
    W_list = best_W.tolist() if isinstance(best_W, np.ndarray) else best_W
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
