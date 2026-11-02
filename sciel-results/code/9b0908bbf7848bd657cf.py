"""Matrix multiplication 2x2 by 2x9 - rank-minimal decomposition search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 36
    
    def make_schoolbook(rank):
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
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank, tol=1e-9):
        """Check if the decomposition is exact."""
        # For each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 
        # the "tensor value" which for matrix mult is 1 if this is a valid term
        # Actually, the tensor T[i,j,k] for matrix multiplication is:
        # The multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k) in the schoolbook
        # But for a general decomposition, we need:
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k]
        # where T[i,j,k] is the matrix multiplication tensor.
        
        # The matrix multiplication tensor for (n x m) * (m x p) -> (n x p):
        # T[i, j, k] contributes to (A*B)_{i,k} via A_{i,j} * B_{j,k}
        # So the "target" tensor has T[i,j,k] = 1 for all i,j,k (each term is a product)
        # Wait, no. The tensor itself is defined such that:
        # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_j sum_c [U[i*m+j,c] A_{i,j}] [V[j*p+k,c] B_{j,k}] W[k*n+i,c]
        # This doesn't quite work. Let me reconsider.
        
        # The correct formulation: We want to compute C = A*B where C_{i,k} = sum_j A_{i,j} B_{j,k}
        # A rank-r decomposition gives r "linear forms" in A and B that multiply to give C.
        # Specifically: C_{i,k} = sum_c (sum_j U[i*m+j,c] * A_{i,j}) * (sum_k' V[j'*p+k',c] * B_{j',k'}) ... 
        # This is getting confused. Let me use the standard tensor definition.
        
        # The multiplication tensor T has indices (i, j, k) for i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 (each entry is 1, representing the product A_{i,j} * B_{j,k})
        # The rank decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Start with schoolbook
    best = make_schoolbook(schoolbook_rank)
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions using randomized optimization
    # The tensor T[i,j,k] = 1 for all i,j,k (n*m*p = 36 constraints)
    # We have r*(n*m + m*p + p*n) = r*(4 + 18 + 18) = r*40 variables
    # For r < 36, we have 40r variables and 36 equations, so r >= 1 is possible in principle
    
    # Use a simple approach: try random initializations and gradient descent
    # But given time constraints, let's try a few strategies
    
    rng = np.random.RandomState(42)
    
    # Strategy: Try ranks from lower to higher, with limited attempts
    # For this small problem, maybe rank 30-35 is achievable?
    
    # Actually, let's think about this differently. The matrix multiplication tensor
    # for 2x2 * 2x9 might have a known or discoverable lower bound.
    
    # Let's try a numerical search with random restarts
    def try_rank(r, max_iter=5000, seed=42):
        if time.time() > DEADLINE - 10:
            return None
        
        rng_local = np.random.RandomState(seed)
        
        # Initialize randomly
        U = rng_local.randn(n*m, r)
        V = rng_local.randn(m*p, r) 
        W = rng_local.randn(p*n, r)
        
        # Normalize to avoid scale issues
        for c in range(r):
            norm = np.sqrt(np.sum(U[:,c]**2) * np.sum(V[:,c]**2) * np.sum(W[:,c]**2))
            if norm > 1e-10:
                scale = 1.0 / (norm ** (1.0/3.0))
                U[:,c] *= scale
                V[:,c] *= scale
                W[:,c] *= scale
        
        # Gradient descent on the error
        # Error = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        
        lr = 0.01
        for it in range(max_iter):
            if time.time() > DEADLINE - 10:
                return None
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # s = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                        # ds/dc for each c: V[j*p+k,c]*W[k*n+i,c] for U
                        # etc.
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        err = s - 1.0
                        for c in range(r):
                            dU[i*m+j][c] += 2 * err * V[j*p+k][c] * W[k*n+i][c]
                            dV[j*p+k][c] += 2 * err * U[i*m+j][c] * W[k*n+i][c]
                            dW[k*n+i][c] += 2 * err * U[i*m+j][c] * V[j*p+k][c]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check convergence
            if it % 100 == 0:
                # Quick check
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            max_err = max(max_err, abs(s - 1.0))
                if max_err < 1e-6:
                    # Found a solution!
                    return (U.tolist(), V.tolist(), W.tolist())
        
        return None
    
    # Try a few ranks below schoolbook
    for r in range(schoolbook_rank - 1, max(schoolbook_rank - 10, 20), -1):
        if time.time() > DEADLINE - 15:
            break
        
        result = try_rank(r, max_iter=2000, seed=42 + r)
        if result is not None:
            U_list, V_list, W_list = result
            # Verify
            if verify_decomposition(U_list, V_list, W_list, r):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                break  # Found lower rank, stop
    
    return best
# EVOLVE-BLOCK-END
