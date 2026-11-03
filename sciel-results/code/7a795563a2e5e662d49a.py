"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 15
    pn = p * n  # 15
    total_terms = n * m * p  # 45
    
    # Schoolbook as fallback
    def schoolbook():
        rank = total_terms
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = total_terms
    
    # Try to find lower rank decompositions
    # Use the fact that for 3x3x3, rank 21 is known (Laderman)
    # For 3x3x5, try ranks from 21 upward
    
    def verify(U, V, W, tol=1e-6):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
                    s = np.sum(val)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def round_to_half(x):
        """Round to nearest half-integer."""
        return np.round(x * 2.0) / 2.0
    
    def try_rank(rank, max_iter=5000, lr=0.01):
        """Try to find a rank-rank decomposition via gradient descent."""
        np.random.seed(42 + rank)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        # Initialize with some structure
        # Set diagonal-ish entries to 1
        for c in range(rank):
            U[c % nm, c] = 1.0
            V[c % mp, c] = 1.0
            W[c % pn, c] = 1.0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute gradients
            # f = sum_{i,j,k} (sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1)^2
            # df/dU[i,j,c] = 2*(sum_c' U[i,j,c']*V[j,k,c']*W[k,i,c] - 1) * V[j,k,c]*W[k,i,c]
            
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err = 2.0 * (s - 1.0)
                        for c in range(rank):
                            grad_U[i*m+j, c] += err * V[j*p+k, c] * W[k*n+i, c]
                            grad_V[j*p+k, c] += err * U[i*m+j, c] * W[k*n+i, c]
                            grad_W[k*n+i, c] += err * U[i*m+j, c] * V[j*p+k, c]
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Check convergence
            if it % 100 == 0:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
                            s = np.sum(s)
                            max_err = max(max_err, abs(s - 1.0))
                if max_err < 1e-4:
                    # Try rounding
                    Ur = round_to_half(U)
                    Vr = round_to_half(V)
                    Wr = round_to_half(W)
                    if verify(Ur, Vr, Wr, tol=1e-9):
                        return Ur, Vr, Wr
                    # Also try without rounding check
                    if verify(U, V, W, tol=1e-6):
                        return Ur, Vr, Wr
        
        return None
    
    # Try Laderman-inspired approach for 3x3x3 core
    # First, let's try to get a good 3x3x3 decomposition, then extend
    
    # Actually, let's try a more structured approach.
    # For 3x3x5, we can think of it as 3x3x3 + 3x3x2
    
    # Try various ranks
    for rank in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, max_iter=3000, lr=0.005)
        if result is not None:
            U, V, W = result
            if verify(U, V, W, tol=1e-9):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                break
    
    # Convert to lists of integers/half-integers
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
