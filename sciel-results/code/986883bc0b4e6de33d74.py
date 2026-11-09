"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U[0][__len__] if U else 0
    # Build the multiplication tensor T[i,j,k] = delta
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # Actually the tensor is T[i,j,k] = 1 (the standard matrix multiplication tensor has T[i,j,k]=1)
    # Let me re-read: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all valid i,j,k
    # And the decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, let me re-read the index conventions.
    # U has shape (n*m, rank), so U[i*m+j, c]
    # V has shape (m*p, rank), so V[j*p+k, c]
    # W has shape (p*n, rank), so W[k*n+i, c]
    # And the tensor element is T[i,j,k] which should equal sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # For matrix multiplication, T[i,j,k] = 1 (it's the all-ones tensor in this context? No...)
    
    # Actually, the matrix multiplication tensor for C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all i,j,k (this is the standard definition)
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _rank_of(U, V, W):
    """Return the rank (number of columns)."""
    if not U:
        return 0
    return len(U[0])

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Start with schoolbook decomposition (rank = n*m*p = 128)
    rank_schoolbook = n * m * p
    
    # Try to find a better decomposition using random search / optimization
    # Strategy: Try to find a low-rank decomposition by starting from a random 
    # initialization and using gradient-based or combinatorial search.
    
    # For now, let's try a structured approach:
    # 1. Start with schoolbook
    # 2. Try to reduce rank by finding linear dependencies
    
    # Actually, let me think about this differently. The matrix multiplication tensor
    # T[i,j,k] = 1 for all i,j,k in the appropriate ranges.
    # We want to find the minimum rank r such that there exist U (n*m x r), V (m*p x r), W (p*n x r)
    # with T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] for all i,j,k.
    
    # This is equivalent to finding a bilinear algorithm for matrix multiplication.
    # For 4x4 * 4x8, the known best bounds are... let me think.
    
    # One approach: Use the fact that we can pad the 4x4 * 4x8 to a larger square problem
    # and use known results, or use direct optimization.
    
    # Let me try a simple approach: start with schoolbook and try to merge terms.
    
    # First, let's just return the schoolbook decomposition as a baseline.
    # Then try to improve if we have time.
    
    def make_schoolbook():
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
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Try to find a better decomposition
    # Approach: Use random vectors and try to fit the tensor
    
    # Let's try a different approach: use the structure of the problem.
    # The tensor T[i,j,k] = 1 can be written as a sum of rank-1 tensors.
    # A rank-1 tensor is u * v * w where u is n*m dimensional, v is m*p dimensional, w is p*n dimensional.
    
    # For the all-ones tensor, we can use the following observation:
    # T[i,j,k] = sum_{a=1}^{n} sum_{b=1}^{m} sum_{c=1}^{p} delta_{i,a} delta_{j,b} delta_{k,c}
    # This is the schoolbook decomposition.
    
    # Can we do better? For matrix multiplication, the rank is related to the
    # number of scalar multiplications needed.
    
    # Let me try a random search approach with verification.
    
    # Try ranks from a lower bound up to the schoolbook rank.
    # Lower bound: The tensor T has a certain tensor rank. For the all-ones tensor
    # of dimensions (n*m) x (m*p) x (p*n), the rank is at least max(n*m, m*p, p*n)
    # but that's not tight.
    
    # Actually, for the matrix multiplication tensor, the rank is at least
    # n*m + m*p + p*n - 2*min(n,m,p) or something like that. Let me just try.
    
    # Let me try a gradient-based approach:
    # Initialize U, V, W randomly and try to minimize the error.
    
    def try_random_decomposition(rank, num_trials=100, seed=None):
        """Try to find a decomposition of the given rank using random initialization."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        for trial in range(num_trials):
            # Random initialization
            U = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(n * m)]
            V = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(m * p)]
            W = [[rng.uniform(-1, 1) for _ in range(rank)] for _ in range(p * n)]
            
            # Simple gradient descent
            lr = 0.01
            for step in range(100):
                # Compute gradient
                dU = [[0.0] * rank for _ in range(n * m)]
                dV = [[0.0] * rank for _ in range(m * p)]
                dW = [[0.0] * rank for _ in range(p * n)]
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                            # dT/dU[i*m+j,c] = V[j*p+k,c] * W[k*n+i,c]
                            # dT/dV[j*p+k,c] = U[i*m+j,c] * W[k*n+i,c]
                            # dT/dW[k*n+i,c] = U[i*m+j,c] * V[j*p+k,c]
                            # Loss = sum_{i,j,k} (T[i,j,k] - 1)^2
                            # dLoss/dU[i*m+j,c] = -2 * (T[i,j,k] - 1) * V[j*p+k,c] * W[k*n+i,c]
                            # etc.
                            t_val = 0.0
                            for c in range(rank):
                                t_val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                            err = t_val - 1.0
                            for c in range(rank):
                                dU[i*m+j][c] += -2 * err * V[j*p+k][c] * W[k*n+i][c]
                                dV[j*p+k][c] += -2 * err * U[i*m+j][c] * W[k*n+i][c]
                                dW[k*n+i][c] += -2 * err * U[i*m+j][c] * V[j*p+k][c]
                
                # Update
                for i in range(n*m):
                    for c in range(rank):
                        U[i][c] -= lr * dU[i][c]
                for j in range(m*p):
                    for c in range(rank):
                        V[j][c] -= lr * dV[j][c]
                for k in range(p*n):
                    for c in range(rank):
                        W[k][c] -= lr * dW[k][c]
            
            # Check if we found a valid decomposition
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try to find a better decomposition
    # Start with a lower rank and increase
    
    # For 4x4 * 4x8, let's try ranks from 50 up to 127
    # But this might be too slow. Let's be strategic.
    
    # Actually, let me think about what's achievable. The schoolbook rank is 128.
    # Can we do better? For square matrices, Strassen's algorithm gives rank 7 for 2x2
    # instead of 8. For larger matrices, the rank is known to be less than n^3.
    
    # For 4x4 * 4x8, let's try to use a block structure.
    # We can think of the 4x8 matrix as two 4x4 blocks.
    # C = A * B where A is 4x4, B is 4x8 = [B1, B2] where B1, B2 are 4x4.
    # C = [A*B1, A*B2]
    # If we could compute A*B1 and A*B2 using the same intermediate products,
    # we might save some multiplications.
    
    # One idea: Use the fact that B1 and B2 share the same row structure.
    # But I'm not sure this helps directly.
    
    # Let me try a different approach: use known results or a smarter search.
    
    # For now, let's just return the schoolbook decomposition if we can't find a better one.
    # The random search above might be too slow for the time limit.
    
    # Let me try a few random decompositions with lower ranks.
    
    # Try rank 100
    if time.time() < deadline - 10:
        result = try_random_decomposition(100, num_trials=10, seed=42)
        if result is not None:
            best_result = result
            best_rank = 100
    
    # Try rank 80
    if time.time() < deadline - 10 and best_rank > 80:
        result = try_random_decomposition(80, num_trials=10, seed=43)
        if result is not None:
            best_result = result
            best_rank = 80
    
    # Try rank 64
    if time.time() < deadline - 10 and best_rank > 64:
        result = try_random_decomposition(64, num_trials=10, seed=44)
        if result is not None:
            best_result = result
            best_rank = 64
    
    return best_result
# EVOLVE-BLOCK-END
