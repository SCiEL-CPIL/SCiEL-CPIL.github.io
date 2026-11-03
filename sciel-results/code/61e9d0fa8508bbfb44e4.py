"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # The multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # Index conventions:
    #   U row index: i*m + j  (i in 0..n-1, j in 0..m-1)
    #   V row index: j*p + k  (j in 0..m-1, k in 0..p-1)
    #   W row index: k*n + i  (k in 0..p-1, i in 0..n-1)
    
    # Schoolbook rank = n*m*p = 54
    # Try to find lower rank
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Function to verify a decomposition
    def verify(U, V, W, rank):
        """Check if U, V, W correctly decompose the multiplication tensor."""
        # For each (i,k), check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta... 
        # Actually, the decomposition means:
        # C[i,k] = sum_j A[i,j]*B[j,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k' V[j*p+k',r]*B[j,k']) * W[k*n+i,r]
        # Wait, that's not right. Let me re-read the convention.
        
        # The tensor decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r
        # where T[i,j,k] is the coefficient for A[i,j]*B[j,k] in computing C[i,k]
        # Actually T[i,j,k] should be 1 if this term contributes to C[i,k], which it always does.
        
        # The correct check: for all (i,j,k), the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] 
        # should equal the original tensor value.
        
        # Original tensor: T[i,j,k] = 1 for all valid (i,j,k) since each term A[i,j]*B[j,k] 
        # contributes to C[i,k] with coefficient 1.
        
        # Wait, that's not right either. The tensor T has indices such that:
        # C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all i,j,k.
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a decomposition using a greedy/iterative approach
    # Start with schoolbook and try to reduce rank
    
    def schoolbook_decomp():
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
    
    # First, try some known low-rank constructions
    
    # For M_{3,3,6}, one approach: exploit that p=6=2*3
    # We can think of B as two 3x3 blocks: B = [B1 | B2]
    # Then C = [A*B1 | A*B2]
    # If we use rank r for 3x3*3x3, we get rank 2r for the full product
    # Best known for M_3 is 23 (Strassen-like), so rank 46?
    # But actually the best known rank for 3x3 matrix multiplication is 23.
    
    # Let me try to construct a rank-23 decomposition for 3x3*3x3 first,
    # then extend to 3x3*3x6.
    
    # Actually, let me try a simpler approach: use the fact that we can share
    # computations between the two 3x3 multiplications.
    
    # Alternative: try random/structured low-rank decompositions and verify
    
    # Let me try rank values from low to high and see if we can find a valid one
    
    # For efficiency, let's try a few candidate ranks
    candidate_ranks = [18, 20, 22, 24, 25, 26, 27, 28, 30, 32, 36, 40, 42, 45, 48, 50, 54]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            continue
            
        # Try to find a decomposition of this rank using iterative refinement
        # Initialize with random or structured values
        
        # Use a structured initialization based on schoolbook
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Initialize using first min(target_rank, n*m*p) schoolbook terms
        num_init = min(target_rank, n*m*p)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < target_rank:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        
        # Try iterative refinement
        for iteration in range(100):
            if time.time() > deadline:
                break
                
            # Check if current decomposition is valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                    
            if valid:
                best_rank = target_rank
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                break
                
            # If not valid, try to fix errors
            # Compute error tensor
            error = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        error[i,j,k] = 1.0 - val
                        
            # Find the largest error
            max_idx = np.unravel_index(np.argmax(np.abs(error)), error.shape)
            i, j, k = max_idx
            err = error[i,j,k]
            
            if abs(err) < 1e-12:
                continue
                
            # Try to adjust one of the rank components
            # Pick a random rank component
            r = np.random.randint(target_rank)
            
            # Gradient-based update
            # dU[i*m+j, r] += eta * V[j*p+k, r] * W[k*n+i, r] * err
            # dV[j*p+k, r] += eta * U[i*m+j, r] * W[k*n+i, r] * err  
            # dW[k*n+i, r] += eta * U[i*m+j, r] * V[j*p+k, r] * err
            
            eta = 0.1
            U[i*m+j, r] += eta * V[j*p+k, r] * W[k*n+i, r] * err
            V[j*p+k, r] += eta * U[i*m+j, r] * W[k*n+i, r] * err
            W[k*n+i, r] += eta * U[i*m+j, r] * V[j*p+k, r] * err
            
        if time.time() > deadline:
            break
    
    # If no better decomposition found, use schoolbook
    if best_U is None:
        best_U, best_V, best_W, best_rank = schoolbook_decomp()
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
