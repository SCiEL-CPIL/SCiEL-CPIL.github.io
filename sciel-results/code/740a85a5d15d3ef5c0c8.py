<<<<<<< SEARCH
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120 - 10  # 110s safety margin
    
    # Schoolbook baseline: rank 54
    R_school = n * m * p
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, r] = 1.0
                V_school[j * p + k, r] = 1.0
                W_school[k * n + i, r] = 1.0
                r += 1
    
    # Try to find a rank-19 decomposition for 3x3 * 3x3
    # Known result: minimal rank for 3x3*3x3 is 19
    # We'll use a known rank-19 decomposition
    
    # Rank-19 decomposition for 3x3 matrix multiplication
    # Based on the known minimal decomposition
    
    # U3: (9, 19), V3: (9, 19), W3: (9, 19)
    # For 3x3 * 3x3 -> 3x3: C[i,k] = sum_j A[i,j] * B[j,k]
    # U indexed by (i,j), V indexed by (j,k), W indexed by (k,i)
    
    U3 = np.zeros((9, 19))
    V3 = np.zeros((9, 19))
    W3 = np.zeros((9, 19))
    
    # Known rank-19 decomposition coefficients (from literature)
    # This is a verified minimal decomposition for 3x3*3x3
    
    # Using the standard rank-19 decomposition:
    # The 19 terms can be constructed as follows:
    
    # First, the 9 diagonal terms (schoolbook part)
    for i in range(3):
        for j in range(3):
            U3[i*3+j, 0] = 1.0
            V3[j*3+i, 0] = 1.0
            W3[i*3+j, 0] = 1.0
    
    # Actually, let me use a proper rank-19 decomposition
    # Based on the work of Landsberg and others
    
    # Here's a verified rank-19 decomposition for 3x3*3x3:
    # I'll use the known explicit form
    
    # Reset
    U3 = np.zeros((9, 19))
    V3 = np.zeros((9, 19))
    W3 = np.zeros((9, 19))
    
    # Term 0-8: Standard diagonal-like terms
    # These handle the "diagonal" contributions
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            U3[idx, i*3+j] = 1.0
            V3[j*3+i, i*3+j] = 1.0
            W3[i*3+j, i*3+j] = 1.0
    
    # Terms 9-18: Additional terms to handle off-diagonal interactions
    # Using known rank-19 structure
    
    # Let me use a different approach: verify if rank 19 is achievable
    # by checking known decompositions
    
    # Actually, the minimal rank for 3x3*3x3 is indeed 19, but 
    # constructing it explicitly is complex. Let me try a 
    # computational approach to verify and construct it.
    
    # For now, let's use a safe approach: 
    # Split B into two 3x3 blocks, use schoolbook for each
    # This gives rank 27 (9+9+9 = 27? No, 9 for first block + 9 for second = 18? 
    # Wait, 3x3*3x3 schoolbook is rank 9, so two blocks give rank 18)
    
    # Actually wait: for 3x3 * 3x3, schoolbook rank is 3*3*3 = 27
    # For 3x3 * 3x6, schoolbook rank is 3*3*6 = 54
    
    # If I split B = [B1 | B2] where B1, B2 are 3x3:
    # C = A*B = [A*B1 | A*B2]
    # A*B1 has rank 27 (schoolbook) or 19 (minimal)
    # A*B2 has rank 27 (schoolbook) or 19 (minimal)
    # Total: 54 (schoolbook) or 38 (if we can share)
    
    # But we can't simply add ranks because the U factors must be consistent.
    # However, if we use the SAME U factors for both multiplications,
    # we can potentially reduce the total rank.
    
    # Let me try: use rank-19 decomposition for each 3x3*3x3
    # but with the SAME U matrix. This requires that the U factors 
    # for both multiplications are identical, which they are since 
    # both use the same A matrix.
    
    # So: U_total = U3 (shape 9x19)
    # V_total = [V3_for_B1, V3_for_B2] (shape 18x19)
    # W_total = [W3_for_B1, W3_for_B2] (shape 18x19)
    
    # But wait, this only works if the decomposition is "linear" in B.
    # In tensor decomposition, C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For B1: C1[i,k] = sum_r U[i,j,r] * V1[j,k,r] * W1[k,i,r]
    # For B2: C2[i,k] = sum_r U[i,j,r] * V2[j,k,r] * W2[k,i,r]
    # If U, W are the same, then total rank is 19, not 38!
    
    # This is the key insight: we can reuse the U and W factors.
    
    # Now I need a valid rank-19 decomposition for 3x3*3x3.
    # Let me construct one using known results.
    
    # Based on the literature, here's a rank-19 decomposition:
    # I'll use the explicit form from known results.
    
    # Actually, let me try a different approach: use a numerical 
    # optimization to find a rank-r decomposition, starting from r=19.
    
    # But that's complex. Let me instead use a known explicit decomposition.
    
    # Here's a verified rank-19 decomposition for 3x3*3x3:
    
    # I'll use the following approach: start with schoolbook (rank 27)
    # and try to reduce by finding linear dependencies.
    
    # For safety, let me just return the schoolbook decomposition
    # which is guaranteed correct at rank 54.
    
    if time.time() > deadline - 5:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to construct a lower-rank decomposition
    # Using the block structure: C = [A*B1 | A*B2]
    
    # For each 3x3*3x3 multiplication, use a rank-r decomposition
    # and share the U factors.
    
    # Let me try rank 20 for 3x3*3x3 (slightly above minimal 19 for safety)
    
    # Actually, let me use a known explicit rank-19 decomposition.
    # Based on the work of Bini, Capovani, Lotti, and Romani (1979),
    # and later improvements.
    
    # Here's a concrete rank-19 decomposition:
    
    U3 = np.zeros((9, 19))
    V3 = np.zeros((9, 19))
    W3 = np.zeros((9, 19))
    
    # Terms 0-8: Handle the "diagonal" A[i,i]*B[i,k] type terms
    for i in range(3):
        for k in range(3):
            r = i * 3 + k
            U3[i*3+i, r] = 1.0
            V3[i*3+k, r] = 1.0
            W3[k*3+i, r] = 1.0
    
    # Terms 9-18: Handle off-diagonal terms
    # This is where it gets complex. Let me use a different strategy.
    
    # I'll use a numerical approach: try to find a rank-r decomposition
    # by solving a system of equations.
    
    # For rank r, we need to find U (9xr), V (9xr), W (9xr) such that:
    # sum_{r=0}^{r-1} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if (i,j,k) is valid, 0 otherwise
    
    # This is a nonlinear system. Let me try a simple approach:
    # Start with random initialization and use gradient descent.
    
    # But given time constraints, let me just return schoolbook.
    
    # Actually, let me try one more thing: use the fact that for 
    # 3x3*3x6, we can potentially do better than 54 by exploiting 
    # the rectangular structure.
    
    # Known lower bound: rank >= max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Upper bound: 54 (schoolbook)
    
    # Let me try rank 36: use schoolbook for each half of B
    # U: (9, 36), V: (18, 36), W: (18, 36)
    
    # For B1 (columns 0-2): use schoolbook with rank 27
    # For B2 (columns 3-5): use schoolbook with rank 27
    # But we can share U factors if we structure it right.
    
    # Actually, the simplest improvement: just use schoolbook but 
    # verify it's correct.
    
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
=======
import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120 - 10  # 110s safety margin
    
    # Schoolbook baseline: rank 54
    R_school = n * m * p
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, r] = 1.0
                V_school[j * p + k, r] = 1.0
                W_school[k * n + i, r] = 1.0
                r += 1
    
    # Build target tensor T[i,j,k] = 1 for all valid indices
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    target_tensor = np.ones((n, m, p))
    
    # Try to find a lower-rank decomposition using numerical optimization
    # Start with schoolbook and try to reduce rank
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = R_school
    
    # Function to verify a decomposition
    def verify_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces the target tensor."""
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        rank = U.shape[1]
        result = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
        return np.allclose(result, target_tensor, atol=1e-10)
    
    # Function to compute error
    def compute_error(U, V, W):
        """Compute the error between the decomposition and target tensor."""
        rank = U.shape[1]
        result = np.zeros((n, m, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        result[i, j, k] += u_r[i, j] * v_r[j, k] * w_r[k, i]
        return np.linalg.norm(result - target_tensor)
    
    # Try to find a rank-r decomposition using gradient descent
    def find_decomposition(target_rank, max_iters=5000, lr=0.01, seed=42):
        """Find a rank-target_rank decomposition using gradient descent."""
        if time.time() > deadline - 20:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U = rng.randn(n * m, target_rank) * 0.1
        V = rng.randn(m * p, target_rank) * 0.1
        W = rng.randn(p * n, target_rank) * 0.1
        
        # Also try initialization from schoolbook (truncated)
        if target_rank <= R_school:
            U_init = U_school[:, :target_rank].copy()
            V_init = V_school[:, :target_rank].copy()
            W_init = W_school[:, :target_rank].copy()
            # Scale to help optimization
            scale = np.sqrt(R_school / target_rank)
            U = U_init * scale
            V = V_init * scale
            W = W_init / scale
        
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = compute_error(U, V, W)
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Compute gradients
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # d/dU[i,j,r] = V[j,k,r] * W[k,i,r] * (result[i,j,k] - target[i,j,k])
                        # d/dV[j,k,r] = U[i,j,r] * W[k,i,r] * (result[i,j,k] - target[i,j,k])
                        # d/dW[k,i,r] = U[i,j,r] * V[j,k,r] * (result[i,j,k] - target[i,j,k])
                        
                        # Compute current contribution
                        contrib = np.zeros(target_rank)
                        for r in range(target_rank):
                            contrib[r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        error = contrib - target_tensor[i, j, k]
                        
                        for r in range(target_rank):
                            U_grad[i*m+j, r] += V[j*p+k, r] * W[k*n+i, r] * error
                            V_grad[j*p+k, r] += U[i*m+j, r] * W[k*n+i, r] * error
                            W_grad[k*n+i, r] += U[i*m+j, r] * V[j*p+k, r] * error
            
            # Update parameters
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            # Compute error
            err = compute_error(U, V, W)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-10:
                break
        
        if best_err < 1e-10:
            return best_U, best_V, best_W
        return None
    
    # Try different target ranks from high to low
    for target_rank in range(R_school - 1, 17, -1):
        if time.time() > deadline - 30:
            break
        
        result = find_decomposition(target_rank, max_iters=3000, lr=0.005, seed=target_rank)
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W):
                best_rank = target_rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                print(f"Found valid decomposition at rank {target_rank}")
                break
    
    return best_result
>>>>>>> REPLACE
