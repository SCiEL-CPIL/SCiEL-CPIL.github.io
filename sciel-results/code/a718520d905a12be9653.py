"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x4 matrix.
    
    Strategy:
    1. Try to find a rank-16 decomposition using a structured search based on
       known results for rectangular matrix multiplication.
    2. If that fails, fall back to rank-20 or higher using a more general approach.
    3. Use a deadline guard to ensure we return within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # The matrix multiplication tensor T has dimensions n*m*p = 2*4*4 = 32
    # We want to find the minimal rank r such that T can be written as sum of r rank-1 tensors
    
    # Known result: For 2x4 by 4x4 multiplication, the rank is at most 16
    # (this follows from the general bound min(n*p, m*n, m*p) * max(n,m,p) / max(n,m,p) ... 
    # actually let's think more carefully)
    
    # For multiplying an (n x m) matrix by an (m x p) matrix, the schoolbook rank is n*m*p.
    # But there are better algorithms. For 2x4 * 4x4:
    # - We can view this as computing 8 outputs, each being a dot product of a row of A (length 4) 
    #   with a column of B (length 4).
    # - The trivial rank is 8*4 = 32 (schoolbook).
    # - But we can do better using Strassen-like techniques or by exploiting structure.
    
    # Let's try a direct approach: parameterize a rank-r decomposition and solve for the coefficients.
    # For small ranks, we can use a greedy or iterative approach.
    
    # First, let's try rank = 16 (which is 2*4*4/2, suggesting some factor of 2 improvement)
    # Actually, let's think about this differently.
    
    # The output C = A @ B where A is 2x4, B is 4x4, so C is 2x4.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We want to write C[i,k] = sum_c U[i*4+j... wait, the indexing is:
    # U has shape (n*m, rank) = (8, rank), indexed by (i,j) -> i*m+j
    # V has shape (m*p, rank) = (16, rank), indexed by (j,k) -> j*p+k  
    # W has shape (p*n, rank) = (8, rank), indexed by (k,i) -> k*n+i
    
    # So: C[i,k] = sum_j A[i,j] * B[j,k] = sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c]
    # Wait, that's not quite right. Let me re-derive.
    
    # The tensor decomposition is:
    # T_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    # where T_{i,j,k} = 1 if this corresponds to a valid term in the matrix product.
    # Actually, the matrix multiplication tensor has T_{i,j,k} = 1 for all i,j,k 
    # (it's the identity tensor in the sense that C_{i,k} = sum_j A_{i,j} B_{j,k}).
    
    # So we need: for all i, j, k: 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And for invalid combinations (which don't exist here since all i,j,k are valid), it would be 0.
    
    # Let's try a constructive approach for small ranks.
    
    # For rank r, we have r variables in each of U, V, W.
    # Total unknowns: r*(n*m + m*p + p*n) = r*(8 + 16 + 8) = 32*r
    # Constraints: n*m*p = 32 equations (one for each (i,j,k) triple)
    
    # So for r >= 1, we have more unknowns than equations, which is good.
    # But the system is nonlinear (trilinear).
    
    # Let's try a simple heuristic: fix W to be a permutation-like structure and solve for U and V.
    
    # Approach: Try rank values from 16 upward, and for each, try to find a valid decomposition
    # using a randomized iterative method.
    
    def try_rank(r, max_iter=5000, seed=42):
        """Try to find a rank-r decomposition."""
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # We want: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        
        # Use a simple iterative method: for each (i,j,k), adjust the factors to reduce the error.
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Compute current values
            errors = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for c in range(r):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        errors[i, j, k] = val - 1.0
            
            if np.max(np.abs(errors)) < 1e-10:
                # Check if we have an exact solution (within tolerance)
                # Verify more carefully
                exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            if abs(val - 1.0) > 1e-9:
                                exact = False
                                break
                        if not exact:
                            break
                    if not exact:
                        break
                if exact:
                    return U, V, W
            
            # Gradient-based update (simplified)
            # For each (i,j,k) with error e, we want to reduce |e|.
            # Update U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] in the direction that reduces error.
            
            lr = 0.01
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = errors[i, j, k]
                        if abs(e) < 1e-12:
                            continue
                        for c in range(r):
                            # Gradient with respect to U[i*m+j, c] is V[j*p+k, c] * W[k*n+i, c]
                            gU = V[j * p + k, c] * W[k * n + i, c]
                            # Gradient with respect to V[j*p+k, c] is U[i*m+j, c] * W[k*n+i, c]
                            gV = U[i * m + j, c] * W[k * n + i, c]
                            # Gradient with respect to W[k*n+i, c] is U[i*m+j, c] * V[j*p+k, c]
                            gW = U[i * m + j, c] * V[j * p + k, c]
                            
                            U[i * m + j, c] -= lr * e * gU
                            V[j * p + k, c] -= lr * e * gV
                            W[k * n + i, c] -= lr * e * gW
        
        return None
    
    # Try ranks from 16 upward
    for r in range(16, 33):
        if time.time() > deadline - 5:
            break
        
        result = try_rank(r, max_iter=2000, seed=r)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = [[float(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[float(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[float(W[i, c]) for c in range(r)] for i in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition of rank 32
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
