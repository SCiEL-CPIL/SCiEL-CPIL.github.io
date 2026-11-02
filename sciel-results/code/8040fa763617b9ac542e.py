"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-minimal exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    The decomposition satisfies:
    T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where T[i, j, k] = 1 if and only if the (i,j,k) entry of the multiplication tensor is 1.
    For matrix multiplication C = A * B where A is n x m and B is m x p,
    C[i, k] = sum_j A[i, j] * B[j, k].
    The tensor T has T[i, j, k] = 1 for all i, j, k (it's the all-ones tensor in the 
    standard bilinear form representation), and the decomposition computes C[i,k] = sum_j A[i,j]B[j,k].
    
    The rank is the number of columns in U, V, W.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with the schoolbook decomposition of rank n*m*p = 105
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
    
    def to_numpy(U_list, V_list, W_list):
        return (np.array(U_list, dtype=float),
                np.array(V_list, dtype=float),
                np.array(W_list, dtype=float))
    
    def verify(U, V, W):
        """Check if the decomposition is exact."""
        U_np, V_np, W_np = to_numpy(U, V, W)
        n, m, p = DIMENSIONS
        # Build the multiplication tensor T
        # T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication, T[i, j, k] = 1 for all i, j, k
        # because C[i,k] = sum_j A[i,j] B[j,k], so the coefficient of A[i,j]B[j,k] is 1.
        
        # Compute the decomposed tensor
        # T_decomp[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        T_decomp = np.zeros((n, m, p), dtype=float)
        for c in range(U_np.shape[1]):
            u_col = U_np[:, c]  # shape (n*m,)
            v_col = V_np[:, c]  # shape (m*p,)
            w_col = W_np[:, c]  # shape (p*n,)
            
            # Reshape to 2D
            u_2d = u_col.reshape(n, m)
            v_2d = v_col.reshape(m, p)
            w_2d = w_col.reshape(p, n)
            
            # For each (i, j, k), add u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
        
        # The target tensor is all ones
        T_target = np.ones((n, m, p), dtype=float)
        return np.allclose(T_decomp, T_target, atol=1e-10)
    
    # Verify the initial solution
    assert verify(U, V, W), "Initial schoolbook decomposition failed verification"
    
    # Try to find a lower rank decomposition using iterative refinement
    # We use a simple approach: try to merge columns that are similar
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try various lower ranks
    for target_rank in range(100, 15, -1):
        if time.time() > deadline:
            break
        
        # Try to construct a decomposition of target_rank
        # Use a random initialization and gradient-free optimization
        
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization with integer/half-integer values
            U_try = [[0] * target_rank for _ in range(n * m)]
            V_try = [[0] * target_rank for _ in range(m * p)]
            W_try = [[0] * target_rank for _ in range(p * n)]
            
            # Initialize with random small values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1, -0.5, 0, 0.5, 1]
            for i in range(n * m):
                for c in range(target_rank):
                    U_try[i][c] = random.choice(vals)
            for j in range(m * p):
                for c in range(target_rank):
                    V_try[j][c] = random.choice(vals)
            for k in range(p * n):
                for c in range(target_rank):
                    W_try[k][c] = random.choice(vals)
            
            # Verify
            if verify(U_try, V_try, W_try):
                found = True
                break
        
        if found:
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = target_rank
            break  # Found a valid decomposition at this rank
    
    # Also try a structured approach: block diagonal by rows
    # Each row i of A (1x5) times B (5x7) gives a 1x7 row of C
    # This requires rank 5 per row, total rank 15
    # But this only works if we can separate the rows completely
    
    if time.time() < deadline - 5:
        # Try the row-wise decomposition: rank = n * m = 3 * 5 = 15
        # For row i, we need to compute C[i, k] = sum_j A[i, j] B[j, k]
        # This is a 1 x 5 x 7 tensor, which has rank 5 (minimum)
        # The full 3 x 5 x 7 tensor can be decomposed as direct sum of 3 such tensors
        
        target_rank = n * m  # 15
        U_try = [[0] * target_rank for _ in range(n * m)]
        V_try = [[0] * target_rank for _ in range(m * p)]
        W_try = [[0] * target_rank for _ in range(p * n)]
        
        # For each row i, assign m columns
        for i in range(n):
            base_col = i * m
            for j in range(m):
                col = base_col + j
                U_try[i * m + j][col] = 1
                V_try[j * p][col] = 1  # This is wrong; V should be indexed by (j, k)
                # Actually, for the 1x5x7 tensor, we need:
                # T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
                # But our convention is U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
                # So for a fixed i, we want:
                # For each (j, k), the coefficient should be 1
                # Using rank m=5 for this block:
                # Let c = j (0 to 4)
                # U[i*m+j, c] = 1 if c == j, else 0
                # V[j*p+k, c] = 1 if c == j, else 0
                # W[k*n+i, c] = 1 if c == j, else 0
                # Then T_decomp[i, j, k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
                # = sum_c (1 if c==j) (1 if c==j) (1 if c==j) = 1 if j is in range, which is always 1
                # But this gives T[i,j,k] = 1 for all j,k, which is correct!
                
                # Wait, let me redo this:
                # For fixed i, we use columns base_col to base_col + m - 1
                # For column c = base_col + j (where j is 0 to m-1):
                #   U[i*m + j'][c] = 1 if j' == j, else 0
                #   V[j'*p + k'][c] = 1 if j' == j, else 0  (for all k')
                #   W[k'*n + i][c] = 1  (for all k')
                # Then T_decomp[i, j', k'] = sum_c U[i*m+j',c] V[j'*p+k',c] W[k'*n+i,c]
                # The only c that contributes is c = base_col + j'
                # U[i*m+j', base_col+j'] = 1
                # V[j'*p+k', base_col+j'] = 1
                # W[k'*n+i, base_col+j'] = 1
                # So T_decomp[i, j', k'] = 1 for all j', k'
                # This is correct!
        
        # Let me rebuild this properly
        U_try = [[0] * target_rank for _ in range(n * m)]
        V_try = [[0] * target_rank for _ in range(m * p)]
        W_try = [[0] * target_rank for _ in range(p * n)]
        
        for i in range(n):
            base_col = i * m
            for j in range(m):
                col = base_col + j
                # U[i*m + j][col] = 1
                U_try[i * m + j][col] = 1
                # V[j*p + k][col] = 1 for all k
                for k in range(p):
                    V_try[j * p + k][col] = 1
                # W[k*n + i][col] = 1 for all k
                for k in range(p):
                    W_try[k * n + i][col] = 1
        
        # Verify this row-wise decomposition
        if verify(U_try, V_try, W_try):
            best_U, best_V, best_W = U_try, V_try, W_try
            best_rank = target_rank
    
    # Final verification
    assert verify(best_U, best_V, best_W), "Final decomposition failed verification"
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
