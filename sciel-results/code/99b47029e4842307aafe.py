"""Matrix multiplication 2x3x6: rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _to_rational_matrix(mat):
    """Convert a numpy matrix to a matrix of Fractions."""
    n, m = mat.shape
    result = [[Fraction(0)] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            result[i][j] = Fraction(mat[i, j])
    return result

def _rational_rank(mat):
    """Compute exact rank of a matrix over rationals using Gaussian elimination."""
    n, m = mat.shape
    A = [row[:] for row in mat]
    rank = 0
    for col in range(m):
        # Find pivot
        pivot_row = -1
        for row in range(rank, n):
            if A[row][col] != 0:
                pivot_row = row
                break
        if pivot_row == -1:
            continue
        # Swap
        A[rank], A[pivot_row] = A[pivot_row], A[rank]
        # Eliminate
        pivot = A[rank][col]
        for row in range(rank + 1, n):
            if A[row][col] != 0:
                factor = A[row][col] / pivot
                for c in range(col, m):
                    A[row][c] -= factor * A[rank][c]
        rank += 1
    return rank

def _build_tensor_matrices(n, m, p):
    """
    Build the three matrix views of the multiplication tensor.
    
    The multiplication tensor T has entries T[i,j,k] = 1 if we're multiplying
    A[i,j] * B[j,k] to get C[i,k].
    
    We need to find rank-r decomposition:
    U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta (appropriate Kronecker structure)
    
    Actually, the standard formulation:
    For matrix multiplication C = A * B where A is n x m, B is m x p, C is n x p:
    
    The tensor T_{i,j,k} for the multiplication is such that:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    A rank-r decomposition means:
    C[i,k] = sum_c U[i*m+? , c] ... 
    
    Let me use the standard formulation more carefully.
    
    The multiplication tensor M_{(i,j),(j,k),(i,k)} - this is the 3-tensor where
    M[i,j,k] = 1 if the indices form a valid multiplication term.
    
    Actually for the tensor decomposition approach:
    
    We want: sum_c U[a,c] * V[b,c] * W[c_idx, c] = T[a,b,c_idx]
    
    where a indexes A (n*m), b indexes B (m*p), c_idx indexes C (n*p),
    and T[a,b,c_idx] = 1 iff a=(i,j), b=(j,k), c_idx=(i,k) for some i,j,k.
    
    Wait, the task says W has shape (p*n, rank) and uses convention W[k*n+i, r].
    So W indexes C as (k, i) where k is column (0..p-1) and i is row (0..n-1).
    So C index = k*n + i, which means C is stored as C[i,k] but indexed as k*n+i.
    
    Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    
    So W[row, col] where row = k*n + i, meaning the C matrix is indexed with
    column index k first, then row index i. So C[k*n+i] corresponds to C[i,k].
    
    The tensor T[a, b, w] where:
    - a = i*m + j (index into A, i in 0..n-1, j in 0..m-1)
    - b = j*p + k (index into B, j in 0..m-1, k in 0..p-1)  
    - w = k*n + i (index into C, k in 0..p-1, i in 0..n-1)
    
    T[a, b, w] = 1 iff there exist i,j,k such that a=i*m+j, b=j*p+k, w=k*n+i.
    Since a determines (i,j) and b determines (j,k), and w determines (k,i),
    T[a,b,w] = 1 iff the j from a matches the j from b, AND the k from b matches 
    the k from w, AND the i from a matches the i from w.
    
    For a rank-r decomposition: T[a,b,w] = sum_c U[a,c]*V[b,c]*W[w,c]
    
    This is a 3-tensor decomposition problem.
    
    One approach: for each fixed w, define T_w as the (n*m) x (m*p) matrix
    T_w[a,b] = T[a,b,w]. Then T_w has rank equal to the number of (i,j,k) tuples
    where w = k*n+i, i.e., where k and i are fixed. For fixed k,i, the valid (j) 
    values are j=0..m-1. So T_w has entries: T_w[i*m+j, j*p+k] = 1 for j=0..m-1.
    This is a matrix with m ones on a "diagonal-like" pattern. Its rank is m (if m>0).
    
    Similarly, for fixed a=(i,j), T_a is a (m*p) x (n*p) matrix... this gets complex.
    
    Let me use a different approach. The rank of the matrix multiplication tensor
    for n x m by m x p is at most n*m*p (schoolbook) and at least max(n*m, m*p, n*p).
    
    For 2x3x6: schoolbook rank = 36. Lower bound = max(6, 18, 12) = 18.
    
    Let me try to find a rank-18 decomposition by solving the system.
    """
    pass

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x6 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # A: 2x3, B: 3x6, C: 2x6
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (18, rank)  
    # W: (p*n, rank) = (12, rank)
    
    # Schoolbook rank = n*m*p = 36
    # Lower bound = max(n*m, m*p, n*p) = max(6, 18, 12) = 18
    
    # Let's try to find the minimal rank by attempting decompositions
    # starting from the lower bound.
    
    def _verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        n, m, p = DIMENSIONS
        # Check: sum_c U[a,c]*V[b,c]*W[w,c] should equal T[a,b,w]
        # T[a,b,w] = 1 iff a=i*m+j, b=j*p+k, w=k*n+i for consistent i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    w = k * n + i
                    s = sum(U[a][c] * V[b][c] * W[w][c] for c in range(rank))
                    if s != 1:
                        return False
        # Check all other entries are 0
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k in range(p):
                        b = j2 * p + k
                        for i2 in range(n):
                            for k2 in range(p):
                                w = k2 * n + i2
                                # Skip the valid entry
                                if i2 == i and j2 == j and k2 == k:
                                    continue
                                s = sum(U[a][c] * V[b][c] * W[w][c] for c in range(rank))
                                if s != 0:
                                    return False
        return True
    
    def _verify_fast(U, V, W, rank):
        """Faster verification using numpy."""
        n, m, p = DIMENSIONS
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i,j,k), check U[i*m+j,:]*V[j*p+k,:]*W[k*n+i,:] sum = 1
        # And for invalid combos, sum = 0
        
        # Build the full tensor check
        # T[a,b,w] = sum_c U[a,c]*V[b,c]*W[w,c]
        # This is a 3-tensor, but we can check slice by slice.
        
        # For each w, T[:, :, w] = U @ diag? No...
        # T[a,b,w] = sum_c U[a,c] * V[b,c] * W[w,c]
        # For fixed w: T[:, :, w] = U * (W[w,:] * V^T) element-wise... 
        # Actually T[:, :, w] = U @ (V * W[w,:])^T where * is element-wise
        # No: T[a,b,w] = sum_c U[a,c] * (V[b,c] * W[w,c])
        # So for fixed w: T_w = U @ (V * W[w,:][None,:])^T
        # where V * W[w,:][None,:] means V[b,c] * W[w,c]
        
        errors = 0.0
        for w in range(p * n):
            k = w // n
            i = w % n
            # V_w[b,c] = V[b,c] * W[w,c]
            Vw = V_arr * W_arr[w][None, :]
            # T_w[a,b] = sum_c U[a,c] * Vw[b,c] = U @ Vw^T
            Tw = U_arr @ Vw.T
            # Check valid entries
            for j in range(m):
                a = i * m + j
                b = j * p + k
                if abs(Tw[a, b] - 1.0) > 1e-9:
                    errors += abs(Tw[a, b] - 1.0)
            # Check invalid entries should be 0
            # Total entries: (n*m) * (m*p) = 6*18 = 108
            # Valid entries for this w: m = 3
            # Invalid: 108 - 3 = 105
            # We can just check all and subtract valid
            total_sum = np.sum(np.abs(Tw))
            valid_sum = 0.0
            for j in range(m):
                a = i * m + j
                b = j * p + k
                valid_sum += abs(Tw[a, b])
            invalid_sum = total_sum - valid_sum
            if invalid_sum > 1e-9:
                errors += invalid_sum
        
        return errors < 1e-9
    
    def _schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    # Try ranks from lower bound upward
    lower_bound = max(n * m, m * p, n * p)
    
    # For small problems, we can try to solve the nonlinear system
    # T[a,b,w] = sum_c U[a,c]*V[b,c]*W[w,c]
    
    # Strategy: Fix one factor and solve for others, or use iterative refinement.
    
    # Actually, let's try a structured approach.
    # The tensor for matrix multiplication has special structure.
    
    # For 2x3x6, let's try rank 18 first.
    # If that fails, try 19, 20, ... up to 36.
    
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try to find better decompositions
    for target_rank in range(lower_bound, best_rank):
        if time.time() > DEADLINE:
            break
        
        success = False
        
        # Method: Use the fact that for fixed w, T_w = U @ (V * W[w,:])^T
        # This is a rank-r matrix factorization of each T_w slice.
        
        # Alternative: iterative approach
        # Initialize randomly, then refine
        
        np.random.seed(42)
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > DEADLINE:
                break
            
            # Initialize U, V, W with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Iterative refinement: alternate fixing two factors and solving for the third
            # This is like a generalized ALS for 3-tensor decomposition
            
            for iteration in range(200):
                if time.time() > DEADLINE:
                    break
                
                # Fix V and W, solve for U
                # T[a,b,w] = sum_c U[a,c] * V[b,c] * W[w,c]
                # For each a: T[a,:,:] = sum_c U[a,c] * (V[:,c] * W[:,c])
                # This is: T[:, :, w] flattened... 
                # 
                # Actually, let's think of it differently.
                # For each (b,w): T[:,b,w] = sum_c U[:,c] * V[b,c] * W[w,c]
                # = U @ (V[b,:]*W[w,:])
                # So for each (b,w) pair, we have a linear system in U.
                # But we need it to hold for all (b,w) simultaneously.
                #
                # Stack all equations:
                # For each (b,w): T[:,b,w] = U @ (V[b,:]*W[w,:])
                # This gives (m*p)*(p*n) equations for U.
                # 
                # Let M be the (m*p)*(p*n) matrix where M[(b,w), c] = V[b,c]*W[w,c]
                # Then T_vec = U @ M^T where T_vec is the (n*m) x (m*p)*(p*n) matrix
                # T_vec[a, (b,w)] = T[a,b,w]
                
                # Build M: shape (m*p*p*n, target_rank)
                # M[b*p*n + w, c] = V[b,c] * W[w,c]
                M = np.zeros((m * p * p * n, target_rank))
                for b in range(m * p):
                    for w in range(p * n):
                        M[b * p * n + w, :] = V[b, :] * W[w, :]
                
                # Build T_vec: shape (n*m, m*p*p*n)
                T_vec = np.zeros((n * m, m * p * p * n))
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        for j2 in range(m):
                            for k in range(p):
                                b = j2 * p + k
                                for i2 in range(n):
                                    for k2 in range(p):
                                        w = k2 * n + i2
                                        if i2 == i and j2 == j and k2 == k:
                                            T_vec[a, b * p * n + w] = 1.0
                
                # Solve U @ M^T = T_vec, i.e., U = T_vec @ (M^T)^+ 
                # Using least squares: U = T_vec @ pinv(M^T)
                # Or: (M @ U^T)^T = T_vec => M @ U^T = T_vec^T
                # U^T = pinv(M) @ T_vec^T
                
                try:
                    U_new = np.linalg.lstsq(M, T_vec.T, rcond=None)[0].T
                    U = U_new
                except:
                    pass
                
                # Fix U and W, solve for V
                # T[a,b,w] = sum_c U[a,c] * V[b,c] * W[w,c]
                # For each (a,w): T[a,:,w] = V @ (U[a,:]*W[w,:])^T
                # M2[(a,w), c] = U[a,c] * W[w,c]
                # T2[b, (a,w)] = T[a,b,w]
                
                M2 = np.zeros((n * m * p * n, target_rank))
                for a in range(n * m):
                    for w in range(p * n):
                        M2[a * p * n + w, :] = U[a, :] * W[w, :]
                
                T2 = np.zeros((m * p, n * m * p * n))
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        for j2 in range(m):
                            for k in range(p):
                                b = j2 * p + k
                                for i2 in range(n):
                                    for k2 in range(p):
                                        w = k2 * n + i2
                                        if i2 == i and j2 == j and k2 == k:
                                            T2[b, a * p * n + w] = 1.0
                
                try:
                    V_new = np.linalg.lstsq(M2, T2.T, rcond=None)[0].T
                    V = V_new
                except:
                    pass
                
                # Fix U and V, solve for W
                # T[a,b,w] = sum_c U[a,c] * V[b,c] * W[w,c]
                # For each (a,b): T[a,b,:] = W @ (U[a,:]*V[b,:])^T
                # M3[(a,b), c] = U[a,c] * V[b,c]
                # T3[w, (a,b)] = T[a,b,w]
                
                M3 = np.zeros((n * m * m * p, target_rank))
                for a in range(n * m):
                    for b in range(m * p):
                        M3[a * m * p + b, :] = U[a, :] * V[b, :]
                
                T3 = np.zeros((p * n, n * m * m * p))
                for i in range(n):
                    for j in range(m):
                        a = i * m + j
                        for j2 in range(m):
                            for k in range(p):
                                b = j2 * p + k
                                for i2 in range(n):
                                    for k2 in range(p):
                                        w = k2 * n + i2
                                        if i2 == i and j2 == j and k2 == k:
                                            T3[w, a * m * p + b] = 1.0
                
                try:
                    W_new = np.linalg.lstsq(M3, T3.T, rcond=None)[0].T
                    W = W_new
                except:
                    pass
                
                # Check convergence
                if iteration % 20 == 0:
                    # Quick error check
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                a = i * m + j
                                b = j * p + k
                                w = k * n + i
                                s = np.dot(U[a, :], V[b, :] * W[w, :])
                                err += abs(s - 1.0)
                    if err < 1e-8:
                        success = True
                        break
            
            if success:
                # Verify fully
                U_list = [[float(x) for x in U[a, :]] for a in range(n * m)]
                V_list = [[float(x) for x in V[b, :]] for b in range(m * p)]
                W_list = [[float(x) for x in W[w, :]] for w in range(p * n)]
                
                if _verify(U_list, V_list, W_list, target_rank):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
        
        if success:
            break
    
    return best_result

# EVOLVE-BLOCK-END
