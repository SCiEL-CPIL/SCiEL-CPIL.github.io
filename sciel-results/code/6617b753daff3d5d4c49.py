import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a low-rank exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    
    Strategy: 
    1. Build the target tensor T for M_{3,3,7}.
    2. Attempt to find a rank-minimal decomposition using a structured approach 
       based on Strassen's algorithm for the 3x3x3 core, extended to 7 columns.
    3. Use a local optimization approach (Gauss-Newton style or gradient descent) 
       to refine a random or structured initialization towards an exact solution.
    4. If no better rank is found, fall back to the schoolbook rank 63.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    
    dim_U = n * m  # 9
    dim_V = m * p  # 21
    dim_W = p * n  # 21
    
    # Build the target tensor T[a,b,c]
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Precompute the linear operator L: (U, V, W) -> residual
    # We want to minimize || T - sum_r U[:,r] V[:,r] W[:,r] ||_F^2
    # This is a tensor completion / rank minimization problem.
    
    # Helper to compute the tensor from U, V, W
    def tensor_from_factors(U, V, W):
        # U: (9, R), V: (21, R), W: (21, R)
        # Result: (9, 21, 21)
        # Efficient way: sum of outer products
        # R = U.shape[1]
        # T = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.einsum('ir,jr,kr->ijk', U, V, W)
    
    def residual(U, V, W):
        T_est = tensor_from_factors(U, V, W)
        return T - T_est
    
    # Try to find a decomposition with rank < 63
    # We know rank(M_{3,3,3}) = 23.
    # For M_{3,3,7}, a trivial bound is 23 + 4*9 = 59? No, that's not how it works.
    # A better bound: M_{3,3,7} can be seen as M_{3,3,3} (for first 3 cols) + 4 * M_{3,3,1}.
    # Rank(M_{3,3,1}) = 9.
    # So rank <= 23 + 4*9 = 59.
    # Can we do better?
    # Strassen's algorithm for 3x3 uses 23 multiplications.
    # If we apply Strassen to each of the 7 columns independently, we get 7*23 = 161 (worse than schoolbook).
    # But we can share the A-products.
    # In Strassen, the 23 products are linear combinations of A and B entries.
    # For M_{3,3,7}, the B entries are different for each column.
    # However, the A entries are the same.
    # So we can precompute the linear combinations of A (which are 23 of them, or fewer if we optimize).
    # Actually, the 23 products in Strassen are of the form L1(A, B_col1) * L2(A, B_col1) ...
    # No, Strassen products are P_k = L_k(A) * M_k(B).
    # Wait, Strassen splits A and B into blocks.
    # For 3x3, it's not a simple block split like 2x2. It's a specific 23-rank decomposition.
    # The 23 rank-1 tensors have specific U, V, W factors.
    # Let's try to construct a rank 23 + 4*9 = 59 decomposition first.
    # Then try to reduce it.
    
    # Actually, let's try a random search / optimization for ranks 50, 55, 59.
    # If we find a rank 59 solution, that's an improvement over 63.
    
    def try_rank(R, max_iters=1000, tol=1e-6):
        # Initialize U, V, W with small random values
        # Use random seed for reproducibility
        np.random.seed(42)
        
        # We want to solve T = sum U_r V_r W_r
        # This is non-convex.
        # We can use gradient descent on the factors.
        # To improve conditioning, we can use the "alternating least squares" or similar.
        
        # Let's try a simple gradient descent on the flattened parameters.
        # Or better, use the fact that for fixed V, W, the optimal U is given by linear least squares.
        # This is the ALS approach.
        
        U = np.random.randn(dim_U, R) * 0.01
        V = np.random.randn(dim_V, R) * 0.01
        W = np.random.randn(dim_W, R) * 0.01
        
        # To help convergence, we can initialize one factor with the target tensor's slices?
        # T is sparse.
        
        # ALS:
        # Fix V, W. Minimize || T - sum_r U_r (V_r \otimes W_r) ||^2
        # This is a linear system for U.
        # Let M_r = V_r \otimes W_r (flattened).
        # Then T_flattened = U @ M^T?
        # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
        # Let T_2[i, (j,k)] = sum_r U[i, r] (V[j, r] W[k, r])
        # Let M be a matrix of size (dim_V*dim_W, R) where M[(j,k), r] = V[j, r] W[k, r]
        # Then T_2 = U @ M^T
        # U = T_2 @ (M^T M)^{-1} M^T
        
        # Similarly for V and W.
        
        for it in range(max_iters):
            # Update U
            # M_U: (dim_V*dim_W, R)
            M_U = np.einsum('jr,kr->jkr', V, W).reshape(dim_V*dim_W, R)
            T_2 = T.reshape(dim_U, dim_V*dim_W)
            # Solve U @ M_U^T = T_2
            # U = T_2 @ pinv(M_U^T)
            try:
                U = np.linalg.lstsq(M_U.T, T_2.T, rcond=None)[0].T
            except:
                pass
            
            # Update V
            # M_V: (dim_U*dim_W, R)
            M_V = np.einsum('ir,kr->ikr', U, W).reshape(dim_U*dim_W, R)
            T_2 = T.reshape(dim_V, dim_U*dim_W).T # Wait, T[i,j,k]. We want T[j, (i,k)]
            # T[i,j,k] = U[i,r] V[j,r] W[k,r]
            # For fixed V, we want to solve for V[j,r]
            # T[i,j,k] = sum_r V[j,r] (U[i,r] W[k,r])
            # Let M_V[(i,k), r] = U[i,r] W[k,r]
            # Then T_2[j, (i,k)] = sum_r V[j,r] M_V[(i,k), r]
            # T_2 = V @ M_V^T
            # V = T_2 @ pinv(M_V^T)
            T_2 = T.transpose(1, 0, 2).reshape(dim_V, dim_U*dim_W)
            try:
                V = np.linalg.lstsq(M_V.T, T_2.T, rcond=None)[0].T
            except:
                pass
            
            # Update W
            # M_W: (dim_U*dim_V, R)
            M_W = np.einsum('ir,jr->ijr', U, V).reshape(dim_U*dim_V, R)
            # T[i,j,k] = sum_r W[k,r] (U[i,r] V[j,r])
            # T_2[k, (i,j)] = sum_r W[k,r] M_W[(i,j), r]
            # T_2 = W @ M_W^T
            T_2 = T.transpose(2, 0, 1).reshape(dim_W, dim_U*dim_V)
            try:
                W = np.linalg.lstsq(M_W.T, T_2.T, rcond=None)[0].T
            except:
                pass
            
            # Check convergence
            T_est = tensor_from_factors(U, V, W)
            err = np.linalg.norm(T - T_est)
            if err < tol:
                return U, V, W, err
        
        return None, None, None, np.inf

    # Try ranks
    # We know 63 is possible.
    # Try 59, 55, 50...
    # But ALS might get stuck in local minima.
    # Let's try a few restarts for lower ranks.
    
    best_rank = 63
    best_factors = None
    
    # Schoolbook fallback
    U_school = np.zeros((dim_U, 63), dtype=float)
    V_school = np.zeros((dim_V, 63), dtype=float)
    W_school = np.zeros((dim_W, 63), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1
                V_school[b, col] = 1
                W_school[c, col] = 1
                col += 1
    
    # Try to find better rank
    # Note: ALS can be slow. We have 120s limit.
    # Let's try rank 59 first.
    
    start_time = time.time()
    deadline = start_time + 110
    
    # Try rank 59
    for R in [59, 55, 50, 45]:
        if time.time() > deadline:
            break
        # Try 3 restarts
        for restart in range(3):
            if time.time() > deadline:
                break
            np.random.seed(100 + R + restart)
            # Use try_rank
            U, V, W, err = try_rank(R, max_iters=500, tol=1e-8)
            if U is not None and err < 1e-6:
                best_rank = R
                best_factors = (U, V, W)
                break
        if best_rank < 63:
            break
    
    # If we found a better rank, use it.
    if best_factors is not None:
        U, V, W = best_factors
        # Round to integers if close?
        # The problem asks for integer/half-integer entries.
        # ALS produces floats. We need to check if they are close to integers/halves.
        # If not, we might need a different approach or accept the float error?
        # The prompt says "integer/half-integer entries that exactly decompose".
        # This implies the solution MUST have integer/half-integer entries.
        # ALS might converge to a solution that is not exactly integer/half-integer.
        # However, for matrix multiplication tensors, the rank factors often have integer entries.
        # Let's check if the found factors are close to integers/halves.
        
        def is_close_to_half_integer(x):
            return np.all(np.abs(x - np.round(2*x)/2) < 1e-5)
        
        if is_close_to_half_integer(U) and is_close_to_half_integer(V) and is_close_to_half_integer(W):
            U = np.round(2*U)/2
            V = np.round(2*V)/2
            W = np.round(2*W)/2
            # Verify exactness
            T_est = tensor_from_factors(U, V, W)
            if np.allclose(T, T_est, atol=1e-5):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
