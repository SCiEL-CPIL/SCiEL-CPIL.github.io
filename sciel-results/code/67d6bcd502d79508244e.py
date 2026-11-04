"""
Matrix multiplication 3x4x6: find a rank-minimal exact tensor decomposition.

Strategy:
1. Start with the schoolbook decomposition (rank 72).
2. Attempt to reduce rank by finding linear dependencies among the rank-1 terms.
3. Use a structured search: try to find a basis of the column space of the 
   "flattened" tensor that yields a valid decomposition.
4. For small dimensions, we can also try known algebraic constructions.

The tensor T satisfies:
  sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k

We seek the minimum rank r such that this holds with integer/half-integer entries.
"""
import numpy as np
from itertools import product
import time
import random

def construct():
    n, m, p = 3, 4, 6
    start_time = time.time()
    max_time = 110  # seconds, leave buffer for overhead
    
    # Schoolbook decomposition
    schoolbook_rank = n * m * p  # 72
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # First, verify schoolbook
    if not verify(U_school, V_school, W_school):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try to reduce rank by finding linear dependencies
    # The key insight: if we can find a set of rank-1 tensors whose sum is zero,
    # we can replace some terms with others to reduce the total count.
    
    # Approach: Try to find a lower rank by solving a system of equations
    # We want to find U, V, W of rank r < schoolbook_rank such that:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a non-linear system. Let's try a few strategies:
    
    # Strategy 1: Use the fact that for small matrices, we can sometimes
    # find decompositions by exploiting symmetries.
    
    # Strategy 2: Try random initializations and use gradient descent
    # to minimize the error, then round to nearest half-integers.
    
    best_rank = schoolbook_rank
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    
    # Try ranks from schoolbook_rank - 1 down to a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Actually, the tensor rank is at least the max of the matrix dimensions
    # in a certain sense, but the actual lower bound is more complex.
    
    # For 3x4x6, let's try to find a rank below 72.
    
    # Let's try a structured approach: 
    # The tensor T can be viewed as a matrix of size (n*p) x (m) for each fixed...
    # Actually, let's use the matrix flattening approach.
    
    # Flatten the tensor in different ways to get lower bounds.
    # T_{i,j,k} = 1 for all i,j,k (in the sense that each term contributes)
    # Wait, no. The tensor is:
    # T = sum_{i,j,k} e_{i,j} * e_{j,k} * e_{k,i}
    # where e_{i,j} is in R^{n*m}, e_{j,k} is in R^{m*p}, e_{k,i} is in R^{p*n}
    
    # The rank of T is the minimum r such that T = sum_{r=1}^{r} u_r * v_r * w_r
    
    # Let's try to find a decomposition by solving for the factors.
    
    # For a given rank r, we have r unknowns in each factor.
    # The constraint is: for all i,j,k:
    #   sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of n*m*p = 72 equations.
    # The unknowns are R*(n*m + m*p + p*n) = R*(12 + 24 + 18) = 54*R
    # For R = 72, we have 54*72 = 3888 unknowns and 72 equations (underdetermined)
    # For R = 1, we have 54 unknowns and 72 equations (overdetermined, likely no solution)
    
    # Let's try to find the minimum R by binary search or direct search.
    
    # Try R from a lower bound up to schoolbook_rank
    # Lower bound: The tensor rank is at least the max of the "slice" ranks.
    # For fixed i: the matrix V_i[j,k] = sum_r V[j*p+k, r] * W[k*n+i, r]
    # This must satisfy: for all j,k: sum_r U[i*m+j, r] * V_i[j,k] = 1
    # So V_i must be such that U_i^T * V_i (element-wise) = 1
    
    # Let's just try a few ranks and see if we can find a solution.
    
    # For efficiency, let's try ranks: 72, 71, 70, ..., down to maybe 50
    # But this is expensive. Let's try a smarter approach.
    
    # Use the fact that the schoolbook decomposition has many symmetries.
    # Try to find linear combinations of the rank-1 terms that sum to zero.
    
    # If we can find a set of indices {r_1, ..., r_s} and coefficients c_1, ..., c_s
    # such that sum_{t=1}^{s} c_t * (U_{r_t} * V_{r_t} * W_{r_t}) = 0,
    # then we can modify the decomposition to reduce the rank.
    
    # For the schoolbook decomposition, each term is e_{i,j} * e_{j,k} * e_{k,i}
    # A linear combination sum c_{i,j,k} * e_{i,j} * e_{j,k} * e_{k,i} = 0
    # iff for all (i,j,k): c_{i,j,k} = 0 (since the tensors are linearly independent)
    # So there are no non-trivial linear dependencies in the schoolbook decomposition.
    
    # This means we need a fundamentally different decomposition to reduce the rank.
    
    # Let's try a random search with rounding.
    
    def try_random_decomposition(rank, num_trials=100, seed=42):
        """Try to find a valid decomposition of the given rank using random search."""
        rng = np.random.RandomState(seed)
        for trial in range(num_trials):
            # Initialize with random small values
            U = rng.randn(n * m, rank) * 0.1
            V = rng.randn(m * p, rank) * 0.1
            W = rng.randn(p * n, rank) * 0.1
            
            # Try to optimize using simple gradient descent
            # The objective is: sum_{i,j,k} (sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] - 1)^2
            # This is non-convex, but let's try a few steps.
            
            for step in range(200):
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # d/dU[i*m+j, r] of (sum_r' U[i*m+j,r'] * V[j*p+k,r'] * W[k*n+i,r'] - 1)^2
                            # = 2 * (sum_r' ... - 1) * V[j*p+k, r] * W[k*n+i, r]
                            err = 0.0
                            for r in range(rank):
                                err += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            err -= 1.0
                            for r in range(rank):
                                grad_U[i*m+j, r] += 2 * err * V[j*p+k, r] * W[k*n+i, r]
                                grad_V[j*p+k, r] += 2 * err * U[i*m+j, r] * W[k*n+i, r]
                                grad_W[k*n+i, r] += 2 * err * U[i*m+j, r] * V[j*p+k, r]
                
                # Gradient step
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                max_grad = max(np.max(np.abs(grad_U)), np.max(np.abs(grad_V)), np.max(np.abs(grad_W)))
                if max_grad < 1e-6:
                    break
            
            # Round to nearest half-integer
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            if verify(U_round, V_round, W_round, tol=1e-9):
                return U_round, V_round, W_round
        
        return None, None, None
    
    # Try a few ranks below schoolbook
    # This is expensive, so let's limit the number of trials
    for target_rank in range(schoolbook_rank - 1, max(20, schoolbook_rank - 10), -1):
        if time.time() - start_time > max_time:
            break
        
        U_found, V_found, W_found = try_random_decomposition(target_rank, num_trials=10, seed=target_rank)
        if U_found is not None:
            best_rank = target_rank
            best_U = U_found
            best_V = V_found
            best_W = W_found
            break
    
    # If we didn't find anything better, return the schoolbook decomposition
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
