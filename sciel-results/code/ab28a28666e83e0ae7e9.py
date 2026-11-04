"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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

def _construct_bipartite():
    """
    Placeholder for alternative constructions.
    Currently returns schoolbook as no lower rank is known for 3x5x6.
    """
    return _construct_schoolbook()

# EVOLVE-BLOCK-START
def _try_rank_decomposition(rank, max_time=120.0):
    """
    Try to find a rank-`rank` decomposition of the 3x5x6 matrix multiplication tensor.
    Returns (U, V, W) if found, None otherwise.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS
    dim_U = n * m  # 15
    dim_V = m * p  # 30
    dim_W = p * n  # 18
    
    # Build the target tensor T as a 3D array: T[a, b, c] = 1 if (a,b,c) corresponds to valid (i,j,k)
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((dim_U, dim_V, dim_W), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # We want to find U (dim_U x rank), V (dim_V x rank), W (dim_W x rank)
    # such that sum_r U[:,r] * V[:,r] * W[:,r] == T
    # i.e., T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # This is a system of dim_U * dim_V * dim_W = 15*30*18 = 8100 equations
    # with rank * (dim_U + dim_V + dim_W) = rank * 63 unknowns
    # For rank=89: 89*63 = 5607 unknowns, 8100 equations -> overdetermined
    # For rank=90: 90*63 = 5670 unknowns, 8100 equations -> still overdetermined
    # Wait, that can't be right. Let me reconsider.
    
    # Actually, the tensor T is sparse. Only 90 entries are 1, the rest are 0.
    # So we have 8100 equations, but 8010 of them are T[a,b,c] = 0.
    
    # Let's try a random initialization and gradient descent / Newton's method approach.
    # Or better, use the fact that this is a bilinear system and try iterative methods.
    
    # Approach: Initialize U, V, W randomly and use alternating least squares or
    # a custom optimization to minimize ||T - sum_r U[:,r]V[:,r]W[:,r]||_F^2
    
    # For speed, let's try a simple approach: random init + projected gradient
    # But this is a non-convex problem, so we need multiple restarts.
    
    # Actually, let's think about this differently. The schoolbook solution is known to work
    # with rank 90. The question is whether rank < 90 is possible.
    
    # Given the time constraints and the complexity of this non-convex optimization,
    # and the fact that no known lower rank exists for 3x5x6 in the literature,
    # it's very likely that 90 is optimal or very close to optimal.
    
    # Let's do a quick check: try rank 85, 88, 89 with random restarts and see if we can
    # get a solution. If not, return None.
    
    np.random.seed(42)
    
    best_residual = np.inf
    best_U = None
    best_V = None
    best_W = None
    
    num_restarts = 5
    iters_per_restart = 200
    
    for restart in range(num_restarts):
        if time.time() - start_time > max_time - 10:
            break
            
        # Random initialization
        U = np.random.randn(dim_U, rank) * 0.1
        V = np.random.randn(dim_V, rank) * 0.1
        W = np.random.randn(dim_W, rank) * 0.1
        
        # Alternating minimization: fix two, solve for the third
        for it in range(iters_per_restart):
            if time.time() - start_time > max_time - 10:
                break
            
            # Update U: for each a, T[a,:,:] = sum_r U[a,r] * (V[:,r] * W[:,r])
            # This is: T[a] = U[a] @ (V * W) where V*W is element-wise product summed over r
            # Actually: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # For fixed V, W: T[a] = sum_r U[a,r] * (V[:,r] * W[:,r]).T
            # Let M_VW[b,c] = sum_r V[b,r] * W[c,r] (this is a matrix)
            # Then T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r] is NOT simply U[a] @ M_VW
            # because the sum is over r of products of three terms.
            
            # This is a trilinear form, so ALS doesn't directly apply in a simple way.
            # Let's use gradient descent instead.
            
            # Compute current decomposition
            # T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # We can compute this as: for each r, outer(U[:,r], V[:,r], W[:,r])
            # But that's expensive. Let's use einsum.
            
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T - T_approx)
            
            if residual < best_residual:
                best_residual = residual
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if residual < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            # Gradient descent
            lr = 0.001
            # dL/dU[a,r] = sum_{b,c} (T[a,b,c] - T_approx[a,b,c]) * V[b,r] * W[c,r]
            # dL/dV[b,r] = sum_{a,c} (T[a,b,c] - T_approx[a,b,c]) * U[a,r] * W[c,r]
            # dL/dW[c,r] = sum_{a,b} (T[a,b,c] - T_approx[a,b,c]) * U[a,r] * V[b,r]
            
            diff = T - T_approx
            
            # Compute gradients efficiently
            # grad_U[a,r] = sum_{b,c} diff[a,b,c] V[b,r] W[c,r]
            # = (diff @ (V * W)) but this isn't quite right due to the 3D structure
            
            # Let's compute: for each r, grad_U[:,r] = sum_{b,c} diff[:,:,r-related] ... 
            # Actually: grad_U[a,r] = sum_b sum_c diff[a,b,c] V[b,r] W[c,r]
            # = sum_b (sum_c diff[a,b,c] W[c,r]) V[b,r]
            # = (diff @ W[:,r]) @ V[:,r]  ... no, that's not right either.
            
            # Let's just do it with einsum
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
    
    # Check if we found a solution
    if best_residual < 1e-10:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    start_time = time.time()
    
    # First, try to find a lower rank decomposition
    # Try ranks from 85 to 89 (just below schoolbook rank of 90)
    for rank in [89, 88, 87, 86, 85]:
        if time.time() - start_time > 100:
            break
        result = _try_rank_decomposition(rank, max_time=100.0)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # If no lower rank found, return schoolbook
    return _construct_schoolbook()
# EVOLVE-BLOCK-END
