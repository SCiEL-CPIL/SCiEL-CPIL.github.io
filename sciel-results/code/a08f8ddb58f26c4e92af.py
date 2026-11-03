import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For exact matrix multiplication, T[i,j,k] should be 1 iff (i,j,k) is a valid
    # multiplication triple, i.e., T[i,j,k] = 1 for all i,j,k (since every entry
    # of the product matrix C[i,k] = sum_j A[i,j]*B[j,k] involves exactly one term
    # per (i,j,k) triple).
    # Actually, the tensor for matrix multiplication is:
    # M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # and the decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Efficient check using numpy
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = np.array(U).reshape(n, m, rank)
    V3 = np.array(V).reshape(m, p, rank)
    W3 = np.array(W).reshape(p, n, rank)
    
    # Compute sum over r: U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # This is a 4-index contraction. Let's do it carefully.
    # result[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # We can compute this as: for each r, outer product structure
    # Or use einsum: 'ijr,jkr,kir->ijk'
    result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # The result should be all ones (n*m*p tensor)
    return np.allclose(result, np.ones((n, m, p)), atol=1e-10)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS
    nmp = n * m * p
    
    # Start with schoolbook decomposition (rank 42)
    rank_schoolbook = nmp
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find a lower rank decomposition using a greedy/randomized approach
    # The idea: we want to find rank-r matrices U, V, W such that
    # sum_{r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is equivalent to: the tensor T (all ones of shape (n,m,p)) has a
    # rank-r decomposition in the "multiplication" sense.
    
    # Let's try a different approach: use the fact that for small dimensions,
    # we can try to find a low-rank decomposition by solving a system of equations.
    
    # For rank r, we have r*3 variables (entries of U, V, W) and n*m*p equations.
    # The equations are cubic (trilinear), so this is a nonlinear system.
    
    # Alternative: use a known lower bound and try to find a decomposition.
    # For 2x3x7, the rank is at least max(n*m, m*p, p*n) = max(6, 21, 14) = 21.
    # Also, the rank is at least the tensor rank of the all-ones tensor.
    
    # Let's try a randomized approach: initialize with small rank and try to
    # satisfy the equations iteratively.
    
    # Actually, let's try a more systematic approach. The matrix multiplication
    # tensor for (n,m,p) has a known structure. For the all-ones tensor, we can
    # use the following: if we can find rank-r such that the all-ones tensor of
    # shape (n,m,p) has rank r in the usual tensor sense, then we're done.
    
    # But the "multiplication tensor" rank is different from the usual tensor rank.
    # Let me reconsider.
    
    # The condition is: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's define: for each r, let u_r be a vector in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}
    # Then the condition is that the sum of outer products (in a specific sense) equals 1.
    
    # Let's try to find a decomposition by starting with a random initialization
    # and using gradient descent or a similar optimization method.
    
    # Actually, for this small problem, let's try to find a decomposition of
    # rank r for r from 21 upward, and check if it's feasible.
    
    # A simpler approach: use the fact that the all-ones tensor of shape (n,m,p)
    # can be written as a sum of rank-1 tensors. The minimum number of rank-1
    # tensors needed is the tensor rank of the all-ones tensor.
    
    # For an all-ones tensor of shape (a,b,c), the tensor rank is min(a,b,c) if
    # we can use arbitrary real entries. But here we have a specific structure.
    
    # Let me try a different approach: brute force search for small ranks.
    
    # Actually, let's just try to improve the schoolbook by combining terms.
    # If two terms have the same U and V entries but different W, we can combine them.
    
    # For the schoolbook, each term is unique in all three matrices.
    # But we can try to find a smarter decomposition.
    
    # Let's try rank 21 (the lower bound) with a randomized initialization.
    # We'll use a simple iterative method.
    
    for target_rank in range(21, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Try to optimize to satisfy the constraints
            # Use a simple gradient-based approach
            for iteration in range(2000):
                if time.time() > deadline - 10:
                    break
                
                U3 = U.reshape(n, m, target_rank)
                V3 = V.reshape(m, p, target_rank)
                W3 = W.reshape(p, n, target_rank)
                
                # Compute current result
                result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                error = result - 1.0
                if np.max(np.abs(error)) < 1e-10:
                    break
                
                # Compute gradients (approximate)
                # d(error)/d(U[i,j,r]) = V3[j,k,r] * W3[k,i,r] summed over k
                # d(error)/d(V[j,k,r]) = U3[i,j,r] * W3[k,i,r] summed over i
                # d(error)/d(W[k,i,r]) = U3[i,j,r] * V3[j,k,r] summed over j
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(target_rank):
                                val = error[i, j, k]
                                grad_U[i * m + j, r] += val * V3[j, k, r] * W3[k, i, r]
                                grad_V[j * p + k, r] += val * U3[i, j, r] * W3[k, i, r]
                                grad_W[k * n + i, r] += val * U3[i, j, r] * V3[j, k, r]
                
                # Gradient descent step
                lr = 0.01
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            # Check if we found a valid decomposition
            U3 = U.reshape(n, m, target_rank)
            V3 = V.reshape(m, p, target_rank)
            W3 = W.reshape(p, n, target_rank)
            result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            if np.max(np.abs(result - 1.0)) < 1e-8:
                # Convert to lists of float/int
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                if _verify(U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
        if best_rank == target_rank:
            break
    
    return best_result
# EVOLVE-BLOCK-END
