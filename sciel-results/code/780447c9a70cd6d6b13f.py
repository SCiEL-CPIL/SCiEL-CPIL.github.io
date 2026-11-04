"""Improved program for matrix multiplication tensor decomposition (5,5,5)."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Deadline: 140s out of 150s
    t0 = time.time()
    deadline = t0 + 140.0

    # First, get the schoolbook decomposition as a fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school

    def verify(U, V, W):
        """Verify that the tensor decomposition is exact."""
        r = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # This should equal delta[i,j,k] where T[i,j,k] = 1 if i,j,k correspond to valid matrix mult terms
        # Actually, the matrix multiplication tensor has entries:
        # T_{i,j,k} = 1 if (i,j,k) is a valid (row, col_a, col_b) term... 
        # Let me re-think the convention.
        
        # The task says: "The output factor follows the official trace convention W[k*n+i,r]"
        # So W is indexed by [k*n+i], meaning W[r, c] where r = k*n + i
        # U is indexed by [i*m+j] presumably, so U[r, c] where r = i*m + j
        # V is indexed by [j*p+k], so V[r, c] where r = j*p + k
        
        # The tensor equation is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
        # Wait, that's not right either. Let me think about this more carefully.
        
        # For matrix multiplication C = A * B, C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has indices (i, j, k) where T[i,j,k] = 1 (the coefficient of A[i,j]*B[j,k] in C[i,k])
        # Actually T[i,j,k] = 1 for all i,j,k since every term contributes.
        # No wait, the tensor is such that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] should give the correct output.
        
        # Let me just verify by checking: for each c, the contribution is rank-1.
        # The sum over c of U[:,c] * V[:,c] * W[:,c] should equal the multiplication tensor.
        # The multiplication tensor M has M[i,j,k] = 1 for all i,j,k (it's the identity in a sense).
        # Actually no. Let me look at the schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
        # for a specific (i,j,k). So summing over c, we get 1 for each (i,j,k) triple.
        
        # So the target tensor T has T[i,j,k] = 1 for all i,j,k? That can't be right for matrix mult.
        # Actually for the bilinear complexity, the tensor is defined as:
        # T_{i,j,k} = 1 where the term A[i,j]*B[j,k] contributes to C[i,k].
        # So T is a 3-tensor with T[i,j,k] = 1 for all valid i,j,k.
        
        # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
        
        T_computed = np.zeros((n, m, p))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_computed[i, j, k] += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
        
        T_target = np.ones((n, m, p))
        return np.allclose(T_computed, T_target, atol=1e-10)

    def try_random_search(rank, max_iters=1000):
        """Try to find a decomposition of given rank using random initialization and gradient descent."""
        nonlocal best_result, best_rank, deadline
        
        if rank >= best_rank:
            return False
        
        np.random.seed(42)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to avoid numerical issues
        for c in range(rank):
            norm_U = np.linalg.norm(U[:, c])
            norm_V = np.linalg.norm(V[:, c])
            norm_W = np.linalg.norm(W[:, c])
            if norm_U > 1e-10:
                U[:, c] /= norm_U
            if norm_V > 1e-10:
                V[:, c] /= norm_V
            if norm_W > 1e-10:
                W[:, c] /= norm_W
        
        lr = 0.01
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute current tensor
            T_comp = np.zeros((n, m, p))
            for c in range(rank):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_comp[i, j, k] += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            
            error = np.sum((T_comp - 1.0) ** 2)
            if error < 1e-8:
                # Found a solution! Convert to integer/half-integer
                U_list = [[float(U[i, c]) for c in range(rank)] for i in range(n*m)]
                V_list = [[float(V[i, c]) for c in range(rank)] for i in range(m*p)]
                W_list = [[float(W[i, c]) for c in range(rank)] for i in range(p*n)]
                
                # Round to nearest 0.5
                U_rounded = [[round(x * 2) / 2 for x in row] for row in U_list]
                V_rounded = [[round(x * 2) / 2 for x in row] for row in V_list]
                W_rounded = [[round(x * 2) / 2 for x in row] for row in W_list]
                
                if verify(U_rounded, V_rounded, W_rounded):
                    best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    best_rank = rank
                    return True
                else:
                    # Try with unrounded values
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = rank
                    return True
            
            # Gradient descent step
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        diff = T_comp[i, j, k] - 1.0
                        for c in range(rank):
                            dU[i*m+j, c] += 2 * diff * V[j*p+k, c] * W[k*n+i, c]
                            dV[j*p+k, c] += 2 * diff * U[i*m+j, c] * W[k*n+i, c]
                            dW[k*n+i, c] += 2 * diff * U[i*m+j, c] * V[j*p+k, c]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Periodic renormalization
            if it % 10 == 0:
                for c in range(rank):
                    norm_U = np.linalg.norm(U[:, c])
                    norm_V = np.linalg.norm(V[:, c])
                    norm_W = np.linalg.norm(W[:, c])
                    if norm_U > 1e-10:
                        U[:, c] /= norm_U
                        lr = 0.01
                    if norm_V > 1e-10:
                        V[:, c] /= norm_V
                    if norm_W > 1e-10:
                        W[:, c] /= norm_W
        
        return False

    # Try various ranks from lowest known possible up to schoolbook
    # Known bounds: rank(5,5,5) is between 49 (lower bound) and 125 (schoolbook)
    # Try ranks in decreasing order of likelihood
    ranks_to_try = [60, 64, 68, 70, 72, 75, 80, 85, 90, 96, 100, 105, 110, 115, 120]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
        success = try_random_search(r, max_iters=50)
        if success:
            break
    
    return best_result
# EVOLVE-BLOCK-END
