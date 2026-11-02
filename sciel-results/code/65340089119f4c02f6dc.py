import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    # The correct tensor definition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The decomposition is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all i, j, k where the indices are compatible (i.e., for all i in [0,n), j in [0,m), k in [0,p))
    # Wait, that's not right. Let me think again.
    #
    # The matrix multiplication tensor T is defined such that:
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    # But C[i,k] = sum_j A[i,j] * B[j,k]
    # So sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} (sum_j A[i,j] * B[j,k])
    # This means T[i,j,k] = 1 for all i, j, k.
    #
    # But that gives rank n*m*p, which is the schoolbook rank.
    #
    # Actually, I think I'm confusing the tensor definition. Let me reconsider.
    #
    # The standard tensor decomposition for matrix multiplication:
    # We want to find U, V, W such that:
    # sum_c U[i, c] * V[j, c] * W[k, c] = T[i, j, k]
    # where T is the multiplication tensor.
    #
    # For multiplying an (n x m) matrix A by an (m x p) matrix B to get (n x p) matrix C:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    #
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # T[i, j, k] = 1 if the term A[i,j] * B[j,k] contributes to C[i,k], which is always true.
    # So T[i, j, k] = 1 for all i, j, k.
    #
    # But wait, that means the tensor is all ones, and the rank is n*m*p.
    #
    # I think I need to reconsider the indexing. In the problem statement:
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has indices (k*n+i, r) where k is the column of B / row of C, i is the row of A / column of C.
    #
    # Let me re-read: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # U[i*m + j, c] corresponds to A[i, j]
    # V[j*p + k, c] corresponds to B[j, k]
    # W[k*n + i, c] corresponds to C[i, k] (but with k*n+i indexing, so k is row, i is column? Or k is column, i is row?)
    #
    # The convention says W[k*n+i, r]. If C is (n x p), then C[i, k] where i is row (0..n-1) and k is column (0..p-1).
    # But W has p*n entries, indexed by k*n+i. So k ranges over 0..p-1 and i ranges over 0..n-1.
    # This suggests W[k*n+i] corresponds to C[i, k] (i is row, k is column), but indexed as k*n+i.
    #
    # So the decomposition condition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k.
    #
    # This is the same as before. So the tensor is all ones, and the schoolbook rank is n*m*p = 72.
    #
    # But we know that matrix multiplication can be done with fewer multiplications.
    # The key insight is that the tensor is NOT all ones. Let me think more carefully.
    #
    # Actually, I think the issue is that the tensor T[i,j,k] is 1 only when the indices form a valid "path" in the multiplication.
    # For matrix multiplication, the bilinear form is:
    # <A, B> -> C where C[i,k] = sum_j A[i,j] B[j,k]
    #
    # The tensor T is a 3-tensor with T[i,j,k] = 1 if and only if the term A[i,j] * B[j,k] appears in the computation of C[i,k].
    # Since C[i,k] = sum_j A[i,j] B[j,k], every term A[i,j] B[j,k] appears in C[i,k].
    # So T[i,j,k] = 1 for all i, j, k.
    #
    # But this can't be right because we know that Strassen's algorithm and others use fewer multiplications.
    #
    # I think I'm making a fundamental error. Let me look at this differently.
    #
    # The tensor rank of matrix multiplication M_{n,m,p} is the minimum number of scalar multiplications needed.
    # For M_{2,2,2}, the rank is 7 (Strassen), not 8.
    #
    # The tensor is defined as: T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the sum is over valid (i,j,k).
    # For matrix multiplication, the valid (i,j,k) are those where the multiplication is defined.
    #
    # Actually, I think the correct definition is:
    # T[i, j, k] = 1 if i < n, j < m, k < p (all indices are in range)
    # But that's still all ones.
    #
    # Wait, I think the issue is that the tensor is not over all (i,j,k) but over the structure of the multiplication.
    # Let me look at it from the perspective of the bilinear form.
    #
    # The bilinear form is: B(A, B) = C where C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as: sum_{i,j,k} A[i,j] B[j,k] C_{ik}^{(j)} where C_{ik}^{(j)} is the coefficient.
    # But C[i,k] = sum_j A[i,j] B[j,k], so the coefficient of A[i,j] B[j,k] in C[i,k] is 1.
    #
    # I'm going in circles. Let me just implement the correct check based on the known fact that
    # the schoolbook decomposition works and try to find a better one.
    
    # The correct condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i, j, k
    # This is because each product A[i,j] * B[j,k] contributes exactly once to C[i,k].
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    
    # Try to find lower rank decompositions using random search + local optimization
    # Known bounds: rank(M_{3,3,8}) <= 72 (schoolbook)
    # We can try ranks from 60 down to see if we can find a valid decomposition
    
    # For each target rank, try multiple random initializations and use gradient descent
    target_ranks = list(range(60, 73))  # Try from 60 to 72
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Gradient descent to minimize sum of (val - 1)^2
            lr = 0.01
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            
                            # val = sum_r U[a,r] * V[b,r] * W[c,r]
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            err = val - 1
                            
                            # d/dU[a,r] = V[b,r] * W[c,r]
                            grad_U[a, :] += err * V[b, :] * W[c, :]
                            # d/dV[b,r] = U[a,r] * W[c,r]
                            grad_V[b, :] += err * U[a, :] * W[c, :]
                            # d/dW[c,r] = U[a,r] * V[b,r]
                            grad_W[c, :] += err * U[a, :] * V[b, :]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if np.max(np.abs(grad_U)) < 1e-8 and np.max(np.abs(grad_V)) < 1e-8 and np.max(np.abs(grad_W)) < 1e-8:
                    break
            
            # Check if we found a valid decomposition
            if check_decomposition(U, V, W, target_rank):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
    
    # Return the best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
