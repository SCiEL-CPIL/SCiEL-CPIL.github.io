"""Matrix multiplication 2x5 x 5x6 rank-minimal decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # The schoolbook decomposition has rank n*m*p = 60
    # We'll try to find a lower rank decomposition
    
    # For matrix multiplication, the tensor T is defined as:
    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The flattening bounds give rank >= max(n*m, m*p, p*n) = max(10,30,12) = 30
    # But we need to check if rank 30 is achievable
    
    # Let's try to construct a rank-30 decomposition
    # Using the "border rank" approach: if we can find a rank-r decomposition
    # that is a limit of rank-r decompositions, we can potentially use it
    
    # For 2x5 x 5x6, let's try rank = 30 (the m*p flattening bound)
    # The idea: use the structure where we factor through the middle dimension
    
    # Actually, for matrix multiplication, the standard lower bound is:
    # rank >= max(n*m, m*p, p*n) but this is not always tight
    # The Strassen algorithm shows that for 2x2x2, rank 7 < 8 = schoolbook
    
    # For our dimensions, let's attempt a constructive approach
    # We'll use the fact that we can potentially share computation
    
    # Let's try rank = 55, 50, 45, 40, 35, 30 in order
    # and check if a valid decomposition exists
    
    # First, let's verify the schoolbook works
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1.0
                    V[j*p + k][col] = 1.0
                    W[k*n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition using optimization
    # We'll use the tensor flattening approach
    
    # For a rank-r decomposition, we need matrices U (n*m x r), V (m*p x r), W (p*n x r)
    # such that for all (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]
    # Wait, that's not quite right. Let me reconsider.
    
    # The correct formulation: The multiplication tensor T has entries
    # T[(i,j), (j,k)] = delta_{i,i'} * delta_{k,k'} for the (i',k') output
    # Actually, the tensor for matrix multiplication is:
    # T[i, j, k, i', j', k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # No, that's the identity tensor.
    
    # Let me reclarify: For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    # The computation is C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i,j,k) and the "fiber" at (i,k) is the linear map
    # that takes (A[i,*], B[*,k]) to C[i,k]
    
    # The correct tensor formulation: We have a 3-tensor T with dimensions (n*p, n*m, m*p)
    # where T[(i,k), (i,j), (j,k)] = delta_{i,i} * delta_{j,j} * delta_{k,k} = 1
    # if the indices are consistent, 0 otherwise
    
    # Actually, the standard formulation: The multiplication tensor mu_{n,m,p}
    # lives in (K^n)^* ⊗ (K^m) ⊗ (K^p)^* ⊗ ... 
    # Let me use the standard: T ∈ (K^{n*m}) ⊗ (K^{m*p}) ⊗ (K^{n*p})^*
    # or equivalently, we factor T as sum of rank-1 tensors
    
    # For our purposes, the decomposition means:
    # Find U (n*m x r), V (m*p x r), W (n*p x r) such that
    # for all (i,j), (j',k), (i',k'): 
    #   sum_c U[i*m+j, c] * V[j'*p+k, c] = W[i'*n+k', c] if j=j' and i=i', k=k'
    #   = 0 otherwise
    # 
    # More precisely: The (i*m+j, j'*p+k) entry of U @ V.T should equal
    #   1 if j=j' AND (i,k) corresponds to output (i',k') with i'=i, k'=k
    #   0 otherwise
    # And W provides the output indexing
    
    # Let me use a different, clearer formulation:
    # We want to express the bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    # as a sum of r rank-1 bilinear maps.
    
    # A rank-1 bilinear map is: (A,B) -> f(A) * g(B) where f: K^{n*m} -> K, g: K^{m*p} -> K
    # and the output is a fixed matrix H (n x p) scaled by f(A)*g(B)
    # 
    # So: C = sum_{c=1}^r f_c(A) * g_c(B) * H_c
    # where H_c are fixed n x p matrices
    
    # Equivalently: C[i,k] = sum_c (sum_{i',j'} U[c, i'*m+j'] * A[i',j']) * (sum_{j'',k'} V[c, j''*p+k'] * B[j'',k']) * W[c, i*n+k]
    # 
    # Wait, I need to be more careful with the indexing convention.
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, r) and W[k*n + i, c] is the coefficient for output C[i,k] in term c.
    
    # Let me define:
    # U: (n*m) x r, where U[i*m+j, c] is the weight for A[i,j] in term c
    # V: (m*p) x r, where V[j*p+k, c] is the weight for B[j,k] in term c  
    # W: (p*n) x r, where W[k*n+i, c] is the weight for C[i,k] in term c
    # 
    # The decomposition requires:
    # sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = 1 if j=j', i=i', k=k'
    # = 0 otherwise
    # 
    # More precisely, for the bilinear form to compute C[i,k] = sum_j A[i,j]*B[j,k]:
    # We need: for all (i,k), sum_j (A[i,j]*B[j,k]) = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j' V[j'*p+k,c]*B[j',k]) * W[k*n+i,c]
    # 
    # This must hold for all A, B, so the coefficients must match:
    # For each c, the term contributes to C[i,k] via:
    #   (sum_j U[i*m+j,c]*A[i,j]) * (sum_j' V[j'*p+k,c]*B[j',k]) * W[k*n+i,c]
    # = W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j' V[j'*p+k,c]*B[j',k])
    # 
    # For this to equal sum_j A[i,j]*B[j,k] when summed over c, we need:
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'} for all i,j,j',k
    # 
    # Wait, that's not right either because U depends on i and V depends on k.
    
    # Let me use the standard tensor rank formulation:
    # The multiplication tensor T ∈ (K^{n*m}) ⊗ (K^{m*p}) ⊗ (K^{n*p})^*
    # has entries: T[(i,j), (j',k), (i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # 
    # A rank-1 decomposition is T = sum_c U_c ⊗ V_c ⊗ W_c where
    # U_c ∈ K^{n*m}, V_c ∈ K^{m*p}, W_c ∈ (K^{n*p})^*
    # 
    # So: sum_c U_c[i*m+j] * V_c[j'*p+k] * W_c[i'*n+k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    # This is the correct formulation. Let's implement a search for low rank.
    
    # Given time constraints, let's try a few specific ranks and use a simple approach
    # For 2x5x6, known results suggest the rank might be around 30-40
    
    # Let's try rank 30 first (the theoretical lower bound from m*p flattening)
    
    def try_rank(r):
        """Try to find a rank-r decomposition using random initialization and gradient descent."""
        if time.time() > deadline:
            return None
        
        # Initialize U, V, W randomly
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # The tensor T has entries T[(i,j), (j',k), (i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        # We want: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[i'*n+k', c] ≈ T[(i,j), (j',k), (i',k')]
        
        # For efficiency, we'll compute the error on a subset of entries
        # The tensor is sparse: T[(i,j), (j',k), (i',k')] is nonzero only when i=i', j=j', k=k'
        # There are n*m*p = 60 such nonzero entries
        
        # Let's compute the error for all nonzero entries
        # Error = sum over nonzero (i,j,j',k,i',k') with i=i', j=j', k=k' of:
        #   (sum_c U[i*m+j,c] * V[j'*p+k,c] * W[i'*n+k',c] - 1)^2
        # Plus we need the zero entries to be approximately 0, but that's a lot of entries
        
        # For a proper check, we need all entries, but that's n*m * m*p * p*n = 10*30*12 = 3600 entries
        # which is manageable
        
        # Let's build the full tensor check
        # For each (a,b,c) where a ∈ [0,n*m), b ∈ [0,m*p), c ∈ [0,p*n):
        #   a = i*m+j, b = j'*p+k, c = i'*n+k'
        #   T[a,b,c] = 1 if i=i' and j=j' and k=k' else 0
        #   We want: sum_r U[a,r] * V[b,r] * W[c,r] ≈ T[a,b,c]
        
        # This is a 3-tensor fitting problem. We can use gradient descent.
        
        lr = 0.01
        for iteration in range(500):
            if time.time() > deadline:
                return None
            
            # Compute gradient
            # dL/dU[a,r] = sum_{b,c} (pred[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            # dL/dV[b,r] = sum_{a,c} (pred[a,b,c] - T[a,b,c]) * U[a,r] * W[c,r]
            # dL/dW[c,r] = sum_{a,b} (pred[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
            
            # pred[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            
            # For efficiency, let's compute using matrix operations
            # pred = U @ V.T @ W.T reshaped appropriately
            # Actually: pred[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # = (U @ V.T)[a, r] * W[c, r] summed over r... no
            # = sum_r U[a,r] * V[b,r] * W[c,r]
            
            # Let's compute the error and gradient numerically for small problems
            # Since n*m=10, m*p=30, p*n=12, r is at most 60, this is feasible
            
            # Build the target tensor T
            T = np.zeros((n*m, m*p, p*n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        T[a, b, c] = 1.0
            
            # Compute prediction
            # pred[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # We can compute this as: for each r, outer(U[:,r], V[:,r]) @ ... 
            # Actually: pred = sum_r np.einsum('ar,br,cr->abc', U, V, W)
            
            pred = np.einsum('ar,br,cr->abc', U, V, W)
            
            error = np.sum((pred - T) ** 2)
            if error < 1e-8:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                # Round to nearest integer or half-integer if close
                for mat in [U_list, V_list, W_list]:
                    for i in range(len(mat)):
                        for j in range(len(mat[0])):
                            val = mat[i][j]
                            if abs(val - round(val)) < 1e-6:
                                mat[i][j] = float(round(val))
                            elif abs(2*val - round(2*val)) < 1e-6:
                                mat[i][j] = float(round(2*val) / 2)
                return {"U": U_list, "V": V_list, "W": W_list}
            
            if iteration % 100 == 0 and iteration > 0:
                pass  # Could log, but skip for speed
            
            # Compute gradient
            # dU[a,r] = 2 * sum_{b,c} (pred[a,b,c] - T[a,b,c]) * V[b,r] * W[c,r]
            # dV[b,r] = 2 * sum_{a,c} (pred[a,b,c] - T[a,b,c]) * U[a,r] * W[c,r]
            # dW[c,r] = 2 * sum_{a,b} (pred[a,b,c] - T[a,b,c]) * U[a,r] * V[b,r]
            
            diff = pred - T
            
            # dU[a,r] = 2 * sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            # = 2 * (diff @ (V * W.T).T)[a,r] ... let's be careful
            # For fixed a,r: sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
            # = sum_b V[b,r] * sum_c diff[a,b,c] * W[c,r]
            # Let temp[a,b,r] = sum_c diff[a,b,c] * W[c,r]
            # Then dU[a,r] = 2 * sum_b temp[a,b,r] * V[b,r]
            
            # temp[a,b,r] = sum_c diff[a,b,c] * W[c,r]
            # = (diff reshaped) contracted with W
            # diff shape: (n*m, m*p, p*n), W shape: (p*n, r)
            # temp[a,b,r] = sum_c diff[a,b,c] * W[c,r]
            # = np.einsum('abc,cr->abr', diff, W)
            
            temp_U = np.einsum('abc,cr->abr', diff, W)  # shape (n*m, m*p, r)
            dU = 2 * np.einsum('abr,br->ar', temp_U, V)  # shape (n*m, r)
            
            # dV[b,r] = 2 * sum_{a,c} diff[a,b,c] * U[a,r] * W[c,r]
            # temp[b,a,r] = sum_c diff[a,b,c] * W[c,r] ... same as above but indexed differently
            # Actually: sum_c diff[a,b,c] * W[c,r] is the same temp
            # dV[b,r] = 2 * sum_a temp[a,b,r] * U[a,r]
            # = 2 * np.einsum('abr,ar->br', temp_U, U)
            
            dV = 2 * np.einsum('abr,ar->br', temp_U, U)  # shape (m*p, r)
            
            # dW[c,r] = 2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
            # temp2[a,b,r] = sum_c ... no
            # dW[c,r] = 2 * sum_{a,b} diff[a,b,c] * U[a,r] * V[b,r]
            # = 2 * np.einsum('abc,ar,br->cr', diff, U, V)
            
            dW = 2 * np.einsum('abc,ar,br->cr', diff, U, V)  # shape (p*n, r)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        return None
    
    # Try decreasing ranks
    best_result = None
    for r in [30, 35, 40, 45, 50, 55, 60]:
        if time.time() > deadline - 5:
            break
        
        result = try_rank(r)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return schoolbook()
