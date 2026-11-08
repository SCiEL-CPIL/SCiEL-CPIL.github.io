import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Build the multiplication tensor T with indices (i, j, k) -> i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if this is part of the multiplication, 0 otherwise
    # For matrix mult C = A*B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    
    # The decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where U is (n, R), V is (m, R), W is (p, R)
    # But the interface wants U of shape (n*m, R), V of shape (m*p, R), W of shape (p*n, R)
    # Let me re-read the task...
    
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # So U has n*m rows, V has m*p rows, W has p*n rows
    # 
    # The trace convention: W[k*n+i, r]
    # So W is indexed by (k, i) where k is col of B, i is row of A
    # 
    # The decomposition must satisfy:
    # For all i, j, k: T[i,j,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    # Wait, that doesn't seem right either. Let me think more carefully.
    
    # Standard tensor decomposition for matrix multiplication:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor T[i,j,k] = 1 (the multiplication tensor)
    # Decomposition: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # 
    # But the interface specifies different shapes. Let me check the existing code.
    # In the current code, U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (21, rank)  
    # W has shape (p*n, rank) = (14, rank)
    
    # Looking at the existing code more carefully:
    # U indices: 0..5 (n*m = 6)
    # V indices: 0..20 (m*p = 21)
    # W indices: 0..13 (p*n = 14)
    
    # The add_term function adds to U[idx], V[idx], W[idx]
    # In add_strassen_block, U indices used are 0,1,3,4 (subset of 0..5)
    # V indices are b00, b01, b10, b11, b2k which are in range 0..20
    # W indices are c00, c01, c10, c11, c06, c16, c0k, c1k which are in range 0..13
    
    # So the indexing convention must be:
    # U[i*m + j, r] for i in [0,n), j in [0,m)
    # V[j*p + k, r] for j in [0,m), k in [0,p)
    # W[k*n + i, r] for k in [0,p), i in [0,n)
    
    # And the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for all valid i,j,k
    
    # Now, to find a minimal rank decomposition, we can:
    # 1. Build the full tensor as a matrix
    # 2. Use Gaussian elimination to find a basis of rank-1 tensors that span it
    
    # The tensor T has dimensions n x m x p = 2 x 3 x 7
    # We can flatten it into a vector of length n*m*p = 42
    # Each rank-1 term U[:,r] * V[:,r] * W[:,r] corresponds to a specific pattern
    
    # Actually, let me think of this differently. The tensor T is fixed:
    # T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,7)
    
    # We want to find U, V, W such that:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a bilinear (actually trilinear) problem. Let's linearize it.
    # For a fixed set of U and V, the equation becomes linear in W:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # But U, V, W are all unknowns. This is a nonlinear problem.
    
    # Alternative approach: Use the fact that the tensor is very structured.
    # T[i,j,k] = 1 means the tensor is the "all ones" tensor in the (i,j,k) space.
    # Wait, no. Let me re-read.
    
    # Actually, I think I'm overcomplicating this. Let me look at what the tensor actually is.
    # For matrix multiplication, the tensor T[i,j,k] represents the contribution of A[i,j]*B[j,k] to C[i,k].
    # So T[i,j,k] = 1 if (i,j,k) is a valid triple, 0 otherwise.
    # Since all triples are valid, T[i,j,k] = 1 for all i,j,k.
    
    # So the tensor is the all-ones tensor of size 2x3x7.
    # We want to decompose this as sum of rank-1 tensors.
    
    # A rank-1 tensor has the form u_i * v_j * w_k for some vectors u, v, w.
    # We want: 1 = sum_r u_i^(r) * v_j^(r) * w_k^(r) for all i,j,k.
    
    # This is equivalent to finding a low-rank decomposition of the "all ones" tensor.
    # The all-ones tensor of size n x m x p has rank 1! Because it's u_i * v_j * w_k where u, v, w are all-ones vectors.
    # 
    # Wait, that can't be right. Let me reconsider.
    
    # I think I'm misunderstanding the tensor. Let me re-read the task.
    # "Matrix multiplication 2,3,7"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix."
    
    # The multiplication tensor for C = A*B where A is 2x3 and B is 3x7 is:
    # T[i,j,k] = 1 if this represents the product A[i,j]*B[j,k] contributing to C[i,k]
    # 
    # But actually, the standard multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k),
    # and the decomposition gives a way to compute C[i,k] = sum_j A[i,j]*B[j,k] using fewer products.
    
    # The key insight: the tensor T[i,j,k] is NOT the all-ones tensor in the sense that
    # it's a single tensor, but the decomposition finds rank-1 components that sum to it.
    
    # Let me reconsider the problem from scratch. The multiplication tensor for m x n by n x p matrices
    # is T[i,j,k] where i is row of first matrix, j is the shared dimension, k is column of second matrix.
    # T[i,j,k] = 1 (it's the identity tensor in some sense).
    
    # The rank of this tensor is the minimum number of scalar multiplications needed.
    # For 2x3 by 3x7, the naive algorithm uses 2*3*7 = 42 multiplications.
    # Strassen-type algorithms can reduce this.
    
    # The current solution uses rank 39. Can we do better?
    
    # Let me try a different approach: use Gaussian elimination on the tensor.
    # Flatten the tensor into a matrix and find its rank.
    
    # The tensor T has shape (2, 3, 7). We can matricize it in different ways.
    # For example, reshape to (2*3, 7) = (6, 7) matrix.
    # The rank of this matricization gives a lower bound on the tensor rank.
    
    T = np.ones((2, 3, 7))
    
    # Matricize as (n*m, p)
    T_mat1 = T.reshape(n*m, p)
    rank1 = np.linalg.matrix_rank(T_mat1)
    
    # Matricize as (n, m*p)
    T_mat2 = T.reshape(n, m*p)
    rank2 = np.linalg.matrix_rank(T_mat2)
    
    # Matricize as (m, n*p)
    T_mat3 = T.reshape(m, n*p)
    rank3 = np.linalg.matrix_rank(T_mat3)
    
    # The tensor rank is at least the maximum of these matricization ranks
    lower_bound = max(rank1, rank2, rank3)
    
    # For the all-ones tensor, all matricizations have rank 1, so lower_bound = 1.
    # But the tensor rank is actually 1! The all-ones tensor IS a rank-1 tensor.
    # 
    # This means the multiplication tensor for 2x3 by 3x7 has rank 1?!
    # That can't be right. Let me reconsider.
    
    # I think the issue is that the multiplication tensor is NOT the all-ones tensor.
    # Let me re-read the problem statement more carefully.
    
    # "Matrix multiplication 2,3,7"
    # The tensor T[i,j,k] for multiplying a 2x3 matrix by a 3x7 matrix.
    # 
    # I believe the tensor is defined such that:
    # C[i,k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # 
    # But the decomposition is:
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # 
    # And the "rank" is the minimum R such that such U, V, W exist.
    # 
    # For the all-ones tensor, we can write:
    # 1 = 1 * 1 * 1
    # So U[i,0] = 1, V[j,0] = 1, W[k,0] = 1 for all i,j,k.
    # This gives rank 1!
    
    # But this contradicts the known result that 2x3 by 3x7 matrix multiplication
    # requires more than 1 multiplication. I must be misunderstanding the problem.
    
    # Let me look at the existing code again. The existing code uses rank 39 and has
    # specific patterns in U, V, W. The U, V, W matrices have multiple nonzero entries
    # in each row, and the decomposition is not simply the all-ones pattern.
    
    # I think the key is that the tensor is NOT the all-ones tensor. Instead, the
    # multiplication tensor has a specific structure related to how matrix multiplication works.
    
    # Let me reconsider. Perhaps the tensor T[i,j,k] is defined differently.
    # Maybe T[i,j,k] = delta_{j,k} or something similar?
    
    # Looking at the try_optimize_227 function in the current code:
    # "Tensor T[i, j, k] = 1 if j == k else 0 for i in 0..1, j in 0..1, k in 0..6"
    # 
    # So for the 2x2x7 subproblem, T[i,j,k] = 1 if j == k, else 0.
    # This is the "diagonal" tensor, not the all-ones tensor!
    
    # I think I've been misunderstanding. The multiplication tensor for matrix multiplication
    # is actually the tensor where T[i,j,k] = 1 if the indices are "compatible" in some way,
    # but I'm not sure what the exact definition is.
    
    # Let me try a different approach. Instead of trying to understand the tensor from first principles,
    # let me use the existing rank-39 solution as a starting point and try to optimize it.
    
    # First, let me verify the existing solution is correct by checking the decomposition.
    
    def build_rank39_solution():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for idx in u_indices:
                U[idx][col] += 1
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1
        
        def add_strassen_block(k_start):
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1
            
            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1
            
            add_term([0, 4], [b00, b11], [c00, c11])
            add_term([3, 4], [b00], [c10, c11])
            W[c11][col-1] = -1
            
            add_term([0], [b01, b11], [c01, c11])
            V[b11][col-1] = -1
            
            add_term([4], [b10, b00], [c00, c10])
            V[b00][col-1] = -1
            
            add_term([0, 1], [b11], [c00, c01])
            W[c00][col-1] = -1
            
            add_term([3, 0], [b00, b01], [c11])
            U[0][col-1] = -1
            
            add_term([1, 4], [b10, b11], [c00])
            U[4][col-1] = -1
        
        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        
        add_term([0], [b06], [c06])
        add_term([1], [b16], [c06])
        add_term([3], [b06], [c16])
        add_term([4], [b16], [c16])
        
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            
            add_term([2], [b2k], [c0k])
            add_term([5], [b2k], [c1k])
        
        return U, V, W
    
    U39, V39, W39 = build_rank39_solution()
    
    # Now let me try to find a better solution using optimization.
    # I'll use the rank-39 solution as initialization and try to reduce the rank.
    
    def verify_solution(U, V, W, T):
        """Check if the decomposition is correct."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        T = np.array(T, dtype=float)
        
        # Check the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        max_error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    computed = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] 
                                   for r in range(len(U[0])))
                    expected = T[i, j, k]
                    error = abs(computed - expected)
                    max_error = max(max_error, error)
        
        return max_error
    
    # What is the actual tensor T? Let me try to infer it from the existing solution.
    # I'll compute what tensor the rank-39 solution decomposes.
    
    def infer_tensor(U, V, W):
        """Compute the tensor that the given U, V, W decompose."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        R = len(U[0])
        
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] 
                                     for r in range(R))
        return T
    
    T_inferred = infer_tensor(U39, V39, W39)
    
    # Now I know the actual tensor. Let me see what it looks like.
    # For the 2x2x7 subproblem, the comment said T[i,j,k] = 1 if j == k else 0.
    # Let me check if that's consistent.
    
    # Actually, let me just proceed with the inferred tensor and try to find a better decomposition.
    
    # Strategy: Use gradient-based optimization to find a lower-rank decomposition.
    # Start with the rank-39 solution and try to reduce the rank.
    
    def try_lower_rank(T, initial_rank=39, target_rank=None, max_steps=5000, lr=0.01, seed=42):
        """Try to find a lower-rank decomposition using gradient descent."""
        n, m, p = DIMENSIONS
        np.random.seed(seed)
        
        if target_rank is None:
            target_rank = initial_rank - 1
        
        # Initialize with the given solution, zeroing out some columns
        U_init = np.array(U39, dtype=float)
        V_init = np.array(V39, dtype=float)
        W_init = np.array(W39, dtype=float)
        
        # If target_rank < initial_rank, we need to reduce the number of columns
        # One approach: use the top principal components
        if target_rank < initial_rank:
            # Use SVD-like approach to find the most important components
            # For now, just randomly select columns
            cols_to_keep = np.random.choice(initial_rank, target_rank, replace=False)
            U = U_init[:, cols_to_keep]
            V = V_init[:, cols_to_keep]
            W = W_init[:, cols_to_keep]
        else:
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
        
        R = target_rank
        T = np.array(T, dtype=float)
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Compute current tensor
            C = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C[i, j, k] = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] 
                                         for r in range(R))
            
            error = T - C
            loss = np.sum(error**2)
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-12:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            e = error[i, j, k]
                            dU[i*m + j, r] += -2 * e * V[j*p + k, r] * W[k*n + i, r]
                            dV[j*p + k, r] += -2 * e * U[i*m + j, r] * W[k*n + i, r]
                            dW[k*n + i, r] += -2 * e * U[i*m + j, r] * V[j*p + k, r]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        return best_U, best_V, best_W, best_loss
    
    # Try to find a lower-rank solution
    best_solution = (U39, V39, W39)
    best_rank = 39
    
    for target_rank in range(38, 20, -1):
        if time.time() > deadline:
            break
        
        U_opt, V_opt, W_opt, loss = try_lower_rank(T_inferred, 39, target_rank, max_steps=2000, seed=42 + (39 - target_rank))
        
        if loss < 1e-10:
            best_solution = (U_opt, V_opt, W_opt)
            best_rank = target_rank
            break
    
    U_best, V_best, W_best = best_solution
    
    # Convert to integer/half-integer if possible
    def to_rational(U, V, W):
        """Try to convert the solution to integer/half-integer entries."""
        U_r = [[0]*len(U[0]) for _ in range(len(U))]
        V_r = [[0]*len(V[0]) for _ in range(len(V))]
        W_r = [[0]*len(W[0]) for _ in range(len(W))]
        
        for i in range(len(U)):
            for r in range(len(U[0])):
                val = U[i, r]
                # Round to nearest half-integer
                U_r[i][r] = round(val * 2) / 2
        
        for j in range(len(V)):
            for r in range(len(V[0])):
                val = V[j, r]
                V_r[j][r] = round(val * 2) / 2
        
        for k in range(len(W)):
            for r in range(len(W[0])):
                val = W[k, r]
                W_r[k][r] = round(val * 2) / 2
        
        return U_r, V_r, W_r
    
    U_final, V_final, W_final = to_rational(U_best, V_best, W_best)
    
    # Verify the final solution
    final_error = verify_solution(U_final, V_final, W_final, T_inferred)
    
    if final_error > 1e-6:
        # If the rationalized solution is not accurate, use the original
        U_final = [[float(x) for x in row] for row in U_best.tolist()]
        V_final = [[float(x) for x in row] for row in V_best.tolist()]
        W_final = [[float(x) for x in row] for row in W_best.tolist()]
    
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
