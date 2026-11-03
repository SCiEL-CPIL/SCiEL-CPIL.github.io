"""
Matrix multiplication 2x4 * 4x8 -> 2x8
Find a rank-minimal exact tensor decomposition.

The multiplication tensor T has entries T[(i,j),(j',k),(i',k')] = 1 if j=j' and i=i' and k=k'
Actually, the correct tensor for matrix multiplication C = A*B where:
  A is n x m (2x4), B is m x p (4x8), C is n x p (2x8)
  C[i,k] = sum_j A[i,j] * B[j,k]

The tensor T has indices (a, b, c) where:
  a = i*m + j  (index into A, shape n*m = 8)
  b = j'*p + k  (index into B, shape m*p = 32)
  c = k'*n + i'  (index into C, shape p*n = 16)

T[a,b,c] = 1 if the (i,j) from a, (j',k) from b, and (i',k') from c satisfy:
  j = j' AND i = i' AND k = k'

So T[(i*m+j), (j'*p+k), (k'*n+i')] = 1 if j=j', i=i', k=k', else 0.

We need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c.

The schoolbook rank is n*m*p = 64.

Let me think about a lower rank decomposition.

Key insight: For fixed j, the tensor factors as:
  T[(i*m+j), (j*p+k), (k*n+i)] = 1 (for all i, k)

So for each j, we have a sub-tensor of size n x p x n*p = 2 x 8 x 16.
This sub-tensor can be written as: sum over i,k of e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}

But we can share across j's.

Actually, let me think differently. The tensor is:
  T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}

This is the standard schoolbook with rank 64.

To reduce rank, we need to find linear combinations.

Let me try a different approach. Consider the tensor as a matrix.
Fix the C index c = k*n + i. Then T[:, :, c] is a matrix of size (n*m) x (m*p) = 8 x 32.
T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i, i.e., a corresponds to (i,j), b to (j,k), c to (i,k).

For fixed c = (i,k), T[:, :, c] has a 1 at position (i*m+j, j*p+k) for j=0..3.
So T[:, :, c] is a matrix with 4 ones (one for each j).

The rank of the full tensor is the minimum number of rank-1 tensors needed.

Let me try to use a Kronecker product structure.

Actually, let me just implement a solver that finds the minimum rank decomposition.

I'll use a systematic approach:
1. Set up the tensor T as a 3D array of shape (8, 32, 16).
2. Try to find a decomposition with rank r by solving a system of equations.

But that's expensive. Let me think of a smarter approach.

Alternative: use the fact that the tensor has a specific structure.

For the 2x4*4x8 case, let me try to find a decomposition with rank less than 64.

Let me try rank 32 first as a starting point.

Actually, let me just implement a general solver using numpy.

The condition is:
  sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]

This is a system of 8*32*16 = 4096 equations.

For rank r, we have 8*r + 32*r + 16*r = 56*r unknowns.
For r=64: 56*64 = 3584 unknowns, 4096 equations. Underdetermined? No, 3584 < 4096.

Wait, that's not right. The system is nonlinear (products of U, V, W entries).

Let me use a different approach: iterative refinement or use a library.

Actually, for this specific problem, let me try to construct a decomposition manually.

The tensor T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) in the appropriate encoding.

Let me try to use a decomposition based on the following idea:
- For each j, we need to "connect" A's j-th column with B's j-th row to produce C.
- The j-th column of A is a vector of length n=2.
- The j-th row of B is a vector of length p=8.
- Their outer product is a 2x8 matrix, which contributes to C.

So C = sum_j A[:,j] * B[j,:]

This is a sum of 4 outer products. Each outer product is a 2x8 matrix.

In tensor form, for each j:
  sum_{i,k} A[i,j] * B[j,k] * (contribution to C[i,k])

The rank-1 tensor for this would be:
  u = e_{i*m+j} (but we need to sum over i)
  
Actually, the outer product A[:,j] * B[j,:] can be written as:
  sum_i sum_k A[i,j] * B[j,k] * e_{i,k}

In the tensor decomposition framework:
  For fixed j, we need: sum_{i,k} (A[i,j]) * (B[j,k]) * (coefficient for C[i,k])

This suggests that for each j, we could use n*p = 16 rank-1 tensors, giving 4*16 = 64 total. That's the schoolbook.

To reduce rank, we need to share across j's.

Let me try a different approach: use a bilinear algorithm based on linear combinations.

Consider using the following idea from fast matrix multiplication:
- Use linear combinations of A's columns and B's rows to reduce the number of products.

For 2x4 * 4x8, let me try to find a decomposition with rank 32.

Actually, let me just implement a numerical solver.

I'll set up the problem as: find U (8 x r), V (32 x r), W (16 x r) such that:
  sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c

I'll use an iterative approach: fix two matrices, solve for the third.

Let me implement this.
"""

import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Build the multiplication tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) with matching indices
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with a reasonable rank and try to reduce
    
    # First, let's get the schoolbook decomposition as a baseline
    # Then try to reduce the rank
    
    # Use an iterative approach to find a decomposition
    # We'll try ranks from low to high and check if a valid decomposition exists
    
    def try_decompose(rank):
        """Try to find a decomposition with the given rank."""
        # Initialize U, V, W with random values
        rng = np.random.RandomState(42)
        
        best_U = None
        best_V = None
        best_W = None
        best_err = float('inf')
        
        for trial in range(50):
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Iteratively refine
            for it in range(200):
                # Fix U, V, solve for W
                # For each c: W[:, c] should satisfy sum_r U[:, r] * V[:, r] * W[r, c] ... 
                # Actually, the condition is:
                # sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
                
                # Reshape: for fixed c, we have a bilinear form in U and V
                # Let's define M_c = T[:, :, c] (shape: n*m x m*p)
                # We need: sum_r (U[:, r] * V[:, r])^T * W[c, r] ... no
                
                # The condition is: for all a, b, c:
                # sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
                
                # For fixed c, define: M_c[a, b] = T[a, b, c]
                # We need: sum_r U[a, r] * V[b, r] * W[c, r] = M_c[a, b]
                
                # This is: sum_r W[c, r] * (U[:, r] * V[:, r]) = M_c (element-wise)
                # Or: sum_r W[c, r] * outer(U[:, r], V[:, r]) = M_c
                
                # So: M_c = sum_r W[c, r] * outer(U[:, r], V[:, r])
                
                # If we fix U and V, this is linear in W:
                # For each c: M_c = sum_r W[c, r] * outer(U[:, r], V[:, r])
                
                # Let's vectorize: for each r, define P_r = outer(U[:, r], V[:, r]) (shape: n*m x m*p)
                # Then: M_c = sum_r W[c, r] * P_r (element-wise)
                
                # So: vec(M_c) = sum_r W[c, r] * vec(P_r)
                
                # This is a linear system: A_c * W[:, c] = vec(M_c)
                # where A_c is a matrix of size (n*m * m*p) x rank
                
                # Similarly for U and V.
                
                # Solve for W:
                # For each c, solve: sum_r W[c, r] * P_r = M_c
                # This is: A * W[:, c] = vec(M_c) where A[r, idx] = P_r.flatten()[idx]
                
                A_mat = np.zeros((n * m * m * p, rank))
                for r in range(rank):
                    P_r = np.outer(U[:, r], V[:, r])
                    A_mat[:, r] = P_r.flatten()
                
                # Solve for W
                W_new = np.zeros((p * n, rank))
                for c in range(p * n):
                    M_c = T[:, :, c]
                    b_vec = M_c.flatten()
                    # Solve A_mat * W[:, c] = b_vec
                    sol, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_new[c, :] = sol
                
                # Solve for U:
                # For fixed a: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
                # For fixed a, define: N_a[b, c] = T[a, b, c]
                # We need: sum_r U[a, r] * (V[:, r] * W[:, r]) = N_a (element-wise)
                # Or: sum_r U[a, r] * outer(V[:, r], W[:, r]) = N_a
                
                B_mat = np.zeros((m * p * p * n, rank))
                for r in range(rank):
                    Q_r = np.outer(V[:, r], W[:, r])
                    B_mat[:, r] = Q_r.flatten()
                
                U_new = np.zeros((n * m, rank))
                for a in range(n * m):
                    N_a = T[a, :, :]
                    b_vec = N_a.flatten()
                    sol, residuals, rank_sol, sv = np.linalg.lstsq(B_mat, b_vec, rcond=None)
                    U_new[a, :] = sol
                
                # Solve for V:
                # For fixed b: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
                # For fixed b, define: P_b[a, c] = T[a, b, c]
                # We need: sum_r V[b, r] * (U[:, r] * W[:, r]) = P_b (element-wise)
                # Or: sum_r V[b, r] * outer(U[:, r], W[:, r]) = P_b
                
                C_mat = np.zeros((n * m * p * n, rank))
                for r in range(rank):
                    R_r = np.outer(U[:, r], W[:, r])
                    C_mat[:, r] = R_r.flatten()
                
                V_new = np.zeros((m * p, rank))
                for b in range(m * p):
                    P_b = T[:, b, :]
                    b_vec = P_b.flatten()
                    sol, residuals, rank_sol, sv = np.linalg.lstsq(C_mat, b_vec, rcond=None)
                    V_new[b, :] = sol
                
                U = U_new
                V = V_new
                W = W_new
            
            # Check the error
            err = 0.0
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        val = 0.0
                        for r in range(rank):
                            val += U[a, r] * V[b, r] * W[c, r]
                        err += (val - T[a, b, c]) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
        
        return best_U, best_V, best_W, best_err
    
    # Try different ranks
    for rank in [14, 16, 20, 24, 28, 32, 40, 48, 56, 64]:
        U, V, W, err = try_decompose(rank)
        if err < 1e-10:
            # Found a valid decomposition
            # Convert to lists
            U_list = [[float(U[i, r]) for r in range(rank)] for i in range(n * m)]
            V_list = [[float(V[i, r]) for r in range(rank)] for i in range(m * p)]
            W_list = [[float(W[i, r]) for r in range(rank)] for i in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition with rank 64
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
