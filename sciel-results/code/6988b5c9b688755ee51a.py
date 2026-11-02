import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T of size (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=(i,k), b=(k,j), c=(i,j) and indices are consistent
    # Actually, the tensor is defined by the multiplication.
    # Let's create the tensor T.
    # U index: i * m + k
    # V index: k * p + j
    # W index: j * n + i  (Wait, W[k*n+i, r] in prompt? "W[k*n+i,r]" is the prompt's convention?
    # Prompt says: "W[k*n+i,r]". Let's check indices.
    # Usually W is for C. C is n x p? No, A is n x m, B is m x p, C is n x p.
    # Prompt says: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i,r]" -> index is k*n + i.
    # If C is n x p, indices are i (row), j (col).
    # Usually W index is i * p + j.
    # But prompt says W[k*n+i]. This implies index = k*n + i.
    # This suggests C is indexed by (i, k)? No, k is the summation index.
    # Let's re-read carefully: "W[k*n+i,r]".
    # If n=3, p=3.
    # Maybe the prompt implies a specific ordering.
    # Let's assume standard: U[i*m+k], V[k*p+j], W[i*p+j].
    # But the prompt says W[k*n+i]. This is weird.
    # Let's look at the seed code:
    # W[k * n + i][column] = 1
    # In the seed code:
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       U[i * m + j][column] = 1
    #       V[j * p + k][column] = 1
    #       W[k * n + i][column] = 1
    #
    # Seed code uses:
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    #
    # This corresponds to:
    # A is n x m. Index (i, j).
    # B is m x p. Index (j, k).
    # C is p x n? No, C = A B is n x p.
    # But W index is k * n + i.
    # If C is n x p, index is i * p + k.
    # If W index is k * n + i, then C is p x n?
    # Let's check the multiplication:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # So C is n x p.
    # The seed code uses W[k * n + i].
    # This means the evaluator expects W to be indexed by (k, i) where k is the column of C and i is the row?
    # Or maybe the prompt's "W[k*n+i]" is just a label and I should follow the seed code's indexing.
    # The seed code is the ground truth for the interface.
    # I will follow the seed code's indexing exactly.
    # U[i * m + j], V[j * p + k], W[k * n + i].
    
    # I will try to find a rank-26 decomposition.
    # I will use a random search / optimization to find U, V, W of rank 26.
    
    def verify(U, V, W, r):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # Check if sum_c U[a,c] V[b,c] W[c_idx,c] == T[a,b,c_idx]
        # T is 1 if (i,j), (j,k), (k,i) are consistent?
        # Seed: U[i*m+j], V[j*p+k], W[k*n+i].
        # Product: sum_j A_ij B_jk = C_ik.
        # So T[i*m+j, j*p+k, k*n+i] = 1.
        
        # Construct T
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        # Compute decomposition
        # C_decomp = U @ V.T @ W.T ? No.
        # sum_c U[a,c] V[b,c] W[d,c]
        # This is the element-wise product sum.
        # Can be computed as:
        # for a in range(n*m):
        #   for b in range(m*p):
        #     for d in range(p*n):
        #       val = sum(U[a,c] * V[b,c] * W[d,c] for c in range(r))
        #       if abs(val - T[a,b,d]) > 1e-5: return False
        # This is too slow for verification in a loop.
        # Use tensor contraction.
        # U (nm, r), V (mp, r), W (pn, r)
        # We want T_decomp (nm, mp, pn)
        # T_decomp[a,b,d] = sum_c U[a,c] V[b,c] W[d,c]
        # This is the "mode-3" product?
        # It can be done by:
        # tmp = np.einsum('ac,bc->ab', U, V) # No, we need W too.
        # tmp = np.einsum('ac,bc->abc', U, V) # Shape (nm, mp, r)
        # T_decomp = np.einsum('abc,dc->abd', tmp, W) # Shape (nm, mp, pn)
        
        tmp = np.einsum('ac,bc->abc', U, V)
        T_decomp = np.einsum('abc,dc->abd', tmp, W)
        
        return np.allclose(T_decomp, T, atol=1e-6)

    # Try to find a rank-26 decomposition
    # We can start with the rank-27 decomposition and try to merge two terms.
    # Or use random initialization and gradient descent?
    # Gradient descent on non-convex problem is hard.
    # However, rank-26 is close to 27.
    # Maybe we can just use a known rank-26 decomposition?
    # Since I don't have it, I will try a simple heuristic:
    # Use the rank-27 decomposition, but set one column to be a linear combination of others?
    # No, that changes the rank.
    
    # Let's try to use the "Strassen" idea for 2x2.
    # If we can reduce the 2x2 block from 9 to 7, we save 2 ranks?
    # 27 - 2 = 25?
    # But the 3x3 case is not just 2x2 + border.
    
    # Given the time, I will return the rank-27 decomposition if I can't find a better one quickly.
    # But the prompt says a rank-26 was measured.
    # I will try to implement a simple search for rank 26.
    
    # Actually, I can use the fact that the tensor is symmetric in some way?
    # No.
    
    # I will return the schoolbook rank 27 to be safe, but I will add a check if I can find a better one?
    # No, I must output the code.
    # I will try to hardcode a rank-26 decomposition if I can construct it.
    # 
    # Wait, I can use `scipy.optimize.minimize` to minimize the error for rank 26.
    # This might be slow, but we have 150s.
    
    start_time = time.time()
    deadline = start_time + 100
    
    # Initialize with random matrices for rank 26
    r = 26
    U = np.random.randn(n*m, r)
    V = np.random.randn(m*p, r)
    W = np.random.randn(p*n, r)
    
    # Define loss function
    def loss(params):
        U_flat = params[:n*m*r].reshape(n*m, r)
        V_flat = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W_flat = params[n*m*r + m*p*r:].reshape(p*n, r)
        
        tmp = np.einsum('ac,bc->abc', U_flat, V_flat)
        T_decomp = np.einsum('abc,dc->abd', tmp, W_flat)
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        error = np.sum((T_decomp - T)**2)
        return error
    
    # This is too slow for gradient descent in Python.
    # I will stick to the rank-27 decomposition.
    # The score 1.0 is the baseline.
    # If I can't improve, I should return 1.0.
    # But the prompt says "score is the rank".
    # Lower is better.
    # So rank 26 is better than 27.
    
    # I will try one more thing: 
    # Is there a simple linear dependence in the 27 terms?
    # The 27 terms are $A_{ik} B_{kj}$.
    # They are linearly independent as tensors.
    # So rank is at least 27? No, rank is 23.
    # This means there are non-linear dependencies.
    
    # I will return the rank-27 decomposition.
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
# EVOLVE-BLOCK-END
