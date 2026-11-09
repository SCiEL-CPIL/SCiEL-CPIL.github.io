import numpy as np
import time

def construct():
    # For multiplying a 4x4 matrix (A) by a 4x6 matrix (B), producing a 4x6 result (C).
    # The tensor decomposition has factors:
    #   U: shape (4*4, rank) = (16, rank)
    #   V: shape (4*6, rank) = (24, rank)
    #   W: shape (6*4, rank) = (24, rank)
    # following the convention W[k*n + i, r] where the output C is 4x6.
    
    # Use the straightforward schoolbook (outer product) decomposition.
    # For C = A @ B where A is (4,4) and B is (4,6):
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # 
    # The rank of the standard decomposition is m = 4 (the shared dimension).
    # U[i*k + x, r] = 1 if i==r and x==k else 0  -> U has shape (16, 4)
    # V[k*6 + j, r] = 1 if k==r and j==j else 0 -> V has shape (24, 4)
    # W[j*4 + i, r] = 1 if j==j and i==i else 0 -> W has shape (24, 4)
    # Wait, let's be precise about the index conventions.
    
    # The task says:
    # U: (n*m, rank) = (4*4, rank) = (16, rank)
    # V: (m*p, rank) = (4*6, rank) = (24, rank)
    # W: (p*n, rank) = (6*4, rank) = (24, rank)
    
    # The matrix multiplication tensor T has entries T[i, k, j] = delta_{i, k, j} in the sense that
    # C[i,j] = sum_k A[i,k] B[k,j].
    
    # A rank-1 decomposition: T = sum_r U_r \otimes V_r \otimes W_r
    # where U_r is a vector of size 16, V_r is size 24, W_r is size 24.
    
    # The standard (schoolbook) decomposition has rank m = 4.
    # For each k in range(4):
    #   U[:, k] has 1 at positions corresponding to (i, k) for all i, i.e., index i*4 + k.
    #   V[:, k] has 1 at positions corresponding to (k, j) for all j, i.e., index k*6 + j.
    #   W[:, k] has 1 at positions corresponding to (j, i) for all i, j, i.e., index j*4 + i.
    
    # Let's verify this:
    # (U \otimes V \otimes W)_{i,k,j} = sum_r U[i*4+k, r] * V[k*6+j, r] * W[j*4+i, r]
    # For r = k:
    #   U[i*4+k, k] = 1
    #   V[k*6+j, k] = 1
    #   W[j*4+i, k] = 1
    # So the sum is 1 for all i, j, k. This matches the identity tensor for matrix multiplication.
    
    # Therefore, rank 4 is achievable with this simple decomposition.
    
    # However, is rank 4 the minimum? 
    # The trivial lower bound is max(n, m, p) = max(4, 4, 6) = 6? No, the lower bound is usually m for C = A*B.
    # Actually, for C = A @ B, the standard rank is m. The lower bound is at least max(n, m, p)? No.
    # The rank of the matrix multiplication tensor M_{n,m,p} is at least max(n, m, p)? 
    # Actually, it's known that R(M_{n,m,p}) >= max(n, m, p). So for 4x4x6, the lower bound is 6.
    
    # Wait, my schoolbook construction above has rank 4, which is less than 6. That can't be right if the lower bound is 6.
    # Let me re-check the lower bound. The rank of M_{n,m,p} is at least max(n, m, p) is a common claim, but let's verify.
    # Actually, the lower bound is max(n, m, p) is not always true. For M_{2,2,2}, rank is 7 (Strassen), which is > 2.
    # For M_{1,m,p}, rank is m*p? No, it's just the standard rank.
    
    # Let's re-examine the schoolbook decomposition.
    # C[i,j] = sum_k A[i,k] B[k,j]
    # The tensor is T[i, k, j].
    # A rank-1 term is U[i, k] * V[k, j] * W[j, i].
    # If we set r = k, then:
    #   U_r[i, k] = delta_{r, k}
    #   V_r[k, j] = delta_{r, k}
    #   W_r[j, i] = 1
    # This doesn't work because W_r would need to be 1 for all i, j, which means it's a constant vector, but then the product would be delta_{r,k} * delta_{r,k} * 1 = delta_{r,k}, which sums to 1 only if r=k is the only non-zero term. 
    # Actually, the schoolbook decomposition is:
    # T[i, k, j] = sum_{r=0}^{m-1} delta_{k,r} * delta_{k,r} * 1? No.
    
    # The correct schoolbook (outer product) decomposition for M_{n,m,p} has rank m:
    # For each r in 0..m-1:
    #   U_r[i, k] = delta_{k, r}
    #   V_r[k, j] = delta_{k, r}
    #   W_r[i, j] = 1
    # Then sum_r U_r[i,k] V_r[k,j] W_r[i,j] = sum_r delta_{k,r} delta_{k,r} * 1 = delta_{k,k} = 1 if k is in range.
    # Wait, this gives 1 for all i, j, k, which is not the multiplication tensor. The multiplication tensor is T[i,k,j] which is 1 if we're computing the (i,j) entry of the product, but the tensor itself is defined such that C = A * B corresponds to contracting T with A and B.
    
    # Let's be very careful. The matrix multiplication tensor T_{n,m,p} is a 3-tensor of size n x m x p such that:
    # C[i,j] = sum_k T[i, k, j] * A[i, k] * B[k, j]
    # In this convention, T[i, k, j] = 1 for all i, k, j. That's not right either.
    
    # The standard definition: The multiplication tensor M_{n,m,p} is defined by
    # M_{n,m,p} = sum_{i=0}^{n-1} sum_{j=0}^{p-1} sum_{k=0}^{m-1} e_{i,k} \otimes e_{k,j} \otimes e_{i,j}
    # where e_{i,k} is the vector with 1 at position (i,k) and 0 elsewhere, etc.
    # So the tensor has components M[i, k, j] = 1 for all valid i, k, j? No, that's the all-ones tensor.
    
    # Actually, the correct tensor for matrix multiplication is:
    # M = sum_{i,k,j} E_{i,k} \otimes F_{k,j} \otimes G_{i,j}
    # where E_{i,k} is the matrix unit for A, F_{k,j} for B, G_{i,j} for C.
    # This is a sum of n*m*p rank-1 terms, which is the "trivial" decomposition of rank n*m*p.
    
    # The schoolbook decomposition is NOT rank m. The schoolbook algorithm uses m multiplies, but the tensor rank is different.
    # The rank of the tensor is the minimum number of rank-1 tensors that sum to M.
    
    # The standard lower bound is R(M_{n,m,p}) >= max(n, m, p). 
    # For M_{4,4,6}, the lower bound is 6.
    
    # So my rank-4 construction above was flawed. Let me reconsider.
    
    # The trivial decomposition has rank 4*4*6 = 96.
    # The "schoolbook" or "outer product" algorithm uses 4 multiplies, but that corresponds to a different tensor decomposition.
    
    # Let's use the known lower bound: rank >= max(4, 4, 6) = 6.
    # Is rank 6 achievable?
    # For M_{1,m,p}, the rank is m (since C[i,j] = A[i,k] B[k,j] with i fixed, it's just a linear combination).
    # Actually, M_{1,m,p} has rank m.
    # M_{n,1,p} has rank p.
    # M_{n,m,1} has rank n.
    
    # So for M_{4,4,6}, the lower bound is 6. Can we achieve rank 6?
    # One approach: use the fact that M_{4,4,6} can be decomposed using the Strassen-like ideas or by embedding.
    
    # Actually, the known best rank for M_{4,4,6} is not well-documented in my immediate memory. 
    # Let's use a safe construction that gives a valid decomposition, even if not minimal.
    # The trivial rank is 96. The schoolbook-like decomposition (using m=4 multiplies) might give rank 4*6 = 24? 
    # Let's think: for each k in 0..3, we compute the outer product A[:, k] @ B[k, :], which is a 4x6 matrix.
    # This outer product has rank 1. So we have 4 such outer products.
    # Each outer product corresponds to a rank-1 term in the tensor? No, the outer product is a rank-1 matrix, but the tensor decomposition is different.
    
    # Let's construct a valid decomposition of rank 24:
    # For each i in 0..3 and k in 0..3, we have a term that contributes to all j in 0..5.
    # U[i*4 + k, r] where r corresponds to (i, k, j)? No, rank is the number of terms.
    
    # Let's use the trivial decomposition of rank 96, which is guaranteed to work.
    # For each (i, k, j) in range(4) x range(4) x range(6):
    #   U[i*4 + k, r] = 1
    #   V[k*6 + j, r] = 1
    #   W[j*4 + i, r] = 1
    # where r is the index of the (i, k, j) triple.
    
    # This gives rank 96. It's not minimal, but it's valid.
    
    # Can we do better? 
    # The "outer product" decomposition: for each k in 0..3, the term is A[:, k] (4x1) times B[k, :] (1x6) = 4x6 matrix.
    # This matrix has rank 1. So we can write it as a sum of 4 rank-1 matrix terms (one for each row or column).
    # Actually, a rank-1 matrix u * v^T can be written as u * v^T, which is already rank 1.
    # So the contribution from k is: (sum_i A[i,k] e_i) * (sum_j B[k,j] e_j)^T = sum_i sum_j A[i,k] B[k,j] e_i e_j^T.
    # This is a sum of 24 rank-1 matrix terms for each k.
    # So total rank would be 4 * 24 = 96, same as trivial.
    
    # Alternatively, we can use the fact that the outer product of a 4-vector and a 6-vector has rank 1, so it corresponds to 1 rank-1 term in the tensor? 
    # No, the tensor M_{n,m,p} is not the sum of outer products of vectors.
    
    # Let's just use the trivial rank-96 decomposition. It's safe and valid.
    
    n, m, p = 4, 4, 6
    rank = n * m * p  # 96
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, r] = 1.0
                V[k * p + j, r] = 1.0
                W[j * n + i, r] = 1.0
                r += 1
    
    # Convert to list of lists with integer entries
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Ensure all entries are integers
    U_list = [[int(x) for x in row] for row in U_list]
    V_list = [[int(x) for x in row] for row in V_list]
    W_list = [[int(x) for x in row] for row in W_list]
    
    return {"U": U_list, "V": V_list, "W": W_list}
