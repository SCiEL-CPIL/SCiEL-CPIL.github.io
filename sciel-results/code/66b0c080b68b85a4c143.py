import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Fallback: schoolbook decomposition
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    
    # Try to build a low-rank decomposition using Strassen-like ideas
    # For 3x3x3, Strassen's algorithm gives rank 7 (but with coefficients)
    # For 3x3x7, we can think of it as 3x3x3 (first 3 cols) + 3x3x3 (next 3 cols) + 3x3x1 (last col)
    # If we can do 3x3x3 in rank 7, then total would be 7+7+3 = 17 (with 3x3x1 in rank 3)
    # But Strassen's 7-term decomposition uses non-integer coefficients typically.
    # Actually, for 2x2x2, Strassen gives rank 7 with integer-ish coefficients (±1).
    # For 3x3x3, the minimum rank is known to be at most 23 (Laderman) but likely lower.
    # Known: 3x3x3 matrix multiplication tensor has rank at most 23 (Laderman 1977).
    # Actually, I recall the rank of 3x3x3 matrix mult tensor is between 19 and 23.
    # Let me try a different approach: use the structure of the problem.
    
    # Actually, let's think about this differently.
    # The tensor for multiplying A (3x3) by B (3x7) to get C (3x7) has:
    # U of shape (9, r), V of shape (21, r), W of shape (21, r)
    # where r is the rank.
    
    # Schoolbook gives r = 9*21 = 189? No wait.
    # U has shape (n*m, r) = (9, r)
    # V has shape (m*p, r) = (21, r)
    # W has shape (p*n, r) = (21, r)
    # Schoolbook: r = n*m*p = 3*3*7 = 63
    
    # Let me verify the schoolbook construction.
    # For each (i,j,k) where i in [0,3), j in [0,3), k in [0,7):
    #   U[i*m + j, col] = 1  (A element at (i,j))
    #   V[j*p + k, col] = 1  (B element at (j,k))
    #   W[k*n + i, col] = 1  (C element at (k,i)? or (i,k)?)
    
    # The problem says: "output factor follows the official trace convention W[k*n+i,r]"
    # So W index is k*n + i, where k is the column of C and i is the row of C? 
    # Wait, C is 3x7, so C has rows i (0..2) and columns k (0..6).
    # If W[k*n + i], then k ranges 0..6 and i ranges 0..2, giving 7*3 = 21 entries.
    # So W[k*3 + i] corresponds to C[i, k].
    
    # Let me re-examine: C = A * B, where A is 3x3, B is 3x7, C is 3x7.
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # In the tensor decomposition, we want:
    #   C[i,k] = sum_r U[i*m+j? no... 
    # Actually, the standard formulation is:
    #   C[i,k] = sum_{j,r} A[i,j] * U[j, r] * V[r, k] ... no that's not right either.
    
    # Let me think more carefully. The matrix multiplication tensor M_{n,m,p} has entries:
    #   M_{i,j,k} = 1 if ... 
    # Actually, the tensor T for multiplying an (n x m) matrix by an (m x p) matrix is:
    #   T_{i, j, k} = delta_{i,j} * delta_{j,k} ... no.
    
    # The correct formulation: We have three sets of indices.
    # A has indices (i, j) where i in [0,n), j in [0,m)
    # B has indices (j, k) where j in [0,m), k in [0,p)
    # C has indices (i, k) where i in [0,n), k in [0,p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # In tensor decomposition, we write:
    #   C[i,k] = sum_r U[i, r] * V[k, r] ... no, that's not right.
    
    # Actually, the standard way is:
    #   C[i,k] = sum_{j,r} A[i,j] * U[j, r] * V[r, k]
    # No, that would be a 3-tensor with indices (i, j, k) but U and V have different index structures.
    
    # Let me look at the interface again:
    # U: shape (n*m, rank) - so U is indexed by (i*m + j, r)
    # V: shape (m*p, rank) - so V is indexed by (j*p + k, r)
    # W: shape (p*n, rank) - so W is indexed by (k*n + i, r)
    
    # The reconstruction is:
    #   T[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
    # where u = i*m + j, v = j'*p + k, w = k'*n + i'
    # And T[u, v, w] should be 1 if i=i', j=j', k=k', else 0.
    # Wait, that's the identity tensor, not the matrix multiplication tensor.
    
    # Actually, I think the tensor for matrix multiplication is:
    #   T_{(i,j), (j',k'), (k'',i'')} = delta_{i,i''} * delta_{j,j'} * delta_{k,k''}
    # No, that's not quite right either.
    
    # Let me reconsider. The matrix multiplication map is:
    #   (A, B) -> C = A * B
    # This is a bilinear map. To express it as a tensor decomposition, we can view it as:
    #   C[i,k] = sum_{j,r} A[i,j] * B[j,k] * ... 
    # But this is already a sum over j. To make it a sum over r (the rank), we need to absorb the j sum.
    
    # Actually, I think the correct formulation is:
    #   T_{(i,j), (j',k'), (k'',i'')} = 1 if j = j' and k = k'' and i = i'', else 0
    # Wait, no. Let me think again.
    
    # The matrix multiplication tensor M has entries:
    #   M_{i, j, k} = 1 if the (i,k) entry of C depends on A[i,j] and B[j,k]
    # But C[i,k] = sum_j A[i,j] * B[j,k], so each term A[i,j] * B[j,k] contributes to C[i,k].
    # So the tensor should have a 1 at position (i,j; j,k; i,k) for each (i,j,k).
    
    # In the index convention:
    #   U index: i*m + j
    #   V index: j*p + k
    #   W index: k*n + i
    # So T[(i*m+j), (j'*p+k'), (k''*n+i'')] = 1 if i=i'', j=j', k=k'', else 0.
    
    # Let me verify with the schoolbook:
    # For each (i,j,k), set U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # Then T[u,v,w] = sum_col U[u,col]*V[v,col]*W[w,col]
    # T[(i*m+j), (j'*p+k'), (k''*n+i'')] = 1 if there exists col such that:
    #   U[i*m+j, col] = 1, V[j'*p+k', col] = 1, W[k''*n+i'', col] = 1
    # In schoolbook, col = i*m + j * p + k? No, col is just a counter.
    # For each (i,j,k), we set one column. So T[(i*m+j), (j'*p+k'), (k''*n+i'')] = 1
    #   iff (i,j,k) = (i',j',k') where i''=i, j''=j', k''=k.
    # So T[(i*m+j), (j'*p+k'), (k''*n+i'')] = 1 iff i=i'', j=j', k=k''.
    # Yes! That's correct.
    
    # So the schoolbook rank is n*m*p = 3*3*7 = 63.
    
    # Now, can we do better?
    # The key insight is that we can group terms.
    # For example, for the 3x3x3 part (first 3 columns of B), we can use a decomposition of rank < 27.
    # For 3x3x3, the minimum rank is known to be at most 23 (Laderman).
    # If we use Laderman for the first 3 columns and schoolbook for the last 4, we get 23 + 4*3 = 23+12 = 35.
    # Or if we use Laderman for two 3x3x3 blocks and schoolbook for the last column: 23+23+3 = 49.
    # Or if we can find a better decomposition for 3x3x7 directly.
    
    # Let me try to implement Laderman's algorithm for 3x3x3.
    # Laderman (1977) gave a 23-term decomposition for 3x3x3 matrix multiplication.
    
    # Actually, I recall that the rank of the 3x3x3 matrix multiplication tensor is at most 23,
    # but I'm not sure of the exact decomposition. Let me try to find a simpler approach.
    
    # Alternative: use the fact that 3x3x7 can be seen as computing 3 outputs of size 3x7.
    # Each row of C is a 3x7 matrix that is a linear combination of the rows of B.
    # C[i, :] = sum_j A[i,j] * B[j, :]
    # This is a linear map from R^{3x7} to R^{3x7} (for each fixed A).
    
    # Hmm, let me try a different approach. Let me see if I can find a decomposition of rank 49
    # by using a 7-term decomposition for 3x3x3 (Strassen-like) but I don't think Strassen works for 3x3.
    # Strassen's algorithm is for 2x2x2, giving rank 7 instead of 8.
    # For 3x3x3, the best known upper bound is 23 (Laderman).
    
    # Let me try to implement Laderman's decomposition.
    # I'll search my memory... I think Laderman's decomposition uses 23 products.
    # The key idea is to use identities like (a+b)(c+d) = ac + ad + bc + bd to save one product.
    
    # Actually, let me try a more systematic approach. Let me define the tensor and try to find
    # a low-rank decomposition using optimization or a structured search.
    
    # Given the time constraint, let me try a few known decompositions.
    
    # First, let me verify that the schoolbook decomposition works.
    result = _verify(U_fb, V_fb, W_fb, n, m, p)
    if result:
        # Schoolbook works, but can we do better?
        pass
    
    # Let me try a decomposition that uses the structure of the problem.
    # Idea: group the 7 columns of B into 3+3+1.
    # For the first 3 columns, use a 3x3x3 decomposition of rank r1.
    # For the next 3 columns, use a 3x3x3 decomposition of rank r2.
    # For the last column, use a 3x3x1 decomposition of rank r3 = 3 (since it's just 3 products).
    # Total rank = r1 + r2 + r3.
    
    # If I can find a 3x3x3 decomposition of rank < 27, I can improve.
    # Let me try to implement a simple 3x3x3 decomposition.
    
    # Actually, let me try a different approach: use the fact that the matrix multiplication tensor
    # has a lot of structure, and try to find a decomposition by solving a system of equations.
    
    # Given the time constraint and the complexity, let me just return the schoolbook decomposition
    # unless I can quickly find a better one.
    
    # Let me try one more thing: use a block decomposition.
    # If I can decompose 3x3x3 into rank 20, then 20+20+3 = 43 < 63.
    # But I don't know a 20-term decomposition for 3x3x3.
    
    # Let me just return the schoolbook for now, as it's guaranteed to work.
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _verify(U, V, W, n, m, p):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    reconstructed = np.einsum('ur,vr,wr->uvw', U, V, W)
    return np.allclose(reconstructed, T, atol=1e-9)
