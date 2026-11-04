"""Matrix multiplication tensor decomposition for 2x4 by 4x8 matrices."""
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    # Rank 64 decomposition using the Kronecker product structure.
    # The multiplication tensor T[i,j,k] = A[i,j]*B[j,k] can be seen as:
    # For each output position (i,k), the bilinear form is sum_j A[i,j]*B[j,k].
    # We use rank 64 = n*m*p by leveraging the fact that we can share factors.
    
    # Actually, let's use a smarter construction. The standard rank is n*m*p = 64.
    # But we can do better by exploiting structure.
    
    # Let's try a direct approach: the tensor is in R^{2} x R^{4} x R^{8} -> output R^{2} x R^{8}
    # The multiplication tensor has entries T[i, j, k] = delta structure...
    
    # Actually the matrix multiplication tensor for C = A*B where A is 2x4, B is 4x8:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) -> contributes to C[i,k]
    # T is a 3-tensor in R^{2 x 4 x 8}
    
    # We need rank decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r in R^2 (for A indices (i,j)), v_r in R^4 (for B indices (j,k))... 
    # Wait, no. The interface says:
    # U: shape (n*m, rank) = (8, rank)
    # V: shape (m*p, rank) = (32, rank)  
    # W: shape (p*n, rank) = (16, rank)
    
    # So U rows are indexed by (i,j) in 0..7
    # V rows are indexed by (j,k) in 0..31
    # W rows are indexed by (k,i) in 0..15 (trace convention W[k*n+i, r])
    
    # The bilinear form is: C[k*n+i, :] ... wait, let me re-read.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W has rows indexed by (k,i) where k is the column of B / output column, i is row of A / output row.
    # And C[i,k] corresponds to position... the decomposition satisfies:
    # The (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C.
    # In tensor form: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over j and r? No...
    
    # Actually the standard formulation: C = sum_r u_r v_r^T w_r^T where
    # C[i,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, that doesn't quite work dimensionally unless we're careful.
    
    # Let me think again. The tensor decomposition of matrix multiplication:
    # We want to represent the bilinear map (A, B) -> C where C = AB.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # A rank-r decomposition gives: C[i,k] = sum_r f_r(A) * g_r(B) * h_r(i,k)
    # where f_r depends on A, g_r depends on B, and h_r gives the (i,k) position.
    
    # In the format specified:
    # U has shape (n*m, rank), so U[a, r] for a = i*m+j
    # V has shape (m*p, rank), so V[b, r] for b = j*p+k
    # W has shape (p*n, rank), so W[c, r] for c = k*n+i
    
    # Then: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's verify with schoolbook: for each (i,j,k), set U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # Then sum over j: only the term where U[i*m+j,c]=1 AND V[j*p+k,c]=1 contributes.
    # For a given c corresponding to (i',j',k'), U[i*m+j,c]=1 only if i=i', j=j'
    # V[j*p+k,c]=1 only if j=j', k=k'
    # So both give j=j', i=i', k=k'. Then W[k*n+i,c] = W[k'*n+i',c]=1
    # So C[i,k] gets contribution 1*1*1=1 only when there exists (i',j',k') with i=i',k=k', and the term is included.
    # This gives C[i,k] = sum over all j such that there's a term for (i,j,k), which is all j.
    # But that's not right - it gives C[i,k] = p (all k values contribute). 
    
    # Hmm, I think I'm misunderstanding. Let me reconsider.
    
    # Actually the correct interpretation: the decomposition satisfies
    # sum_{i,j,k} T[i,j,k] * x_i * y_j * z_k = sum_r (sum_{i,j} U[i,j,r] x_i y_j) * (sum_k V[j,k,r]... 
    
    # No, let me just verify with the seed program that scores 1.0 and replicate its logic
    # exactly, since that's a valid rank 64 solution. Then I'll try to improve.
    
    # Schoolbook rank 64:
    rank = 64
    U = np.zeros((8, rank))
    V = np.zeros((32, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for i in range(2):
        for j in range(4):
            for k in range(8):
                U[i*4+j, col] = 1
                V[j*8+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
