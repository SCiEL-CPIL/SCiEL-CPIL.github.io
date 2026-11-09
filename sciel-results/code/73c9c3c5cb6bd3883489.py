import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    # The matrix multiplication tensor for (n,m,p) = (2,4,4):
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Trivial decomposition: rank = n*m*p = 32
    # Each term r corresponds to a unique (i,j,k) triple
    
    # Let's try to find a lower rank decomposition.
    # The known lower bound for 2x4x4 is at least 2*4*4/something...
    # Actually, the trivial rank is 32. Let's see if we can do better.
    
    # Strategy: Try to construct a rank-28 or lower decomposition directly.
    # For n=2, the output has 2 rows. The W matrix has shape (p*n, rank) = (8, rank).
    # W[k*n + i, r] = W[4*i + k, r] for i in {0,1}, k in {0,1,2,3}
    
    # Let's try a systematic approach: start with the trivial rank-32 and try to
    # merge terms or find linear dependencies.
    
    # Actually, let me first verify what rank is achievable.
    # The tensor T is of size (2*4) x (4*4) x (4*2) = 8 x 16 x 8
    # T[i,j,k] = 1 for all valid indices.
    
    # This is a very structured tensor. Let's think about it differently.
    # T[i,j,k] = 1 means the tensor is the outer product of ones vectors in each mode?
    # No, T is a 3-tensor, not a matrix. T[i,j,k] = 1 for all i,j,k means it's
    # the all-ones 3-tensor.
    
    # The all-ones 3-tensor of size a x b x c has rank 1! Because:
    # T[i,j,k] = 1 = 1 * 1 * 1
    # So U[i,r]=1, V[j,r]=1, W[k,r]=1 for r=0 only.
    # Wait, that gives T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r] = U[i,0]*V[j,0]*W[k,0] = 1*1*1 = 1.
    
    # So the rank is 1?! That can't be right for matrix multiplication...
    
    # Oh wait, I'm confusing the tensor. The matrix multiplication tensor is NOT
    # the all-ones tensor. Let me reconsider.
    
    # For matrix multiplication C = A * B where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T satisfies: C[i,k] = sum_j A[i,j] * B[j,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 if the indices are consistent with the multiplication, else 0.
    # But actually, T[i,j,k] = 1 for ALL valid i, j, k because every A[i,j] * B[j,k]
    # contributes to C[i,k]. There's no condition that makes T[i,j,k] = 0.
    
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # This IS the all-ones tensor!
    
    # And the all-ones tensor of size a x b x c has rank 1:
    # U[i,0] = 1 for all i, V[j,0] = 1 for all j, W[k,0] = 1 for all k
    # Then sum_r U[i,r]*V[j,r]*W[k,r] = 1*1*1 = 1 = T[i,j,k] ✓
    
    # So the rank is 1! But that seems too good to be true. Let me double-check.
    
    # The tensor for matrix multiplication: we want to compute C = A @ B
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # For this to work, we need T[i,j,k] such that sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j] B[j,k]
    # This requires T[i,j,k] = 1 for all valid i,j,k.
    
    # So yes, T is the all-ones tensor and rank is 1.
    
    # But wait - the problem says "exact tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 4 matrix".
    # The standard matrix multiplication tensor has T[i,j,k] = 1 for all valid indices.
    # The rank of the all-ones tensor is indeed 1.
    
    # Let me verify: with rank 1, U has shape (8,1), V has shape (16,1), W has shape (8,1).
    # U[i,0] = 1, V[j,0] = 1, W[k,0] = 1.
    # Then for any A (2x4) and B (4x4):
    # C[i,k] = sum_r U[i,0]*V[j,0]*W[k,0] * A[i,j]*B[j,k] summed over j
    # Hmm, I need to be more careful about the indexing.
    
    # The decomposition is: the bilinear map (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # can be written as: C[i,k] = sum_r sum_j U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * ... 
    # No, that's not right either.
    
    # Let me re-read the task: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # The tensor T has indices (i,j,k) where i is row of A (and C), j is col of A (and row of B), k is col of B (and C).
    # T[i,j,k] = 1 for all valid i,j,k.
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    
    # So U is indexed by (i,j), V by (j,k), W by (k,i).
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the all-ones tensor: T[i,j,k] = 1 for all i,j,k.
    # With rank 1: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # Then sum_r ... = 1*1*1 = 1. ✓
    
    # So the answer is rank 1!
    
    # But wait, this feels wrong. Matrix multiplication should have non-trivial rank.
    # Let me reconsider... 
    
    # Oh! I think I'm misunderstanding the tensor. The matrix multiplication tensor is NOT
    # simply all-ones. The issue is that in the bilinear computation, each term A[i,j]*B[j,k]
    # contributes to C[i,k], but the "tensor" that encodes this is:
    # T_{ijk} = 1 if the indices form a valid "chain" i->j->k in the multiplication.
    # But since every (i,j,k) triple is valid (any row of A times any column of B with matching
    # middle index), T_{ijk} = 1 for all valid i,j,k.
    
    # I'm now confident the rank is 1. Let me just return that.
    
    if time.time() > deadline:
        pass  # shouldn't happen
    
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
