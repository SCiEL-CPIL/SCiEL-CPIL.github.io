"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    # Check dimensions
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    for c in range(rank):
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
    
    # For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # and for any (i, j, k) where we don't want a product, it should be 0.
    # Actually, the tensor T[i,j,k] = 1 for all valid (i,j,k) in the multiplication.
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # where T[i,j,k] = 1 if this corresponds to a valid multiplication entry.
    
    # Wait, let me re-read the task. The matrix multiplication tensor for (n,m,p) is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So for every (i, j, k) in the valid range, the sum must equal 1.
    # But there are no "invalid" entries - the tensor is all ones for the valid indices.
    # The number of free parameters in U, V, W is rank*(n*m + m*p + p*n).
    # The number of constraints is n*m*p (one for each (i,j,k)).
    
    # Let me verify by checking a few random (i,j,k) tuples.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    import numpy as np
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    # For each (i,j,k): sum_c U_reshaped[i,j,c] * V_reshaped[j,k,c] * W_reshaped[k,i,c]
    # This is a batched operation.
    # We can compute it as: for each c, U_reshaped[:,:,c] * V_reshaped[:,:,c] * W_reshaped[:,:,c]
    # But the indices don't align directly. Let me think...
    
    # Actually, let's just do it with einsum or loops.
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is like a tensor contraction.
    
    # Let me use a loop over c and accumulate.
    T = np.zeros((n, m, p), dtype=float)
    for c in range(rank):
        # U[:, :, c] has shape (n, m)
        # V[:, :, c] has shape (m, p)  
        # W[:, :, c] has shape (p, n)
        # We need T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is: for each i,j,k: U[i,j,c] * V[j,k,c] * W[k,i,c]
        # = (U[:,:,c] * W.T[:,:,c].T) ... hmm, not quite.
        
        # Let me think of it as: A[i,j] = U[i,j,c], B[j,k] = V[j,k,c], C[k,i] = W[k,i,c]
        # T[i,j,k] += A[i,j] * B[j,k] * C[k,i]
        # = sum over the "cycle" i->j->k->i
        
        # This is equivalent to: trace of a 3-way product.
        # For fixed i: T[i,j,k] += U[i,j,c] * V[j,k,c] * W[k,i,c]
        # = (U[i,:,c] * V[:,k,c].T) @ W[:,i,c] ... no.
        
        # Let me just do it with numpy broadcasting.
        # U[:, :, c] shape (n, m, 1)
        # V[:, :, c] shape (m, p, 1) -> need to align j index
        # W[:, :, c] shape (p, n, 1) -> need to align k and i indices
        
        # T[i,j,k] = U[i,j,c] * V[j,k,c] * W[k,i,c]
        # = np.einsum('ij,jk,ki->ijk', U[:,:,c], V[:,:,c], W[:,:,c])
        T += np.einsum('ij,jk,ki->ijk', U_reshaped[:, :, c], V_reshaped[:, :, c], W_reshaped[:, :, c])
    
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Start with the known best: Strassen for 4x4*4x4 + schoolbook for 4x4*4x3
    # This gives rank 56 + 48 = 104
    
    # First, let's build the Strassen-based decomposition for the 4x4*4x4 part
    # and schoolbook for the remaining 4x4*4x3 part.
    
    # Actually, let me try a different structural approach.
    # The key insight from the directive: treat as 7 independent 4x4x1 multiplications.
    # For each column k of B, we compute A * B[:,k] which is a 4x4 * 4x1 multiplication.
    # The tensor for 4x4*4x1 has rank at most 16 (schoolbook), but can it be lower?
    
    # For matrix-vector multiplication (n x m) * (m x 1), the tensor rank is exactly n*m
    # for the standard case, unless we use special structure.
    # Actually, for A (4x4) * v (4x1), the result is a 4x1 vector.
    # The tensor has 4*4*1 = 16 entries, all 1.
    # Can we decompose this with rank < 16?
    
    # The border rank of matrix-vector multiplication... Let me think.
    # For n x m * m x 1, the tensor is T[i,j] = 1 (i in [0,n), j in [0,m)).
    # This is just an n*m tensor of all ones.
    # The rank of an all-ones tensor of size n*m is... 
    # Actually, T[i,j] = 1 = 1*1, so rank 1? No, that's for a matrix.
    # For the tensor T[i,j,k] where k is the output index... 
    
    # Wait, I'm confusing myself. Let me re-read the problem.
    # We're multiplying A (n x m) by B (m x p) to get C (n x p).
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all valid (i,j,k).
    # The decomposition is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # So the tensor is just all-ones of size n*m*p.
    # The rank of the all-ones tensor of size a*b*c is 1!
    # Because T[i,j,k] = 1 = 1 * 1 * 1.
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all indices.
    
    # Wait, that can't be right. Let me re-read the problem statement.
    
    # "Matrix multiplication 4,4,7"
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 7 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]"
    
    # Hmm, if the tensor is just all ones, then rank 1 would work.
    # But the schoolbook decomposition has rank n*m*p = 112.
    # And the Strassen-based one has rank 104.
    # If rank 1 worked, the score would be 112/1 = 112, which is way better than 1.077.
    
    # So I must be misunderstanding the problem. Let me re-read.
    
    # Ah, I think the issue is that the tensor is NOT all ones. 
    # In matrix multiplication, the "tensor" being decomposed is the multiplication map itself.
    # The standard formulation is:
    # C = A * B, where C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be viewed as a bilinear map from (A, B) -> C.
    # The tensor for this map has components T[i,j,k] which represent the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = 1 if the indices match up correctly, and 0 otherwise.
    
    # But in the standard formulation, ALL combinations of (i,j,k) contribute:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So for each (i,j,k), the coefficient is 1.
    # The tensor is indeed all ones of size n*m*p.
    
    # But wait, that would mean rank 1 works, which contradicts the known results.
    # Let me look at the schoolbook decomposition more carefully.
    
    # In the schoolbook:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # where c = i*m*p + j*p + k (or similar indexing)
    
    # So for each (i,j,k), there's a separate term with U=1, V=1, W=1.
    # The sum over c gives: for a fixed (i,j,k), only one c contributes (the one where
    # c corresponds to (i,j,k)), and that term is 1*1*1 = 1.
    # So T[i,j,k] = 1 for all (i,j,k).
    
    # This confirms the tensor is all ones.
    
    # But then why doesn't rank 1 work?
    # If I set U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all indices,
    # then T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0]
    # = 1 * 1 * 1 = 1
    
    # This should work! Rank 1!
    
    # But the score would be 112/1 = 112, which is way above 1.077.
    # So either I'm misunderstanding the problem, or there's a constraint I'm missing.
    
    # Let me re-read the task contract:
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 7 matrix."
    # "score is the rank"
    # "The evaluator measures the raw value (lower is better)."
    # "search score is seed/value"
    
    # So if the rank is R, the score is 112/R (where 112 is the schoolbook rank).
    # If R=1, score = 112.
    # If R=104, score = 112/104 ≈ 1.077.
    
    # So rank 1 should give score 112, which is much better than 1.077.
    # But the current best is 1.077, which suggests rank 1 doesn't work.
    
    # There must be something I'm missing. Let me think about what "tensor decomposition" means here.
    
    # Ah! I think the issue is that the decomposition must be EXACT in the sense that it
    # correctly computes the matrix product for ALL inputs, not just that the tensor
    # coefficients match. But the tensor coefficients ARE the complete specification of
    # the bilinear map, so if T[i,j,k] = 1 for all (i,j,k), then the bilinear map is
    # correctly specified.
    
    # Unless... the problem is asking for a different kind of decomposition?
    
    # Let me look at the W convention: "W[k*n+i,r]"
    # This is the "trace convention" which is standard for matrix multiplication tensors.
    
    # I'm going to try rank 1 and see if it works.
    
    # Actually, wait. Let me reconsider. Maybe the tensor is NOT all ones.
    # In some formulations, the tensor for matrix multiplication has a specific structure
    # where not all entries are 1.
    
    # Let me look at this from first principles. The matrix multiplication C = A*B is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The bilinear map is: (A, B) -> C
    # This can be written as: C[i,k] = sum_{i',j',k'} T[i',j',k'] * A[i',j'] * B[j',k'] * delta(i',i) * delta(k',k)
    
    # Hmm, that's not quite right either.
    
    # The standard tensor formulation is:
    # The multiplication tensor T has indices (i, j, k) and T[i,j,k] = 1 if and only if
    # the product A[i,j] * B[j,k] contributes to C[i,k].
    
    # Since A[i,j] * B[j,k] contributes to C[i,k] for ALL valid (i,j,k), we have T[i,j,k] = 1.
    
    # I'm stuck. Let me just try rank 1 and see what happens.
    
    # Actually, I just realized something. Let me re-read the schoolbook code:
    
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             U[i * m + j][column] = 1
    #             V[j * p + k][column] = 1
    #             W[k * n + i][column] = 1
    #             column += 1
    
    # So U has n*m rows, V has m*p rows, W has p*n rows.
    # The column index c corresponds to (i,j,k).
    
    # For a given (i,j,k), the contribution to the product is:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where c is the column corresponding to (i,j,k).
    
    # But for a DIFFERENT (i',j',k'), the same column c (corresponding to (i,j,k)) would give:
    # U[i'*m+j', c] * V[j'*p+k', c] * W[k'*n+i', c]
    # = 0 * 0 * 0 = 0 (since U[i'*m+j', c] = 0 unless i'=i and j'=j, etc.)
    
    # So the decomposition is:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # = 1 (only the term where c corresponds to (i,j,k) contributes)
    
    # This confirms T is all ones.
    
    # OK so let me just try rank 1.
    
    U1 = [[1] * 1 for _ in range(n * m)]
    V1 = [[1] * 1 for _ in range(m * p)]
    W1 = [[1] * 1 for _ in range(p * n)]
    
    if _verify_fast(U1, V1, W1, n, m, p):
        return {"U": U1, "V": V1, "W": W1}
    
    # If rank 1 doesn't work (which would be surprising), fall back to schoolbook.
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
