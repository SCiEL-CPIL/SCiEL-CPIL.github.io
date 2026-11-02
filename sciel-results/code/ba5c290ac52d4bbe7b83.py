"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # For matrix multiplication of (n x m) by (m x p), the tensor rank
    # is well-studied. For 3x3x5, we can use a decomposition based on
    # the structure of the multiplication tensor.
    #
    # The tensor T[i,j,k] = 1 for all valid i,j,k means:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # A rank-r decomposition satisfies:
    # T[i,j,k] = sum_{t=1}^{r} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    #
    # We need to find the minimal r such that this holds exactly.
    #
    # For small dimensions, we can use known results or construct
    # explicit decompositions. Let's try to find a low-rank decomposition
    # by exploiting the algebraic structure.
    
    # Strategy: Use the fact that for matrix multiplication, there are
    # known fast algorithms. For 3x3 matrices, Strassen's algorithm
    # gives rank 7. But we have 3x3 * 3x5, so the output is 3x5.
    
    # Let's think about this differently. We need:
    # C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1,2}, k in {0,1,2,3,4}
    
    # The tensor has entries T[i,j,k] = 1 for all i,j,k.
    # We want: sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 for all i,j,k.
    
    # This is a system of n*m*p = 45 equations in 3*r unknowns
    # (r entries each for U, V, W).
    
    # Let's try to find a decomposition with small r.
    # 
    # Key insight: The tensor T[i,j,k] = 1 is actually a rank-1 tensor
    # in the (i,j,k) space if we ignore the index coupling in the
    # decomposition. But the coupling U[i*m+j], V[j*p+k], W[k*n+i]
    # makes it non-trivial.
    
    # Let me try a constructive approach. Consider:
    # U[i*m+j, t] depends on both i and j
    # V[j*p+k, t] depends on both j and k  
    # W[k*n+i, t] depends on both k and i
    
    # For rank 1: U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1 for all i,j,k
    # This requires U[i*m+j,0] = a(i,j), V[j*p+k,0] = b(j,k), W[k*n+i,0] = c(k,i)
    # such that a(i,j) * b(j,k) * c(k,i) = 1 for all i,j,k.
    # 
    # If we set a(i,j) = 1, b(j,k) = 1, c(k,i) = 1, this works!
    # But wait - is this valid? Let's check:
    # sum_t U[i*m+j,t] V[j*p+k,t] W[k*n+i,t] = 1*1*1 = 1. Yes!
    #
    # So rank 1 should work? That seems too good. Let me verify with
    # the actual matrix multiplication formula.
    
    # Actually, I think I'm confusing the tensor definition. Let me be
    # more careful. The matrix multiplication tensor for (n x m) * (m x p)
    # is defined such that:
    # (A*B)[i,k] = sum_{j=1}^{m} A[i,j] B[j,k]
    #
    # In tensor decomposition form, we write:
    # (A*B)[i,k] = sum_{t=1}^{r} (sum_{j} U[i*m+j,t] A[i,j]) * 
    #                (sum_{j} V[j*p+k,t] B[j,k]) * W[k*n+i,t]
    #
    # No wait, that's not right either. The standard form is:
    # (A*B)[i,k] = sum_{t=1}^{r} (sum_{j} U[i,j,t] A[i,j]) * 
    #                (sum_{j} V[j,k,t] B[j,k]) * W[i,k,t]
    #
    # Hmm, but the task specifies U has shape (n*m, rank), V has shape
    # (m*p, rank), W has shape (p*n, rank). So U is indexed by (i*m+j),
    # V by (j*p+k), W by (k*n+i).
    
    # I think the correct interpretation is that the tensor T[i,j,k]
    # satisfies:
    # sum_{j} T[i,j,k] A[i,j] B[j,k] = (A*B)[i,k]
    #
    # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    #
    # And the decomposition is:
    # T[i,j,k] = sum_{t} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # If this is correct, then rank 1 works with all ones. But this
    # contradicts known results about matrix multiplication tensor rank.
    
    # Let me reconsider. Maybe the tensor is not all ones. Maybe the
    # correct tensor for matrix multiplication has a different structure.
    
    # Actually, I think the issue is that the tensor for matrix
    # multiplication is NOT simply T[i,j,k] = 1. The tensor encodes
    # HOW the multiplication works, and for standard multiplication,
    # it IS T[i,j,k] = 1. But the rank of this tensor (with the
    # specific index structure in the decomposition) is what matters.
    
    # Let me just try to find a decomposition computationally for small
    # ranks and see what works.
    
    def try_decomposition(r):
        """Try to find a rank-r decomposition."""
        # Variables: U (n*m x r), V (m*p x r), W (p*n x r)
        # Constraints: for all i,j,k: sum_t U[i*m+j,t] V[j*p+k,t] W[k*n+i,t] = 1
        
        # This is a system of nonlinear equations. Let's use a
        # different approach: fix two matrices and solve for the third.
        
        # Actually, let's use a smarter approach. Note that the
        # constraint is:
        # For fixed i,k: sum_j sum_t U[i*m+j,t] V[j*p+k,t] W[k*n+i,t] = 1
        #
        # Wait, no. The constraint is for EACH (i,j,k) triple:
        # sum_t U[i*m+j,t] V[j*p+k,t] W[k*n+i,t] = 1
        
        # Let me try a rank-1 solution first.
        # U[i*m+j, 0] = a[i,j], V[j*p+k, 0] = b[j,k], W[k*n+i, 0] = c[k,i]
        # Need: a[i,j] * b[j,k] * c[k,i] = 1 for all i,j,k
        
        # Set a[i,j] = 1 for all i,j. Then need b[j,k] * c[k,i] = 1 for all j,k,i.
        # This means b[j,k] = 1/c[k,i] for all i, which requires c[k,i] to be
        # independent of i, so c[k,i] = c[k]. Then b[j,k] = 1/c[k].
        # Set c[k] = 1 for all k, then b[j,k] = 1 for all j,k.
        # So rank 1 works with all ones!
        
        # But this can't be right. Let me double-check by considering
        # what the decomposition actually computes.
        
        # I think the confusion is about what the decomposition means
        # for the actual multiplication. Let me look at this from
        # the perspective of the number of multiplications.
        
        # In a rank-r decomposition, the number of scalar multiplications
        # is r (one per term in the sum). For standard matrix multiplication
        # of (n x m) by (m x p), we need n*m*p multiplications in the
        # naive algorithm.
        
        # If rank 1 worked, we'd only need 1 multiplication, which is
        # clearly impossible for non-trivial matrix multiplication.
        
        # So I must be misunderstanding the tensor definition. Let me
        # reconsider.
        
        # I think the correct tensor for matrix multiplication is not
        # T[i,j,k] = 1, but rather the tensor that encodes the bilinear
        # form. The matrix multiplication is:
        # C[i,k] = sum_j A[i,j] B[j,k]
        #
        # This is a bilinear map from (A, B) to C. The tensor associated
        # with this bilinear map has entries M[i,j,k] where the bilinear
        # form is:
        # sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k] * (something)
        
        # Actually, I think the standard formulation is:
        # The multiplication tensor M for (n x m) * (m x p) has:
        # M[i,j,k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k]
        #
        # And indeed, A[i,j] B[j,k] contributes to C[i,k] with coefficient 1.
        # So M[i,j,k] = 1 for all valid i,j,k.
        
        # The tensor rank of M is the minimal r such that:
        # M[i,j,k] = sum_{t=1}^{r} U[i,j,t] V[j,k,t] W[k,i,t]
        
        # And if this is correct, then rank 1 works with all ones.
        
        # But this contradicts the fact that matrix multiplication has
        # tensor rank > 1. So either:
        # 1. The tensor is not all ones, or
        # 2. The index structure in the decomposition is different
        
        # Let me try a different approach: just use the schoolbook
        # decomposition which definitely works, and see if we can improve.
        
        pass
    
    # Let's just use the schoolbook decomposition as a safe fallback.
    # Rank = n * m * p = 3 * 3 * 5 = 45
    
    r = n * m * p
    U = np.zeros((n * m, r), dtype=np.int32)
    V = np.zeros((m * p, r), dtype=np.int32)
    W = np.zeros((p * n, r), dtype=np.int32)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
