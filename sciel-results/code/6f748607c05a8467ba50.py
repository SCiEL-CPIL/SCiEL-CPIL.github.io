<<<<<<< SEARCH
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 1
    U = [[1] * rank for _ in range(n * m)]
    V = [[1] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Standard matrix multiplication tensor decomposition
    # T[i, j, k] = sum_l A[i, l] * B[l, k]
    # Decomposition: T = sum_r U[i, r] * V[l, r] * W[k, r]
    # where r indexes the summation index l (0 to m-1)
    # But the interface requires:
    # U: (n*m, rank) -> indexed by (i, l) or similar?
    # V: (m*p, rank) -> indexed by (l, k) or similar?
    # W: (p*n, rank) -> indexed by (k, i) or similar?
    
    # Let's look at the shapes:
    # U is (n*m, rank). The tensor T is of size (n, m, p).
    # Usually, the flattening is row-major.
    # If U[i*m + l, r], V[l*p + k, r], W[k*n + i, r]
    # Then sum_r U[i*m + l, r] * V[l*p + k, r] * W[k*n + i, r]
    # We want this to equal delta_{i, i'} delta_{l, l'} delta_{k, k'} ? No.
    # The multiplication tensor T_{i, l, k} = delta_{i, i} ... no.
    # T_{i, l, k} = 1 if we are summing over l?
    # Actually, the tensor for matrix multiplication C = A * B is:
    # C_{i, k} = sum_l A_{i, l} B_{l, k}
    # This is a bilinear map. The tensor T is in U^* x V^* x W^* ?
    # The task says: "exactly decompose the matrix multiplication tensor"
    # The standard decomposition has rank m.
    # Let's construct the identity decomposition.
    # T_{i, l, k} = 1 if the indices match the summation structure.
    # Actually, the "multiplication tensor" T is defined such that
    # <T, A x B> = tr(A B^T)? No, C = A B.
    # The tensor T_{i, l, k} has value 1 if it corresponds to the term A_{i,l} B_{l,k}.
    # So T_{i, l, k} = 1 for all i, l, k? No, that's not right.
    # The tensor T is a 3-tensor of size n x m x p.
    # T[i, l, k] = 1 if the operation is C[i, k] += A[i, l] * B[l, k].
    # So T[i, l, k] = 1 for all valid i, l, k.
    # Wait, if T[i, l, k] = 1 for all i, l, k, then the decomposition is:
    # sum_r U[i, r] V[l, r] W[k, r] = 1 for all i, l, k?
    # That would require rank 1 with all ones.
    # But the error was "nonzero entries outside the target support".
    # This implies the target tensor T is NOT all ones.
    
    # Let's re-read the interface.
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This suggests the flattening of W is by k and i.
    # W shape is (p*n, rank).
    # If W[k*n + i, r], then the third index of the tensor is k, and the first is i?
    # Or is the tensor T[i, k, l]?
    
    # Standard convention for matrix multiplication tensor T:
    # T is in R^{n x m x p}.
    # T[i, j, k] = 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    # So T[i, j, k] = 1 for all i, j, k.
    # Why would the seed fail?
    # Seed: U all 1s (n*m x 1), V all 1s (m*p x 1), W all 1s (p*n x 1).
    # Sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 * 1 * 1 = 1.
    # This matches T[i, j, k] = 1.
    # So why the error?
    
    # Perhaps the flattening order is different.
    # If U is flattened as (i, j) -> i*m + j.
    # If V is flattened as (j, k) -> j*p + k.
    # If W is flattened as (k, i) -> k*n + i.
    # Then the product is 1.
    
    # Maybe the "target support" implies that the tensor is sparse?
    # No, matrix multiplication tensor is dense.
    
    # Let's check the error: "tensor has nonzero entries outside the target support".
    # This usually means the reconstructed tensor has non-zeros where the target is zero,
    # or the target has non-zeros where the reconstruction is zero?
    # Or it means the indices in the factor matrices don't align with the tensor structure.
    
    # Let's try a different interpretation.
    # Maybe the tensor is T[i, k, j]?
    # If T[i, k, j] = 1, and we reconstruct:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This still gives 1 for all i, j, k.
    
    # Is it possible the rank 1 solution is considered "trivial" and rejected?
    # Or is there a constraint that U, V, W must be sparse?
    # "integer/half-integer entries"
    
    # Let's look at the "official trace convention W[k*n+i,r]".
    # This confirms W's index is (k, i).
    # What about U and V?
    # U is (n*m, rank). Likely (i, j) -> i*m + j.
    # V is (m*p, rank). Likely (j, k) -> j*p + k.
    
    # If the target tensor is indeed all ones, rank 1 should work.
    # Unless... the evaluator checks for minimal rank? No, it scores rank.
    # Unless the evaluator expects the tensor to be 0 elsewhere?
    # But matrix multiplication tensor is 1 everywhere.
    
    # Wait, is it possible the tensor is defined as T[i, j, k] = delta_{i, k}? No.
    
    # Let's try to construct the standard rank-m decomposition.
    # Rank m.
    # U[i*m + j, r] = 1 if j == r, else 0.
    # V[j*p + k, r] = 1 if j == r, else 0.
    # W[k*n + i, r] = 1 for all k, i?
    # Then sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # = sum_r delta_{j, r} delta_{j, r} * 1
    # = delta_{j, j} = 1.
    # This also gives 1 everywhere.
    
    # So both rank 1 and rank m give the all-ones tensor.
    # Why did rank 1 fail?
    
    # Hypothesis: The flattening of U or V is different.
    # If U is (j, i) -> j*n + i?
    # If V is (k, j) -> k*m + j?
    
    # Let's assume the standard row-major flattening for the matrix dimensions.
    # A is n x m. Flatten A to vector of length n*m.
    # B is m x p. Flatten B to vector of length m*p.
    # C is n x p. Flatten C to vector of length n*p.
    # The bilinear map is C = A B.
    # vec(C) = (I_p x A) vec(B)? No.
    # vec(C) = (B^T x I_n) vec(A).
    # This is a linear map, not bilinear.
    # The bilinear form is <C, X> = tr(X^T A B) = tr(B X^T A) = tr(A B X^T).
    # This can be written as sum_{i, j, k} A_{i, j} B_{j, k} C_{k, i}.
    # So the tensor T[i, j, k] = 1.
    
    # If the tensor is all ones, rank 1 is correct.
    # Why the error?
    # "tensor has nonzero entries outside the target support"
    # This error message is specific. It might mean that the evaluator is checking
    # if the reconstructed tensor matches the target tensor *exactly* in terms of support.
    # If the target tensor is all ones, support is all indices.
    # If my reconstruction is all ones, support is all indices.
    
    # Is it possible the target tensor is NOT all ones?
    # What if the tensor is T[i, j, k] = 1 if i==j and j==k? No.
    
    # Let's look at similar problems.
    # Often, the "multiplication tensor" refers to the structure of the matrix product.
    # Maybe the evaluator expects the factors to be sparse?
    # "rank-minimal exact tensor decomposition"
    
    # If rank 1 fails, maybe the flattening is different.
    # Let's try the rank-m decomposition which is the standard "obvious" one.
    # It uses m factors.
    
    n, m, p = DIMENSIONS
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[1] * rank for _ in range(p * n)] # W is 1 for all?
    
    # If U[i*m + j, r] = delta_{j, r}
    # And V[j*p + k, r] = delta_{j, r}
    # And W[k*n + i, r] = 1
    # Sum_r delta_{j, r} delta_{j, r} * 1 = 1.
    
    # Let's populate U and V.
    for i in range(n):
        for j in range(m):
            idx_u = i * m + j
            U[idx_u][j] = 1
    
    for j in range(m):
        for k in range(p):
            idx_v = j * p + k
            V[idx_v][j] = 1
    
    # W is all ones.
    # This gives rank m.
    # Is rank m minimal?
    # For n=3, m=4, p=7, the minimal rank is likely m=4 (or close).
    # Rank 1 gave an error.
    # Maybe the error was due to integer types? No, lists of ints.
    
    # Let's try this rank-m construction.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
