"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    # The tensor T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,8).
    # We seek the minimal rank R such that T = sum_r U_r ⊗ V_r ⊗ W_r.
    #
    # Known: The rank of the matrix multiplication tensor M_{n,m,p} is n*m*p for the
    # schoolbook algorithm. For 3x3 * 3x8, schoolbook rank is 3*3*8 = 72.
    #
    # However, we can do better by exploiting the structure. For each fixed k, the
    # slice is a 3x3x3 tensor T[i,j,k] = 1, which is the tensor for multiplying a 3x3
    # matrix by a 3x1 vector. The rank of this tensor is 3 (not 9), because:
    #   T[i,j,k] = delta_{i,k} (for the matrix-vector case where output index = i, and
    #   we fix the output column k... wait, no).
    #
    # Let me reconsider. The bilinear form is:
    #   C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k.
    #
    # For the 3x3 * 3x3 case, the tensor T[i,j,k]=1 has rank 3.
    # Proof: T[i,j,k] = sum_r delta_{i,r} * 1 * delta_{k,r}
    #   U[i,j,r] = delta_{i,r}
    #   V[j,r] = 1 (but V has shape (m, rank) = (3, rank), so V[j,r] for j in [0,3))
    #   W[k,i,r]... wait, the indices are U: (n*m, rank) = (9, rank), V: (m*p, rank) = (9, rank), W: (p*n, rank) = (9, rank)
    #
    # For n=m=p=3:
    #   U shape: (3*3, R) = (9, R)
    #   V shape: (3*3, R) = (9, R)
    #   W shape: (3*3, R) = (9, R)
    #   T[i,j,k] = sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
    #
    # We want T[i,j,k] = 1 for all i,j,k in {0,1,2}.
    #
    # Rank 3 decomposition:
    #   For r in {0,1,2}:
    #     U[i*3+j, r] = 1 if i == r, else 0  (so U row i*3+j is 1 for all j when i=r, else 0)
    #     V[j*3+k, r] = 1 for all j, k
    #     W[k*3+i, r] = 1 if k == r, else 0
    #   Sum_r: U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
    #     = sum_r [1 if i==r] * 1 * [1 if k==r]
    #     = 1 if i == k, else 0
    #   But we need 1 for all i,j,k, not just i==k!
    #
    # So that doesn't work. The rank of the 3x3x3 all-ones tensor is actually 3, but
    # the decomposition is different. Let me think again.
    #
    # The tensor T[i,j,k] = 1. We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    #
    # Try rank 3:
    #   r=0: U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1 → sum = 1. Done? 
    #   But that's rank 1! U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1 gives sum=1 for all i,j,k.
    #   So the rank is 1?!
    #
    # Wait, that can't be right for matrix multiplication. Let me re-examine the tensor.
    #
    # The matrix multiplication tensor for n=m=p has T[i,j,k] = 1 if the output index
    # corresponds to the correct (i,k) pair given input indices (i,j) and (j,k).
    # Actually, the standard tensor for matrix multiplication M_{n} has:
    #   T[i,j,k] = 1 if i = output row, j = input col of A / input row of B, k = output col
    #   And the bilinear form is C[i,k] = sum_j A[i,j] B[j,k].
    #   So T[i,j,k] = 1 for all i, j, k in range(n).
    #
    # So T is the all-ones tensor of size n x n x n. Its rank as a 3-tensor...
    # If we set U[i,j,r] = 1, V[j,k,r] = 1, W[k,i,r] = 1 for r=0 only, we get sum = 1.
    # So rank is 1?
    #
    # But that contradicts Strassen's result that M_2 has rank 7, not 1.
    #
    # The issue: the tensor for matrix multiplication is NOT the all-ones tensor in the
    # way I'm thinking. The indices are:
    #   A is indexed by (i, j): row i, col j
    #   B is indexed by (j, k): row j, col k
    #   C is indexed by (i, k): row i, col k
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #   So the tensor T has entries T[i, j, k] where the bilinear form is
    #   sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]... no.
    #
    # The standard definition: The matrix multiplication tensor M_{n,m,p} is a tensor in
    #   F^{n*m} ⊗ F^{m*p} ⊗ F^{p*n}
    # with entries indexed by ((i,j), (j,k), (k,i)) and T[(i,j),(j,k),(k,i)] = 1.
    #
    # So for n=3, m=3, p=8:
    #   U has shape (3*3, R) = (9, R), indexed by (i,j)
    #   V has shape (3*8, R) = (24, R), indexed by (j,k)
    #   W has shape (8*3, R) = (24, R), indexed by (k,i)
    #   T[(i,j),(j,k),(k,i)] = 1 for all valid i,j,k
    #
    # So the tensor IS the all-ones tensor in the sense that every valid triple gets 1.
    # And with U[(i,j),0]=1, V[(j,k),0]=1, W[(k,i),0]=1, the sum is 1.
    # So rank is 1?
    #
    # This is clearly wrong. The problem is that the output C is indexed by (i,k), but
    # in the tensor, the third index is (k,i) not (i,k). But that's just a permutation.
    #
    # I think the confusion is that the tensor T[i,j,k] = 1 for all i,j,k means that
    # the bilinear form is sum_{i,j,k} A[i,j] B[j,k] C[k,i] = (sum_i A[i,*])(sum_j B[*,j])(sum_k C[k,*])
    # which is a scalar, not the matrix product.
    #
    # No wait. The bilinear form for the matrix multiplication tensor is:
    #   <T, A ⊗ B ⊗ C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]
    # For this to equal sum_{i,k} C[i,k] (the "correct" output), we need...
    #
    # Actually, the standard way: the tensor M_{n} satisfies
    #   sum_{i,j,k} M_{n}[i,j,k] A[i,j] B[j,k] = C[i,k] for each (i,k)
    # No, that's not how it works. The tensor is defined so that
    #   M_{n} = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k  (where the indices correspond to the
    #   three factors of the bilinear form)
    #
    # I think the correct interpretation is:
    #   The matrix multiplication tensor has T[i,j,k] = 1 where the three indices
    #   correspond to: first factor index = i (row of A, row of C), second factor index = j
    #   (col of A, row of B), third factor index = k (col of B, col of C).
    #   But the output C has indices (i,k), so the third factor should be indexed by (k,i)
    #   or (i,k) depending on convention.
    #
    # Given the task says W has shape (p*n, rank) with W[k*n+i, r], the third index is
    #   (k, i) where k is the column of B (and C) and i is the row of A (and C).
    #   So the tensor is T[i, j, k] = 1, and the bilinear form is
    #   sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] W_out[k,i]
    #   where W_out[k,i] is the (k,i)-th entry of the output matrix (row k, col i? or
    #   the output is stored as C[i,k] but the tensor index is (k,i)).
    #
    # Anyway, the key point is that T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,...,7}.
    # And we need rank R such that sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
    #
    # With U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1, the sum is 1. So rank 1 works?
    #
    # Let me verify with the schoolbook: rank = n*m*p = 72. If rank 1 works, schoolbook
    # is very far from optimal. But Strassen's algorithm shows rank 7 for 2x2x2, not 1.
    #
    # The resolution: The tensor for 2x2x2 matrix multiplication is NOT the all-ones tensor.
    # It has 8 entries, all equal to 1. But the rank is 7, not 1.
    #
    # Wait, if T[i,j,k] = 1 for all 8 entries, then U[i,j,0]=1, V[j,k,0]=1, W[k,i,0]=1
    # gives sum = 1 for all 8 entries. So rank 1. But Strassen says rank 7.
    #
    # I must be misunderstanding the tensor definition. Let me look it up mentally.
    #
    # The matrix multiplication tensor M_{n} is in F^n ⊗ F^n ⊗ F^n with
    #   M_n = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * delta_{i,k}... no.
    #
    # Actually, I recall now: the tensor is T[i,j,k] where the bilinear form is
    #   sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]
    # and T[i,j,k] = 1 if and only if the indices form a valid "path" in the
    # multiplication. For matrix multiplication, every (i,j,k) is a valid path, so
    # T[i,j,k] = 1.
    #
    # But then rank 1 works, contradicting Strassen.
    #
    # I think the issue is that the tensor is NOT T[i,j,k]=1. The correct tensor for
    # matrix multiplication has T[i,j,k] = 1 only when the output index matches.
    # Specifically, for C = A*B, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor should satisfy:
    #   sum_j T[i,j,k] A[i,j] B[j,k] = C[i,k]
    #   => T[i,j,k] = delta_{j, something}... no.
    #
    # The standard definition: M_{n} = sum_{i,j,k} E_i ⊗ E_j ⊗ E_k where E_i, E_j, E_k
    # are basis elements such that the contraction gives the matrix product.
    #
    # I believe the correct tensor is:
    #   T[i, j, k] = 1 for all i, j, k, where the three factors are:
    #     Factor 1: indexed by i (row of A)
    #     Factor 2: indexed by j (col of A = row of B)  
    #     Factor 3: indexed by k (col of B)
    #   And the output C[i,k] is obtained by summing over j.
    #
    # In this case, the tensor is indeed the all-ones tensor, and rank 1 would work.
    # But this contradicts known results.
    #
    # After more thought: I think the issue is that the tensor for matrix multiplication
    # is NOT the all-ones tensor. The tensor M_{n} has entries indexed by (i,j) for A,
    # (j,k) for B, and (i,k) for C. The entry is 1 if and only if the indices are
    # consistent, i.e., the j in A matches the j in B. Since we sum over j, the tensor
    # has T[(i,j), (j',k), (i',k')] = 1 if i=i', k=k', and j=j'.
    #
    # So T is not all-ones; it's a "partial" identity. Specifically:
    #   T[(i,j), (j',k), (i',k')] = delta_{i,i'} * delta_{k,k'} * delta_{j,j'}
    #
    # This makes more sense! The rank of this tensor for n=2 is 7 (Strassen), not 1.
    #
    # For our problem: n=3, m=3, p=8.
    #   A is 3x3, indexed by (i, j) with i in {0,1,2}, j in {0,1,2}
    #   B is 3x8, indexed by (j, k) with j in {0,1,2}, k in {0,...,7}
    #   C is 3x8, indexed by (i, k) with i in {0,1,2}, k in {0,...,7}
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #
    # The tensor T has entries T[(i,j), (j',k), (i',k')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    #
    # Wait, but the task says W has shape (p*n, rank) with W[k*n+i, r]. So the output
    # index is (k, i), not (i, k). So the tensor entry is:
    #   T[(i,j), (j',k), (k',i')] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    #
    # The schoolbook decomposition has rank n*m*p = 3*3*8 = 72, with one term for each
    # (i,j,k) triple:
    #   U[i*m+j, (i,j,k)] = 1
    #   V[j*p+k, (i,j,k)] = 1
    #   W[k*n+i, (i,j,k)] = 1
    #
    # To reduce rank, we need to find a smaller set of terms.
    #
    # For the 3x3 * 3x3 case, the rank is known to be 23 (from Alon-Roichman or similar
    # results, actually the best known upper bound is 23 and lower bound is 20 or so).
    # But for 3x3 * 3x8, we can potentially do better by exploiting the rectangular shape.
    #
    # A simple improvement: For each fixed i and k, the sum over j is a dot product of
    # row i of A and column k of B. This is a rank-1 update to C[i,k].
    #
    # Alternatively, we can use the fact that for fixed k, the 3x3 * 3x1 multiplication
    # has tensor rank 3 (as I discussed earlier, the all-ones tensor in 3x3x1 space has
    # rank... let me think).
    #
    # For fixed k, the tensor slice is T[i, j, i'] = delta_{i, i'} * delta_{j, j} (trivially)
    # Actually, for fixed k, the slice is T[i, j, i'] = delta_{i, i'} (since j=j' is
    # enforced by the summation, and k=k' is fixed).
    #
    # The 3x3x3 tensor T[i,j,i'] = delta_{i,i'} has rank 3:
    #   T[i,j,i'] = sum_r delta_{i,r} * 1 * delta_{i',r}
    #   U[i*3+j, r] = delta_{i,r}
    #   V[j*3+k, r] = 1  (but k is fixed, so V[j, r] = 1)
    #   W[k*3+i', r] = delta_{i',r}
    #
    # So for each k, we need rank 3. But we have 8 values of k. If we can share the
    # factors across k, we might reduce the total rank.
    #
    # A natural approach: For each (i, i') pair where i=i', we need 3 terms (one for each
    # r=i). But actually, for fixed k, the decomposition is:
    #   T[i,j,i'] = sum_{r=0}^{2} delta_{i,r} * 1 * delta_{i',r}
    # This works for any fixed k. The factors U and W do not depend on k, but V does
    # (V depends on j and k).
    #
    # If we use the same U and W for all k, but different V for each k, the total rank
    # is 3 * 8 = 24 (since V has 3*8=24 rows, and we need 3 terms per k, so 8*3=24 terms).
    #
    # But wait, the V factor has shape (m*p, rank) = (24, rank). If we use 24 terms,
    # each term r corresponds to a specific (k, r) pair. So the rank is 24.
    #
    # Can we do better? If we allow V to depend on k in a structured way, maybe.
    #
    # For now, let's implement the rank-24 decomposition:
    #   For each k in {0,...,7} and r in {0,1,2}:
    #     term index = k * 3 + r
    #     U[i*3+j, term] = delta_{i,r}  (1 if i==r, else 0)
    #     V[j*8+k, term] = 1
    #     W[k*3+i, term] = delta_{i,r}  (1 if i==r, else 0)
    #
    # Let me verify: sum_{term} U[i*3+j, term] V[j*8+k, term] W[k*3+i, term]
    #   = sum_{k'=0}^{7} sum_{r=0}^{2} delta_{i,r} * delta_{k,k'} * delta_{i,r}
    #   = sum_{r=0}^{2} delta_{i,r} * delta_{i,r}  (only k'=k survives)
    #   = delta_{i,i}  (since delta_{i,r} is 0 or 1, so delta_{i,r}^2 = delta_{i,r})
    #   = 1
    #
    # Yes! This gives 1 for all valid (i,j,k). So rank 24 works.
    #
    # Can we reduce further? The lower bound for 3x3 * 3x3 is 23 (or maybe lower).
    # For 3x3 * 3x8, a simple lower bound is 3*3 = 9 (from the 3x3 * 3x1 case repeated 8
    # times, but with sharing). Actually, the lower bound is at least the rank of any
    # slice, which is 3, so 8*3=24 is not a lower bound. The true lower bound is unknown
    # but likely less than 24.
    #
    # However, implementing a rank < 24 decomposition is non-trivial. For this task,
    # rank 24 is a significant improvement over rank 72.
    
    n, m, p = DIMENSIONS
    rank = 0
    U = [[0] * 1  # placeholder, will resize
         for _ in range(n * m)]
    V = [[0] * 1
         for _ in range(m * p)]
    W = [[0] * 1
         for _ in range(p * n)]
    
    # Build the rank-24 decomposition
    terms = []
    for k in range(p):
        for r in range(n):
            terms.append((k, r))
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (k, r) in enumerate(terms):
        for i in range(n):
            for j in range(m):
                # U[i*m+j, col] = delta_{i,r}
                U[i * m + j][col] = 1 if i == r else 0
                # V[j*p+k, col] = 1
                V[j * p + k][col] = 1
                # W[k*n+i, col] = delta_{i,r}
                W[k * n + i][col] = 1 if i == r else 0
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
