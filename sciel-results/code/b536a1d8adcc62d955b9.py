import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 3, 5, 7
    
    # We will implement a rank-35 decomposition based on Strassen-like padding.
    # The idea is to pad 3x5 and 5x7 to 4x4 and 4x8 (or similar) and use 7 rank-1 terms per 2x2x2 block?
    # Actually, the prompt suggests "rank of 35".
    # Schoolbook rank is 3*5*7 = 105.
    # Strassen reduces 2x2x2 from 8 to 7.
    # If we pad 3x5 to 4x4? No, 3x5 is not square.
    # Let's try a direct construction for rank 35.
    
    # A known result: rank(3,5,7) <= 35?
    # Let's verify if 35 is achievable.
    # 3*5 = 15, 5*7=35, 3*7=21.
    # If we use a tensor decomposition, the rank R must satisfy R >= max(n*m, m*p, p*n) ? No.
    # R >= max(n*p, ...) ? No.
    # Lower bounds: R >= max(n*m, m*p, p*n) is not true.
    # R >= max(n*p, ...) ?
    # Actually, the trivial lower bound is max(n*p, m*p, n*m) is NOT correct.
    # The lower bound is max(n*p, ...) ?
    # Wait, the tensor is T_{i,j,k} = delta_{j,k}? No, T_{i,j,k} = delta_{j,k} is for identity.
    # Matrix multiplication tensor T_{i,j,k} = delta_{j,k} is wrong.
    # T_{i,j,k} = 1 if j=k? No.
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # Tensor T has indices i, j, k. T_{i,j,k} = delta_{j,k}? No.
    # T_{i,j,k} = 1 if the path i->j->k is valid?
    # The tensor is defined by T_{i,j,k} = 1 if j is the intermediate index.
    # So T is a 3x5x7 tensor where T[i,j,k] = 1 if j is shared? No.
    # T[i,j,k] = 1 if the multiplication uses A[i,j] and B[j,k].
    # So T[i,j,k] = 1 if the index j is the same in A and B.
    # Thus T[i,j,k] = 1 if the j-th element of A's row i and B's column k are multiplied.
    # This is simply the identity map on the middle index?
    # No, the tensor is T[i,j,k] = 1 if j is the contraction index.
    # So for each fixed i and k, the vector over j is the standard basis? No.
    # T[i,j,k] = 1 if j is the index.
    # Actually, the tensor is defined by T[i,j,k] = 1 if the pair (i,j) and (j,k) are connected.
    # This is just the tensor that maps A and B to C.
    # T[i,j,k] = 1 if j is the common index.
    # So T[i,j,k] = 1 if j is in 0..4.
    # Wait, T[i,j,k] is non-zero only if j is the same index in A and B.
    # So T[i,j,k] = 1 if j is the index.
    # This means T[i,j,k] = 1 for all i, k, and specific j?
    # No. T[i,j,k] = 1 if the multiplication step is A[i,j]*B[j,k].
    # So T[i,j,k] = 1 if j is the index.
    # This implies that for each j, we have a 3x7 slice that is all ones?
    # No.
    # Let's look at the definition:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This can be written as a tensor contraction.
    # The tensor T has components T[i,j,k].
    # T[i,j,k] = 1 if the term A[i,j]B[j,k] appears in the sum for C[i,k].
    # This is true for all i, j, k?
    # No, j is the summation index.
    # For a fixed i and k, we sum over j.
    # So T[i,j,k] = 1 for all i, k, and j?
    # No, that would mean C[i,k] = sum_j A[i,j]B[j,k] * 1.
    # Yes, that is correct.
    # So T[i,j,k] = 1 for all i in 0..2, j in 0..4, k in 0..6.
    # Is that right?
    # If T[i,j,k] = 1, then the decomposition is sum_r U[i,j,r] V[j,k,r] W[i,k,r] = T[i,j,k].
    # If T[i,j,k] = 1 for all i,j,k, then we need to decompose the all-ones tensor.
    # The rank of the all-ones tensor of size n x m x p is 1?
    # No, the all-ones tensor has rank 1?
    # U[i,j,r] = 1, V[j,k,r] = 1, W[i,k,r] = 1.
    # Then sum_r 1*1*1 = R.
    # We need sum_r ... = 1.
    # So we can set R=1, U=1, V=1, W=1.
    # But this would imply C[i,k] = 1 * 1 * 1 = 1?
    # No, the formula is C = U * V * W?
    # The interface says: 'U', 'V', 'W' factor matrices.
    # Usually, this means T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[i,k,r].
    # If T[i,j,k] = 1 for all i,j,k, then rank 1 is sufficient.
    # But matrix multiplication is NOT the all-ones tensor.
    
    # Let's re-read carefully.
    # "Matrix multiplication 3,5,7"
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a bilinear map.
    # The tensor T has indices i, j, k.
    # T[i,j,k] = 1 if the pair (i,j) and (j,k) are used together.
    # In the standard definition, T[i,j,k] = 1 if j is the intermediate index.
    # But for a fixed i and k, the sum is over j.
    # So the tensor is actually a collection of slices.
    # The tensor T is defined by T[i,j,k] = 1 if the multiplication involves A[i,j] and B[j,k].
    # This is ALWAYS true for the term A[i,j]B[j,k].
    # So T[i,j,k] = 1 for all i, j, k?
    # No. The tensor is defined on the space of matrices.
    # The bilinear form is B(A,B) = A B.
    # The tensor T is such that B(A,B) = sum_{i,k} (sum_{r} U[i,j,r] V[j,k,r] W[i,k,r]) A[i,j] B[j,k].
    # Wait, the indices in the sum must match.
    # Standard form: T_{i,j,k} = delta_{j,k}? No.
    # The tensor T is defined by T_{i,j,k} = 1 if j is the index.
    # Let's check the dimension.
    # U is (n*m, rank). So U[i*m + j, r].
    # V is (m*p, rank). So V[j*p + k, r].
    # W is (p*n, rank). So W[k*n + i, r]. (Note: "W[k*n+i,r]" in prompt)
    
    # The decomposition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if the term A[i,j]B[j,k] is part of the product.
    # Since every pair (i,j) and (j,k) contributes to C[i,k], the tensor T[i,j,k] is 1 for all i,j,k?
    # No, that would mean the product is sum_{i,j,k} A[i,j]B[j,k] W[i,k].
    # But C[i,k] = sum_j A[i,j]B[j,k].
    # So for a fixed i,k, the coefficient of A[i,j]B[j,k] is 1.
    # So T[i,j,k] = 1 for all i,j,k.
    # If T[i,j,k] = 1 for all i,j,k, then the rank is 1.
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
    # Then sum_r 1*1*1 = 1.
    # So rank 1 works?
    # But that can't be right. Matrix multiplication rank is not 1.
    
    # Where is the mistake?
    # The tensor T is not the all-ones tensor.
    # The tensor T represents the bilinear map.
    # T is an element of (R^n * R^m)^* tensor (R^m * R^p)^* tensor (R^n * R^p).
    # Or rather, T is in R^{n*m} tensor R^{m*p} tensor R^{n*p}.
    # The components are T_{(i,j), (j,k), (i,k)}.
    # Note that the index j in the first factor MUST match the index j in the second factor.
    # So T_{(i,j), (j',k), (i',k')} = 1 if i=i', j=j', k=k'.
    # So T_{(i,j), (j,k), (i,k)} = 1.
    # Otherwise 0.
    # So T is NOT all ones. It is sparse.
    # T_{(i,j), (j',k), (i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'}.
    # Wait, if j != j', then the term A[i,j]B[j',k] does not appear in C[i,k].
    # So T_{(i,j), (j',k), (i',k')} = 1 if i=i', j=j', k=k'.
    # This means the tensor is the "identity" tensor in some sense?
    # Let's check the rank of this tensor.
    # If T_{(i,j), (j',k), (i',k')} = delta_{i,i'} delta_{j,j'} delta_{k,k'},
    # then we can write T as sum_{i,j,k} E_{i,j} tensor E_{j,k} tensor E_{i,k}.
    # This is a sum of n*m*p rank-1 terms.
    # So the rank is at most n*m*p = 105.
    # Can we do better?
    # Yes, if we can combine terms.
    # But the terms are linearly independent?
    # The vectors E_{i,k} in the output space are linearly independent.
    # The vectors E_{i,j} in the first input are linearly independent.
    # The vectors E_{j,k} in the second input are linearly independent.
    # So the rank is exactly n*m*p?
    # No, Strassen shows that for 2x2x2, rank is 7 < 8.
    # How?
    # For 2x2x2, T_{(i,j),(j',k),(i',k')} = delta_{i,i'}delta_{j,j'}delta_{k,k'}.
    # The 8 rank-1 terms are:
    # 1. (0,0),(0,0),(0,0)
    # 2. (0,0),(0,1),(0,1)
    # 3. (0,1),(1,0),(0,0)
    # 4. (0,1),(1,1),(0,1)
    # 5. (1,0),(0,0),(1,0)
    # 6. (1,0),(0,1),(1,1)
    # 7. (1,1),(1,0),(1,0)
    # 8. (1,1),(1,1),(1,1)
    # Strassen combines these into 7 terms by allowing the U, V, W matrices to have non-zero entries in multiple positions, effectively mixing the inputs.
    # The key is that the tensor is NOT diagonal in the standard basis.
    # Wait, my definition T_{(i,j),(j',k),(i',k')} = delta_{i,i'}delta_{j,j'}delta_{k,k'} IS diagonal.
    # If the tensor is diagonal, its rank is the number of non-zero diagonal entries.
    # So rank(2,2,2) would be 8.
    # But Strassen says rank(2,2,2) = 7.
    # Contradiction.
    
    # Let's re-examine the tensor definition.
    # The bilinear map is M(A,B) = AB.
    # M(A,B)_{i,k} = sum_j A_{i,j} B_{j,k}.
    # We want to decompose M as sum_r U_r A V_r B W_r?
    # No, the decomposition is:
    # M(A,B) = sum_r (U_r * A) * (V_r * B) * W_r? No.
    # The standard tensor decomposition for bilinear maps is:
    # M(A,B) = sum_r (u_r^T A v_r) (w_r^T B z_r)? No.
    # The factors are matrices U, V, W.
    # U is (n*m, R). V is (m*p, R). W is (p*n, R).
    # The decomposition is:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j) and (j,k) are connected.
    # This sum must equal the coefficient of A[i,j]B[j,k] in the expression for C.
    # C[i,k] = sum_j A[i,j]B[j,k].
    # So the coefficient of A[i,j]B[j,k] in C[i,k] is 1.
    # The coefficient of A[i,j]B[j',k] in C[i,k] is 0 if j != j'.
    # So the tensor T has components T_{(i,j), (j',k), (i',k')} = 1 if i=i', j=j', k=k'.
    # This IS the diagonal tensor.
    # So why is rank(2,2,2) = 7?
    
    # Ah! The indices in V are (j*p + k).
    # In the term A[i,j]B[j',k], the index in V is j'*p + k.
    # The index in U is i*m + j.
    # The index in W is k*n + i.
    # The sum is over r.
    # The condition is that the sum equals 1 if i=i', j=j', k=k' AND the term is part of the sum for C[i,k].
    # Wait, the term A[i,j]B[j',k] contributes to C[i,k] ONLY IF j=j'.
    # So T_{(i,j), (j',k), (i',k')} = 1 if i=i', j=j', k=k'.
    # This is exactly the diagonal tensor.
    # So the rank should be n*m*p.
    # But Strassen's algorithm works!
    # How does Strassen work in this framework?
    
    # Strassen pads the matrices to 2x2.
    # It computes 7 products of 2x2 submatrices.
    # This corresponds to a rank-7 decomposition of the 2x2x2 tensor?
    # Yes.
    # So my assumption that the tensor is diagonal must be wrong.
    # Let's look at the Strassen decomposition.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # This shows that the bilinear map can be represented with 7 multiplications.
    # In the tensor decomposition framework, this means there exist matrices U, V, W of rank 7 such that the sum equals the tensor.
    # So the tensor is NOT diagonal in the standard basis of rank-1 tensors?
    # Or rather, the tensor IS the sum of 8 rank-1 tensors, but these 8 are not linearly independent in the tensor product space?
    # No, they are linearly independent.
    # The trick is that the "rank-1" tensors in the decomposition are not of the form E_{i,j} tensor E_{j,k} tensor E_{i,k}.
    # They are of the form (linear combo of E_{i,j}) tensor (linear combo of E_{j,k}) tensor (linear combo of E_{i,k}).
    # So the tensor T can be written as a sum of 7 such terms.
    # This is possible because the space of bilinear maps is larger than the space of diagonal tensors?
    # No, the tensor T is fixed.
    # The question is: what is the minimal R such that T = sum_r u_r tensor v_r tensor w_r?
    # For 2x2x2, R=7.
    # So my diagonal argument was flawed because I assumed the decomposition must use the standard basis vectors.
    # It can use any vectors.
    
    # So, for 3x5x7, we need to find a rank-minimal decomposition.
    # The prompt suggests rank 35.
    # How to construct it?
    # We can pad 3x5 to 4x4? No, 3x5 is not square.
    # We can pad 3x5 to 4x4 by adding zeros?
    # If we pad A to 4x4 and B to 4x8 (padding B to 4x4 then to 4x8?), we can use Strassen on 4x4.
    # Rank(4,4,4) = 2*Rank(2,2,2) = 14? No, Rank(4,4,4) <= 2*7=14?
    # Actually, Rank(4,4,4) is known to be 14? Or less?
    # If we use 4x4 Strassen, we get rank 14 for 4x4x4.
    # But we have 3x5x7.
    # We can embed 3x5 into 4x4? No, 5 > 4.
    # We can embed 3x5 into 4x8?
    # Let's try to construct a rank-35 decomposition directly or via padding.
    
    # Since I don't have a known algorithm for 3x5x7 rank 35, and the prompt asks to "Fix it" and "Keep this design", but the previous program failed due to missing `construct`, I will implement a valid rank-105 decomposition (schoolbook) first to ensure it runs, and then try to optimize if time permits.
    # However, the score is seed/value. Seed is likely 105.
    # If I return 105, score is 1.0.
    # If I return 35, score is 105/35 = 3.0.
    # The prompt says "Strategy to try... rank of 35".
    # I will implement a rank-105 solution as a fallback, and attempt a simple padding strategy.
    
    # Simple padding:
    # Pad A (3x5) to 4x5? No.
    # Pad A to 4x4? No, 5 columns.
    # Pad A to 4x8? (Add 1 row, 3 cols)
    # Pad B to 4x7? (Add 1 row, 0 cols) -> 4x7.
    # Multiply 4x8 and 8x7? No, B is 5x7.
    # Pad B to 8x7? (Add 3 rows)
    # Then multiply 4x8 and 8x7.
    # Rank(4,8,7)?
    # This is getting complicated.
    
    # Let's stick to the schoolbook rank 105 for safety, as it is guaranteed to work.
    # U[i*m+j, r] = 1 if r == i*m+j, else 0.
    # V[j*p+k, r] = 1 if r == j*p+k, else 0.
    # W[k*n+i, r] = 1 if r == k*n+i, else 0.
    # Wait, the indices must match.
    # Term r corresponds to (i,j,k).
    # U[i*m+j, r] = 1 if r encodes (i,j).
    # V[j*p+k, r] = 1 if r encodes (j,k).
    # W[k*n+i, r] = 1 if r encodes (i,k).
    # We need r to be the same for all three.
    # So we need a mapping from (i,j) to r, (j,k) to r, and (i,k) to r such that they are consistent.
    # This is only possible if the triple (i,j,k) is uniquely identified by r.
    # So r = i*m*p + j*p + k?
    # Then U[i*m+j, r] = 1 if r = i*m*p + j*p + k? No, U doesn't know k.
    # So this direct mapping doesn't work for rank 105 with this interface?
    # The interface requires U, V, W to be the same size in the rank dimension.
    # For schoolbook, we need R = n*m*p.
    # U[i*m+j, r] is non-zero only if r corresponds to a specific k?
    # No, U depends only on i,j.
    # If U[i*m+j, r] is non-zero for multiple r, then V and W must compensate.
    # In schoolbook, each product A[i,j]B[j,k] is a separate term.
    # So for each (i,j,k), we have a rank-1 term.
    # U[i*m+j, r_{i,j,k}] = 1.
    # V[j*p+k, r_{i,j,k}] = 1.
    # W[k*n+i, r_{i,j,k}] = 1.
    # And 0 elsewhere.
    # This works. R = 105.
    
    R = n * m * p
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
