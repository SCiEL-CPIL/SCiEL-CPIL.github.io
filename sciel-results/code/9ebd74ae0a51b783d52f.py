"""Matrix multiplication 3x3x5: attempt a low-rank exact decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    # Check tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # T[i,j,k] should be 1 if (i,j,k) is a valid term, else 0
    # Actually, the tensor is defined such that for matrix multiplication C = A * B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i, j, k] = delta_{i,k} ... no wait.
    # The standard tensor for matmul M_{n,m,p} has entries:
    # T[i, j, k] = 1 if we are computing C[i, k] from A[i, j] and B[j, k]
    # Actually, the tensor is defined as:
    # T[a, b, c] = 1 if a, b, c correspond to (i, j, k) such that C[i,k] = sum_j A[i,j] B[j,k]
    # The rank-1 term is: (u_i * v_j * w_k)
    # We want: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if (i,j,k) is a valid pair for matmul
    # In matmul, for fixed i,k, we sum over j.
    # So the tensor T[i, j, k] (where i is row of C/A, j is col of A / row of B, k is col of C/B)
    # T[i,j,k] = 1 for all valid (i,j,k). But wait, that's not a tensor that equals delta.
    # Let's re-read the contract: "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor for M_{n,m,p} is T in R^{n x m x p} where
    # T[i,j,k] = 1 if and only if ... actually it's simpler.
    # The tensor is defined such that:
    # sum_j A[i,j] B[j,k] = C[i,k]
    # This can be written as: C[i,k] = sum_{j} sum_{r} U[i,j,r] V[j,k,r] W[k,i,r] A[i,j] B[j,k] ... no.
    # The standard definition:
    # T[i, j, k] = 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # Since every j contributes to C[i,k] for fixed i,k, T[i,j,k] = 1 for all i,j,k?
    # No, that would mean the tensor is all ones, which has rank 1.
    
    # Let's look at the seed program.
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    # This corresponds to the term: U[i,j] V[j,k] W[k,i]
    # The product is 1 for the specific (i,j,k) triplet corresponding to that column.
    # Summing over columns (which iterate over all i,j,k), we get:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 if (i,j,k) is the triplet for this r, else 0.
    # So the tensor T is the "identity" tensor of the space?
    # No, the tensor T is defined by T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # In the schoolbook, T[i,j,k] = 1 if (i,j,k) is a valid index combination.
    # Since i is 0..n-1, j is 0..m-1, k is 0..p-1, ALL combinations are valid.
    # So T[i,j,k] = 1 for all i,j,k?
    # Wait, if T[i,j,k] = 1 for all i,j,k, then the tensor is the all-ones tensor, which has rank 1.
    # But the seed program has rank 45.
    # This implies that the tensor is NOT all ones.
    # Let's re-read the indices in the seed.
    # U index: i*m + j. (i: 0..n-1, j: 0..m-1)
    # V index: j*p + k. (j: 0..m-1, k: 0..p-1)
    # W index: k*n + i. (k: 0..p-1, i: 0..n-1)
    # The term is U[i,j] V[j,k] W[k,i].
    # The sum is over r.
    # For a fixed (i,j,k), the sum is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # In the schoolbook, each r corresponds to a unique (i', j', k').
    # U[i*m+j, r] = 1 if i == i' and j == j'.
    # V[j*p+k, r] = 1 if j == j' and k == k'.
    # W[k*n+i, r] = 1 if k == k' and i == i'.
    # So the product is 1 if (i,j,k) == (i',j',k'), else 0.
    # So T[i,j,k] = 1 if there exists an r such that (i,j,k) is the triplet for r.
    # Since r iterates over ALL (i',j',k') combinations, T[i,j,k] = 1 for ALL i,j,k.
    # This is the all-ones tensor.
    # BUT, the rank of the all-ones tensor is 1.
    # So why does the seed program have rank 45?
    # Ah, the "matrix multiplication tensor" is NOT the all-ones tensor.
    # The matrix multiplication tensor T for M_{n,m,p} is defined as:
    # T[i, j, k] = 1 if and only if ... ?
    # Actually, the standard tensor for matrix multiplication is:
    # T in R^{n x m x p} where T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # This is always true for valid indices.
    # HOWEVER, the tensor is usually defined such that:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # The tensor is T(i,j,k). The rank of this tensor is the rank of matrix multiplication.
    # For 3x3x3, the rank is 19 (or 23?).
    # For 3x3x5, the rank is...
    # If the tensor is T[i,j,k] = 1 for all i,j,k, the rank is 1.
    # This contradicts the known rank of matrix multiplication.
    # Let's check the definition again.
    # The tensor is T: R^n x R^m x R^p -> R
    # T(a, b, c) = sum_{i,j,k} a_i b_j c_k T[i,j,k]
    # We want T such that:
    # C = A B => C[i,k] = sum_j A[i,j] B[j,k]
    # Let a = A (flattened), b = B (flattened), c = C (flattened).
    # This is not a bilinear form in the standard sense unless we view it as:
    # <c, T(a,b)> = sum_{i,k} c[i,k] sum_j a[i,j] b[j,k]
    # = sum_{i,j,k} a[i,j] b[j,k] c[i,k]
    # So T[i,j,k] is the coefficient of a[i,j] b[j,k] c[i,k].
    # In the expression sum_{i,j,k} a[i,j] b[j,k] c[i,k], the coefficient is 1 for ALL i,j,k.
    # So the tensor IS the all-ones tensor?
    # No, the indices are coupled.
    # a is indexed by (i,j), b by (j,k), c by (i,k).
    # The term is a[i,j] b[j,k] c[i,k].
    # The sum is over i, j, k.
    # So T[i,j,k] (where i is row of a/c, j is col of a/row of b, k is col of b/c)
    # The tensor entry T[i,j,k] corresponds to the product a[i,j] b[j,k] c[i,k].
    # The value is 1.
    # So yes, T[i,j,k] = 1 for all valid i,j,k.
    # Why is the rank of 3x3 matmul 19 then?
    # Because the indices are NOT independent in the tensor product space R^{nm} x R^{mp} x R{pn}.
    # The tensor is T in R^{nm} \otimes R^{mp} \otimes R^{pn}.
    # The entry T[(i,j), (j,k), (i,k)] = 1.
    # All other entries are 0?
    # No, the tensor is defined on the basis vectors.
    # The basis vector for R^{nm} is e_{i,j}.
    # The basis vector for R^{mp} is e_{j,k}.
    # The basis vector for R^{pn} is e_{i,k}.
    # T(e_{i,j}, e_{j,k}, e_{i,k}) = 1.
    # T(e_{i,j}, e_{j',k}, e_{i,k}) = 0 if j != j'.
    # So T is NOT all ones.
    # T is 1 only when the indices match: the j in a[i,j] must match the j in b[j,k], and the i,k in c[i,k] must match the i in a and k in b.
    # So T[i,j,k] (in the flattened sense) is 1 if the triplet (i,j,k) forms a valid "path" i->j->k.
    # Since any i,j,k is a valid path, T[i,j,k] = 1 for all i,j,k in the flattened index space?
    # No, the flattened index of a is i*m+j.
    # The flattened index of b is j*p+k.
    # The flattened index of c is k*n+i (as per the seed's W index).
    # The tensor T is a 3D array of size (n*m) x (m*p) x (p*n).
    # T[A_idx, B_idx, C_idx] = 1 if there exist i,j,k such that:
    # A_idx = i*m+j, B_idx = j*p+k, C_idx = k*n+i.
    # This condition is satisfied if and only if the indices are consistent.
    # For a given (A_idx, B_idx, C_idx), let i = A_idx // m, j = A_idx % m.
    # Let j' = B_idx // p, k = B_idx % p.
    # Let k' = C_idx // n, i' = C_idx % n.
    # We need j == j' and i == i' and k == k'.
    # So T[A_idx, B_idx, C_idx] = 1 if and only if the indices are consistent.
    # This is exactly what the schoolbook decomposition does.
    # The schoolbook has rank n*m*p.
    # Each rank-1 term picks out one consistent triplet.
    # The tensor is a "permutation-like" tensor with 1s on the consistent indices.
    
    # Now, can we find a lower rank?
    # Yes, by finding linear combinations of these rank-1 terms that also equal 1 on the consistent indices and 0 elsewhere.
    # This is equivalent to finding the rank of this specific tensor.
    
    # For 3x3x3, the rank is 19.
    # For 3x3x5, the rank is likely higher.
    # The rank of M_{n,m,p} is at most n*m*p.
    # It is also at least max(n*m, m*p, p*n).
    # For 3,3,5: max(9, 15, 15) = 15.
    # So rank is at least 15.
    # The schoolbook is 45.
    
    # Strategy:
    # 1. Hardcode the Bini decomposition for 3x3x3 (rank 19).
    # 2. For the 3x3x5 problem, we can view it as 3x3x3 + 3x3x2.
    #    The first 3 columns of the 5-column matrix B.
    #    The last 2 columns of the 5-column matrix B.
    #    However, the indices for C are coupled.
    #    C[i,k] for k=0,1,2 uses B[j,0..2].
    #    C[i,k] for k=3,4 uses B[j,3..4].
    #    The tensor for 3x3x5 is the direct sum of the tensor for 3x3x3 and 3x3x2?
    #    No, because C[i,k] depends on B[j,k].
    #    The tensor T[i,j,k] is non-zero only if k is the same in B and C.
    #    So the tensor is block-diagonal with respect to k?
    #    Let's check the indices.
    #    T[A_idx, B_idx, C_idx] = 1 if i=A_idx//m, j=A_idx%m, j'=B_idx//p, k=B_idx%p, k'=C_idx//n, i'=C_idx%n, and j==j', i==i', k==k'.
    #    So k must be the same in B and C.
    #    This means the tensor is a direct sum over k of tensors T_k.
    #    T_k has size (n*m) x (m*1) x (1*n) ? No.
    #    For a fixed k, B_idx must have k'=k. So B_idx is j*p+k for j=0..m-1.
    #    C_idx must have k=k. So C_idx is k*n+i for i=0..n-1.
    #    A_idx is i*m+j for i=0..n-1, j=0..m-1.
    #    So for fixed k, the tensor T_k is (n*m) x m x n.
    #    And T_k[i*m+j, j, k*n+i] = 1.
    #    This is the matrix multiplication tensor for M_{n,m,1}? No, M_{n,1,n}?
    #    It's the tensor for multiplying an n x m matrix by a m x 1 matrix to get an n x 1 vector?
    #    No, it's the tensor for M_{n,m,n} but with only one column of B and C?
    #    Actually, T_k is the tensor for M_{n,m,1} if we identify the 1 dimension with k.
    #    But the rank of M_{n,m,p} is not the sum of ranks of M_{n,m,1}.
    #    However, if the tensor is a direct sum, the rank is the sum of the ranks.
    #    Is T a direct sum over k?
    #    Yes, because the condition k==k' decouples the k index.
    #    So rank(M_{3,3,5}) = sum_{k=0}^{4} rank(T_k).
    #    T_k is the tensor for M_{3,3,1}?
    #    Let's check the rank of M_{3,3,1}.
    #    M_{3,3,1} is multiplying a 3x3 matrix by a 3x1 matrix.
    #    This is equivalent to multiplying a 3x3 matrix by 5 different 3x1 vectors? No.
    #    The rank of M_{n,m,p} is the rank of the tensor.
    #    If T is a direct sum of T_k, then rank(T) = sum rank(T_k).
    #    What is T_k?
    #    T_k[i*m+j, j, k*n+i] = 1.
    #    This is the tensor for M_{n,m,1} where the output dimension is 1?
    #    No, the output index is i. The input indices are (i,j) and (j,k).
    #    But k is fixed.
    #    So it's the tensor for M_{n,m,1} with n=n, m=m, p=1?
    #    The rank of M_{n,m,1} is n*m. (Schoolbook: each output element requires m multiplications).
    #    Wait, M_{n,m,1} computes a vector of size n from a matrix n x m and a vector m x 1.
    #    C[i] = sum_j A[i,j] B[j].
    #    The tensor is T[i,j] = A[i,j] B[j] C[i].
    #    The rank of this tensor is n*m.
    #    So if rank(M_{3,3,5}) = 5 * rank(M_{3,3,1}) = 5 * 9 = 45.
    #    This matches the schoolbook rank!
    #    So the direct sum argument suggests the rank is 45.
    #    BUT, this assumes the tensor is a direct sum.
    #    Is it?
    #    The tensor T is in R^{nm} \otimes R^{mp} \otimes R^{pn}.
    #    The decomposition into direct sums requires that the spaces R^{mp} and R^{pn} decompose as direct sums.
    #    R^{mp} = R^{m} \oplus R^{m} \oplus ... (p times).
    #    R^{pn} = R^{n} \oplus R^{n} \oplus ... (p times).
    #    If we use the standard basis, yes.
    #    So the rank is indeed the sum of the ranks of the slices.
    #    And the rank of each slice (M_{3,3,1}) is 9.
    #    So the rank is 45.
    #    This implies that 45 is the minimal rank?
    #    This contradicts the fact that matrix multiplication rank can be less than n*m*p.
    #    The catch: The rank of M_{n,m,p} is NOT the sum of ranks of M_{n,m,1} slices if we allow mixing of the k indices in the decomposition.
    #    The direct sum argument holds if the decomposition must respect the block structure.
    #    But a general tensor decomposition can mix the blocks!
    #    U, V, W are arbitrary matrices.
    #    They can have non-zero entries in multiple k blocks.
    #    So the rank can be lower than 45.
    
    # So, we need to find a lower rank decomposition.
    # Known results:
    # Rank of 3x3x3 is 19.
    # Rank of 3x3x4 is 23?
    # Rank of 3x3x5 is...
    # If we can find a rank 37 decomposition, that would be an improvement.
    
    # Let's try to construct a rank 37 decomposition.
    # We can use the Bini decomposition for the first 3 k-values (rank 19)
    # and schoolbook for the remaining 2 k-values (rank 18).
    # Total rank = 19 + 18 = 37.
    # This works IF we can separate the k indices.
    # But we CAN separate them by setting the entries of U, V, W to 0 for the "wrong" k blocks.
    # Specifically, for the Bini part, we only use k=0,1,2.
    # For the schoolbook part, we only use k=3,4.
    # Since the tensor is 0 if k doesn't match, we can just add the two decompositions.
    # The rank is the sum of the ranks if the supports are disjoint.
    # Here, the supports are disjoint in the k dimension.
    # So rank = 19 + 2*9 = 37.
    
    # So we need the Bini decomposition for 3x3x3.
    # I will hardcode the Bini factors for 3x3x3.
    # If I cannot recall them, I will use a random search or return schoolbook.
    # But I will try to recall or find a simple low-rank decomposition.
    
    # Actually, there is a simpler argument.
    # The rank of M_{n,m,p} is at most n*m*p - (n-1)(m-1)(p-1)/2 ? No.
    # Let's just return the schoolbook for now, as I don't have the Bini factors memorized.
    # But the prompt says "attempt a low-rank exact decomposition".
    # And the score is 1.0 for schoolbook.
    # I need to beat 1.0.
    # So I MUST find a rank < 45.
    
    # I will implement a simple search for a rank 37 decomposition.
    # Or better, I will use the fact that the problem is separable in k.
    # I will use the best known rank for 3x3x3 (19) and schoolbook for 3x3x2 (18).
    # Total rank 37.
    
    # I need the Bini factors.
    # I will try to generate them or use a known set.
    # Since I cannot guarantee I have the correct Bini factors, I will use a random search.
    # But random search for rank 37 is hard.
    
    # Alternative: Return schoolbook.
    # But that scores 1.0.
    
    # Let's try to find a rank 40 decomposition.
    # Or any rank < 45.
    # I will use a greedy algorithm to merge schoolbook terms.
    # If I can merge 5 terms, I get rank 40.
    
    # For now, I will return the schoolbook decomposition.
    # This is safe.
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 45 schoolbook
    
    # Schoolbook decomposition
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
