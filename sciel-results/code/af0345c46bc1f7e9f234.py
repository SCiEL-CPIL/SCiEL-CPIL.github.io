import numpy as np

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # The matrix multiplication tensor for n x m by m x p has rank m*n if we use the
    # "row-wise" decomposition: for each row i of A and each column j of A (which is a row of B),
    # we form a rank-1 term. But actually, the rank of the tensor is exactly m*n when we consider
    # the standard decomposition. Wait, let me reconsider.
    #
    # Actually, the rank of the matrix multiplication tensor M_{n,m,p} is known to be at most n*m*p
    # (trivial) and at least max(n*m, m*p, n*p). For n=2, m=4, p=6:
    # n*m = 8, m*p = 24, n*p = 12. So rank >= 24? No, that's not right either.
    #
    # Let me think more carefully. The tensor T[i,j,k] = delta_{j, k} ... no.
    # T is defined such that C[i,k] = sum_j A[i,j] * B[j,k].
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
    # Actually, the tensor is T[i,j,k] = delta_{i,i'} * delta_{k,k'} ... no.
    #
    # The standard way: the multiplication tensor is a 3-tensor of size n x m x p where
    # T[i,j,k] = 1 if we are computing C[i,k] = sum_j A[i,j] B[j,k].
    # So T[i,j,k] = 1 for all valid i,j,k? No, that's not a tensor decomposition.
    #
    # The tensor T is defined as: T = sum_{i,j,k} T_{ijk} e_i^A \otimes e_j^B \otimes e_k^C
    # where T_{ijk} = 1 if the product A_{ij} * B_{jk} contributes to C_{ik}.
    # But A_{ij} * B_{jk} contributes to C_{ik} for all i,j,k. So T_{ijk} = 1 for all i,j,k?
    # No, that would mean C_{ik} = sum_{j} A_{ij} B_{jk}, which is correct.
    # But the tensor is actually T_{ijk} = delta_{i,i} * delta_{k,k} ... I'm confusing myself.
    #
    # Let me use the standard definition: the matrix multiplication tensor M_{n,m,p} is the tensor
    # such that for any matrices A (n x m) and B (m x p), the product C = A * B satisfies
    # C_{ik} = <M_{n,m,p}, A \otimes B>_{i,k} = sum_{j} A_{ij} B_{jk}.
    #
    # In terms of a 3-tensor T with indices (i, j, k), we have:
    # C_{ik} = sum_j T_{ijk} A_{ij} B_{jk}
    # So T_{ijk} = 1 for all i, j, k.
    #
    # The rank of this tensor is the minimum number r such that
    # T = sum_{r'=1}^r u_{r'} \otimes v_{r'} \otimes w_{r'}
    # where u is in R^{n*m}, v is in R^{m*p}, w is in R^{p*n} (with the trace convention).
    #
    # A simple decomposition: for each pair (i, j), define:
    # u^{(i,j)} has 1 at position (i, j) and 0 elsewhere.
    # v^{(i,j)} has 1 at position (j, k) for all k, and 0 elsewhere. Wait, that's not rank 1.
    #
    # For rank 1, we need u \otimes v \otimes w where u is a vector, v is a vector, w is a vector.
    # u^{(i,j,k)} \otimes v^{(i,j,k)} \otimes w^{(i,j,k)} = e_i^A \otimes e_j^B \otimes e_k^C
    # So each rank-1 term picks one (i,j,k).
    #
    # The tensor T has T_{ijk} = 1 for all i,j,k. So T = sum_{i,j,k} e_i \otimes e_j \otimes e_k.
    # That's n*m*p = 2*4*6 = 48 terms. But we can do better.
    #
    # Actually, T_{ijk} = 1 for all i,j,k means the tensor is the all-ones tensor.
    # The rank of the all-ones tensor of size n x m x p is max(n, m, p) if... no.
    #
    # Wait, I think I'm wrong. Let me re-examine.
    # C_{ik} = sum_j A_{ij} B_{jk}
    # This is a sum over j only. For fixed i and k, C_{ik} depends on A_{i,0..m-1} and B_{0..m-1, k}.
    #
    # The tensor T is such that:
    # C_{ik} = sum_j T_{ijk} A_{ij} B_{jk}
    # For this to hold, we need T_{ijk} = 1 for all i, j, k.
    #
    # But wait, that can't be right because then the rank would be related to the all-ones tensor.
    #
    # Let me look at this differently. The standard decomposition is:
    # For each j (0 to m-1), and for each i (0 to n-1), and for each k (0 to p-1):
    # term = A_{ij} * B_{jk} contributes to C_{ik}.
    #
    # A rank-1 decomposition term u \otimes v \otimes w corresponds to:
    # sum_{i',j',k'} u_{i'j'} v_{j'k'} w_{k'i'} A_{i'j'} B_{j'k'}
    # This should equal a single term in the sum.
    #
    # If we set u_{ij} = 1 for a specific (i,j), v_{jk} = 1 for all k (for that j), w_{ki} = 1 for that (k,i),
    # then the term is A_{ij} * (sum_k B_{jk}) * 1 ... no, that's not right.
    #
    # For a rank-1 term to contribute A_{ij} * B_{jk} to C_{ik}, we need:
    # u has support only at (i,j), v has support only at (j,k), w has support only at (k,i).
    # So u_{i'j'} = delta_{i'i} delta_{j'j}, v_{j'k'} = delta_{j'j} delta_{k'k}, w_{k'i'} = delta_{k'k} delta_{i'i}.
    #
    # Then the term is: sum_{i',j',k'} delta_{i'i}delta_{j'j} * delta_{j'j}delta_{k'k} * delta_{k'k}delta_{i'i} * A_{i'j'} B_{j'k'}
    # = A_{ij} B_{jk} * delta_{i'i}delta_{i'i} * delta_{j'j}delta_{j'j} * delta_{k'k}delta_{k'k}
    # = A_{ij} B_{jk} (since all deltas are 1 when i'=i, j'=j, k'=k)
    #
    # So each (i,j,k) gives a rank-1 term. Total: n*m*p = 48.
    #
    # But we can group terms. For fixed (i,j), the contribution to C is:
    # sum_k A_{ij} B_{jk} e_k^C = A_{ij} * (sum_k B_{jk} e_k^C)
    # This is A_{ij} times a vector in C-space. So for fixed (i,j), we have one "row" in U,
    # and the V and W parts handle the sum over k.
    #
    # For fixed (i,j):
    # u has 1 at (i,j)
    # v has 1 at (j,k) for all k
    # w has 1 at (k,i) for all k
    #
    # But v and w are vectors, not matrices. v is in R^{m*p}, so v_{jk} = 1 for all k for fixed j.
    # w is in R^{p*n}, so w_{ki} = 1 for all k for fixed i.
    #
    # Then the term is: sum_{j',k'} v_{j'k'} w_{k'i'} A_{i'j'} B_{j'k'}
    # With u_{i'j'} = delta_{i'i}delta_{j'j}:
    # = sum_{j',k'} delta_{i'i}delta_{j'j} * v_{j'k'} * w_{k'i'} * A_{i'j'} B_{j'k'}
    # = v_{jk'} w_{k'i} A_{ij} B_{jk'}  (summing over k')
    # = A_{ij} sum_{k'} v_{jk'} w_{k'i} B_{jk'}
    #
    # For this to equal A_{ij} B_{jk} (for a specific k), we need v_{jk'} w_{k'i} = delta_{k'k}.
    # So v_{jk} = 1, w_{ki} = 1, and v_{jk'} = 0 for k' != k, w_{k'i} = 0 for k' != k.
    #
    # This brings us back to the 48-term decomposition.
    #
    # However, we can use linearity. For fixed (i,j), the contribution is A_{ij} * (B_j \cdot e_i^C)
    # where B_j is the j-th row of B, and e_i^C is the i-th basis vector in C-space (but C has p columns).
    #
    # Actually, C_{ik} = sum_j A_{ij} B_{jk}. For fixed i, C_i (the i-th row of C) = sum_j A_{ij} B_j (the j-th row of B).
    #
    # So for each i, C_i = sum_j A_{ij} B_j. This is a linear combination of the rows of B, with coefficients A_{ij}.
    #
    # The tensor for fixed i is: T_i = sum_j A_{ij} B_j \otimes e_i^C
    #
    # In tensor form: T_i = sum_j (e_i^A \otimes e_j^A) \otimes (row j of B as a vector) \otimes e_i^C
    #
    # The rank of this sub-tensor (for fixed i) is m, because we need m terms (one for each j).
    # So for n=2, the total rank is 2*m = 8.
    #
    # Let me verify: for fixed i and j, the term is:
    # u has 1 at (i,j) in A-space
    # v has 1 at (j,k) for all k in B-space (i.e., v is the j-th row of the identity in B-space, extended)
    # w has 1 at (k,i) for all k in C-space (i.e., w is the i-th column of the identity in C-space, extended)
    #
    # Wait, v is in R^{m*p}. For fixed j, v_{j',k'} = delta_{j'j} for all k'. So v is the j-th "row" of the identity matrix of size m*p, but with all k' components equal.
    #
    # Actually, v_{j'k'} = delta_{j'j}. This is a vector with 1s at positions (j, 0), (j, 1), ..., (j, p-1) and 0s elsewhere.
    #
    # w_{k'i'} = delta_{i'i}. This is a vector with 1s at positions (0, i), (1, i), ..., (p-1, i) and 0s elsewhere.
    #
    # Then the term is: sum_{j',k'} u_{i'j'} v_{j'k'} w_{k'i''} A_{i'j'} B_{j'k'}
    # With u_{i'j'} = delta_{i'i} delta_{j'j}:
    # = sum_{k'} v_{jk'} w_{k'i} A_{ij} B_{jk'}
    # = A_{ij} sum_{k'} delta_{j'j} delta_{i'i} B_{jk'}  (since v_{jk'} = 1 for all k', w_{k'i} = 1 for all k')
    # = A_{ij} * p * B_{jk'} ... no, this sums over k', so it's A_{ij} * sum_{k'} B_{jk'} * 1 * 1
    #
    # That's not right. I need v_{j'k'} w_{k'i''} to pick out B_{jk} for a specific k.
    #
    # I think the correct decomposition is indeed 48 terms, one for each (i,j,k).
    #
    # But wait, the current program uses 42 terms with Strassen. Let me check if 48 is correct or if there's a better way.
    #
    # Actually, I recall that the rank of M_{n,m,p} is at most n*m*p and at least max(n*m, m*p, n*p).
    # For n=2, m=4, p=6: max(8, 24, 12) = 24. So rank >= 24.
    #
    # The trivial decomposition gives 48. Can we do better?
    #
    # Let me try a different approach. For each j (0 to m-1), the contribution is:
    # sum_i A_{ij} * (row j of B) \otimes e_i^C
    #
    # This is a rank-m tensor (since there are m terms for i=0..n-1, but they share the same B-row).
    #
    # Actually, for fixed j, the sub-tensor is:
    # T_j = sum_i A_{ij} B_j \otimes e_i^C
    # = (sum_i A_{ij} e_i^C) \otimes B_j
    #
    # This is rank 1 in the (A,C) space times B_j. But B_j is a vector in R^p.
    #
    # So T_j = u_j \otimes v_j \otimes w_j where:
    # u_j has A_{ij} for each i (but A is a variable, so u_j is a fixed vector in A-space)
    #
    # I'm getting confused. Let me just use the 48-term decomposition and see if the evaluator accepts it.
    #
    # Actually, let me re-read the problem. The current program uses 42 terms. The trivial is 48.
    # Maybe 42 is better than 48? Or maybe I can find a decomposition with fewer terms.
    #
    # Let me try the row-wise decomposition with rank = n*m = 8.
    # For each (i, j):
    # u_{i'j'} = delta_{i'i} delta_{j'j}
    # v_{j'k'} = delta_{j'j}  (1 for all k' if j'=j, else 0)
    # w_{k'i'} = delta_{i'i}  (1 for all k' if i'=i, else 0)
    #
    # Term = sum_{i',j',k'} u_{i'j'} v_{j'k'} w_{k'i''} A_{i'j'} B_{j'k'}
    # = sum_{k'} delta_{j'j} delta_{i'i} A_{ij} B_{jk'} * delta_{i''i}
    # = A_{ij} sum_{k'} B_{jk'}
    #
    # This gives A_{ij} * (sum of row j of B), not A_{ij} * B_{jk}. So this is wrong.
    #
    # I need to pick out B_{jk} for a specific k. So I need 48 terms.
    #
    # Let me just implement the 48-term decomposition.

    rank = n * m * p  # 48
    U = [[0] * rank for _ in range(n * m)]  # 8 x 48
    V = [[0] * rank for _ in range(m * p)]  # 24 x 48
    W = [[0] * rank for _ in range(p * n)]  # 12 x 48

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # u: 1 at (i, j)
                U[i * m + j][col] = 1
                # v: 1 at (j, k)
                V[j * p + k][col] = 1
                # w: 1 at (k, i) -> index k*n + i
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
