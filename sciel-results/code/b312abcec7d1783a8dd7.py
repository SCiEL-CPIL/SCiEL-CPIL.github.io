import numpy as np
import time
from scipy.linalg import svd

DIMENSIONS = [2, 3, 6]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n, m, p = DIMENSIONS  # 2, 3, 6

    # The multiplication tensor T has entries T[i, b, k] where:
    # i in {0,...,n-1}, b in {0,...,m-1}, k in {0,...,p-1}
    # T[i,b,k] = 1 if this is a valid (i,b,k) triple for the identity
    # Actually, the tensor for matrix multiplication C = A * B where
    # A is n x m, B is m x p, C is n x p:
    # C[i,k] = sum_b A[i,b] * B[b,k]
    # The tensor T has T[i, b, k] = 1 (it's the identity tensor)
    # The decomposition is: sum_l U[i*m+b, l] * V[b*p+k, l] * W[k*n+i, l]
    # = delta(i,i') * delta(b,b') * delta(k,k')
    # Wait, that's not right. Let me re-read the contract.
    #
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor equation is:
    # sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l] = 1 iff a==i and d==k and b==c
    #
    # So the tensor T has T[a, b, c, d, i, k] = 1 iff a==i, b==c, d==k
    # This is just the identity tensor! T[i, b, k] = delta(i,i)*delta(b,b)*delta(k,k)
    # No wait, the tensor is: T[i, b, k] where the multiplication gives
    # C[i,k] = sum_b A[i,b] B[b,k]
    # So the "tensor" is really: for each (i,b,k) triple, the contribution is A[i,b]*B[b,k]
    # The rank-1 decomposition: sum_l U[i,b,l] * V[b,k,l] * W[k,i,l]
    # should equal delta(i,i')*delta(b,b')*delta(k,k')... no.
    #
    # Actually the tensor is: T[i, b, k] = 1 for all valid (i,b,k)
    # because C[i,k] = sum_b A[i,b] * B[b,k]
    # The rank-1 terms are: U[i,b,l] * V[b,k,l] * W[k,i,l]
    # and we need sum_l U[i,b,l]*V[b,k,l]*W[k,i,l] = 1 for all (i,b,k)
    # Wait no, that's not right either. Let me re-read.
    #
    # From the verify function:
    # for a in range(n), b in range(m), c in range(m), d in range(p), i in range(n), k in range(p):
    #   target = 1.0 if (a == i and d == k and b == c) else 0.0
    #   s = sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l]
    #
    # So: sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l] = delta(a,i) * delta(b,c) * delta(d,k)
    #
    # This means the tensor T is: T[a, b, c, d, i, k] = delta(a,i)*delta(b,c)*delta(d,k)
    # But this is a 6-index tensor. However, it factors as:
    # T = sum over (a,b,c,d,i,k) of delta(a,i)*delta(b,c)*delta(d,k) * e_a e_b e_c e_d e_i e_k
    #
    # The rank-1 decomposition in the tensor sense:
    # T[a,b,c,d,i,k] = sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l]
    #
    # This is equivalent to saying the 3-tensor (with indices (ab), (cd), (ki)) has rank R.
    # Let me define:
    # X = flatten: index (a,b) -> a*m+b, range [0, n*m)
    # Y = flatten: index (c,d) -> c*p+d, range [0, m*p)  
    # Z = flatten: index (k,i) -> k*n+i, range [0, p*n)
    #
    # T[X, Y, Z] = 1 if (a==i and b==c and d==k) where X=am+b, Y=cp+d, Z=kn+i
    #
    # So T[X, Y, Z] = 1 iff the indices are consistent.
    #
    # The rank of this tensor is what we want to minimize.
    #
    # Key insight: This tensor is actually the "matrix multiplication tensor" M_{n,m,p}
    # and its rank is mu(n,m,p).
    #
    # For n=2, m=3, p=6:
    # mu(2,3,6) <= mu(2,3,3) * 2 = 11 * 2 = 22 (by block structure)
    # mu(2,3,6) <= mu(2,3,2) * 3 = ? * 3
    # mu(2,3,6) <= mu(2,2,6) * ? 
    #
    # Actually, mu(2,3,6) <= mu(2,3,3) * 2 = 22
    # Also mu(2,3,6) <= mu(2,2,2) * 3 * 3 = 7*9 = 63 (crude)
    # mu(2,3,6) <= mu(3,3,3) * 2 = 7*2 = 14? No, that's not a valid reduction.
    #
    # Let me think about this differently.
    # mu(n,m,p) <= mu(n,m,p1) * ceil(p/p1)
    # mu(2,3,6) <= mu(2,3,3) * 2 = 22
    # mu(2,3,6) <= mu(2,3,2) * 3
    # What is mu(2,3,2)?
    # mu(2,3,2) = min rank for 2x3 * 3x2 = 2x2
    # This is at most 2*3*2 = 12 (schoolbook)
    # Can we do better? mu(2,2,2) = 7, so mu(2,3,2) <= 7 + 2*2 = 11? 
    # Actually mu(2,3,2) <= mu(2,2,2) + 2*2 = 7+4 = 11
    # Or mu(2,3,2) <= 2*mu(2,2,2) - ? 
    #
    # Let me just compute the tensor numerically and find its rank via SVD.

    # Build the tensor T of shape (n*m, m*p, p*n)
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12

    T = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for k in range(p):
                # T[a*m+b, b*p+k, k*n+a] = 1
                T[a * m + b, b * p + k, k * n + a] = 1.0

    # Now find the rank of this 3-tensor.
    # The rank of a 3-tensor is not simply the max of its matricization ranks.
    # But we can use the following approach:
    # 1. Compute SVD of each matricization to get a lower bound on rank
    # 2. Try to find a rank-R decomposition by solving a system of equations

    # Matricizations:
    T12_3 = T.reshape(nm * mp, pn)  # (108, 12)
    T1_23 = T.reshape(nm, mp * pn)  # (6, 216)
    T2_13 = T.reshape(mp, nm * pn)  # (18, 72)

    # SVD of matricizations gives lower bounds
    _, s1, _ = svd(T12_3, full_matrices=False)
    _, s2, _ = svd(T1_23, full_matrices=False)
    _, s3, _ = svd(T2_13, full_matrices=False)

    rank_lb1 = np.sum(s1 > 1e-10)
    rank_lb2 = np.sum(s2 > 1e-10)
    rank_lb3 = np.sum(s3 > 1e-10)

    tensor_rank_lb = max(rank_lb1, rank_lb2, rank_lb3)

    # The actual tensor rank is >= max of matricization ranks
    # For this specific tensor, let's figure out what it is.
    # 
    # T[a*m+b, b*p+k, k*n+a] = 1
    # 
    # Fix (a,b): T[a*m+b, b*p+k, k*n+a] = 1 for all k in [0,p)
    # So for fixed (a,b), the slice has p non-zero entries.
    # 
    # Fix (b,k): T[a*m+b, b*p+k, k*n+a] = 1 for all a in [0,n)
    # So for fixed (b,k), the slice has n non-zero entries.
    #
    # Fix (a,k): T[a*m+b, b*p+k, k*n+a] = 1 for all b in [0,m)
    # So for fixed (a,k), the slice has m non-zero entries.
    #
    # Total non-zero entries: n * m * p = 2 * 3 * 6 = 36
    #
    # The rank of this tensor... let me think.
    # Each rank-1 term U[:,l] * V[:,l] * W[:,l] can cover at most
    # (non-zero entries in U[:,l]) * (non-zero entries in V[:,l]) * (non-zero entries in W[:,l])
    # entries, but they must align.
    #
    # Actually, a rank-1 term U[:,l] * V[:,l] * W[:,l] produces a tensor where
    # T[x,y,z] = U[x,l]*V[y,l]*W[z,l]
    # For this to match our T (which has 0/1 entries), we need:
    # U[x,l]*V[y,l]*W[z,l] = 1 iff (x,y,z) is a valid triple
    # 
    # A valid triple is (a*m+b, b*p+k, k*n+a) for some a,b,k.
    #
    # So a rank-1 term can cover a set of triples where:
    # a*m+b is in support of U[:,l]
    # b*p+k is in support of V[:,l]  
    # k*n+a is in support of W[:,l]
    # AND the indices are consistent (same a,b,k)
    #
    # The maximum number of triples a single rank-1 term can cover:
    # If U[:,l] has support {a1*m+b1, a2*m+b2, ...}, V[:,l] has support {c1*p+d1, ...}, W[:,l] has support {e1*n+f1, ...}
    # Then the term covers triples where (a,b) is in U-support, (c,d) is in V-support, (e,f) is in W-support
    # AND a=f, b=c, d=e.
    #
    # So the number of covered triples = number of (a,b,k) such that:
    # a*m+b in U-support, b*p+k in V-support, k*n+a in W-support
    #
    # To maximize coverage, we want large supports that are "aligned".
    #
    # For example, if U[:,l] has support {0*m+0, 0*m+1, 0*m+2, 1*m+0, 1*m+1, 1*m+2} (all of A row 0 and 1, i.e., all 6 entries)
    # V[:,l] has support {0*p+0, 0*p+1, ..., 2*p+5} (all 18 entries)
    # W[:,l] has support {0*n+0, 0*n+1, 1*n+0, 1*n+1, ..., 5*n+1} (all 12 entries)
    # Then the term covers all 36 triples. But U[:,l]*V[:,l]*W[:,l] = 1 for all (x,y,z) in the supports,
    # which is 6*18*12 = 1296 entries, but we only want 36 of them to be 1 and the rest 0.
    # This doesn't work because a rank-1 term can't selectively zero out entries.
    #
    # So each rank-1 term must have a "diagonal" structure.
    #
    # Let me think about what the minimum rank could be.
    #
    # Lower bound: The tensor has 36 non-zero entries. A rank-1 term with 
    # |supp(U)|, |supp(V)|, |supp(W)| non-zero entries can cover at most
    # min(|supp(U)|, |supp(V)|, |supp(W)|) triples in the best case (if perfectly aligned).
    # Actually no, it can cover up to the number of consistent (a,b,k) triples.
    #
    # Let me just try to find a decomposition of various ranks by solving linear systems.

    # Approach: For a given rank R, try to find U, V, W with integer/half-integer entries.
    # We can use the following:
    # 1. Start with a known upper bound (e.g., 22 from mu(2,3,3)*2)
    # 2. Try to reduce the rank by solving linear systems with sparsity constraints

    # First, let's establish that rank 22 works (block construction from rank-11 for 2x3x3)
    # Actually, let me first try to find the exact rank by trying ranks from the lower bound up.

    # Let me use a different approach: 
    # The tensor T can be written as a sum of rank-1 tensors.
    # We can set up the problem as: find the minimum R such that there exist
    # U (6 x R), V (18 x R), W (12 x R) with the tensor equation satisfied.
    #
    # This is equivalent to: the tensor T has tensor rank <= R.
    #
    # One way to find a decomposition: use the "slice rank" or "quantum rank" bounds,
    # but for small tensors, we can just try to solve it directly.
    #
    # Let me try a constructive approach:
    # mu(2,3,6) <= mu(2,3,3) * 2 = 22
    # mu(2,3,6) <= mu(2,2,6) + mu(2,1,6) ... not helpful
    # mu(2,3,6) <= mu(2,3,2) * 3
    # 
    # What is mu(2,3,2)?
    # 2x3 * 3x2 = 2x2
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    #
    # This is like 2x3 * 3x2. We can use Strassen for the 2x2*2x2 part (7 terms)
    # plus 2 more terms for the third column/row. So mu(2,3,2) <= 9.
    # Actually, let me think... 
    # C = A * B where A is 2x3, B is 3x2.
    # Split A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # Split B = [B1; B2] where B1 is 2x2, B2 is 1x2
    # C = A1*B1 + A2*B2
    # A1*B1 can be done in 7 terms (Strassen)
    # A2*B2 is 2x1 * 1x2 = 2x2, which is an outer product, rank 1? No, 
    # A2*B2 has entries (A2[0]*B2[0], A2[0]*B2[1], A2[1]*B2[0], A2[1]*B2[1])
    # This requires 4 multiplications in general, but as a tensor it's rank 4? 
    # No, A2*B2 = sum_l ... it's a 2x1 * 1x2 multiplication.
    # The tensor for 2x1 * 1x2 has 4 non-zero entries, and rank 4 (each entry needs its own term).
    # Wait, no. A2 is 2x1 (a column vector), B2 is 1x2 (a row vector).
    # C = A2 * B2 is a 2x2 matrix. C[i,j] = A2[i] * B2[j].
    # This is an outer product. As a tensor: T[i, j, k, l] where... 
    # Actually for 2x1 * 1x2: A has 2 entries, B has 2 entries, C has 4 entries.
    # C[i,j] = A[i] * B[j]
    # Rank-1 decomposition: sum_l U[i,l] * V[j,l] * W[i,j,l]
    # We need sum_l U[i,l] * V[j,l] * W[i,j,l] = A[i]*B[j] for all i,j.
    # This is just the outer product, which has rank 1 as a matrix but as a tensor...
    # Actually, C[i,j] = A[i] * B[j] is already rank 1 in the tensor sense!
    # U[i,l] = A[i], V[j,l] = B[j], W[i,j,l] = 1 for l=0.
    # Wait, that gives sum_l U[i,l]*V[j,l]*W[i,j,l] = A[i]*B[j]*1 = A[i]*B[j]. Yes!
    # So 2x1 * 1x2 has tensor rank 1.
    #
    # So mu(2,3,2) <= 7 + 1 = 8.
    # Can we do better? mu(2,3,2) >= ?
    # The tensor has 2*3*2 = 12 non-zero entries.
    # Lower bound: max matricization rank.
    # Let me compute: T shape is (2*3, 3*2, 2*2) = (6, 6, 4)
    # T[a*3+b, b*2+k, k*2+a] = 1
    #
    # Matricization 1: (6*6, 4) = (36, 4). The rank is at most 4.
    # Matricization 2: (6, 6*4) = (6, 24). Rank at most 6.
    # Matricization 3: (6, 6*4) = (6, 24). Rank at most 6.
    #
    # So mu(2,3,2) >= 6. And we have mu(2,3,2) <= 8.
    #
    # Let me try to find mu(2,3,2) = 7 or 8.
    #
    # Actually, for our problem, let me just try to find a good decomposition for 2x3x6
    # by trying ranks from the lower bound up.
    #
    # Lower bound for 2x3x6: max matricization rank.
    # T shape: (6, 18, 12)
    # Matricization 1: (6*18, 12) = (108, 12). Rank <= 12.
    # Matricization 2: (6, 18*12) = (6, 216). Rank <= 6.
    # Matricization 3: (18, 6*12) = (18, 72). Rank <= 18.
    # So tensor rank >= 12.
    #
    # Upper bound: 22 (from mu(2,3,3)*2)
    #
    # Let me try to find a decomposition of rank 12, 13, ..., 22.
    #
    # Actually, let me think about this more carefully.
    # mu(2,3,6) <= mu(2,3,3) * 2 = 22
    # mu(2,3,6) <= mu(2,2,6) * ? 
    # mu(2,2,6): 2x2 * 2x6 = 2x6
    # This is 2x2 * 2x6. We can think of it as 3 copies of 2x2*2x2.
    # mu(2,2,6) <= 3 * 7 = 21.
    # Then mu(2,3,6) <= mu(2,2,6) + mu(2,1,6) = 21 + ?
    # mu(2,1,6): 2x1 * 1x6 = 2x6, which is an outer product, rank 1.
    # So mu(2,3,6) <= 21 + 1 = 22. Same as before.
    #
    # Alternatively: mu(2,3,6) <= mu(2,3,2) * 3
    # If mu(2,3,2) = 7, then mu(2,3,6) <= 21.
    # If mu(2,3,2) = 8, then mu(2,3,6) <= 24.
    #
    # Let me try to construct a rank-21 decomposition.
    # 
    # Actually, let me try a completely different approach.
    # I'll use the fact that the tensor is sparse and try to find a decomposition
    # by solving a system of polynomial equations (or using a greedy approach).
    #
    # Greedy approach:
    # 1. Start with the full tensor T.
    # 2. At each step, find the best rank-1 term to subtract.
    # 3. Repeat until T is zero.
    #
    # The "best" rank-1 term is the one that maximizes the number of non-zero
    # entries it can explain (i.e., the number of (x,y,z) where T[x,y,z] != 0
    # and U[x]*V[y]*W[z] != 0 for the chosen U,V,W).
    #
    # But this greedy approach might not be optimal. Let me try it anyway.

    # Actually, let me just try to construct explicit decompositions of various ranks.
    #
    # Key insight: mu(2,3,6) can be bounded using the "submultiplicativity" and
    # "additivity" of tensor rank.
    #
    # mu(n,m,p) <= mu(n,m,p1) + mu(n,m,p-p1)  [additivity]
    # mu(n,m,p) <= mu(n1,m,p) * mu(n-n1,m,p)  [not quite, but]
    # mu(n,m,p) <= mu(n,m,p1) * mu(n,m,p2) if p = p1*p2? No.
    #
    # The correct submultiplicativity is:
    # mu(n,m,p) <= mu(n1,m,p) * mu(n-n1,m,p) is NOT correct.
    # 
    # What IS correct:
    # mu(n,m,p) <= mu(n,m,p1) + mu(n,m,p2) where p = p1 + p2
    # mu(n,m,p) <= mu(n1,m,p) + mu(n2,m,p) where n = n1 + n2
    # mu(n,m,p) <= mu(n,m1,p) + mu(n,m2,p) where m = m1 + m2
    #
    # And: mu(n,m,p) <= mu(n1,m,p1) * mu(n2,m2,p2) if n=n1*n2, m=m1*m2, p=p1*p2
    # (Kronecker product)
    #
    # So: mu(2,3,6) <= mu(2,3,3) * 2 = 22 (using additivity: mu(2,3,6) <= mu(2,3,3) + mu(2,3,3))
    # mu(2,3,6) <= mu(2,3,2) * 3 (using additivity: mu(2,3,6) <= mu(2,3,2) + mu(2,3,2) + mu(2,3,2))
    #
    # If mu(2,3,2) = 7, then mu(2,3,6) <= 21.
    #
    # Let me try to find mu(2,3,2).
    # 2x3 * 3x2 = 2x2
    # T shape: (6, 6, 4)
    # T[a*3+b, b*2+k, k*2+a] = 1
    #
    # Let me enumerate all non-zero entries:
    # a=0,b=0,k=0: T[0, 0, 0] = 1
    # a=0,b=0,k=1: T[0, 1, 2] = 1
    # a=0,b=1,k=0: T[1, 2, 1] = 1
    # a=0,b=1,k=1: T[1, 3, 3] = 1
    # a=0,b=2,k=0: T[2, 4, 0] = 1
    # a=0,b=2,k=1: T[2, 5, 2] = 1
    # a=1,b=0,k=0: T[3, 0, 1] = 1
    # a=1,b=0,k=1: T[3, 1, 3] = 1
    # a=1,b=1,k=0: T[4, 2, 0] = 1
    # a=1,b=1,k=1: T[4, 3, 2] = 1
    # a=1,b=2,k=0: T[5, 4, 1] = 1
    # a=1,b=2,k=1: T[5, 5, 3] = 1
    #
    # 12 non-zero entries.
    #
    # Let me try to find a rank-7 decomposition.
    # 
    # Actually, let me just go with the known upper bound and try to find a good decomposition.
    # I'll try ranks from 12 to 22 and see which ones work.
    #
    # For each rank R, I'll try to find U, V, W by solving a system of equations.
    # The system is: for each (x,y,z) where T[x,y,z] = 1:
    #   sum_l U[x,l] * V[y,l] * W[z,l] = 1
    # For each (x,y,z) where T[x,y,z] = 0:
    #   sum_l U[x,l] * V[y,l] * W[z,l] = 0
    #
    # This is a system of polynomial equations in the entries of U, V, W.
    # It's nonlinear, so I'll use a different approach.
    #
    # Approach: Fix U and V, then solve for W linearly.
    # Given U (6 x R) and V (18 x R), the equation becomes:
    # For each (x,y,z): sum_l U[x,l] * V[y,l] * W[z,l] = T[x,y,z]
    # This is linear in W! So for fixed U, V, we can solve for W.
    #
    # But we also need the equations to be consistent.
    #
    # Let me try a different approach: use the "tensor rank via SVD" method.
    # 
    # Actually, let me just try to construct explicit decompositions.
    #
    # For rank 22 (2 copies of rank-11 for 2x3x3):
    # This is guaranteed to work. Let me construct it.
    #
    # For 2x3x3, a rank-11 decomposition:
    # I'll use the known result that mu(2,3,3) = 11.
    # One explicit decomposition:
    #
    # Actually, let me just use the schoolbook approach and see if I can do better.
    # Schoolbook rank for 2x3x6 is 2*3*6 = 36.
    # The current program achieves rank 34 (score 1.0588 = 36/34).
    #
    # Let me try to achieve rank 22.
    #
    # Construction: 2x3x6 = 2x3x3 + 2x3x3 (block diagonal in the p dimension)
    # mu(2,3,6) <= mu(2,3,3) + mu(2,3,3) = 11 + 11 = 22
    #
    # So I need a rank-11 decomposition for 2x3x3.
    #
    # Let me construct one.
    # 2x3 * 3x3 = 2x3
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    #
    # Let me use Strassen for the 2x2*2x2 part (first 2 columns of B, first 2 rows of B):
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    #
    # Strassen:
    # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])B[0,0]
    # M3 = A[0,0](B[1,0]-B[1,1])
    # M4 = A[1,1](B[0,1]-B[0,0])
    # M5 = (A[0,0]+A[0,1])B[1,1]
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[1,0])
    # M7 = (A[0,1]-A[1,1])(B[0,1]+B[1,1])
    #
    # C[0,0] = M1 - M3 - M4 + M6
    # C[0,1] = M1 + M2 - M5 - M7
    # C[1,0] = -M2 + M3 + M5 + M7
    # C[1,1] = M1 + M4 - M6 + M7
    #
    # Now the remaining terms:
    # C[0,0] += A[0,2]B[2,0]
    # C[0,1] += A[0,2]B[2,1]
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    # C[1,0] += A[1,2]B[2,0]
    # C[1,1] += A[1,2]B[2,1]
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    #
    # That's 8 more terms (A[0,2]B[2,0], A[0,2]B[2,1], A[0,0]B[0,2], A[0,1]B[1,2], A[0,2]B[2,2], A[1,2]B[2,0], A[1,2]B[2,1], A[1,0]B[0,2], A[1,1]B[1,2], A[1,2]B[2,2])
    # Wait, that's 10 terms. 7 + 10 = 17. Not 11.
    #
    # I need to be smarter. Let me combine terms.
    #
    # C[0,0] += A[0,2]B[2,0]
    # C[0,1] += A[0,2]B[2,1]
    # These can be combined: A[0,2]*(B[2,0] for C[0,0], B[2,1] for C[0,1])
    # As a rank-1 term: U[0,2]=1, V[2,0]=1, V[2,1]=1, W[0,0]=1, W[0,1]=1
    # This gives A[0,2]*B[2,0] for C[0,0] and A[0,2]*B[2,1] for C[0,1]. 
    # But it also gives A[0,2]*B[2,0] for C[0,1] and A[0,2]*B[2,1] for C[0,0], which are wrong.
    #
    # So I can't combine them that way. I need separate terms.
    #
    # Let me try a different approach. Let me use the fact that:
    # mu(2,3,3) <= mu(2,2,3) + mu(2,1,3)
    # mu(2,1,3): 2x1 * 1x3 = 2x3, outer product, rank 1.
    # mu(2,2,3): 2x2 * 2x3 = 2x3
    # mu(2,2,3) <= mu(2,2,2) + mu(2,2,1) = 7 + 1 = 8
    # mu(2,2,1): 2x2 * 2x1 = 2x1, which is 2 dot products, rank 2.
    # So mu(2,2,3) <= 7 + 2 = 9.
    # mu(2,3,3) <= 9 + 1 = 10.
    #
    # Hmm, that gives 10, not 11. Let me double-check.
    # mu(2,2,1): 2x2 * 2x1 = 2x1
    # C[0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1] = A[1,0]B[0,0] + A[1,1]B[1,0]
    # This is 2x2 * 2x1. The tensor has 2*2*1 = 4 non-zero entries.
    # T shape: (2*2, 2*1, 1*2) = (4, 2, 2)
    # T[a*2+b, b*1+k, k*1+a] = 1
    # a=0,b=0,k=0: T[0,0,0]=1
    # a=0,b=1,k=0: T[1,1,1]=1
    # a=1,b=0,k=0: T[2,0,1]=1
    # a=1,b=1,k=0: T[3,1,0]=1
    #
    # This is 4 independent entries, so rank >= 4? No, the tensor rank could be less.
    # Actually, each entry is independent, so rank = 4.
    # Wait, no. Let me check if rank 2 works.
    # T[0,0,0]=1, T[1,1,1]=1, T[2,0,1]=1, T[3,1,0]=1
    # A rank-1 term U[x]*V[y]*W[z] can cover at most 1 entry (since the entries are at different (x,y,z) positions).
    # Actually, a rank-1 term U[:,l]*V[:,l]*W[:,l] produces a tensor where T[x,y,z] = U[x,l]*V[y,l]*W[z,l].
    # For this to match our T (which has 0/1 entries), we need U[x,l]*V[y,l]*W[z,l] = 1 for the 4 specific (x,y,z) and 0 elsewhere.
    # This is only possible if U[:,l], V[:,l], W[:,l] are such that the product is 1 at those 4 points and 0 elsewhere.
    # This requires |supp(U[:,l])| * |supp(V[:,l])| * |supp(W[:,l])| >= 4 and the supports to be aligned.
    #
    # For example, if U[:,l] has support {0,1}, V[:,l] has support {0,1}, W[:,l] has support {0,1},
    # then the product is 1 at (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1).
    # But we only want 4 of these to be 1. So a single rank-1 term can't do it.
    #
    # So mu(2,2,1) = 4? That seems high. Let me reconsider.
    #
    # Actually, I think I'm confusing tensor rank with something else.
    # The tensor rank of the matrix multiplication tensor M_{n,m,p} is mu(n,m,p).
    # For M_{2,2,1}: 2x2 * 2x1 = 2x1
    # mu(2,2,1) = ?
    # 
    # C[0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1] = A[1,0]B[0,0] + A[1,1]B[1,0]
    #
    # This is equivalent to: C = A * B where A is 2x2, B is 2x1.
    # We can write B = [B[0,0], B[1,0]]^T.
    # C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #
    # Let's try rank 2:
    # Term 1: U[0,0]=1, U[1,0]=0, U[2,0]=0, U[3,0]=0 (A[0,0])
    #          V[0,0]=1, V[1,0]=0 (B[0,0])
    #          W[0,0]=1, W[1,0]=0 (C[0])
    # Term 2: U[0,1]=0, U[1,1]=0, U[2,1]=0, U[3,1]=0 ... 
    #
    # This is getting complicated. Let me just go with the known result and construct a decomposition.
    #
    # I'll use the following approach:
    # 1. Construct a rank-22 decomposition for 2x3x6 (2 copies of rank-11 for 2x3x3)
    # 2. Try to reduce the rank by finding a better decomposition
    #
    # For the rank-11 decomposition of 2x3x3, let me use a known construction.
    #
    # Actually, let me just try to solve the problem numerically.
    # I'll use the following approach:
    # 1. Initialize U, V, W randomly.
    # 2. Alternately update U, V, W to minimize the error.
    # 3. Round to nearest integers/half-integers.
    # 4. Verify.
    #
    # This is a tensor decomposition algorithm (like ALS - Alternating Least Squares).

    def try_rank(R, max_iters=100, seed=42):
        """Try to find a rank-R decomposition using ALS."""
        np.random.seed(seed)
        
        U = np.random.randn(nm, R) * 0.1
        V = np.random.randn(mp, R) * 0.1
        W = np.random.randn(pn, R) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Update W: for each z, W[z, :] should satisfy
            # sum_l U[x,l] * V[y,l] * W[z,l] = T[x,y,z] for all x,y
            # This is: (U^T * V_elementwise)[:, :, l] * W[z, l] = T[:, :, z]
            # Let M_z[x,y] = sum_l U[x,l] * V[y,l] * W[z,l]
            # We want M_z = T[:,:,z]
            # This is linear in W[z,:]:
            # For each (x,y): sum_l U[x,l] * V[y,l] * W[z,l] = T[x,y,z]
            # Let A_z[l] = sum_{x,y} U[x,l] * V[y,l] * (something)
            # Actually, this is a linear system in W[z,:]:
            # For fixed z, the equations are:
            # sum_l (U[x,l] * V[y,l]) * W[z,l] = T[x,y,z] for all (x,y)
            # This is: (U * V^T elementwise) @ W[z,:] = T[:,:,z].flatten()
            # But U*V^T is (nm x mp), and we need to contract over (x,y) pairs.
            # 
            # Let me define: for each l, let A_l[x,y] = U[x,l] * V[y,l]
            # Then sum_l A_l[x,y] * W[z,l] = T[x,y,z]
            # This is a linear system: for each z, W[z,:] is in the row space of {A_l}
            # 
            # Let me set up the linear system properly.
            # For fixed z:
            # sum_l W[z,l] * (U[:,l] * V[:,l]) = T[:,:,z].reshape(nm*mp)
            # where (U[:,l] * V[:,l]) is the outer product U[:,l] outer V[:,l] flattened.
            #
            # So: sum_l W[z,l] * outer(U[:,l], V[:,l]).flatten() = T[:,:,z].flatten()
            #
            # This is: M @ W[z,:] = T[:,:,z].flatten()
            # where M is (nm*mp x R) with M[x*mp+y, l] = U[x,l] * V[y,l]
            
            M = np.zeros((nm * mp, R))
            for x in range(nm):
                for y in range(mp):
                    for l in range(R):
                        M[x * mp + y, l] = U[x, l] * V[y, l]
            
            # Solve M @ W[z,:] = T[:,:,z].flatten()
            for z in range(pn):
                rhs = T.reshape(nm * mp, pn)[:, z]
                # Least squares
                W[z, :], residuals, rank, sv = np.linalg.lstsq(M, rhs, rcond=None)
            
            # Similarly update U and V
            # For fixed x: sum_l U[x,l] * V[y,l] * W[z,l] = T[x,y,z] for all y,z
            # sum_l U[x,l] * (V[:,l] * W[:,l]) = T[x,:,:].reshape(mp*pn)
            # M2 @ U[x,:] = T[x,:,:].flatten()
            # where M2[y*pn+z, l] = V[y,l] * W[z,l]
            
            M2 = np.zeros((mp * pn, R))
            for y in range(mp):
                for z in range(pn):
                    for l in range(R):
                        M2[y * pn + z, l] = V[y, l] * W[z, l]
            
            for x in range(nm):
                rhs = T[x, :, :].reshape(mp * pn)
                U[x, :], residuals, rank, sv = np.linalg.lstsq(M2, rhs, rcond=None)
            
            # Update V
            # For fixed y: sum_l U[x,l] * V[y,l] * W[z,l] = T[x,y,z] for all x,z
            # sum_l V[y,l] * (U[:,l] * W[:,l]) = T[:,y,:].reshape(nm*pn)
            # M3 @ V[y,:] = T[:,y,:].flatten()
            # where M3[x*pn+z, l] = U[x,l] * W[z,l]
            
            M3 = np.zeros((nm * pn, R))
            for x in range(nm):
                for z in range(pn):
                    for l in range(R):
                        M3[x * pn + z, l] = U[x, l] * W[z, l]
            
            for y in range(mp):
                rhs = T[:, y, :].reshape(nm * pn)
                V[y, :], residuals, rank, sv = np.linalg.lstsq(M3, rhs, rcond=None)
        
        # Round to nearest integers or half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Verify
        def verify_decomp(U, V, W):
            for x in range(nm):
                for y in range(mp):
                    for z in range(pn):
                        s = 0.0
                        for l in range(R):
                            s += U[x, l] * V[y, l] * W[z, l]
                        target = T[x, y, z]
                        if abs(s - target) > 1e-6:
                            return False
            return True
        
        if verify_decomp(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        # Try rounding to integers only
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        
        if verify_decomp(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        return None

    # Try ranks from lower bound up
    best_result = None
    for R in range(int(tensor_rank_lb), 37):
        if time.time() > deadline:
            break
        
        result = try_rank(R, max_iters=50, seed=R)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: use the schoolbook decomposition (rank 36)
    U = np.zeros((nm, nm * mp * pn))
    V = np.zeros((mp, nm * mp * pn))
    W = np.zeros((pn, nm * mp * pn))
    
    idx = 0
    for a in range(n):
        for b in range(m):
            for k in range(p):
                U[a * m + b, idx] = 1.0
                V[b * p + k, idx] = 1.0
                W[k * n + a, idx] = 1.0
                idx += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }


if __name__ == "__main__":
    construct()
