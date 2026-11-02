import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]


def build_target_tensor(n, m, p):
    """T[a*m+b, c*p+d, k*n+i] = 1 iff a==i and b==c and d==k"""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for k in range(p):
                        for i in range(n):
                            if a == i and b == c and d == k:
                                T[a * m + b, c * p + d, k * n + i] = 1.0
    return T


def verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Faster verification using matrix operations."""
    rank = U.shape[1]
    T_recon = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for l in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:, l], V[:, l], W[:, l])
    
    T_target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for k in range(p):
                        for i in range(n):
                            if a == i and b == c and d == k:
                                T_target[a * m + b, c * p + d, k * n + i] = 1.0
    
    return np.allclose(T_recon, T_target, atol=tol)


def build_strassen_222():
    """
    Standard Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    U index: a*2+b (A row a, col b)
    V index: c*2+d (B row c, col d)  
    W index: k*2+i (C row k, col i)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = -1; W[2, 2] = 1
    
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[3, 3] = 1
    
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = -1; W[3, 4] = 1
    
    # M6 = (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1; W[3, 5] = -1
    
    # M7 = (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[1, 6] = -1; W[3, 6] = 1
    
    return U, V, W


def build_rank33_block():
    """
    Build rank-33 decomposition using 3 independent 2x3x2 blocks.
    This is the baseline from the inspiration.
    """
    n, m, p = 2, 3, 2
    nU, nV, nW = n * m, m * p, p * n  # 6, 6, 4
    
    # For 2x3x2: A(2x3) * B(3x2) = C(2x2)
    # We can view this as: for each column d of B, compute A * B[:,d]
    # But better: use Strassen on the 2x2x2 core + handle the extra column of A
    
    # Actually, let's build it from scratch using the tensor structure.
    # T[a*3+b, c*2+d, k*2+i] = 1 iff a==i, b==c, d==k
    
    # Nonzero entries: for a in {0,1}, b in {0,1,2}, d in {0,1}:
    #   i=a, c=b, k=d
    #   So: T[a*3+b, b*2+d, d*2+a] = 1
    # That's 2*3*2 = 12 nonzero entries.
    
    # Rank lower bound: max(n*m, m*p, p*n) = max(6,6,4) = 6
    # But actually for matrix mult, rank >= n*m*p / max(n,m,p) ... no that's not right.
    # Known: rank(2x2x2) = 7, rank(2x3x2) should be around 9-11
    
    # Let's use a systematic approach:
    # The tensor is: sum over b of A[:,b] * B[b,:] 
    # = A[:,0]*B[0,:] + A[:,1]*B[1,:] + A[:,2]*B[2,:]
    # Each term A[:,b]*B[b,:] is a 2x2 matrix = outer product of A column and B row
    # But we can't just do 3 outer products (rank 3) because they're not independent in the tensor sense.
    # Wait, actually they ARE independent rank-1 terms!
    # T = sum_b (e_a * e_b for A) (e_b * e_d for B) (e_d * e_a for C)
    # = sum_b sum_a sum_d A_{a,b} B_{b,d} C_{d,a}
    # In tensor form: T[i,j,k] where i=a*3+b, j=c*2+d, k=d'*2+a'
    # T[i,j,k] = 1 iff a==a', b==c, d==d'
    # = sum_b delta(a,a') delta(b,c) delta(d,d')
    # This is exactly: sum_b (u_b ⊗ v_b ⊗ w_b) where
    #   u_b has 1 at position a*3+b for all a... no wait.
    
    # Let me re-derive. The matrix multiplication tensor for n x m times m x p:
    # T[i,j,k] = 1 iff the (i,j,k) entry is "active"
    # i = a*m + b (row a, col b of A)
    # j = c*p + d (row c, col d of B)  
    # k = k'*n + i' (row k', col i' of C)
    # Active when: a = i', b = c, d = k'
    # So T[a*m+b, b*p+d, d*n+a] = 1
    
    # This can be written as: sum over b of (sum over a of e_{a*m+b}) ⊗ (sum over d of e_{b*p+d}) ⊗ (sum over a,d of e_{d*n+a} where the a and d are linked)
    
    # Hmm, let me think differently. For fixed b:
    # The terms are: for each (a,d), T[a*m+b, b*p+d, d*n+a] = 1
    # This is a set of n*p entries for each b.
    # For n=2, p=2: 4 entries per b, 3 b's = 12 total.
    
    # Can we write this as sum of rank-1 tensors?
    # A rank-1 tensor is u ⊗ v ⊗ w where u, v, w are vectors.
    # (u ⊗ v ⊗ w)[i,j,k] = u[i]*v[j]*w[k]
    
    # For the 2x3x2 case, let's try to find a decomposition.
    # We need: sum_l U[i,l]*V[j,l]*W[k,l] = T[i,j,k]
    
    # Let's try rank 9 (3 terms per "block" of the 2x2 core):
    # Actually, let's just use the known result or compute it.
    
    # For 2x3x2, I'll use the following approach:
    # Split into two 2x2x2 blocks (columns 0,1 of B) but they share the same A.
    # No wait, B has shape 3x2, so it's 3 rows and 2 columns.
    # A is 2x3: 2 rows, 3 columns.
    # C is 2x2: 2 rows, 2 columns.
    
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # This is like: for each pair (row of C, col of C), sum over b of A[row,b]*B[b,col]
    # = sum_b (A[:,b] ⊗ B[b,:] ⊗ something)
    
    # Let's use the "sum over b" structure:
    # C = sum_b A[:,b] * B[b,:]  (outer product of A's column b and B's row b)
    # In tensor decomposition language, each b gives a contribution that is:
    #   (column b of A) ⊗ (row b of B) ⊗ (identity on C indices)
    # But the "identity on C indices" means for each (k,i) in C, the coefficient is 1
    # if k=d and i=a... this is getting circular.
    
    # Let me just directly construct a rank-9 decomposition for 2x3x2:
    # Use Strassen for the 2x2x2 part (rows 0,1 of A; rows 0,1 of B; all of C)
    # Plus 2 extra terms for the third column of A (b=2).
    
    # 2x2x2 Strassen: 7 terms
    # For b=2: A[0,2]*B[2,0] -> C[0,0], A[0,2]*B[2,1] -> C[0,1]
    #          A[1,2]*B[2,0] -> C[1,0], A[1,2]*B[2,1] -> C[1,1]
    # These 4 terms can be done with 2 rank-1 tensors:
    #   Term 8: U[2,8]=1 (A[0,2]), V[4,8]=1 (B[2,0]), W[0,8]=1 (C[0,0])
    #   Wait, that's only 1 entry. We need 4 entries.
    #   Actually: A[0,2]*B[2,0] contributes to C[0,0] only.
    #   A[0,2]*B[2,1] contributes to C[0,1] only.
    #   So we need 4 separate rank-1 terms? No, we can combine:
    #   Term 8: U[2,8]=1, V[4,8]=1, W[0,8]=1  (A[0,2]*B[2,0] -> C[0,0])
    #   Term 9: U[2,8]=1, V[5,8]=1, W[1,8]=1  -- no, same term can't have two V entries
    #   
    #   Actually each rank-1 term has one U index, one V index, one W index.
    #   So A[0,2]*B[2,0] -> C[0,0] is one rank-1 term.
    #   A[0,2]*B[2,1] -> C[0,1] is another.
    #   That's 4 terms for b=2, giving rank 7+4=11.
    
    # But we can do better by combining with the Strassen terms.
    # Let's just use rank 11 for 2x3x2.
    
    # Actually, let me try a different approach. Let's use the fact that
    # 2x3x2 can be embedded in 3x3x2 or handled differently.
    
    # For now, let's use the simple approach: 3 independent 2x2x2 blocks won't work
    # because A has 3 columns and B has 3 rows, so the "middle" dimension is 3.
    
    # Let me just hard-code a known good decomposition.
    # For 2x3x2, the rank is known to be 9 (I believe).
    
    # Let me try to construct it:
    # C[0,0] = A00*B00 + A01*B10 + A02*B20
    # C[0,1] = A00*B01 + A01*B11 + A02*B21
    # C[1,0] = A10*B00 + A11*B10 + A12*B20
    # C[1,1] = A10*B01 + A11*B11 + A12*B21
    
    # Using Strassen-like combinations:
    # M1 = (A00+A11)(B00+B10) -> contributes to C[0,0],C[0,1],C[1,0],C[1,1]
    #   U: A00, A11 -> indices 0, 4
    #   V: B00, B10 -> indices 0, 2
    #   W: C00, C01, C10, C11 -> indices 0,1,2,3
    #   But Strassen M1 gives: C[0,0]+=M1, C[0,1]-=M1, C[1,0]+=M1, C[1,1]-=M1
    #   Wait no, standard Strassen:
    #   M1 = (A11-A01)(B01+B11)
    #   M2 = (A00+A01)B11
    #   M3 = (A10+A11)B00
    #   M4 = (A00-A11)(B00+B11)
    #   M5 = (A01+A11)(B10-B00)
    #   M6 = (A00+A01)(B10-B11)
    #   M7 = (A10-A00)(B01+B11)
    #   C00 = M4+M6+M7-M2
    #   C01 = M1+M2
    #   C10 = M3+M4
    #   C11 = M1-M3+M5+M4  -- let me double check
    
    # Standard Strassen for 2x2:
    # M1 = (A11 - A01) * (B01 + B11)
    # M2 = (A00 + A01) * B11
    # M3 = (A10 + A11) * B00
    # M4 = (A00 - A11) * (B00 + B11)
    # M5 = (A01 + A11) * (B10 - B00)
    # M6 = (A00 + A01) * (B10 - B11)
    # M7 = (A10 - A00) * (B01 + B11)
    # C00 = M4 + M6 + M7 - M2
    # C01 = M1 + M2
    # C10 = M3 + M4
    # C11 = M1 - M3 + M5 + M4
    
    # For 2x3x2, the "2x2 core" uses A columns 0,1 and B rows 0,1.
    # Then we need to handle A column 2 and B row 2.
    
    # Let me just use the schoolbook for 2x3x2: rank 12 (3*2*2=12)
    # Or try to get rank 9.
    
    # Actually, for the full 2x3x6 problem, let me use a different strategy.
    # Use the tensor product structure:
    # 2x3x6 = 2x3x(2*3) 
    # We can view B as a 3x2 block matrix with 3 blocks of 3x2, or 3x6 as 3x(2*3)
    
    # Let me use the following approach:
    # 1. Build a 2x3x3 decomposition with rank r
    # 2. Use it twice to get 2x3x6 with rank 2r
    
    # For 2x3x3: C(2x3) = A(2x3) * B(3x3)
    # C[0,0] = A00*B00 + A01*B10 + A02*B20
    # C[0,1] = A00*B01 + A01*B11 + A02*B21
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,0] = A10*B00 + A11*B10 + A12*B20
    # C[1,1] = A10*B01 + A11*B11 + A12*B21
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    
    # Schoolbook: rank 18
    # Can we do better?
    
    # Let me try to use Strassen on the 2x2x2 sub-blocks and combine.
    # The 2x3x3 can be seen as: for each output column j (0,1,2),
    # C[:,j] = A * B[:,j]
    # Each C[:,j] is a 2x1 vector = A(2x3) * B[:,j](3x1)
    # = A[:,0]*B[0,j] + A[:,1]*B[1,j] + A[:,2]*B[2,j]
    
    # So C = sum_j (A * B[:,j]) e_j^T where e_j is the j-th standard basis vector in R^3
    
    # In tensor form: T = sum_j (sum_b A[:,b] B[b,j]) ⊗ e_j
    # = sum_j sum_b (A[:,b] ⊗ B[b,j] ⊗ e_j)
    
    # This is 3*3 = 9 terms if we don't combine, but each "term" is actually
    # a 2x1x1 tensor, not rank-1 in the full 2x3x3 sense.
    
    # Let me just go with the block approach: 2x3x3 with rank 13 (from inspiration 1)
    # and use it twice for rank 26.
    
    # Actually, let me try a completely different approach.
    # Use the fact that the tensor has a specific structure and try to find
    # a low-rank decomposition using linear algebra.
    
    # The tensor T is in R^{6 x 18 x 12}.
    # We want to find U(6,r), V(18,r), W(12,r) such that
    # T[i,j,k] = sum_l U[i,l] V[j,l] W[k,l]
    
    # This is equivalent to: for each l, the matrix T[:,:,l] = U[:,l] V[:,l]^T ... no.
    # T[i,j,k] = sum_l U[i,l] V[j,l] W[k,l]
    # For fixed k: T[:,:,k] = sum_l W[k,l] (U[:,l] V[:,l]^T)
    # So T[:,:,k] is a linear combination of the rank-1 matrices U[:,l] V[:,l]^T.
    
    # Similarly for fixed i: T[i,:,:] = sum_l U[i,l] (V[:,l] W[:,l]^T)
    
    # The key insight: the tensor is "structured" - it comes from matrix multiplication.
    # The minimal rank for n x m x p matrix multiplication is known for small cases.
    # For 2x3x6, I don't know the exact minimal rank, but it's at most 36 (schoolbook).
    
    # Let me try a heuristic search:
    # 1. Start with schoolbook (rank 36)
    # 2. Try to merge terms by finding linear dependencies
    
    # Actually, let me try a different structural approach:
    # Use the "Kronecker product" structure.
    # 2x3x6 = (2x3x2) ⊗ (1x1x3) in some sense?
    # No, that's not quite right.
    
    # Let me try: 2x3x6 can be decomposed as follows:
    # Split the 6 output columns into 3 groups of 2.
    # For each group g (0,1,2), the output columns are 2g and 2g+1.
    # C[:,2g:2g+2] = A * B[:,2g:2g+2]
    # This is a 2x3 times 3x2 multiplication, giving a 2x2 result.
    # So we have 3 independent 2x3x2 problems.
    # If rank(2x3x2) = r, then rank(2x3x6) <= 3r.
    
    # What is rank(2x3x2)?
    # C(2x2) = A(2x3) * B(3x2)
    # This is equivalent to: for each (i,j) in C, C[i,j] = sum_b A[i,b] B[b,j]
    # The tensor has 2*3*2 = 12 nonzero entries.
    # Lower bound: rank >= max(2*3, 3*2, 2*2) / something... 
    # Actually, a simple lower bound is rank >= max(n*m, m*p, p*n) / max(n,m,p)
    # = max(6,6,4) / 3 = 2. That's too weak.
    # Better: rank >= n*m*p / max(n,m,p)^2 = 12/9 < 2. Still weak.
    # Known: rank(2x2x2) = 7. For 2x3x2, I'll guess rank is around 9-11.
    
    # Let me just try to find a good decomposition for 2x3x2 first.
    # Then use 3 copies for 2x3x6.
    
    # For 2x3x2, let me try rank 9:
    # U: (6, 9), V: (6, 9), W: (4, 9)
    # U index: a*3+b
    # V index: c*2+d
    # W index: k*2+i
    
    # Nonzero entries of T: T[a*3+b, b*2+d, d*2+a] = 1 for a in {0,1}, b in {0,1,2}, d in {0,1}
    
    # Let me try to construct this:
    # For b=0: T[0,0,0]=1, T[0,2,2]=1, T[3,0,0]=1, T[3,2,2]=1
    #   (a=0,b=0,d=0): T[0,0,0]
    #   (a=0,b=0,d=1): T[0,2,2]
    #   (a=1,b=0,d=0): T[3,0,0]
    #   (a=1,b=0,d=1): T[3,2,2]
    # For b=1: T[1,2,0]=1, T[1,4,2]=1, T[4,2,0]=1, T[4,4,2]=1
    #   (a=0,b=1,d=0): T[1,2,0]
    #   (a=0,b=1,d=1): T[1,4,2]
    #   (a=1,b=1,d=0): T[4,2,0]
    #   (a=1,b=1,d=1): T[4,4,2]
    # For b=2: T[2,4,0]=1, T[2,6,2]=1, T[5,4,0]=1, T[5,6,2]=1
    #   (a=0,b=2,d=0): T[2,4,0]
    #   (a=0,b=2,d=1): T[2,6,2]
    #   (a=1,b=2,d=0): T[5,4,0]
    #   (a=1,b=2,d=1): T[5,6,2]
    
    # That's 12 nonzero entries.
    
    # Let me try a rank-9 decomposition:
    # Use 3 terms for each b, sharing structure.
    
    # Actually, let me just use the schoolbook for 2x3x2 (rank 12) and then
    # use 3 copies for 2x3x6 (rank 36). That's the same as the schoolbook for 2x3x6.
    
    # To do better, I need to find a lower-rank decomposition.
    
    # Let me try a different approach: use the "bilinear" structure more explicitly.
    # The matrix multiplication tensor for n x m x p has rank at most n*m*p.
    # But it can be lower. For 2x2x2, it's 7 < 8.
    
    # For 2x3x6, let me try to use the following decomposition:
    # 1. Use Strassen for the 2x2x2 core (using A rows 0,1; B rows 0,1; C cols 0,1)
    # 2. Handle the remaining dimensions separately
    
    # This is getting complex. Let me just use a simple but effective approach:
    # Use the "sum over b" structure to get rank = m * rank(2x1x6) or something.
    
    # Actually, let me try the following:
    # C = A * B
    # = sum_b (A[:,b] * B[b,:])
    # Each term A[:,b] * B[b,:] is a 2x6 matrix (outer product of a 2x1 and 1x6 vector).
    # In tensor decomposition, this is NOT a single rank-1 term because it involves
    # multiple entries of U, V, W.
    
    # Let me just go with the block approach and try to optimize the block size.
    
    # For 2x3x6, use 2 blocks of 2x3x3.
    # Need rank(2x3x3) first.
    
    # 2x3x3: C(2x3) = A(2x3) * B(3x3)
    # Schoolbook: rank 18.
    # Can we do better?
    
    # Let me try to construct a rank-13 decomposition for 2x3x3.
    # (The inspiration mentions rank 13 for 2x3x3.)
    
    # Actually, let me just implement a search for a low-rank decomposition
    # using a simple iterative method.
    
    # For now, let's use the safe approach: 3 blocks of 2x3x2 with rank 11 each = rank 33.
    # Or 2 blocks of 2x3x3 with rank 13 each = rank 26.
    
    # Let me try to build a rank-13 decomposition for 2x3x3.
    
    # 2x3x3:
    # C[0,0] = A00*B00 + A01*B10 + A02*B20
    # C[0,1] = A00*B01 + A01*B11 + A02*B21
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,0] = A10*B00 + A11*B10 + A12*B20
    # C[1,1] = A10*B01 + A11*B11 + A12*B21
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    
    # U: (6, 13), V: (9, 13), W: (6, 13)
    # U index: a*3+b
    # V index: c*3+d
    # W index: k*3+i
    
    # Let me use Strassen for the 2x2x2 sub-blocks:
    # Block (0,0): A cols 0,1; B rows 0,1; C cols 0,1 -> 7 terms
    # Block (0,1): A cols 0,1; B rows 0,1; C cols 0,1 but shifted -> same 7 terms?
    # No, the C indices are different.
    
    # This is getting too complex to do by hand. Let me use a computational approach.
    
    # I'll use a simple greedy algorithm to find a low-rank decomposition.
    return None


def try_greedy_decomposition(n, m, p, max_rank, deadline=None):
    """
    Try to find a low-rank decomposition using a greedy approach.
    Start with schoolbook and try to merge terms.
    """
    nU, nV, nW = n * m, m * p, p * n
    
    # Build target tensor
    T = build_target_tensor(n, m, p)
    
    # Start with schoolbook decomposition
    U = np.zeros((nU, n * m * p))
    V = np.zeros((nV, n * m * p))
    W = np.zeros((nW, n * m * p))
    idx = 0
    for a in range(n):
        for b in range(m):
            for d in range(p):
                U[a * m + b, idx] = 1
                V[b * p + d, idx] = 1
                W[d * n + a, idx] = 1
                idx += 1
    
    rank = n * m * p
    
    # Try to reduce rank by finding linear dependencies
    # This is a complex problem, so let's use a simpler heuristic.
    
    # Actually, let's just return the schoolbook for now.
    # The real work is in finding a better decomposition.
    
    return U, V, W, rank


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Strategy: Use 2 blocks of 2x3x3, each with rank 13.
    # Total rank: 26.
    
    # Build 2x3x3 decomposition with rank 13.
    # I'll use a combination of Strassen and additional terms.
    
    n3, m3, p3 = 2, 3, 3
    nU3, nV3, nW3 = n3 * m3, m3 * p3, p3 * n3  # 6, 9, 6
    
    # For 2x3x3, let's use the following approach:
    # 1. Use Strassen for the 2x2x2 core (A cols 0,1; B rows 0,1; C cols 0,1)
    # 2. Add terms for the third column of A (b=2) and third row of B (c=2)
    # 3. Add terms for the third column of C (k=2)
    
    U3 = np.zeros((nU3, 13))
    V3 = np.zeros((nV3, 13))
    W3 = np.zeros((nW3, 13))
    
    # Strassen for 2x2x2 core (A[0:2,0:2], B[0:2,0:2], C[0:2,0:2])
    # U index: a*3+b, V index: c*3+d, W index: k*3+i
    
    # M1 = (A11-A01)(B01+B11)
    U3[4, 0] = 1; U3[1, 0] = -1  # A[1,1]=idx4, A[0,1]=idx1
    V3[1, 0] = 1; V3[4, 0] = 1  # B[0,1]=idx1, B[1,1]=idx4
    W3[1, 0] = 1; W3[2, 0] = 1  # C[0,1]=idx1, C[1,1]=idx4... wait
    # W index: k*3+i, so C[0,1]=idx1, C[1,1]=idx4
    W3[1, 0] = 1; W3[4, 0] = 1
    
    # M2 = (A00+A01)B11
    U3[0, 1] = 1; U3[1, 1] = 1
    V3[4, 1] = 1  # B[1,1]=idx4
    W3[2, 1] = 1; W3[4, 1] = 1  # C[0,2]... no, C[0,1]=idx1, C[1,1]=idx4
    # Wait, I need to be more careful with indices.
    # W index: k*3+i where k is row, i is col of C.
    # C[0,0]=0, C[0,1]=1, C[0,2]=2, C[1,0]=3, C[1,1]=4, C[1,2]=5
    
    # Let me restart the Strassen assignment with correct indices.
    U3 = np.zeros((nU3, 13))
    V3 = np.zeros((nV3, 13))
    W3 = np.zeros((nW3, 13))
    
    # M1 = (A11-A01)(B01+B11)
    # A11 = U[1*3+1] = U[4], A01 = U[0*3+1] = U[1]
    # B01 = V[0*3+1] = V[1], B11 = V[1*3+1] = V[4]
    # Contributes to C[0,1] and C[1,1]
    # C[0,1] = W[0*3+1] = W[1], C[1,1] = W[1*3+1] = W[4]
    U3[4, 0] = 1; U3[1, 0] = -1
    V3[1, 0] = 1; V3[4, 0] = 1
    W3[1, 0] = 1; W3[4, 0] = 1
    
    # M2 = (A00+A01)B11
    # A00 = U[0], A01 = U[1], B11 = V[4]
    # Contributes to C[0,1] and C[1,1]
    U3[0, 1] = 1; U3[1, 1] = 1
    V3[4, 1] = 1
    W3[1, 1] = 1; W3[4, 1] = 1
    
    # M3 = (A10+A11)B00
    # A10 = U[3], A11 = U[4], B00 = V[0]
    # Contributes to C[0,0] and C[1,0]
    U3[3, 2] = 1; U3[4, 2] = 1
    V3[0, 2] = 1
    W3[0, 2] = 1; W3[3, 2] = 1
    
    # M4 = (A00-A11)(B00+B11)
    # A00 = U[0], A11 = U[4], B00 = V[0], B11 = V[4]
    # Contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    U3[0, 3] = 1; U3[4, 3] = -1
    V3[0, 3] = 1; V3[4, 3] = 1
    W3[0, 3] = 1; W3[1, 3] = 1; W3[3, 3] = 1; W3[4, 3] = 1
    
    # M5 = (A01+A11)(B10-B00)
    # A01 = U[1], A11 = U[4], B10 = V[3], B00 = V[0]
    # Contributes to C[0,0] and C[1,0]
    U3[1, 4] = 1; U3[4, 4] = 1
    V3[3, 4] = 1; V3[0, 4] = -1
    W3[0, 4] = 1; W3[3, 4] = 1
    
    # M6 = (A00+A01)(B10-B11)
    # A00 = U[0], A01 = U[1], B10 = V[3], B11 = V[4]
    # Contributes to C[0,0] and C[1,0]
    U3[0, 5] = 1; U3[1, 5] = 1
    V3[3, 5] = 1; V3[4, 5] = -1
    W3[0, 5] = 1; W3[3, 5] = 1
    
    # M7 = (A10-A00)(B01+B11)
    # A10 = U[3], A00 = U[0], B01 = V[1], B11 = V[4]
    # Contributes to C[0,1] and C[1,1]
    U3[3, 6] = 1; U3[0, 6] = -1
    V3[1, 6] = 1; V3[4, 6] = 1
    W3[1, 6] = 1; W3[4, 6] = 1
    
    # Now we need to handle the third column of A (b=2) and third row of B (c=2).
    # And the third column of C (k=2).
    
    # C[0,2] = A00*B02 + A01*B12 + A02*B22
    # C[1,2] = A10*B02 + A11*B12 + A12*B22
    
    # B02 = V[0*3+2] = V[2], B12 = V[1*3+2] = V[5], B22 = V[2*3+2] = V[8]
    # A02 = U[0*3+2] = U[2], A12 = U[1*3+2] = U[5]
    
    # M8 = (A00+A10)B02
    U3[0, 7] = 1; U3[3, 7] = 1
    V3[2, 7] = 1
    W3[2, 7] = 1; W3[5, 7] = 1
    
    # M9 = (A01+A11)B12
    U3[1, 8] = 1; U3[4, 8] = 1
    V3[5, 8] = 1
    W3[2, 8] = 1; W3[5, 8] = 1
    
    # M10 = A02*B22
    U3[2, 9] = 1
    V3[8, 9] = 1
    W3[2, 9] = 1
    
    # M11 = A12*B22
    U3[5, 10] = 1
    V3[8, 10] = 1
    W3[5, 10] = 1
    
    # But we also need C[0,0] and C[1,0] to include contributions from b=2:
    # C[0,0] += A02*B20
    # C[1,0] += A12*B20
    # B20 = V[2*3+0] = V[6]
    
    # M12 = (A02+A12)B20
    U3[2, 11] = 1; U3[5, 11] = 1
    V3[6, 11] = 1
    W3[0, 11] = 1; W3[3, 11] = 1
    
    # And C[0,1] and C[1,1] need contributions from b=2:
    # C[0,1] += A02*B21
    # C[1,1] += A12*B21
    # B21 = V[2*3+1] = V[7]
    
    # M13 = (A02+A12)B21
    U3[2, 12] = 1; U3[5, 12] = 1
    V3[7, 12] = 1
    W3[1, 12] = 1; W3[4, 12] = 1
    
    # Now verify this 2x3x3 decomposition
    if verify_fast(U3, V3, W3, 2, 3, 3, tol=1e-6):
        # Great! Use it twice for 2x3x6
        U_full = np.zeros((6, 26))
        V_full = np.zeros((18, 26))
        W_full = np.zeros((12, 26))
        
        for block in range(2):
            offset = block * 13
            U_full[:, offset:offset+13] = U3
            
            # V: shape (m*p, rank) = (3*6, rank)
            # V index: b*6 + d where b in [0,3), d in [0,6)
            # For block j (columns 3j to 3j+2 of B), V[b*6 + 3j + k, l] = V3[b*3 + k, l]
            for b in range(3):
                for k in range(3):
                    V_full[b * 6 + block * 3 + k, offset:offset+13] = V3[b * 3 + k]
            
            # W: shape (p*n, rank) = (6*2, rank) = (12, rank)
            # W index: k*2 + i where k in [0,6), i in [0,2)
            # For block j, output columns 3j to 3j+2
            for k in range(3):
                for i in range(2):
                    W_full[(block * 3 + k) * 2 + i, offset:offset+13] = W3[k * 2 + i]
        
        if verify_fast(U_full, V_full, W_full, n, m, p, tol=1e-6):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
    
    # Fallback: schoolbook rank 36
    U_sb = np.zeros((6, 36))
    V_sb = np.zeros((18, 36))
    W_sb = np.zeros((12, 36))
    idx = 0
    for a in range(n):
        for b in range(m):
            for d in range(p):
                U_sb[a * m + b, idx] = 1
                V_sb[b * p + d, idx] = 1
                W_sb[d * n + a, idx] = 1
                idx += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }


if __name__ == "__main__":
    construct()
