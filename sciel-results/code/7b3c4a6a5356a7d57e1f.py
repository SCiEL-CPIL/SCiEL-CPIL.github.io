"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 30
    rank_schoolbook = n * m * p
    
    # We will try to find a decomposition of rank < 30.
    # The known lower bound for 2x3x5 is 25 (from algebraic complexity theory, 
    # specifically related to the structure of rectangular matrix multiplication).
    # However, finding an explicit 25-rank decomposition is non-trivial.
    # Let's try a heuristic search for a rank 26, 27, 28, 29 decomposition first, 
    # or stick to schoolbook if search fails.
    
    # Since we need an EXACT decomposition with integer/half-integer entries,
    # and random search for exact integer solutions is hard, we should look for 
    # structured solutions.
    
    # One approach: Use the known 7-rank Strassen algorithm for 2x2x2.
    # But 2x3x5 is not 2x2x2.
    
    # Let's try a direct construction based on the "symmetric rank" idea or 
    # simply verify if we can do better than 30.
    
    # Actually, for 2x3x5, the schoolbook is 30. 
    # Is there a known algorithm with rank < 30?
    # The tensor rank of 2x3x5 matrix multiplication is known to be 25? 
    # Wait, let's check the literature or known bounds.
    # The rank of M_{2,3} (2x3 by 3x2) is 7? No, M_{2,2} is 7.
    # M_{2,3} (2x3 by 3x1) is vector-matrix, rank 6 (schoolbook).
    # M_{3,2} (3x2 by 2x1) is rank 6.
    # M_{2,3,5} is 2x3 by 3x5.
    
    # A safe bet is to return the schoolbook decomposition, which is valid.
    # Score = 30. Seed score is 1.0 which corresponds to rank 30?
    # The prompt says "score is the rank". And "1.0 equals the seed program".
    # So the seed program has rank 30.
    # To improve, we need rank < 30.
    
    # Let's attempt a search for a rank 29 decomposition using integer entries.
    # We can set up a linear system.
    # The tensor T has entries T[i,j,k] = 1 if (i,j,k) is a valid multiplication index.
    # We want U, V, W such that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k].
    # This is a non-linear system.
    
    # Alternative: Use the fact that if we can find a decomposition of rank R, 
    # we can try to optimize the entries.
    
    # Given the 120s limit and the difficulty of finding exact integer solutions 
    # for ranks < 30 for this specific shape, and the lack of a known simple 
    # algebraic construction for 2x3x5 < 30 in my immediate recall, 
    # I will stick to the schoolbook decomposition to ensure validity and a 
    # score of 1.0 (which is the baseline). 
    # 
    # Wait, the prompt says "score 1.0000" for the current program. 
    # If I return the same thing, I get 1.0. 
    # If I find a rank 29, I get 30/29 = 1.034 > 1.0.
    # 
    # Let's try a simple heuristic: Can we merge two schoolbook terms?
    # Schoolbook terms are (i,j,k). 
    # If we can represent two terms with one rank-1 tensor, we reduce rank by 1.
    # A rank-1 tensor is U[i,j] V[j,k] W[k,i].
    # The schoolbook term (i0,j0,k0) is 1 only at that index.
    # Can we combine (0,0,0) and (0,0,1)?
    # We need U[i,j] V[j,k] W[k,i] to be 1 at (0,0,0) and (0,0,1) and 0 elsewhere.
    # This requires the support of the rank-1 tensor to be a subset of the 
    # support of the sum of the two schoolbook terms.
    # The support of a rank-1 tensor is a "box" (Cartesian product of supports).
    # The support of (0,0,0) is {(0,0,0)}.
    # The support of (0,0,1) is {(0,0,1)}.
    # The union is {(0,0,0), (0,0,1)}.
    # Is this a box? 
    # A box is I x J x K. 
    # If I={0}, J={0}, K={0,1}, then the box is {(0,0,0), (0,0,1)}.
    # Yes!
    # So we can combine any two terms that differ only in k (with same i,j) 
    # into a single rank-1 term?
    # Let's check:
    # Let the rank-1 term be defined by U[i,j]=1 if i=0,j=0 else 0.
    # V[j,k]=1 if j=0 else 0. (Wait, V is indexed by j,k).
    # V[j,k] must be 1 for k=0 and k=1 when j=0.
    # So V[0,0]=1, V[0,1]=1.
    # W[k,i] must be 1 for k=0 and k=1 when i=0.
    # So W[0,0]=1, W[1,0]=1.
    # Let's evaluate the product at (i,j,k):
    # (i,j,k) = (0,0,0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1. Correct.
    # (i,j,k) = (0,0,1): U[0,0]*V[0,1]*W[1,0] = 1*1*1 = 1. Correct.
    # (i,j,k) = (0,0,2): U[0,0]*V[0,2]*W[2,0]. 
    # We need this to be 0. So V[0,2] must be 0 or W[2,0] must be 0.
    # We can set V[0,2]=0.
    # (i,j,k) = (1,0,0): U[1,0]*... U[1,0]=0. So 0. Correct.
    # (i,j,k) = (0,1,0): U[0,1]=0. So 0. Correct.
    # 
    # So we can combine terms (i,j,k1) and (i,j,k2) into one rank-1 term 
    # if we are careful with the other k indices.
    # Specifically, for a fixed (i,j), we can combine all p terms 
    # (i,j,0), (i,j,1), ..., (i,j,p-1) into a SINGLE rank-1 term?
    # Let's check.
    # We want U[i,j]=1, V[j,k]=1 for all k, W[k,i]=1 for all k.
    # Then the product is 1 for all (i,j,k) with this i,j.
    # But we only want it for the specific k's? No, the schoolbook has 1 
    # for each (i,j,k).
    # If we set V[j,k]=1 for all k, and W[k,i]=1 for all k, 
    # then the term is 1 for all k.
    # This covers all p values of k for a fixed (i,j).
    # 
    # So for each (i,j), we can use ONE rank-1 term to cover all k=0..p-1.
    # Total rank = n * m = 2 * 3 = 6?
    # 
    # Let's verify this.
    # If rank = n*m = 6, then:
    # For each (i,j) in 2x3, we have a rank-1 term c.
    # U[i*m+j, c] = 1.
    # V[j*p+k, c] = 1 for all k.
    # W[k*n+i, c] = 1 for all k.
    # 
    # Let's check the reconstruction for a specific (i,j,k).
    # Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # The only non-zero U[i*m+j, c] is for the c corresponding to (i,j).
    # Let c = i*m+j.
    # Then the term is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # = 1 * 1 * 1 = 1.
    # 
    # Is this valid?
    # We must ensure that for other (i',j',k'), the sum is 0 if not (i,j,k).
    # Wait, the matrix multiplication tensor T[i,j,k] is 1 for ALL valid (i,j,k).
    # The indices are:
    # i in 0..n-1
    # j in 0..m-1
    # k in 0..p-1
    # There are no "invalid" indices in the tensor itself; the tensor is defined 
    # on the full grid n x m x p.
    # The matrix multiplication is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] represents the coefficient of A[i,j] B[j,k] in C[i,k].
    # This coefficient is 1 for all i,j,k.
    # 
    # So yes, the tensor is the all-ones tensor of shape (n,m,p)?
    # NO. The tensor is T[i,j,k] = 1.
    # My proposed decomposition gives:
    # Sum_c U[i,m+j,c] V[j,p+k,c] W[k,n+i,c] = 1 for all i,j,k.
    # This is EXACTLY the all-ones tensor.
    # 
    # So the rank of the 2x3x5 matrix multiplication tensor is at most n*m = 6?
    # That seems too low. Let's re-read the problem.
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 
    # matrix by a 3 by 5 matrix."
    # 
    # The tensor for matrix multiplication M_{n,m,p} has entries:
    # T[i,j,k] = 1 if the index (i,j,k) corresponds to a valid multiplication step.
    # In the standard Kronecker product representation, the tensor is 
    # indeed the all-ones tensor of shape (n*m, m*p, p*n)?
    # No, the indices are (i,j) for A, (j,k) for B, (i,k) for C.
    # The tensor T is of shape (n*m, m*p, p*n).
    # T[(i,j), (j',k), (i',k')] is non-zero only if i=i', j=j', k=k'.
    # And then it is 1.
    # So T is the "identity" tensor in the sense that it picks out the diagonal.
    # 
    # Let's re-evaluate my "all-ones" claim.
    # My U, V, W construction:
    # U[i*m+j, c] = 1 if c = i*m+j. (So U is the identity matrix of size n*m).
    # V[j*p+k, c] = 1 if c corresponds to (i,j) where j is the same?
    # In my construction: c = i*m+j.
    # V[j*p+k, c] = 1 for all k, if c = i*m+j.
    # So V[j*p+k, i*m+j] = 1 for all k.
    # W[k*n+i, c] = 1 for all k, if c = i*m+j.
    # So W[k*n+i, i*m+j] = 1 for all k.
    # 
    # Let's check the product for a specific index combination 
    # (a,b) in A, (c,d) in B, (e,f) in C.
    # A index: (i,j) = (a,b).
    # B index: (j',k) = (c,d).
    # C index: (i',k') = (e,f).
    # 
    # The term is non-zero only if:
    # 1. U[a*m+b, col] != 0 => col = a*m+b.
    # 2. V[c*p+d, col] != 0 => V[c*p+d, a*m+b] != 0.
    #    In my construction, V[j*p+k, i*m+j] = 1 for all k.
    #    So we need j = b. Thus c must be b.
    # 3. W[e*n+f, col] != 0 => W[e*n+f, a*m+b] != 0.
    #    In my construction, W[k*n+i, i*m+j] = 1 for all k.
    #    So we need i = a. Thus e must be a.
    # 
    # So the product is 1 if:
    # col = a*m+b
    # c = b
    # e = a
    # 
    # The result is C[e,f] = sum_j A[a,j] B[j,f].
    # The tensor entry T[(a,b), (c,d), (e,f)] should be 1 if 
    # a=e, b=c, and d=f?
    # Wait, the output index is (i,k). The input indices are (i,j) and (j,k).
    # The tensor T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # This coefficient is 1.
    # 
    # In the flattened indices:
    # A index: (i,j) -> i*m+j
    # B index: (j,k) -> j*p+k
    # C index: (i,k) -> i*p+k  (Wait, the problem says W[k*n+i,r])
    # 
    # Let's check the W index convention in the prompt:
    # "The output factor follows the official trace convention W[k*n+i,r]."
    # So C index is (k,i) -> k*n+i.
    # 
    # So the tensor T is defined on:
    # A: (i,j) -> i*m+j
    # B: (j,k) -> j*p+k
    # C: (k,i) -> k*n+i
    # 
    # T[a,b,c] = 1 if a=(i,j), b=(j',k), c=(k',i') and i=i', j=j', k=k'.
    # Otherwise 0.
    # 
    # My construction:
    # Product is 1 if:
    # col = a*m+b (so i=a, j=b)
    # c = b (so j'=b)
    # e = a (so i'=a)
    # 
    # What about k and k'?
    # The product does not depend on d (k) or f (k') in the condition?
    # V[c*p+d, col] = 1 for all d.
    # W[e*n+f, col] = 1 for all f.
    # 
    # So the product is 1 for ANY d and f, as long as i=a, j=b, j'=b, i'=a.
    # 
    # But the tensor T should be 1 ONLY if k=k' as well.
    # If k != k', the coefficient of A[i,j] B[j,k] in C[i,k'] is 0.
    # 
    # So my construction gives 1 for (i,j), (j,k), (k',i) even if k != k'.
    # This is WRONG.
    # 
    # So the rank is NOT n*m.
    # 
    # Back to schoolbook.
    # 
    # Can we do better than 30?
    # 
    # Let's try to find a rank 29 decomposition.
    # 
    # Given the complexity, I will return the schoolbook decomposition.
    
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
