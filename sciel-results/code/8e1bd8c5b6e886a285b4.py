"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank decomposition for 2x2 * 2x9 matrix multiplication.
    Returns dict with U (4 x rank), V (18 x rank), W (18 x rank) as lists.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # 2, 2, 9
    # U: (n*m) x rank = 4 x rank
    # V: (m*p) x rank = 18 x rank
    # W: (p*n) x rank = 18 x rank
    # The tensor T[i, j, k, i', j', k'] = delta_{i,i'} * delta_{k,k'}
    # We need: T[i,j,k,i',j',k'] = sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k*n+i, r]
    
    # Note: W index is k*n + i, U index is i*m + j, V index is j'*p + k'
    
    # The tensor has entries T[i,j,k,i',j',k'] = 1 if i==i' and k==k', else 0
    # (j and j' are free indices for the multiplication)
    
    # Schoolbook: rank = n*m*p = 36
    # We know that for small matrix multiplication, the rank is often less than n*m*p.
    
    # Let me think about this more carefully.
    # The matrix multiplication tensor M_{n,m,p} has entries:
    # M[i, j, k, i', j', k'] = delta_{i,i'} * delta_{k,k'}
    # where i in [0,n), j in [0,m), k in [0,p), i' in [0,n), j' in [0,m'), k' in [0,p)
    
    # Wait, let me re-read the task. The output factor follows W[k*n+i, r].
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    
    # The decomposition is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form with the given indexing:
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] * A[i',j'] * B[j',k']... 
    
    # Actually, let me think of it as:
    # The bilinear form for each (i,k) output entry is: sum_j A[i,j] * B[j,k]
    # The rank is the minimum number of bilinear forms needed to express all n*p output entries.
    
    # For 2x2 * 2x9, we have 2*9 = 18 output entries.
    # Each output (i,k) requires sum over j of A[i,j]*B[j,k].
    
    # A key insight: for fixed i, the 9 outputs C[i,0..8] = sum_j A[i,j] * B[j,0..8]
    # This is like multiplying a 1x2 vector by a 2x9 matrix.
    
    # For a 1x2 * 2x1 multiplication, rank is 2 (schoolbook) or possibly less?
    # Actually, 1xn * nx1 has rank n (I believe).
    
    # Let me try to find a rank-18 decomposition first (which is better than 36).
    
    # Construction idea:
    # For each (i, k) pair, we need C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # 
    # If we use rank 2 per (i,k) pair, that's 18*2 = 36 (schoolbook).
    # But we can share across k for the same i.
    
    # For fixed i, C[i,0..8] = A[i,0]*[B[0,0],...,B[0,8]] + A[i,1]*[B[1,0],...,B[1,8]]
    # This is a linear combination of two 1x9 vectors with coefficients A[i,0] and A[i,1].
    
    # The rank of the "1x2 * 2x9" matrix multiplication tensor...
    # Let me think of it as a 3-tensor of shape (1*2, 2*9, 9*1) = (2, 18, 9)
    # No wait, the dimensions are:
    # U has 2 rows (for i=0,1 with j=0,1... no, U is (n*m) = 4 rows)
    
    # Let me restart with clearer indexing.
    # U[i*m + j, r] for i in 0..1, j in 0..1  (4 rows)
    # V[j*p + k, r] for j in 0..1, k in 0..8  (18 rows)  
    # W[k*n + i, r] for k in 0..8, i in 0..1  (18 rows)
    
    # The condition: for all i, i', k, k':
    # sum_{j, j'} U[i*m+j, r] V[j'*p+k', r] W[k*n+i', r] ... 
    
    # Hmm, I think the standard tensor decomposition is:
    # T_{i,j,k, i',j',k'} = sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k*n+i', r]
    # should equal delta_{i,i'} * delta_{k,k'}
    
    # Let me verify with schoolbook:
    # For each (i, j, k), create a term r = i*m*p + j*p + k
    # U[i*m+j, r] = 1
    # V[j*p+k, r] = 1  
    # W[k*n+i, r] = 1
    # Then sum_r U[i*m+j, r] V[j'*p+k', r] W[k'*n+i', r]
    # = 1 if there exists r such that:
    #   i*m+j = i*m+j AND j*p+k = j'*p+k' AND k'*n+i' = k*n+i
    # From first: trivially true for the same term
    # From second: j = j' and k = k'
    # From third: k' = k and i' = i
    # So sum = 1 if i=i', k=k', and j=j' (but j is summed over)... 
    
    # Wait, I think I'm confusing myself. Let me look at this differently.
    
    # The matrix multiplication tensor for M_{n,m,p} is defined as:
    # M = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k}
    # where e_{i,j} is the (i,j) entry of the first matrix, etc.
    
    # In the flattened indexing:
    # First factor index: i*m + j (for A[i,j])
    # Second factor index: j*p + k (for B[j,k])
    # Third factor index: k*n + i (for C[i,k])
    
    # So M_{(i*m+j), (j*p+k), (k*n+i)} = 1, and 0 otherwise.
    
    # The rank is the minimum R such that we can write M as sum of R rank-1 tensors.
    # A rank-1 tensor is u ⊗ v ⊗ w, meaning M_{a,b,c} = sum_r u[a,r]*v[b,r]*w[c,r].
    
    # For schoolbook: R = n*m*p = 36
    # u[a, (i,j,k)] = 1 if a = i*m+j
    # v[b, (i,j,k)] = 1 if b = j*p+k
    # w[c, (i,j,k)] = 1 if c = k*n+i
    
    # Now, can we do better?
    
    # Key observation: The tensor has a product structure.
    # M = (sum_i e_{i,i}) ⊗ I_m ⊗ (sum_k e_{k,k}) ... no that's not quite right.
    
    # Actually, let's think of it as:
    # C = A * B where A is 2x2, B is 2x9, C is 2x9.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # For each i (0 or 1), we need to compute a 1x9 vector as a linear combination.
    # The "cost" per i is the rank of the 1x2 * 2x9 multiplication.
    
    # For 1xn * nxm, the rank is known to be n (for n >= 2, I think, or maybe nm?).
    # Actually, for 1x1 * 1x1, rank is 1.
    # For 1x2 * 2x1, rank is 2 (I believe this is optimal).
    # For 1x2 * 2x9, by submultiplicativity or other bounds...
    
    # Let me try to construct a low-rank decomposition directly.
    
    # Idea: Use the fact that for each i, C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # We can precompute some combinations of A entries.
    
    # Let's try rank R = 18 (2 per i, but shared across k somehow).
    
    # Actually, I recall that the rank of matrix multiplication (1, n, m) is n for n >= 2.
    # This is because you need at least n terms to express sum_{j=1}^n a_j * b_j.
    # And n terms suffice (schoolbook).
    
    # So for each i, the rank contribution is at least 2.
    # Total rank at least 2*2 = 4? No, that doesn't account for the k dimension.
    
    # Wait, for 1x2 * 2x9, the tensor is (2, 18, 9) in flattened form.
    # The rank of this tensor... 
    
    # Let me think about it as a 3-tensor T of shape (2, 18, 9).
    # T[j, j'*9+k, k] = delta_{j, j'} ... no.
    
    # I think I need to be more careful. Let me just try to find a good decomposition
    # through search or known results.
    
    # Known result: The rank of M_{2,2,2} is 7 (Strassen).
    # The rank of M_{2,2,n} for various n...
    
    # For M_{2,2,9}, one approach is to use Strassen-like methods.
    
    # Let me try a direct construction.
    
    # We need: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1 and k=0..8
    
    # Let's define some intermediate quantities:
    # P0 = A[0,0] * B[0,0]  (this is part of C[0,0])
    # ...
    
    # Actually, let me try a different approach. Use the fact that:
    # (A00 + A01) * (B0k - B1k) + A00*B0k + A01*B1k = A00*B0k + A01*B0k - A00*B1k + A01*B1k
    # = (A00 + A01)*B0k - A00*B1k + A01*B1k... this is getting complicated.
    
    # Let me just try to find the minimum rank through a systematic approach.
    
    # For the 1x2 * 2x9 case:
    # We have 2 input variables for A (A0, A1) and 18 for B (B00..B08, B10..B18), 
    # and 9 outputs (C0..C8) where Ck = A0*B0k + A1*B1k.
    
    # The rank is at least 2 (since we need at least 2 bilinear terms for a generic 1x2*2x1).
    # But with 9 outputs, the rank might be higher.
    
    # Actually, I think the rank of M_{1,2,9} is 9 (one term per output, but each term can handle both A entries).
    # No wait, that's not right either.
    
    # Let me try: for each k, use 2 terms:
    # Term k,0: u=1 for A0, v=1 for B0k, w=1 for Ck
    # Term k,1: u=1 for A1, v=1 for B1k, w=1 for Ck
    # This gives rank 18 for the 1x2*2x9 part, and 36 total.
    
    # Can we share? For a fixed k, Ck = A0*B0k + A1*B1k.
    # This is a bilinear form in (A0, A1) and (B0k, B1k) with output Ck.
    # The rank of this 2x2x1 tensor is 2 (you need 2 terms).
    
    # So for each k, minimum rank is 2, total minimum is 18 for the 1x2*2x9 part.
    # For 2x2*2x9, total minimum is 36? That can't be right since Strassen reduces 2x2*2x2 from 8 to 7.
    
    # Ah, I see. The key is that Strassen shares terms across different output entries.
    
    # For 2x2*2x9, let me try to use Strassen's method for the 2x2 block and extend.
    
    # Strassen for 2x2*2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 2x2*2x9, we can apply a similar idea.
    
    # Let me try to construct a decomposition with rank less than 36.
    
    # Actually, let me just implement the schoolbook and return it, since I'm not confident
    # in finding a better one quickly. The score of 1.0 means schoolbook is the baseline.
    
    # Wait, the task says "score is the rank" and "1.0 equals the seed program". So if I
    # find rank < 36, my score > 1.0.
    
    # Let me try to find a rank-18 or rank-24 decomposition.
    
    # Idea: For each i, compute C[i,0..8] = A[i,0]*B[0,0..8] + A[i,1]*B[1,0..8]
    # Use 2 terms per i, but make the terms depend on all 9 k values.
    
    # Term (i, 0): u[A[i,0]] = 1, v[B[j,k]] for all j,k, w[C[i,k]] for all k
    # But this would require the v and w to be non-zero in many places.
    
    # Let me try: for i=0:
    # Term 0: u[0,0]=1, u[0,1]=0, u[1,0]=0, u[1,1]=0  (only A[0,0])
    #         v[0,k]=1 for all k, v[1,k]=0  (only B[0,k])
    #         w[k,0]=1 for all k, w[k,1]=0  (contributes to C[0,k])
    # 
    # Term 1: u[0,0]=0, u[0,1]=1, u[1,0]=0, u[1,1]=0  (only A[0,1])
    #         v[0,k]=0, v[1,k]=1 for all k  (only B[1,k])
    #         w[k,0]=1 for all k, w[k,1]=0  (contributes to C[0,k])
    
    # This is just schoolbook for i=0, using 2 terms (one for each j).
    # Similarly for i=1, another 2 terms. Total 4 terms? That seems too few.
    
    # Oh wait, I see the issue. Each term can only handle one (j) at a time.
    # So for i=0, I need 2 terms (for j=0 and j=1), each handling all 9 k values.
    # For i=1, another 2 terms. Total 4 terms.
    
    # But wait, that would mean rank 4 for the entire 2x2*2x9 multiplication!
    # That can't be right. Let me check.
    
    # Term 0: U[0,0]=1 (A[0,0]), V[0*9+k, 0]=1 for k=0..8 (B[0,k]), W[k*2+0, 0]=1 for k=0..8 (C[0,k])
    # Term 1: U[0,1]=1 (A[0,1]), V[1*9+k, 1]=1 for k=0..8 (B[1,k]), W[k*2+0, 1]=1 for k=0..8 (C[0,k])
    # Term 2: U[1,0]=1 (A[1,0]), V[0*9+k, 2]=1 for k=0..8 (B[0,k]), W[k*2+1, 2]=1 for k=0..8 (C[1,k])
    # Term 3: U[1,1]=1 (A[1,1]), V[1*9+k, 3]=1 for k=0..8 (B[1,k]), W[k*2+1, 3]=1 for k=0..8 (C[1,k])
    
    # Now check: for output C[i,k], the contribution is:
    # sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k*n+i', r] * ... 
    
    # I think I'm confusing the tensor contraction. Let me think again.
    
    # The tensor M has entries M[a, b, c] where:
    # a = i*m + j (index into A)
    # b = j'*p + k' (index into B)
    # c = k*n + i (index into C)
    
    # M[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
    # More precisely: M[i*m+j, j*p+k, k*n+i] = 1.
    
    # For a rank-1 decomposition: M[a,b,c] = sum_r u[a,r] * v[b,r] * w[c,r]
    
    # For term 0: u[0,0]=1, u[other,0]=0
    #             v[0*9+k, 0]=1 for k=0..8, v[other,0]=0
    #             w[k*2+0, 0]=1 for k=0..8, w[other,0]=0
    
    # Then (u ⊗ v ⊗ w)[a,b,c] = u[a,0]*v[b,0]*w[c,0]
    # This is 1 if a=0 (i=0,j=0), b is in {0*9+0, ..., 0*9+8} = {0,...,8} (j'=0, k'=0..8),
    # and c is in {0*2+0, ..., 8*2+0} = {0,2,4,...,16} (k=0..8, i=0).
    
    # So this term contributes 1 to M[0, b, c] for b in 0..8 and c in {0,2,4,...,16}.
    # But M[0, b, c] should be 1 only if b = j*p+k and c = k*n+i where i=0,j=0.
    # So b should be 0*p+k = k for k=0..8, and c should be k*2+0 = 2k for k=0..8.
    # So M[0, k, 2k] = 1 for k=0..8.
    
    # The term 0 gives 1 for M[0, k, 2k] for k=0..8. 
    
    # But we also need M[0, k, 2k'] for k' != k to be 0. Term 0 gives 0 there (since c=2k' != 2k).
    
    # So term 0 correctly handles the (i=0, j=0) part!
    
    # Similarly, term 1 handles (i=0, j=1): M[1, 9+k, 2k] = 1 for k=0..8.
    # Term 2 handles (i=1, j=0): M[2, k, 2k+1] = 1 for k=0..8.
    # Term 3 handles (i=1, j=1): M[3, 9+k, 2k+1] = 1 for k=0..8.
    
    # So the total sum of 4 terms gives exactly the matrix multiplication tensor!
    
    # Wait, that means the rank is 4?! That seems way too low. Let me double-check.
    
    # Hmm, but this doesn't account for the fact that the multiplication should give:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor M encodes this. If my 4-term decomposition is correct, then rank is 4.
    # But I know that 2x2*2x2 has rank 7, which is more than 4. So something is wrong.
    
    # Let me re-examine. For 2x2*2x2, the tensor has entries M[i*m+j, j*p+k, k*n+i] = 1.
    # With n=m=p=2:
    # M[0,0,0]=1 (i=0,j=0,k=0)
    # M[0,1,2]=1 (i=0,j=0,k=1) -> wait, k*n+i = 1*2+0 = 2
    # M[1,0,1]=1 (i=0,j=1,k=0) -> k*n+i = 0*2+0 = 0? No, i=0,j=1,k=0: M[0*2+1, 1*2+0, 0*2+0] = M[1,2,0]=1
    # ...
    
    # I think I made an error in the indexing. Let me be very careful.
    
    # A is n x m = 2x2, so A[i,j] for i in 0..1, j in 0..1.
    # B is m x p = 2x2, so B[j,k] for j in 0..1, k in 0..1.
    # C is n x p = 2x2, so C[i,k] for i in 0..1, k in 0..1.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor M is a 3-tensor with indices (i,j), (j,k), (i,k).
    # Flattened:
    # First index: i*m + j (0..3)
    # Second index: j*p + k (0..3)
    # Third index: k*n + i (0..3)
    
    # M[i*m+j, j*p+k, k*n+i] = 1
    
    # For n=m=p=2:
    # (i,j,k) = (0,0,0): M[0, 0, 0] = 1
    # (i,j,k) = (0,0,1): M[0, 1, 2] = 1  (since k*n+i = 1*2+0 = 2)
    # (i,j,k) = (0,1,0): M[1, 2, 0] = 1  (since j*p+k = 1*2+0 = 2, k*n+i = 0*2+0 = 0)
    # (i,j,k) = (0,1,1): M[1, 3, 2] = 1
    # (i,j,k) = (1,0,0): M[2, 0, 1] = 1  (k*n+i = 0*2+1 = 1)
    # (i,j,k) = (1,0,1): M[2, 1, 3] = 1
    # (i,j,k) = (1,1,0): M[3, 2, 1] = 1
    # (i,j,k) = (1,1,1): M[3, 3, 3] = 1
    
    # Now, my 4-term construction for n=m=p=2:
    # Term 0: u[0]=1, v[0]=1, v[1]=1 (B[0,0] and B[0,1]), w[0]=1, w[2]=1 (C[0,0] and C[0,1])
    #   Contributes to M[0, 0, 0], M[0, 1, 2]
    # Term 1: u[1]=1, v[2]=1, v[3]=1, w[0]=1, w[2]=1
    #   Contributes to M[1, 2, 0], M[1, 3, 2]
    # Term 2: u[2]=1, v[0]=1, v[1]=1, w[1]=1, w[3]=1
    #   Contributes to M[2, 0, 1], M[2, 1, 3]
    # Term 3: u[3]=1, v[2]=1, v[3]=1, w[1]=1, w[3]=1
    #   Contributes to M[3, 2, 1], M[3, 3, 3]
    
    # This covers all 8 entries of M! So rank <= 4 for 2x2*2x2?
    # But Strassen says rank is 7, and I know it's at least 7.
    
    # There must be an error in my understanding. Let me check the definition again.
    
    # Oh! I think the issue is that the tensor M should satisfy:
    # M[a, b, c] = 1 if a corresponds to A[i,j], b to B[j,k], and c to C[i,k]
    # with the SAME j and SAME k and SAME i.
    
    # But in my construction, term 0 contributes to M[0, 0, 0] and M[0, 1, 2].
    # M[0, 1, 2]: a=0 means i=0,j=0. b=1 means j'=0,k'=1. c=2 means k=1,i'=0.
    # For this to be part of the multiplication, we need j=j' (0=0 ✓), k=k' (1=1 ✓), i=i' (0=0 ✓).
    # So M[0, 1, 2] = 1 is correct.
    
    # So my construction seems correct. But this contradicts the known rank of 7.
    
    # I must be misunderstanding the tensor rank or the problem formulation.
    
    # Let me look at this from a different angle. The bilinear complexity of matrix multiplication
    # (2,2,2) is 7, meaning you need 7 bilinear forms. Each bilinear form is of the form
    # u · A · v where u and v are fixed vectors.
    
    # In the tensor decomposition, each rank-1 term u ⊗ v ⊗ w corresponds to a bilinear form
    # (sum_a u[a] A[a]) * (sum_b v[b] B[b]) which contributes w[c] to C[c].
    
    # So for term 0 in my construction:
    # sum_a u[a] A[a] = A[0,0]
    # sum_b v[b] B[b] = B[0,0] + B[0,1]
    # Contribution to C: w[0]*(A[0,0]*(B[0,0]+B[0,1])) + w[2]*(A[0,0]*(B[0,0]+B[0,1]))
    # = (A[0,0]*B[0,0] + A[0,0]*B[0,1]) * [1 for C[0,0], 1 for C[0,1]]
    
    # But the correct contribution to C[0,0] should be A[0,0]*B[0,0] + A[0,1]*B[1,0].
    # And to C[0,1] should be A[0,0]*B[0,1] + A[0,1]*B[1,1].
    
    # My term 0 gives A[0,0]*B[0,0] + A[0,0]*B[0,1] to C[0,0] and the same to C[0,1].
    # That's wrong! C[0,0] should get A[0,0]*B[0,0], not A[0,0]*B[0,1].
    
    # Ah, I see the issue. The tensor M[a,b,c] = 1 means that the product of the a-th entry of A
    # and the b-th entry of B contributes to the c-th entry of C. But the contribution is just
    # A[a] * B[b] (with coefficient 1).
    
    # So M[0, 1, 0] should be 1 if A[0,0] * B[0,1] contributes to C[0,0].
    # But A[0,0] * B[0,1] contributes to C[0,1], not C[0,0]!
    
    # So M[0, 1, 0] should be 0, and M[0, 1, 2] should be 1.
    
    # Let me recompute:
    # A[0,0] * B[0,0] -> C[0,0]: M[0, 0, 0] = 1
    # A[0,0] * B[0,1] -> C[0,1]: M[0, 1, 2] = 1  (C[0,1] has index k=1, i=0, so k*n+i = 2)
    # A[0,1] * B[1,0] -> C[0,0]: M[1, 2, 0] = 1  (B[1,0] has index j=1,k=0, so j*p+k = 2)
    # A[0,1] * B[1,1] -> C[0,1]: M[1, 3, 2] = 1
    # A[1,0] * B[0,0] -> C[1,0]: M[2, 0, 1] = 1  (C[1,0] has index k=0, i=1, so k*n+i = 1)
    # A[1,0] * B[0,1] -> C[1,1]: M[2, 1, 3] = 1
    # A[1,1] * B[1,0] -> C[1,0]: M[3, 2, 1] = 1
    # A[1,1] * B[1,1] -> C[1,1]: M[3, 3, 3] = 1
    
    # So the tensor is as I had. And my 4-term decomposition seems to work.
    
    # But this contradicts the known result. Let me check if my decomposition actually works
    # by computing the resulting C.
    
    # C[i,k] = sum_{a,b} M[a,b,c] * A[a] * B[b] where c = k*n+i
    
    # For C[0,0] (c=0):
    # M[0,0,0]*A[0]*B[0] + M[1,2,0]*A[1]*B[2] + M[2,0,0]*A[2]*B[0] + ...
    # = 1*A[0]*B[0] + 1*A[1]*B[2] + 0 + ...
    # = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    
    # For C[0,1] (c=2):
    # M[0,1,2]*A[0]*B[1] + M[1,3,2]*A[1]*B[3] + ...
    # = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    
    # This is correct! So the tensor decomposition with 4 terms works for 2x2*2x2?
    
    # I'm confused now. Let me just implement this and see if it's valid.
    
    # For 2x2*2x9:
    # Term 0: A[0,0] * (sum_k B[0,k]) -> C[0,k] for all k
    #   u[0]=1, v[k]=1 for k=0..8, w[2k]=1 for k=0..8
    # Term 1: A[0,1] * (sum_k B[1,k]) -> C[0,k] for all k
    #   u[1]=1, v[9+k]=1 for k=0..8, w[2k]=1 for k=0..8
    # Term 2: A[1,0] * (sum_k B[0,k]) -> C[1,k] for all k
    #   u[2]=1, v[k]=1 for k=0..8, w[2k+1]=1 for k=0..8
    # Term 3: A[1,1] * (sum_k B[1,k]) -> C[1,k] for all k
    #   u[3]=1, v[9+k]=1 for k=0..8, w[2k+1]=1 for k=0..8
    
    # Let me verify C[0,0] (c=0):
    # Terms with w[c]=1: term 0 (w[0]=1), term 1 (w[0]=1)
    # Term 0: u[0]*v[0]*w[0] = 1*1*1 = 1, so contributes A[0,0]*B[0,0]
    # Term 1: u[1]*v[9]*w[0] = 1*1*1 = 1, so contributes A[0,1]*B[1,0]
    # Total: A[0,0]*B[0,0] + A[0,1]*B[1,0] ✓
    
    # C[0,3] (c=6):
    # Terms with w[6]=1: term 0 (w[6]=1), term 1 (w[6]=1)
    # Term 0: A[0,0]*B[0,3]
    # Term 1: A[0,1]*B[1,3]
    # Total: A[0,0]*B[0,3] + A[0,1]*B[1,3] ✓
    
    # This looks correct! So the rank is 4 for 2x2*2x9?
    
    # But wait, this seems too good to be true. Let me check if this is actually a valid
    # tensor decomposition by checking the rank more carefully.
    
    # The issue might be that I'm not correctly understanding what the tensor represents.
    
    # Actually, I think the problem is that the tensor M is not what I think it is.
    # Let me re-read the task.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 2 matrix by a 2 by 9 matrix."
    
    # The tensor for matrix multiplication (2,2,9) is a 3-tensor T of shape (2*2, 2*9, 9*2) = (4, 18, 18).
    # T[a, b, c] = 1 if a = i*2+j, b = j*9+k, c = k*2+i for some i,j,k, else 0.
    
    # My 4-term decomposition:
    # Term 0: u[0]=1, v[0..8]=1, w[0,2,4,...,16]=1
    # Term 1: u[1]=1, v[9..17]=1, w[0,2,4,...,16]=1
    # Term 2: u[2]=1, v[0..8]=1, w[1,3,5,...,17]=1
    # Term 3: u[3]=1, v[9..17]=1, w[1,3,5,...,17]=1
    
    # Let me check T[0, 1, 6]:
    # a=0: i=0,j=0
    # b=1: j'=0,k'=1
    # c=6: k=3,i'=0
    # For T[0,1,6] to be 1, we need j=j' (0=0 ✓), k=k' (1=3 ✗). So T[0,1,6] = 0.
    
    # My decomposition: term 0 gives u[0]*v[1]*w[6] = 1*1*1 = 1.
    # But T[0,1,6] should be 0!
    
    # So my decomposition is WRONG. The issue is that term 0 contributes to T[0,1,6] 
    # (A[0,0]*B[0,1] contributing to C[0,3]), but it should only contribute to T[0,1,2]
    # (A[0,0]*B[0,1] contributing to C[0,1]).
    
    # I see the error now. The w vector should not have 1 for all even indices. It should
    # have 1 only for the specific c that corresponds to the correct (k,i).
    
    # So for term 0 (A[0,0] * B[0,k]), the contribution to C should be C[0,k] for the SAME k.
    # So w should have w[k*2+0] = 1 for the specific k, not for all k.
    
    # This means I can't share across k in a single term. I need a separate term for each k.
    
    # So the minimum rank is at least 2*9 = 18 (for each i, and for each k, one term per j).
    # Actually, for each (i,k), I need 2 terms (one for each j), so total 2*2*9 = 36.
    
    # But Strassen shows that for 2x2*2x2, we can do better than 8. So there must be a way
    # to share across k.
    
    # The key insight from Strassen is to use linear combinations of A and B entries.
    
    # For now, let me just return the schoolbook decomposition (rank 36) since I'm not
    # confident in finding a better one within the time limit.
    
    n, m, p = DIMENSIONS
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
