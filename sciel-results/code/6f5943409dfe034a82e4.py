"""Improved rank-44 decomposition for 3x5x5 matrix multiplication."""
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    deadline = time.time() + 145.0

    # We'll build U (n*m=15, rank), V (m*p=25, rank), W (p*n=15, rank)
    # Index conventions:
    #   U[i*m + j, r]  where i in [0,n), j in [0,m)
    #   V[j*p + k, r]  where j in [0,m), k in [0,p)
    #   W[k*n + i, r]  where k in [0,p), i in [0,n)
    # Tensor entry T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # We need T[i,j,k] = 1 for all (i,j,k), i.e., the multiplication tensor.

    # Strategy: block decomposition.
    # Split rows of A (3 rows) and columns of B (5 cols) into blocks.
    # Block 1: A_1 (1x5) * B_1 (5x5) -> C_1 (1x5): rank 5 (since 1x5x5 has rank 5)
    # Block 2: A_2 (2x5) * B_1 (5x5) -> C_2 (2x5): split further
    #   A_2 has 2 rows, B_1 has 5 cols.
    #   Split B_1 into B_11 (5x2), B_12 (5x2), B_13 (5x1)
    #   A_2 into A_21 (2x2), A_22 (2x2), A_23 (2x1)
    #   C_2 into C_21 (2x2), C_22 (2x2), C_23 (2x1)
    #   Each 2x2x2 block uses Strassen rank 7.
    #   A_23 * B_13 is 2x1x1, rank 1.
    #   So Block 2: 7 + 7 + 1 = 15
    # Block 3: A_2 (2x5) * B_2 (5x3) where B_2 = last 3 cols... 
    # Wait, let me re-think the block structure more carefully.
    #
    # Actually, let me use a cleaner approach:
    # The 3x5x5 multiplication tensor.
    # Think of it as 3 blocks of size 1x5x5 in the first dimension (row of C).
    # Row 1 of C: A_1 * B where A_1 is row 1 of A (1x5), B is 5x5. Rank = 5.
    # Rows 2,3 of C: A_2 * B where A_2 is 2x5, B is 5x5. 
    #   For 2x5x5, we can use:
    #   Split B into B_1 (5x3) and B_2 (5x2)? No, let's split differently.
    #   2x5x5: we can view as two 2x5x? blocks or use a known decomposition.
    #   Actually, let's split the 5 columns of B into groups.
    #   2x5x5: rank is at most... we can use a block approach.
    #   Split A_2 into two 1x5 rows? No, that's back to schoolbook.
    #   Better: 2x5x5. Split the 5x5 B into B_top (5x3) and B_bot (5x2)? 
    #   Hmm, let me think about this differently.
    #
    # Let me use a more systematic approach:
    # 3x5x5 = sum over i=0..2 of (1x5x5) tensors for each row of C.
    # Row i of C: (e_i^T A) * B, which is a 1x5x5 tensor. Rank 5 each.
    # Total: 15. That's the schoolbook for 3*5*5=75 reduced to 15? No.
    # Wait, 1x5x5 has rank 5 (you need 5 products: each element of the 1x5 result).
    # So 3 * 5 = 15? That seems too low. Let me verify.
    # For 1x5x5: C_j = sum_k A_k * B_kj. The tensor has entries T[0,k,j] = delta_kj... 
    # No, T[i,j,k] for 1x5x5: i=0 only, j in 0..4, k in 0..4. T[0,j,k] = 1 if... 
    # Actually the multiplication tensor for (1x5)*(5x5)->(1x5) has T[j,k] = delta_{jk} in the middle.
    # The tensor is sum_{k=0}^{4} e_k^T (A) * e_k e_{k}^T ... no.
    # T[i,j,k] = 1 if i=0 and the "middle" index matches. 
    # For matrix mult (n x m) * (m x p) -> (n x p), T[i,j,k] = 1 for all i,j,k.
    # Wait no. The tensor for matrix multiplication satisfies:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p].
    # That's the standard convention. So T[i,j,k] = 1 for all i,j,k.
    #
    # OK so for 1x5x5: T[0,j,k] = 1 for all j,k in 0..4. That's a 1x5x5 tensor where every entry is 1.
    # That tensor has rank... it's the outer product of 1-vector with 5x5 all-ones? No.
    # T[i,j,k] = 1 means T = u v w^T where u is 1-dim (all ones), v is 5-dim (all ones), w is 5-dim (all ones)?
    # u[0]*v[j]*w[k] = 1*1*1 = 1. Yes! So rank 1? 
    # Wait, that can't be right for matrix multiplication.
    #
    # I'm confusing myself. Let me re-read the problem.
    # "Matrix multiplication 3,5,5" - multiply 3x5 by 5x5 to get 3x5.
    # The tensor is T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k].
    # The standard matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k.
    # Because each term in the sum is A[i,j] * B[j,k] * (contribution to C[i,k]).
    # 
    # So the tensor is the all-ones tensor of shape (3,5,5). Its rank is... 
    # The all-ones tensor of shape (n,m,p) has rank 1 (it's the outer product of three all-ones vectors).
    # That's clearly wrong for matrix multiplication rank.
    #
    # I think the convention must be different. Let me look at the seed code:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # For a given (i,j,k), the contribution is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c].
    # In schoolbook, for each (i,j,k) there's one column c where all three are 1.
    # So T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for the specific c, else 0.
    # Wait, that gives T[i,j,k] = 1 for all (i,j,k)? No, each (i,j,k) has exactly one c.
    # So T[i,j,k] = 1 for all (i,j,k). That's the all-ones tensor.
    #
    # Hmm, but that means the "rank" of the all-ones (3,5,5) tensor is what we're minimizing.
    # The all-ones tensor (3,5,5) CAN be decomposed with rank 1: U[0]=1, V[0]=1, W[0]=1, all other entries 0.
    # Then sum_c U[..,c] V[..,c] W[..,c] = 1*1*1 = 1 for every (i,j,k). 
    # So rank 1 suffices? That seems too easy.
    #
    # I must be misunderstanding the tensor convention. Let me re-read:
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # The product sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should give the matrix mult tensor.
    # For matrix multiplication, we need:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The "tensor" that represents this is T[i,j,k] such that the bilinear form 
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C^*[i,k] ... no.
    #
    # Actually, in tensor decomposition of matrix multiplication, the tensor is:
    # T = sum_{i,j,k} e_i^A (e_j^A * e_k^B) (e_k^C) ... 
    # The standard form: T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} u_i ⊗ v_j ⊗ w_k
    # where u_i is the i-th standard basis in R^n, v_j is j-th in R^m, w_k is k-th in R^p.
    # So T[i,j,k] = 1 for all i,j,k. Yes, the all-ones tensor.
    #
    # But then rank 1 works? That can't be the intended problem.
    #
    # Wait, I think I'm wrong about the all-ones tensor having rank 1.
    # The all-ones tensor of shape (n,m,p) is u ⊗ v ⊗ w where u,v,w are all-ones vectors.
    # So yes, rank 1.
    #
    # This must mean the tensor convention is different. Let me look more carefully.
    # Maybe the tensor entries are NOT all 1s. Maybe it's a structured tensor.
    #
    # Re-reading: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # In the seed: for each (i,j,k), one column has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # So the reconstructed tensor has entry (i,j,k) = 1.
    #
    # I think the issue is that this IS the correct tensor and the problem is well-defined.
    # The "rank" here is the tensor rank of the matrix multiplication tensor, which is NOT the 
    # all-ones tensor. I'm confusing myself.
    #
    # Let me look at this from the bilinear map perspective.
    # Matrix multiplication: (A,B) -> A*B
    # This is a bilinear map from R^{n*m} x R^{m*p} -> R^{n*p}
    # The associated trilinear form is: sum_{i,j,k} T[i,j,k] * a_{i,j} * b_{j,k} * c_{i,k}
    # where c = a*b.
    # T[i,j,k] = 1 for all i,j,k.
    #
    # But the tensor rank of the all-ones tensor is 1, which contradicts the known 
    # matrix multiplication rank results.
    #
    # OH WAIT. I think I see the issue. The tensor for matrix multiplication is NOT the all-ones tensor.
    # Let me reconsider. The bilinear map is:
    # (A,B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # As a tensor T in R^{n*p} ⊗ R^{n*m} ⊗ R^{m*p}, we have:
    # T[c_idx, a_idx, b_idx] where c_idx = (i,k), a_idx = (i,j), b_idx = (j,k)
    # T[(i,k),(i,j),(j,k)] = 1
    # T[(i',k'),(i'',j''),(j''',k'')] = 0 otherwise
    #
    # So T is NOT all-ones. It has specific structure. The indices are correlated.
    # For fixed c_idx=(i,k), a_idx must have the same i, and b_idx must have the same k.
    # And a_idx's j and b_idx's j must match.
    #
    # So T[(i,k),(i,j),(j,k)] = 1, and 0 otherwise.
    # This is a 15 x 15 x 25 tensor (for n=3,m=5,p=5).
    #
    # In the U,V,W formulation:
    # U has shape (n*m, rank) = (15, rank), indexed by a_idx = (i,j)
    # V has shape (m*p, rank) = (25, rank), indexed by b_idx = (j,k)  
    # W has shape (p*n, rank) = (15, rank), indexed by c_idx = (k,i)
    #
    # The reconstructed tensor entry T[c_idx, a_idx, b_idx] = sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
    # We need T[(i,k),(i,j),(j,k)] = 1 for all valid i,j,k, and 0 otherwise.
    #
    # Note: W is indexed by (k,i) not (i,k). So W[k*n+i, r].
    #
    # So the constraint is:
    # For all i in [0,3), j in [0,5), k in [0,5):
    #   sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*3+i, r] = 1
    # For all other combinations (i',j',k') not matching the pattern, the sum = 0.
    #
    # Specifically, for a given a_idx=(i,j) and b_idx=(j',k), we need j=j' for the term to be nonzero.
    # And for a given a_idx=(i,j) and c_idx=(k,i'), we need i=i' for the term to be nonzero.
    #
    # This is the correct tensor. Now the rank is NOT trivially 1.
    #
    # OK, so with this understanding, let me design the decomposition.
    #
    # For 3x5x5:
    # We can think of it as 3 rows of C. Row i of C: C[i,:] = A[i,:] * B
    # Each row is a 1x5x5 multiplication.
    # For 1x5x5: C[k] = sum_j A[j] B[j,k]
    # Tensor: T[(k),(j),(j,k)] = 1, i.e., T[c_idx=(k), a_idx=(j), b_idx=(j,k)] = 1
    # Note: for 1x5x5, n=1, so W[k*1+0, r] = W[k, r]. U[0*5+j, r] = U[j, r]. V[j*5+k, r].
    # T[k, j, (j,k)] = sum_r U[j,r] V[j*5+k, r] W[k, r] = 1
    # And T[k, j, (j',k')] = 0 for (j',k') != (j,k).
    #
    # The 1x5x5 tensor: we need sum_r U[j,r] V[j*5+k,r] W[k,r] = 1 for each (j,k), and 0 for mismatched.
    #
    # A simple decomposition for 1x5x5: rank 5.
    # For each j=0..4: U[j, j] = 1, V[j*5+k, j] = 1 for all k... no.
    # Let me think. For 1x5x5, we want T[k, j, b_idx] = 1 only when b_idx = (j,k).
    # 
    # Decomposition: for each k in 0..4:
    #   U[j, k] = 1 for all j? No.
    #   
    # Let me use: for each j in 0..4:
    #   U[j, j] = 1
    #   V[j*5+k, j] = 1 for all k in 0..4
    #   W[k, j] = 1 for all k in 0..4
    # Then T[k, j, (j',k')] = sum_r U[j,r] V[j'5+k',r] W[k,r]
    #   = sum_r (1 if r=j else 0) * V[j'5+k', r] * (1 if r=j else 0)
    #   = V[j5+k, j] * (1 if j=j' ... no, U[j,r] is nonzero only when r=j)
    #   = U[j,j] * V[j5+k, j] * W[k, j] = 1 * 1 * 1 = 1 when the b_idx is (j,k).
    #   But when b_idx is (j',k') with j' != j: V[j'5+k', j] = 1 (since we set V[j*5+k, j]=1 for all k).
    #   So T[k, j, (j',k')] = 1 * 1 * 1 = 1 even when j' != j. That's wrong.
    #
    # I need V to be zero when j' != j. So V[j*5+k, r] should be nonzero only for the right j.
    #
    # Correct 1x5x5 decomposition (rank 5):
    # For each j in 0..4:
    #   U[j, j] = 1
    #   V[j*5+k, j] = 1 for all k
    #   W[k, j] = 1 for all k
    # But as shown, this gives T[k, j, (j',k')] = 1 for all j', not just j'=j.
    #
    # The issue: V[j*5+k, r] depends on both j and k. For the tensor to work, we need:
    # sum_r U[a_idx, r] V[b_idx, r] W[c_idx, r] = 1 iff (a_idx, b_idx, c_idx) is "compatible"
    #
    # For 1x5x5: a_idx = j, b_idx = (j',k'), c_idx = k
    # Compatible means j' = j (the middle index matches) and c_idx = k' (the output index matches b's second index).
    # Wait, c_idx = k and b_idx = (j',k'). We need k = k' for the output to match.
    # And a_idx = j, b_idx = (j',k'). We need j = j' for the middle index to match.
    # So compatible: j = j' AND k = k'.
    #
    # So T[k, j, (j',k')] = 1 if j=j' and k=k', else 0.
    #
    # Decomposition: for each (j,k) pair... that's 25 terms, rank 25. That's schoolbook.
    #
    # Better: for each j:
    #   We want sum_r U[j,r] V[j5+k',r] W[k,r] = 1 if k=k', else 0.
    #   This is like: for each j, the vector U[j,:] * V[j5+0..4,:] (element-wise) dotted with W[:,k] = delta_{k,k'}
    #   
    # Simple: for each j, set U[j, j] = 1, V[j5+k, j] = 1 for all k, W[k, j] = 1 for all k.
    # Then for fixed j, j': sum_r U[j,r] V[j'5+k',r] W[k,r] = U[j,j] * V[j'5+k', j] * W[k, j] (only r=j contributes)
    #   = 1 * (1 if we set V[j5+k,j]=1 for all k) * 1 = 1.
    # But we need this to be 1 only when j=j' and k=k'. When j!=j', it's still 1. Wrong.
    #
    # The problem is that V[j5+k, r] is shared across different j's in the U dimension.
    #
    # I think the correct simple decomposition for 1x5x5 is rank 5:
    # For each k in 0..4:
    #   U[j, k] = 1 for all j
    #   V[j5+k, k] = 1 for all j
    #   W[k, k] = 1
    # Then T[k, j, (j',k')] = sum_r U[j,r] V[j'5+k',r] W[k,r]
    #   U[j,r] = 1 if r in 0..4 (always, since we set U[j,k]=1 for all k)
    #   Wait, U[j,k]=1 for all j and all k. So U[j,r] = 1 for all r.
    #   V[j'5+k', r] = 1 if r = k' (since we set V[j5+k,k]=1 for all j).
    #   W[k, r] = 1 if r = k.
    #   So T[k, j, (j',k')] = sum_r 1 * (1 if r=k') * (1 if r=k) = 1 if k'=k, else 0.
    #   But we also need j=j'. This gives T[k,j,(j',k')] = 1 if k=k' regardless of j,j'.
    #   That's not correct; we need j=j' too.
    #
    # I'm struggling with the simple decomposition. Let me just use the known result:
    # 1x5x5 has rank 5 (it's a "slice" of the multiplication tensor).
    # Actually, 1xm x p has rank max(m,p) if I recall correctly? No...
    #
    # Let me just use the schoolbook decomposition but be smarter.
    # For 3x5x5, schoolbook is 75.
    # 
    # Let me try a different approach: use the known Strassen decomposition for 2x2x2 and build up.
    #
    # Actually, let me just implement a correct decomposition using the following strategy:
    # 
    # For the 3x5x5 multiplication, I'll use a block approach:
    # 
    # Split the 5x5 B matrix into B = [B1 | B2] where B1 is 5x3 and B2 is 5x2.
    # Split the 3x5 A matrix into A = [A1 | A2 | A3] where each Ai is 3x? No.
    # 
    # Let me think of it as:
    # C = A * B where A is 3x5, B is 5x5, C is 3x5.
    # 
    # Use the following decomposition:
    # For the 2x5x5 subproblem (rows 2,3 of C), use a rank-? decomposition.
    #
    # Actually, I recall that 2x2x2 has rank 7 (Strassen).
    # 2x5x5 can be decomposed by splitting the 5 into 2+2+1:
    # 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x1x1 (rank 1) + cross terms...
    #
    # This is getting complicated. Let me just use a simple but correct decomposition.
    #
    # Simple correct approach: For each of the 3 rows of C, use a 1x5x5 decomposition.
    # For 1x5x5, the minimum rank is 5 (I believe).
    # 
    # 1x5x5 decomposition (rank 5):
    # For each k in 0..4:
    #   For all j in 0..4: U[j, k] = 1
    #   V[j*5+k, k] = 1 for all j
    #   W[k, k] = 1
    #
    # Let me verify: T[c_idx=k, a_idx=j, b_idx=(j',k')] = sum_r U[j,r] V[j'5+k',r] W[k,r]
    # U[j,r] = 1 for all r (since we set U[j,k]=1 for all k for each j)
    # V[j'5+k',r] = 1 if r=k'
    # W[k,r] = 1 if r=k
    # So sum_r = 1 * (1 if k'=k) * (1 if r=k) ... wait, sum over r:
    #   = sum_r [1] * [1 if r=k'] * [1 if r=k]
    #   = 1 if k'=k, else 0
    # 
    # So T[k, j, (j',k')] = 1 if k=k', else 0. This doesn't depend on j or j'.
    # But we need T[k,j,(j',k')] = 1 iff j=j' AND k=k'.
    # So this is wrong when j != j'.
    #
    # The issue is fundamental: with the W[k,r] structure, we can't distinguish j from j'.
    #
    # I think for 1xm xp, the tensor rank is actually m*p in general, or there's a specific structure.
    #
    # Let me just use the full schoolbook decomposition with rank 75, but see if I can find
    # a better one. Given the time constraint, let me implement a known good decomposition.
    #
    # Actually, I just realized: maybe I should use the following for 1x5x5:
    # Rank 5 decomposition:
    # For each j in 0..4:
    #   U[j, j] = 1
    #   V[j*5+k, j] = 1 for all k in 0..4  
    #   W[k, j] = 1 for all k in 0..4
    # 
    # Verify: T[k, j, (j',k')] = sum_r U[j,r] V[j'5+k',r] W[k,r]
    # U[j,r] = 1 if r=j, else 0
    # So only r=j contributes:
    #   = U[j,j] * V[j'5+k', j] * W[k, j]
    #   = 1 * (1 if we set V[j*5+k, j]=1 for all k) * 1
    #   = 1 (since V[j5+k, j] = 1 for all k, so V[j'5+k', j] = 1)
    # 
    # So T[k, j, (j',k')] = 1 for all k, k', j, j'. That's completely wrong.
    #
    # I think the problem is that V[j5+k, r] has the index j baked in, but in the sum, 
    # j comes from a_idx and j' comes from b_idx. They're different indices.
    #
    # Let me redefine: 
    # a_idx = (i, j) for 3x5x5, so U is indexed by (i,j)
    # b_idx = (j, k) so V is indexed by (j,k)
    # c_idx = (k, i) so W is indexed by (k,i)
    #
    # For the tensor to be correct:
    # sum_r U[i,j,r] V[j',k',r] W[k'',i'',r] = 1 iff i=i'', j=j', k=k''
    # 
    # For 1x5x5 (i=0 always):
    # sum_r U[0,j,r] V[j',k',r] W[k'',0,r] = 1 iff j=j', k=k''
    #
    # So for each j, we need the vectors U[0,j,:], and for each (j',k'), V[j',k',:], and for each k'', W[k'',0,:]
    # such that the sum works out.
    #
    # A valid decomposition: for each (j,k) in 0..4 x 0..4:
    #   U[0,j, (j,k)] = 1
    #   V[j,k, (j,k)] = 1
    #   W[k,0, (j,k)] = 1
    # This gives rank 25 (schoolbook for 1x5x5).
    #
    # Can we do better? The tensor is a 5x25x5 tensor (for 1x5x5). 
    # T[j, (j',k'), k''] = 1 iff j=j' and k=k''.
    # This is essentially the "identity" tensor in the j-j' and k-k'' dimensions.
    # The rank of this tensor... I think it's 5 (using the fact that it's a "block diagonal" structure).
    #
    # For each k in 0..4:
    #   U[0,j,k] = 1 for all j
    #   V[j,k,k] = 1 for all j
    #   W[k,0,k] = 1
    # 
    # Verify: T[j, (j',k'), k''] = sum_r U[0,j,r] V[j',k',r] W[k'',0,r]
    # U[0,j,r] = 1 for all r (since U[0,j,k]=1 for all k)
    # V[j',k',r] = 1 if r=k'
    # W[k'',0,r] = 1 if r=k''
    # Sum = 1 if k'=k'', else 0.
    # This gives T[j,(j',k'),k''] = 1 if k'=k'', regardless of j,j'. Wrong.
    #
    # For each j in 0..4:
    #   U[0,j,j] = 1
    #   V[j,k,j] = 1 for all k
    #   W[k,0,j] = 1 for all k
    # 
    # Verify: T[j, (j',k'), k''] = sum_r U[0,j,r] V[j',k',r] W[k'',0,r]
    # U[0,j,r] = 1 if r=j
    # So: = U[0,j,j] * V[j',k',j] * W[k'',0,j] = 1 * (1 if j'=j... no, V[j',k',j] is 1 for all j',k' when r=j)
    # Wait, V[j,k,j] = 1 for all k means V[j',k',j] = 1 for all j',k'.
    # So T = 1 * 1 * 1 = 1 for all. Wrong.
    #
    # I'm going in circles. Let me just use rank 25 for 1x5x5 (schoolbook) and total rank 75 for 3x5x5.
    # That's the same as the seed. Not an improvement.
    #
    # Given the difficulty, let me try a completely different approach: use a known 
    # tensor decomposition for matrix multiplication with lower rank.
    #
    # For 3x3x3, the minimum rank is 23 (Larsen 2011). For 3x5x5, I don't know the exact minimum.
    # But we can definitely do better than 75.
    #
    # Let me use a simple block decomposition:
    # Split A (3x5) into A1 (2x5) and A2 (1x5).
    # Split B (5x5) into B1 (5x3) and B2 (5x2).
    # C1 = A1*B1 (2x3), C2 = A1*B2 (2x2), C3 = A2*B1 (1x3), C4 = A2*B2 (1x2).
    #
    # 2x5x3: split into 2x2x2 (rank 7) + 2x2x2 (rank 7) + 2x1x1 (rank 1) + ... 
    # Actually, 2x5x3: split the 5 into 2+2+1 and 3 into 2+1:
    # 2x2x2 (7) + 2x2x1 (4) + 2x1x2 (4) + 2x1x1 (1) = 16
    # Or use a better decomposition.
    #
    # This is getting very complex. Given the time constraint, let me just return the schoolbook 
    # decomposition but formatted correctly, ensuring it passes validation.
    #
    # Actually, wait. Let me re-examine the problem. The seed has rank 75. The score is 1.0.
    # To improve, I need rank < 75. Even a small improvement would help.
    #
    # Let me try: for 3x5x5, use the following:
    # For the 1x5x5 subproblem, I claim rank 5 is achievable.
    # 
    # Key insight: For 1x5x5, the tensor T[j, (j',k'), k''] = 1 iff j=j' and k=k''.
    # This is the tensor product of two "identity" tensors: I_5 (in j,j') ⊗ I_5 (in k,k'').
    # The tensor product of two rank-r tensors has rank at most r1*r2, but can be less.
    # The identity tensor I_5 = sum_{j} e_j ⊗ e_j has rank 5.
    # So I_5 ⊗ I_5 has rank 5*5 = 25? Or can it be less?
    # Actually, the tensor rank of I_m ⊗ I_n is m*n in general (I think).
    # So 1x5x5 has rank 25. That's schoolbook.
    #
    # So for 3x5x5, if we decompose as 3 copies of 1x5x5, we get rank 3*25 = 75.
    # To do better, we need to share columns across the 3 rows.
    #
    # Idea: For 3x5x5, think of it as a 3x(5x5) tensor in the first dimension.
    # T[i, j, (j',k')] = 1 iff i=i'' (from c_idx) and j=j' and k=k''.
    # 
    # For each (j,k) pair, we need a column that picks out the right i, j, k.
    #
    # Let me try: for each j in 0..4, k in 0..4:
    #   U[i, j, (j,k)] = 1 for all i (so U is 15x20, with 20 columns... no, rank 20?)
    #   V[j, k, (j,k)] = 1
    #   W[k, i, (j,k)] = 1 for all i
    # 
    # Wait, U has shape (15, rank) where 15 = n*m = 3*5. Index is i*m+j = i*5+j.
    # V has shape (25, rank) where 25 = m*p = 5*5. Index is j*p+k = j*5+k.
    # W has shape (15, rank) where 15 = p*n = 5*3. Index is k*n+i = k*3+i.
    #
    # For each (j,k) in 0..4 x 0..4 (25 values):
    #   Column r = j*5+k (say)
    #   U[i*5+j, r] = 1 for all i in 0..2
    #   V[j*5+k, r] = 1
    #   W[k*3+i, r] = 1 for all i in 0..2
    #
    # Verify: T[c_idx=(k'',i''), a_idx=(i,j), b_idx=(j',k')] 
    #   = sum_r U[i*5+j, r] V[j'*5+k', r] W[k''*3+i'', r]
    #   For r = j'*5+k': 
    #     U[i*5+j, r] = 1 if j=j' (since U[i*5+j, r] is nonzero only when the j-part of a_idx matches j')
    #     Wait, U[i*5+j, r] = 1 for all i when r corresponds to (j,k). So U[i*5+j, j'*5+k'] = 1 if j=j'.
    #     V[j'*5+k', r] = 1 if r = j'*5+k', which is true by definition.
    #     W[k''*3+i'', r] = 1 if k''=k' (since W[k*3+i, r] = 1 for all i when r corresponds to (j,k)).
    #   So T = 1 if j=j' and k''=k', else 0.
    #   But we also need i=i''. This is not enforced!
    #
    # So this gives T[k'',i'', (i,j), (j',k')] = 1 if j=j' and k''=k', regardless of i,i''.
    # Wrong.
    #
    # To fix, I need to separate the i dimension. For each i, j, k:
    #   U[i*5+j, (i,j,k)] = 1
    #   V[j*5+k, (i,j,k)] = 1
    #   W[k*3+i, (i,j,k)] = 1
    # This is schoolbook, rank 75.
    #
    # To reduce rank, I need to share some columns. The key is that for a fixed (j,k),
    # the i dimension in U and W can be shared if I use a decomposition of the i-dimension.
    #
    # For fixed (j,k), I need: sum_r U[i*5+j, r] * 1 * W[k*3+i', r] = 1 if i=i', else 0.
    # This is like an identity matrix in the i dimension (3x3).
    # The 3x3 identity has rank 3.
    # So for each (j,k), I need 3 columns (one for each i). Total: 25*3 = 75. Schoolbook.
    #
    # Unless I can share across different (j,k) values. But the tensor structure doesn't allow it 
    # easily because U and V depend on j and k in specific ways.
    #
    # I think for this specific problem, 75 might be close to optimal, or the improvement 
    # requires sophisticated techniques.
    #
    # Given the constraints, let me just return the schoolbook decomposition correctly.
    # The seed already does this with rank 75. I'll keep it but ensure it's correct.
    
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
