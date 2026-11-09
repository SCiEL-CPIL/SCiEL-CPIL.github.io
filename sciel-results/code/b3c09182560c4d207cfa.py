"""Matrix multiplication 3,4,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # The matrix multiplication tensor has shape (n*m) x (m*p) x (p*n)
    # We need to find rank-minimal decomposition.
    # 
    # Known bounds:
    # - Schoolbook: n*m*p = 72
    # - Strassen-like: can do better
    # - For 3x4 * 4x6, we can try various approaches
    #
    # Strategy: Try to find a low-rank decomposition by:
    # 1. Start with schoolbook (rank 72)
    # 2. Try to reduce rank by finding linear dependencies
    # 3. Use the fact that we can pad to square matrices and use known results
    #
    # For 3x4 * 4x6:
    # - Pad to 4x4 * 4x4: rank at most 7 (Strassen)
    # - But we need exact decomposition for the specific dimensions
    #
    # Let's try a systematic approach:
    # 1. Build the multiplication tensor
    # 2. Try to find a low-rank decomposition using various methods
    
    # First, let's establish what we know about the tensor structure.
    # The multiplication tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    
    # Approach: Use the fact that we can decompose via block methods.
    # 
    # Key insight: We can use the following strategy:
    # - The 3x4 matrix A and 4x6 matrix B give a 3x6 result C
    # - We can think of this as computing specific bilinear forms
    #
    # Let's try a direct construction with reduced rank.
    #
    # Method 1: Try rank 60 (savings of 12 from schoolbook)
    # Method 2: Try rank 56, 52, etc.
    #
    # Actually, let's think about this more carefully.
    # The tensor is in R^{12} x R^{24} x R^{18}
    # 
    # A key observation: we can use the following identity for matrix multiplication.
    # If we have A (n x m) and B (m x p), then C = A * B.
    #
    # For our case: A is 3x4, B is 4x6, C is 3x6.
    #
    # Let's try to find a decomposition with rank less than 72.
    #
    # One approach: use the "winograd" style or "strassen" style recurrences.
    #
    # Let me try a different approach: use the fact that we can express
    # the product using fewer multiplications by exploiting the structure.
    #
    # For a 2x2 * 2x2 multiplication, Strassen uses 7 multiplications instead of 8.
    # 
    # For our 3x4 * 4x6 problem, let's try to use a block-based approach.
    #
    # Actually, let's try a more systematic search.
    # We'll construct the tensor and try to find its rank or a low-rank approximation.
    #
    # Since we need EXACT decomposition, we can't just do numerical SVD.
    # We need integer or half-integer entries.
    
    # Let's try a constructive approach with known patterns.
    #
    # Key idea: We can pad the 3x4 matrix to 4x4 by adding a zero row,
    # and the 4x6 matrix to 4x4 by... no, that doesn't work directly.
    #
    # Better approach: Use the following decomposition strategy.
    # 
    # Let's denote the entries of A as a_{i,j} for i in [0,2], j in [0,3]
    # Let's denote the entries of B as b_{j,k} for j in [0,3], k in [0,5]
    # Then c_{i,k} = sum_j a_{i,j} * b_{j,k}
    #
    # We need to express each c_{i,k} as a sum of products of linear forms
    # in the a's and b's.
    #
    # The schoolbook uses 3*4*6 = 72 products.
    #
    # Let's try to find a decomposition with fewer products.
    #
    # One approach: group the computations.
    #
    # For each i (3 choices) and k (6 choices), we have:
    # c_{i,k} = a_{i,0}*b_{0,k} + a_{i,1}*b_{1,k} + a_{i,2}*b_{2,k} + a_{i,3}*b_{3,k}
    #
    # That's 4 multiplications per output entry, 18 output entries, total 72.
    #
    # Can we share some computations?
    #
    # Idea: For a fixed i, the four terms a_{i,j}*b_{j,k} for j=0,1,2,3 and varying k
    # are like multiplying a 1x4 vector by a 4x6 matrix.
    #
    # For a 1x4 * 4x6 multiplication, the schoolbook uses 4*6=24 multiplications.
    # But we can potentially do better.
    #
    # Actually, for a 1xm * mxp multiplication, the rank is known to be mp
    # in general (since each output depends on all m*p products in a distinct way).
    # Wait, that's not quite right.
    #
    # For 1x4 * 4x6: c_k = sum_j a_j * b_{j,k}
    # This is just a dot product for each k. The tensor here has rank 4*6=24?
    # No, actually it's less. Let me think...
    #
    # The tensor for 1x4 * 4x6 * (1x6) has T[j, (j,k), k] = 1
    # This is actually just the "diagonal" in some sense.
    # The rank of this tensor is 4 (not 24)!
    # Because c_k = sum_j a_j * b_{j,k}, and we can write this as:
    # c_k = sum_j (a_j) * (b_{j,k})
    # So we need 4 "multiplications" (products of linear forms), one for each j.
    # Wait, no. Each product a_j * b_{j,k} is a separate multiplication.
    # But a_j is the same across all k, while b_{j,k} varies with k.
    #
    # Actually, the tensor T in R^4 x R^{24} x R^6 with T[j, j*p+k, k] = 1
    # has rank 4. Because:
    # T = sum_{j=0}^{3} u_j ⊗ v_j ⊗ w_j
    # where u_j is the j-th basis vector in R^4,
    # v_j has 1 at position j*p+k for all k, 0 elsewhere,
    # w_k is the k-th basis vector in R^6.
    #
    # Wait, that's not quite right either. Let me be more careful.
    #
    # The multiplication tensor for 1xm * mxp -> 1xp is:
    # T ∈ R^m ⊗ R^{mp} ⊗ R^p
    # T[j, j*p+k, k] = 1
    #
    # This can be decomposed as:
    # T = sum_{j=0}^{m-1} e_j ⊗ f_j ⊗ g_j
    # where e_j is the j-th standard basis vector in R^m,
    # f_j ∈ R^{mp} has f_j[j*p+k] = 1 for all k, 0 otherwise,
    # g_j ∈ R^p... wait, g should depend on k, not j.
    #
    # Actually: T[j, l, k] = 1 if l = j*p+k, else 0.
    # So T = sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_j ⊗ e_{j*p+k} ⊗ e_k
    # That's a decomposition of rank m*p.
    #
    # But can we do better? 
    # T[j, l, k] = delta(l, j*p+k)
    # 
    # Consider: T = sum_{j=0}^{m-1} e_j ⊗ (sum_{k=0}^{p-1} e_{j*p+k} ⊗ e_k)
    # The second factor is a vector in R^{mp} ⊗ R^p.
    # So T = sum_{j=0}^{m-1} u_j ⊗ v_j where u_j = e_j and v_j = sum_k e_{j*p+k} ⊗ e_k
    # This gives rank m.
    #
    # Wait, but the tensor is in R^m ⊗ R^{mp} ⊗ R^p, so a rank-1 term is
    # u ⊗ v ⊗ w where u ∈ R^m, v ∈ R^{mp}, w ∈ R^p.
    # 
    # T[j, l, k] = sum_{r} U[j,r] * V[l,r] * W[k,r]
    #
    # For the 1xm*mxp case: T[j, l, k] = 1 iff l = j*p+k
    # 
    # Can we write this with rank < m*p?
    # 
    # T[j, l, k] = delta_{l, j*p+k}
    # 
    # Note that delta_{l, j*p+k} = 1 if l mod p = k and l div p = j
    # 
    # This is a very structured tensor. Let's see if rank m works.
    # 
    # Try rank m: T = sum_{j=0}^{m-1} e_j ⊗ f_j ⊗ g_j
    # Then T[j', l, k] = sum_j delta_{j',j} * f_j[l] * g_j[k]
    # = f_{j'}[l] * g_{j'}[k]
    # We need this to equal delta_{l, j'*p+k}
    # So f_{j'}[l] * g_{j'}[k] = delta_{l, j'*p+k}
    # 
    # For this to hold: if l = j'*p+k, then f_{j'}[j'*p+k]*g_{j'}[k] = 1
    # If l != j'*p+k, then f_{j'}[l]*g_{j'}[k] = 0
    #
    # From the second condition: for fixed j', if l != j'*p+k for some k,
    # we need f_{j'}[l] * g_{j'}[k] = 0 for all such k.
    # 
    # If g_{j'}[k] != 0 for some k, then f_{j'}[l] = 0 for all l != j'*p+k.
    # But we need f_{j'}[j'*p+k]*g_{j'}[k] = 1, so f_{j'}[j'*p+k] != 0.
    # 
    # So f_{j'} is nonzero only at positions j'*p+k where g_{j'}[k] != 0.
    # And g_{j'}[k] can be nonzero for some subset of k values.
    #
    # If g_{j'}[k] = 1 for all k, then f_{j'}[l] = 1 if l = j'*p+k for some k, i.e.,
    # f_{j'}[j'*p+k] = 1 for all k, and 0 otherwise.
    # Then f_{j'}[j'*p+k]*g_{j'}[k] = 1*1 = 1 for l=j'*p+k. Good.
    # And for l != j'*p+k, f_{j'}[l] = 0, so product is 0. Good.
    #
    # So rank m works for the 1xm*mxp case!
    #
    # This means for our problem:
    # c_{i,k} = sum_j a_{i,j} * b_{j,k}
    # For fixed i, this is a 1x4 * 4x6 multiplication, which has rank 4.
    # 
    # But we have 3 different i values, and the a_{i,j} are different for each i.
    # So the full tensor is:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # 
    # Can we decompose this with rank 3*4 = 12? Let's see.
    # 
    # T = sum_{i,j} e_{i*m+j} ⊗ f_{i,j} ⊗ g_{i,j}
    # where f_{i,j}[l] = 1 if l = j*p+k for some k, 0 otherwise (independent of i!)
    # and g_{i,j}[k*n+i'] = 1 if i'=i, 0 otherwise... 
    #
    # Wait, let me re-examine.
    # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    #
    # So T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    #
    # This is the schoolbook decomposition with rank n*m*p = 72.
    #
    # Can we reduce this? The key is that for fixed i and j, the sum over k is:
    # sum_k e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # = e_{i*m+j} ⊗ (sum_k e_{j*p+k} ⊗ e_{k*n+i})
    #
    # The term in parentheses is a vector in R^{m*p} ⊗ R^{p*n}.
    # Let's call it h_{i,j}.
    # 
    # So T = sum_{i,j} e_{i*m+j} ⊗ h_{i,j}
    #
    # But h_{i,j} is a vector in R^{m*p} ⊗ R^{p*n}, not a simple tensor.
    # For T to be a sum of simple tensors, we need h_{i,j} to be a simple tensor
    # (or we need to further decompose it).
    #
    # h_{i,j} = sum_k e_{j*p+k} ⊗ e_{k*n+i}
    #
    # This is the same structure as the 1xp * px1 case... no wait.
    # h_{i,j} ∈ R^{mp} ⊗ R^{pn}
    # h_{i,j}[l, c] = 1 if l = j*p+k and c = k*n+i for some k
    #
    # So h_{i,j}[l, c] = 1 iff l div p = j, l mod p = k, c div n = i, c mod n = k
    # i.e., l div p = j AND c div n = i AND l mod p = c mod n
    #
    # This is a rank-p tensor (in the l,c space) because:
    # h_{i,j} = sum_k e_{j*p+k} ⊗ e_{k*n+i}
    #
    # So the full decomposition is:
    # T = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # which is rank n*m*p = 72.
    #
    # To reduce the rank, we need to find a different decomposition.
    #
    # Let's try a different approach. What if we use the following identity?
    # 
    # For the 1x4 * 4x6 case (fixed i), we showed rank 4 suffices.
    # The decomposition is:
    # c_k = sum_j a_j * b_{j,k}
    # Using rank-4 decomposition:
    # For each j, define:
    #   U_j = e_j (in R^4, the "a" space)
    #   V_j = sum_k e_{j*p+k} (in R^{24}, the "b" space) -- wait, this doesn't work
    #
    # Let me redo this. For 1xm * mxp -> 1xp:
    # Input: a ∈ R^m, b ∈ R^{mp} (indexed as b[j,k])
    # Output: c ∈ R^p, c[k] = sum_j a[j] * b[j,k]
    #
    # Tensor: T ∈ R^m ⊗ R^{mp} ⊗ R^p, T[j, j*p+k, k] = 1
    #
    # Rank-m decomposition:
    # T = sum_{j=0}^{m-1} u_j ⊗ v_j ⊗ w_j
    # where u_j = e_j ∈ R^m
    #       v_j ∈ R^{mp}: v_j[l] = 1 if l = j*p+k for some k (i.e., l in [j*p, (j+1)*p))
    #       w_j ∈ R^p: w_j[k] = 1 for all k
    #
    # Check: sum_j u_j[a_idx] * v_j[b_idx] * w_j[c_idx]
    # = sum_j delta(a_idx, j) * (1 if b_idx in [j*p, (j+1)*p) else 0) * 1
    # = 1 if a_idx = j AND b_idx in [j*p, (j+1)*p) for some j
    # = 1 if b_idx div p = a_idx
    # 
    # But we need T[a_idx, b_idx, c_idx] = 1 iff b_idx = a_idx*p + c_idx
    # i.e., b_idx div p = a_idx AND b_idx mod p = c_idx
    #
    # The above gives 1 if b_idx div p = a_idx, regardless of c_idx. That's wrong!
    #
    # So rank m does NOT work for the 1xm*mxp case. Let me reconsider.
    #
    # Actually, I think I made an error. Let me re-examine.
    #
    # T[j, l, k] = 1 iff l = j*p + k
    #
    # For a rank-r decomposition: T = sum_{r'} U[j,r'] * V[l,r'] * W[k,r']
    #
    # We need: sum_{r'} U[j,r'] * V[l,r'] * W[k,r'] = 1 iff l = j*p+k
    #
    # This is a very specific pattern. The tensor T is actually the "identity"
    # in a certain sense. Let's think about it as a matrix.
    #
    # Fix k: T[:,:,k] is a matrix in R^m x R^{mp} with T[j, j*p+k, k] = 1.
    # This matrix has 1s at positions (j, j*p+k) for j=0,...,m-1.
    #
    # The rank of T[:,:,k] is m (since the rows are linearly independent).
    #
    # For the full tensor, the rank is at least the maximum rank of any slice.
    # So rank(T) >= m.
    #
    # But is rank(T) = m? Let's try to construct a rank-m decomposition.
    #
    # T = sum_{j=0}^{m-1} u_j ⊗ v_j ⊗ w_j
    #
    # T[j', l, k] = sum_j delta(j', j) * v_j[l] * w_j[k] = v_{j'}[l] * w_{j'}[k]
    #
    # We need v_{j'}[l] * w_{j'}[k] = 1 iff l = j'*p + k
    #
    # So for each j', we need:
    # v_{j'}[j'*p+k] * w_{j'}[k] = 1 for all k
    # v_{j'}[l] * w_{j'}[k] = 0 for all l != j'*p+k, all k
    #
    # From the second condition: if w_{j'}[k] != 0, then v_{j'}[l] = 0 for all l != j'*p+k.
    # But we need v_{j'}[j'*p+k'] * w_{j'}[k'] = 1 for all k', so v_{j'}[j'*p+k'] != 0.
    #
    # If w_{j'}[k] = 1 for all k, then v_{j'}[l] must be 0 for all l not in {j'*p, j'*p+1, ..., j'*p+p-1}.
    # And v_{j'}[j'*p+k] = 1 for all k.
    #
    # Then v_{j'}[l] * w_{j'}[k] = 1 if l = j'*p+k', and 0 otherwise.
    # But we need it to be 1 specifically when l = j'*p+k (matching k), not just any k'.
    #
    # So this gives v_{j'}[j'*p+k'] * w_{j'}[k] = 1 for all k, k', which is wrong.
    #
    # Therefore, rank m does NOT work. The rank must be higher.
    #
    # In fact, I believe the rank of the 1xm*mxp multiplication tensor is m*p.
    # This is because the tensor is essentially the "flattening" of an identity-like structure.
    #
    # Let me verify for small cases:
    # 1x1 * 1x1 -> 1x1: rank 1. m*p = 1. ✓
    # 1x2 * 2x1 -> 1x1: c = a0*b0 + a1*b1. Rank 2. m*p = 2. ✓
    # 1x2 * 2x2 -> 1x2: c0 = a0*b00 + a1*b10, c1 = a0*b01 + a1*b11
    # Tensor in R^2 x R^4 x R^2: T[0,0,0]=1, T[0,2,1]=1, T[1,1,0]=1, T[1,3,1]=1
    # Rank? This is a 2x4x2 tensor. Let's see if rank 2 works.
    # T = u1⊗v1⊗w1 + u2⊗v2⊗w2
    # T[0,0,0] = u1[0]v1[0]w1[0] + u2[0]v2[0]w2[0] = 1
    # T[0,1,0] = u1[0]v1[1]w1[0] + u2[0]v2[1]w2[0] = 0
    # T[0,2,0] = u1[0]v1[2]w1[0] + u2[0]v2[2]w2[0] = 0
    # T[0,3,0] = u1[0]v1[3]w1[0] + u2[0]v2[3]w2[0] = 0
    # T[0,0,1] = u1[0]v1[0]w1[1] + u2[0]v2[0]w2[1] = 0
    # T[0,1,1] = u1[0]v1[1]w1[1] + u2[0]v2[1]w2[1] = 0
    # T[0,2,1] = u1[0]v1[2]w1[1] + u2[0]v2[2]w2[1] = 1
    # T[0,3,1] = u1[0]v1[3]w1[1] + u2[0]v2[3]w2[1] = 0
    # ... and similarly for T[1,...]
    #
    # This looks like it would require rank 4 (one for each nonzero entry).
    # So rank = m*p = 4 for this case. ✓
    #
    # So the 1xm*mxp tensor has rank m*p.
    #
    # This means for our problem, the schoolbook rank of 72 might be optimal
    # or close to optimal. But let's see if we can do slightly better.
    #
    # Actually, wait. The full tensor is not just a sum of independent 1xm*mxp tensors.
    # The structure is:
    # T[i*m+j, j*p+k, k*n+i] = 1
    #
    # For different (i,j) pairs, the "k" index in the third factor is coupled with i.
    # Specifically, the third factor index is k*n+i, which depends on both k and i.
    #
    # This coupling might allow some rank reduction.
    #
    # Let's try a different approach: use known results for small matrix multiplication.
    #
    # For 3x4 * 4x6, we can try:
    # 1. Pad to 4x4 * 4x4: use Strassen (rank 7) for the 4x4 part, but we need to
    #    account for the zero padding.
    # 2. Use a custom decomposition.
    #
    # Actually, let's try a constructive approach with a specific rank.
    #
    # I'll try to find a decomposition with rank 60 (a 12% reduction from 72).
    #
    # Here's a strategy: use the following identity.
    # For each pair (i, k), we have c_{i,k} = sum_j a_{i,j} b_{j,k}.
    # 
    # Instead of computing all 72 products separately, we can try to share some.
    #
    # Idea: For a fixed j, the products a_{i,j} * b_{j,k} for all i,k form a
    # 3x6 matrix of products. We can't really share within this.
    #
    # For a fixed i, the products a_{i,j} * b_{j,k} for all j,k form a 1x6 vector.
    # We need 4 products to compute this (one for each j).
    #
    # For a fixed k, the products a_{i,j} * b_{j,k} for all i,j form a 3x1 vector.
    # We need 4 products to compute this (one for each j).
    #
    # I don't see an obvious way to reduce below 72 with simple sharing.
    #
    # Let's try a different tactic: use the fact that we can use linear combinations.
    #
    # Define new variables:
    # For the A matrix (3x4):
    #   a_{i,j} for i=0,1,2 and j=0,1,2,3
    #
    # For the B matrix (4x6):
    #   b_{j,k} for j=0,1,2,3 and k=0,1,2,3,4,5
    #
    # We want to compute c_{i,k} = sum_j a_{i,j} b_{j,k}
    #
    # Let's try to use the following:
    # Define s_{i,j} = a_{i,j} (no change)
    # Define t_{j,k} = b_{j,k} (no change)
    #
    # This doesn't help. Let's try a more creative approach.
    #
    # What if we use the following identity for each i:
    # c_{i,0} = a_{i,0}b_{0,0} + a_{i,1}b_{1,0} + a_{i,2}b_{2,0} + a_{i,3}b_{3,0}
    # c_{i,1} = a_{i,0}b_{0,1} + a_{i,1}b_{1,1} + a_{i,2}b_{2,1} + a_{i,3}b_{3,1}
    # ...
    #
    # We can write:
    # [c_{i,0}, c_{i,1}, ..., c_{i,5}] = [a_{i,0}, a_{i,1}, a_{i,2}, a_{i,3}] * B
    #
    # This is a 1x4 vector times a 4x6 matrix. As we established, this requires
    # 4*6 = 24 "scalar multiplications" in the schoolbook sense.
    #
    # But wait, in the tensor decomposition sense, the rank of this sub-tensor is 24.
    # And we have 3 such sub-tensors (one for each i), giving total rank 72.
    #
    # However, the sub-tensors are not independent! They share the B matrix.
    # The A matrix has different rows for each i, but the B matrix is the same.
    #
    # This is the key: the B matrix is shared across all i.
    #
    # So the full tensor is:
    # T[i*m+j, j*p+k, k*n+i] = 1
    #
    # Let's try to decompose this differently.
    #
    # T = sum_{j,k} U_jk ⊗ V_jk ⊗ W_jk
    #
    # where for each (j,k):
    # U_jk[i*m+j'] = 1 if j'=j, 0 otherwise  (independent of k!)
    # V_jk[j'*p+k'] = 1 if j'=j and k'=k, 0 otherwise
    # W_jk[k'*n+i'] = 1 if k'=k and i'=i... wait, this depends on i.
    #
    # Hmm, the third factor depends on both i and k, which makes it hard to separate.
    #
    # Let's try yet another approach. I'll use a numerical method to find a
    # low-rank decomposition, then try to round to integers/half-integers.
    #
    # Actually, since we need EXACT decomposition, let's try a different strategy.
    #
    # I'll use the following approach:
    # 1. Start with the schoolbook decomposition (rank 72).
    # 2. Try to find linear dependencies among the rank-1 terms.
    # 3. If we find a dependency, we can reduce the rank.
    #
    # A linear dependency would be:
    # sum_{r} alpha_r * (u_r ⊗ v_r ⊗ w_r) = 0
    # with not all alpha_r = 0.
    #
    # In the schoolbook decomposition:
    # term_{i,j,k} = e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    #
    # A dependency would be:
    # sum_{i,j,k} alpha_{i,j,k} * e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i} = 0
    #
    # This means: for all (a, b, c), sum_{i,j,k: i*m+j=a, j*p+k=b, k*n+i=c} alpha_{i,j,k} = 0
    #
    # But for each (a, b, c), there is at most one (i, j, k) that satisfies the conditions
    # (since i = a div m, j = a mod m, k = b mod p, and we need k*n+i = c, i = c div n).
    #
    # So the only "dependencies" are trivial (each term appears at most once).
    # This means the schoolbook decomposition is already minimal in this sense.
    #
    # But that doesn't mean the rank is 72! It just means the schoolbook terms are
    # linearly independent as tensors. The rank could still be lower if there's a
    # different set of rank-1 tensors that sum to T.
    #
    # This is a subtle point. The rank of a tensor is the minimum number of rank-1
    # tensors needed to represent it, not the number of terms in a particular decomposition.
    #
    # So we need to search for a different decomposition.
    #
    # Given the complexity of this problem, let's try a practical approach:
    # 1. Use a numerical rank-finding algorithm (e.g., iterative SVD or tensor train).
    # 2. Try to find a decomposition with rank < 72.
    # 3. Round to integers/half-integers and verify.
    #
    # Let's implement this.
    
    # Build the multiplication tensor
    # T has shape (n*m, m*p, p*n) = (12, 24, 18)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find a low-rank decomposition.
    # We'll use a simple iterative approach:
    # 1. Start with a random or structured initialization.
    # 2. Use gradient descent or alternating least squares to minimize the error.
    # 3. Try different ranks and keep the best.
    
    def try_decomposition(T, rank, max_iters=200, tol=1e-10):
        """Try to find a rank-rank decomposition of T."""
        shape = T.shape
        # Initialize with random values
        U = np.random.randn(shape[0], rank) * 0.1
        V = np.random.randn(shape[1], rank) * 0.1
        W = np.random.randn(shape[2], rank) * 0.1
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Update U
            # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # Fix V, W: T = U * (V ⊙ W)^T in some sense
            # Reshape T to (shape[0], shape[1]*shape[2])
            T_mat = T.reshape(shape[0], shape[1]*shape[2])
            VW = (V * W).T.reshape(rank, shape[1]*shape[2])  # r x (s1*s2)
            # U * VW = T_mat
            U_new, residuals, rank_u, sv = np.linalg.lstsq(VW, T_mat.T, rcond=None)
            U = U_new.T
            
            # Update V
            T_mat2 = T.reshape(shape[1], shape[0]*shape[2])
            UW = (U * W).T.reshape(rank, shape[0]*shape[2])
            V_new, residuals, rank_v, sv = np.linalg.lstsq(UW, T_mat2.T, rcond=None)
            V = V_new.T
            
            # Update W
            T_mat3 = T.reshape(shape[2], shape[0]*shape[1])
            UV = (U * V).T.reshape(rank, shape[0]*shape[1])
            W_new, residuals, rank_w, sv = np.linalg.lstsq(UV, T_mat3.T, rcond=None)
            W = W_new.T
            
            # Compute error
            T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(T - T_approx)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
        
        return best_U, best_V, best_W, best_err
    
    # Try ranks from 72 down to some minimum
    # We know the rank is at least max(n*m, m*p, p*n) = 24
    # And at most n*m*p = 72
    
    best_result = None
    best_rank = n*m*p  # schoolbook
    
    # First, let's try the schoolbook decomposition as a fallback
    U_school = np.zeros((n*m, n*m*p))
    V_school = np.zeros((m*p, n*m*p))
    W_school = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try to find better decompositions
    for rank in range(n*m*p - 1, 23, -1):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for trial in range(3):
            np.random.seed(rank * 100 + trial)
            U, V, W, err = try_decomposition(T, rank, max_iters=100, tol=1e-10)
            
            if err < 1e-8:
                # Found a valid decomposition!
                # Now try to round to integers or half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_approx = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_approx, T, atol=1e-10):
                    best_result = (U_rounded, V_rounded, W_rounded, rank)
                    best_rank = rank
                    break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W, rank = U_school, V_school, W_school, n*m*p
    else:
        U, V, W, rank = best_result
    
    # Convert to lists
    U_list = [[U[i, j] for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[V[i, j] for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[W[i, j] for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
