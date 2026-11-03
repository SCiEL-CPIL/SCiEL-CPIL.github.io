"""Matrix multiplication 2x5x5 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s wall-clock limit, use 120s budget
    
    n, m, p = DIMENSIONS  # 2, 5, 5
    
    # The multiplication tensor T has indices (i, j, k) for i in [0,n), j in [0,m), k in [0,p)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find rank-r decomposition:
    #   T = sum_{c=0}^{r-1} U_c * V_c * W_c
    # where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
    #
    # The schoolbook rank is n*m*p = 50.
    # For 2x5x5, the minimum rank is known to be 10 (from literature on small matrix multiplication).
    # Let's try to find a rank-10 decomposition.
    
    # Strategy: For 2x5x5, we can think of it as computing two 1x5x5 products that share structure.
    # A 1x5x5 multiplication has rank at most 5 (just the 5 products).
    # For 2x5x5, if the two rows are independent, rank could be up to 10.
    # But there might be sharing possible.
    
    # Actually, let's think more carefully. The tensor for 2x5x5:
    # Output C[i,k] = sum_j A[i,j] * B[j,k] for i=0,1 and k=0..4
    # So we have 10 outputs, each a sum of 5 products.
    # The inputs are A[0,0..4], A[1,0..4], B[0..4,0..4] = 10 + 25 = 35 inputs
    # But in tensor decomposition, we factor the trilinear form.
    
    # Let's use a direct approach: try to find a rank-r decomposition using linear algebra.
    # For rank r, we need vectors u_c in R^{10}, v_c in R^{25}, w_c in R^{10}
    # such that T = sum_c u_c * v_c * w_c (outer product)
    
    # The key insight for 2x5x5: 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can write this as: for each k, we compute a 2x1 vector:
    #   C[:,k] = A * B[:,k]
    # where A is 2x5 and B[:,k] is 5x1.
    
    # A rank-10 decomposition would use exactly 10 scalar multiplications.
    # One natural approach: use 5 multiplications for each output column independently,
    # but that gives rank 10 only if there's no sharing.
    
    # Actually, let's try a different approach. For the 2x5x5 case,
    # we can use the following strategy:
    # - For each pair (i, k), C[i,k] = sum_j A[i,j] * B[j,k]
    # - We need to compute 10 such sums, each involving 5 products.
    # - Total "naive" products: 50
    # - With rank 10, we use 10 intermediate products that are linear combinations.
    
    # Let's try to construct a rank-10 decomposition directly.
    # The idea: for each k (column of B), we need to compute A * B[:,k].
    # This is a 2x5 times 5x1 multiplication, which has rank at most 5.
    # If we can do each column with 5 products, and the columns are independent,
    # total rank = 5 * 2 = 10? No, that's not right either.
    
    # Let me think about this differently. The tensor T_{i,j,k} = delta_{appropriate}.
    # For 2x5x5: T has nonzero entries at positions (i*m+j, j*p+k, k*n+i) for all valid i,j,k.
    # That's 2*5*5 = 50 nonzero entries.
    
    # A rank-1 decomposition corresponds to one scalar multiplication.
    # For rank r, we need r such decompositions.
    
    # Let's try a constructive approach. Consider:
    # For each j (0..4), define intermediate product P_j = (some linear combo of A) * (some linear combo of B)
    # Then each C[i,k] is a linear combination of the P_j's.
    
    # Actually, the cleanest approach for 2x5x5:
    # Use the fact that we can compute A*B where A is 2x5 and B is 5x5.
    # One approach: 
    #   For j=0..4, compute t_j = A[0,j]*B[j,0] + A[1,j]*B[j,0] ... no that doesn't work.
    
    # Let me try a more systematic approach. I'll set up the linear system.
    # For rank r, we need:
    #   For all (a, b, c) where a in [0,10), b in [0,25), c in [0,10):
    #   T[a,b,c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t]
    #
    # This is a tensor equation. For fixed (a,c), we have:
    #   T[a,:,c] = sum_t U[a,t]*W[c,t] * V[:,t]
    #
    # So for each (a,c), the vector T[a,:,c] (length 25) must be in the span of {V[:,t]} weighted by U[a,t]*W[c,t].
    
    # For the 2x5x5 case, T[a,b,c] = 1 iff:
    #   a = i*5+j, b = j*5+k, c = k*2+i for some i in {0,1}, j in {0..4}, k in {0..4}
    
    # Let's enumerate the nonzero entries:
    # i=0: a=j, b=5*j+k, c=2*k
    #   j=0: a=0, b=k, c=2k for k=0..4 -> (0,0,0),(0,1,2),(0,2,4),(0,3,6),(0,4,8)
    #   j=1: a=1, b=5+k, c=2k for k=0..4 -> (1,5,0),(1,6,2),(1,7,4),(1,8,6),(1,9,8)
    #   j=2: a=2, b=10+k, c=2k for k=0..4 -> (2,10,0),(2,11,2),(2,12,4),(2,13,6),(2,14,8)
    #   j=3: a=3, b=15+k, c=2k for k=0..4 -> (3,15,0),(3,16,2),(3,17,4),(3,18,6),(3,19,8)
    #   j=4: a=4, b=20+k, c=2k for k=0..4 -> (4,20,0),(4,21,2),(4,22,4),(4,23,6),(4,24,8)
    # i=1: a=5+j, b=5*j+k, c=2*k+1
    #   j=0: a=5, b=k, c=2k+1 for k=0..4 -> (5,0,1),(5,1,3),(5,2,5),(5,3,7),(5,4,9)
    #   j=1: a=6, b=5+k, c=2k+1 for k=0..4 -> (6,5,1),(6,6,3),(6,7,5),(6,8,7),(6,9,9)
    #   j=2: a=7, b=10+k, c=2k+1 for k=0..4 -> (7,10,1),(7,11,3),(7,12,5),(7,13,7),(7,14,9)
    #   j=3: a=8, b=15+k, c=2k+1 for k=0..4 -> (8,15,1),(8,16,3),(8,17,5),(8,18,7),(8,19,9)
    #   j=4: a=9, b=20+k, c=2k+1 for k=0..4 -> (9,20,1),(9,21,3),(9,22,5),(9,23,7),(9,24,9)
    
    # Now, for rank 10, let's try a specific structure.
    # Observe that for fixed k, the entries involve all j for both i=0 and i=1.
    # For fixed k, the (a,b) pairs are:
    #   i=0: (j, 5j+k) for j=0..4, with c=2k
    #   i=1: (5+j, 5j+k) for j=0..4, with c=2k+1
    # 
    # So for each k, we have 10 nonzero entries in the tensor slice.
    # The V vectors for these entries all have the same b-index pattern but different a and c.
    
    # Let's try a rank-10 decomposition with the following structure:
    # For t = 0..9, let t = 2*j + s where j in {0..4}, s in {0,1}
    # Define:
    #   V[:, t] has a 1 at position b = 5*j + (t//2) ... hmm, let me think differently.
    
    # Alternative: Let's use the structure where each "multiplication" t corresponds to
    # a specific (j, k) pair, and we combine the two i-values.
    # But that gives 25 terms, not 10.
    
    # Better approach: For each k (0..4), we need to compute two dot products:
    #   C[0,k] = dot(A[0,:], B[:,k])
    #   C[1,k] = dot(A[1,:], B[:,k])
    # 
    # We can compute these using 5 multiplications per k if we use the following trick:
    #   Let x_j = A[0,j] + A[1,j] and y_j = A[0,j] - A[1,j]
    #   Then C[0,k] = sum_j (x_j + y_j)/2 * B[j,k]
    #   C[1,k] = sum_j (x_j - y_j)/2 * B[j,k]
    #   So C[0,k] + C[1,k] = sum_j x_j * B[j,k]  (5 products)
    #   And C[0,k] - C[1,k] = sum_j y_j * B[j,k]  (5 products)
    # Total: 10 products for all 5 columns.
    
    # Wait, that's 10 products total (5 for the sum, 5 for the difference), and we can recover
    # both C[0,k] and C[1,k] from these. So rank 10!
    
    # Let me formalize this:
    # For k=0..4:
    #   S_k = sum_{j=0}^{4} (A[0,j]+A[1,j]) * B[j,k]  -> this is 5 products, one per j
    #   D_k = sum_{j=0}^{4} (A[0,j]-A[1,j]) * B[j,k]  -> this is 5 products, one per j
    # Then:
    #   C[0,k] = (S_k + D_k) / 2
    #   C[1,k] = (S_k - D_k) / 2
    
    # So we have 10 intermediate products:
    #   For j=0..4: P_j = (A[0,j]+A[1,j]) * B[j,k] ... wait, but B[j,k] depends on k.
    
    # I made an error. Let me redo:
    # S_k = sum_j (A[0,j]+A[1,j]) * B[j,k]
    # This is a dot product of the vector (A[0,0]+A[1,0], ..., A[0,4]+A[1,4]) with B[:,k].
    # To compute S_k for all k, we need the full matrix product of a 1x5 vector with a 5x5 matrix,
    # which takes 25 products naively, or 5 products if we precompute... no.
    
    # Actually, S_k = sum_j x_j * B[j,k] where x_j = A[0,j]+A[1,j].
    # Computing all 5 values of S_k requires 25 scalar multiplications (x_j * B[j,k] for each j,k).
    # So that doesn't help.
    
    # Let me reconsider. The correct approach for rank 10:
    # We need 10 scalar multiplications total.
    
    # Here's a better idea. Use the following 10 products:
    # For j=0..4:
    #   t_j = A[0,j] * (B[j,0] + B[j,1] + B[j,2] + B[j,3] + B[j,4])  ... no, this doesn't give individual outputs.
    
    # OK let me try yet another approach. Let's just solve the linear system numerically.
    # For rank r, we can try to find U, V, W by optimization or by solving a system of equations.
    
    # Actually, for the 2x5x5 case, the minimum rank is known to be 10 from the literature
    # (this is a small case that has been studied). Let me try to find an explicit rank-10 decomposition.
    
    # One known approach: use the "Strassen-like" decomposition.
    # For 2x2x2, Strassen gives rank 7 instead of 8.
    # For 2x5x5, there might be a similar reduction.
    
    # Let me try a direct constructive approach:
    # The 10 outputs are C[i,k] for i=0,1 and k=0..4.
    # Each C[i,k] = sum_j A[i,j] * B[j,k].
    
    # Consider the following 10 intermediate products:
    # For k=0..4:
    #   u_k = A[0,k] * B[k,0] + A[0,k] * B[k,1] + ... no, this mixes things.
    
    # Let me try:
    # For j=0..4:
    #   p_j = (A[0,j] + A[1,j]) * (B[j,0] + B[j,1] + B[j,2] + B[j,3] + B[j,4])
    # This gives 5 products, but they don't directly give the outputs.
    
    # I think the cleanest approach is to use a numerical solver. Let me set up the problem as finding
    # a rank-r decomposition using iterative methods.
    
    # For now, let's return the schoolbook decomposition of rank 50, and then try to reduce.
    # Actually, let me try rank 10 with a specific structure.
    
    # Here's a valid rank-10 decomposition for 2x5x5:
    # Use the following 10 products (indexed by (j, s) where j in 0..4, s in {0,1}):
    #   For s=0 (even): product involves A[0,j] and B[j,k]
    #   For s=1 (odd): product involves A[1,j] and B[j,k]
    # But that's just the schoolbook 50 products compressed... no.
    
    # Let me just use a simple approach: return rank 50 schoolbook, which is guaranteed correct.
    # Then if time permits, try to find a lower rank.
    
    rank = n * m * p  # 50
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
    
    # Now try to find a lower rank decomposition using linear algebra.
    # For rank r, we need to solve:
    #   T[a,b,c] = sum_{t=0}^{r-1} U[a,t] * V[b,t] * W[c,t]
    #
    # This is a nonlinear system. We can try a greedy or iterative approach.
    # 
    # Alternative: use the fact that for 2x5x5, a rank-10 decomposition exists.
    # Let me try to construct one explicitly.
    
    # Construction for rank 10:
    # The key idea: for each column k of B, compute the 2-vector A * B[:,k].
    # We can do this with 5 multiplications per column if we use a clever basis.
    # But 5 * 5 = 25, not 10.
    
    # Wait, I think I was overcomplicating. Let me re-examine.
    # For 2x5x5, the tensor has 50 nonzero entries.
    # A rank-r decomposition uses r rank-1 tensors.
    # The minimum rank is the tensor rank.
    
    # For 2x2x2, the tensor rank is 7 (Strassen).
    # For 2x5x5, I believe the tensor rank is 10, but let me verify with a construction.
    
    # Here's a rank-10 construction:
    # Define 10 intermediate values:
    # For j = 0, 1, 2, 3, 4:
    #   t_j = A[0,j] * B[j,0] + A[0,j] * B[j,1] + A[0,j] * B[j,2] + A[0,j] * B[j,3] + A[0,j] * B[j,4]
    #   s_j = A[1,j] * B[j,0] + A[1,j] * B[j,1] + A[1,j] * B[j,2] + A[1,j] * B[j,3] + A[1,j] * B[j,4]
    # 
    # No, that's still 50 products.
    
    # I think the correct approach is:
    # The tensor for 2x5x5 can be decomposed as follows:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # This is equivalent to: for each j, the contribution to C is:
    #   C += A[:,j] * B[j,:]  (outer product of A[:,j] (2x1) and B[j,:] (1x5))
    # 
    # So C = sum_{j=0}^{4} A[:,j] * B[j,:]
    # 
    # Each term A[:,j] * B[j,:] is a 2x5 matrix, and it's a rank-1 tensor in the appropriate sense.
    # But A[:,j] is a vector and B[j,:] is a vector, so their outer product is rank 1 as a matrix,
    # but as a tensor in the (i,j,k) space, it's not a single rank-1 term.
    
    # Actually, A[:,j] * B[j,:] = sum_{i,k} A[i,j]*B[j,k] * e_i * e_k
    # In the tensor T[i,j,k], this contributes to T[i,j,k] for all i,k at fixed j.
    # That's 2*5 = 10 entries for each j, and 5 values of j, giving 50 entries total.
    
    # For a rank-1 tensor decomposition, each term is u * v * w where u is in R^{10}, v in R^{25}, w in R^{10}.
    # A single rank-1 term u*v*w has at most |nonzero(u)|*|nonzero(v)|*|nonzero(w)| nonzero entries.
    
    # I think for this problem, the minimum rank is indeed 10, and a construction is:
    # For each k=0..4, use 2 products (one for each i value), giving 10 total.
    # But each "product" would need to be a product of a linear combination of A's and a linear combination of B's.
    
    # Let me try: for each k, compute:
    #   p_k^0 = (linear combo of A[0,:]) * (linear combo of B[:,k])
    #   p_k^1 = (linear combo of A[1,:]) * (linear combo of B[:,k])
    # Then C[i,k] is a linear combination of the p_k's.
    
    # For this to work with just 2 products per k, we need:
    #   C[0,k] = alpha * p_k^0 + beta * p_k^1
    #   C[1,k] = gamma * p_k^0 + delta * p_k^1
    # where p_k^0 = (sum_j a_j * A[0,j] + sum_j b_j * A[1,j]) * (sum_l c_l * B[l,k])
    # and similarly for p_k^1.
    
    # This is getting complicated. Let me just use a numerical approach to find a rank-10 decomposition.
    
    # Actually, let me try a much simpler approach. I'll use the following known result:
    # For 2x5x5, a rank-10 decomposition can be constructed as follows:
    # 
    # For j=0..4:
    #   Let U_j be a vector in R^{10} with U_j[0*5+j] = 1 and U_j[1*5+j] = 1
    #   Let V_j be a vector in R^{25} with V_j[j*5+0] = 1, V_j[j*5+1] = 1, ..., V_j[j*5+4] = 1
    #   ... no, this doesn't work.
    
    # I'll just return the schoolbook decomposition for now, which is safe and correct.
    # The rank is 50, which gives score 1.0.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
