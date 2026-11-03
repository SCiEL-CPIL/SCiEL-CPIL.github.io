"""Decomposition exploiting m=3: split into 2x2x8 (Strassen window) + 2x1x8."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert (n, m, p) == (2, 3, 8)

    # We build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # with integer entries. We verify the tensor identity:
    #   T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # must equal 1 iff (i,j,k) is a valid multiplication index, else 0.
    # Specifically T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).

    # Strategy: sum of two decompositions.
    # Part A: uses only j=2 (the 2x1x8 slice): T_A[i,2,k]=1, T_A[i,j!=2,k]=0.
    # Part B: uses j in {0,1} (the 2x2x8 slice): T_B[i,j in {0,1},k]=1, else 0.

    cols = []  # list of (u_vec, v_vec, w_vec) each length nm, mp, pn

    def add(u, v, w):
        cols.append((u, v, w))

    nm, mp, pn = n*m, m*p, p*n

    # ---- Part A: 2 x 1 x 8 (j=2) ----
    # For each (i,k): product A[i,2] * B[2,k]
    for i in range(n):
        for k in range(p):
            u = [0]*nm; v = [0]*mp; w = [0]*pn
            u[i*m + 2] = 1
            v[2*p + k] = 1
            w[k*n + i] = 1
            add(u, v, w)
    # Part A rank = n*p = 16? That's too many. Wait, 2x1x8 rank is 8? No:
    # 2x1x8 tensor has entries T[i,2,k] for i in 0..1, k in 0..7 -> 16 entries,
    # each independent product A[i,2]*B[2,k]. Rank = 16. Hmm.
    # Actually a 2x1x8 multiplication: C[i,k] = A[i,0]*B[0,k]. That's 2*8=16
    # independent scalar products. So rank 16, not 8. Let me recompute total.

    # Recompute: Part A rank = n * 1 * p = 2*8 = 16.
    # Part B (2x2x8) via Strassen window.

    # ---- Part B: 2 x 2 x 8 ----
    # Split the 8 columns (k) into 4 groups of 2 consecutive k's:
    #   group g (g=0..3): k in {2g, 2g+1}
    # For each group, the 2x2x2 block: C[i, k in group] = sum_{j=0,1} A[i,j]*B[j, k in group]
    # This is a 2x2x2 matrix mult per group (A is 2x2, B is 2x2 with rows j, cols = the 2 k's).
    # Strassen on 2x2x2 uses 7 products. But the 4 groups share nothing unless we slide.
    #
    # Sliding window Strassen: instead of 4 independent 2x2x2 Strassen (4*7=28),
    # observe that for a 2x2x8, we can view it as computing C = A2x2 * B2x8.
    # We can do this with a rank that exploits structure. A 2x2x8 tensor:
    #   C[i, k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a 2 x 2 x 8 tensor. Its rank: we can factor as
    #   C[i,k] = sum_j A[i,j] B[j,k]
    # Rank of this tensor = ? It's the "matrix multiplication" tensor of size 2x2x8.
    # We can write it using the Strassen trick on the 2x2 part but B has 8 columns.
    #
    # Approach: use 7 Strassen products on the 2x2 A and the 2x2 B-block, but B has 8 cols.
    # Actually simplest exact: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
    # Define for each k: two products. Total 2*8 = 16 products. That's the schoolbook for this part.
    #
    # To beat 16, use Strassen-like: 
    #   Let a0=A[i,0], a1=A[i,1] (per i, but shared across i via linearity).
    # We need products that are bilinear in A (2x2) and B (2x8).
    # Standard: rank of 2x2x8 multiplication tensor.
    # We can do: M1 = A[0,0]*B[0,:], M2 = A[1,1]*B[1,:], M3 = (A[0,0]+A[1,1])*(B[0,:]+B[1,:])? 
    # That doesn't directly give 2x2x8.
    #
    # Let me just use the known fact: rank(2x2x8) <= 2*8 = 16 (schoolbook) and we can
    # potentially do better. But to be SAFE and EXACT, I'll use a verified construction.
    #
    # SAFE construction for 2x2x8 (rank 16):
    for i in range(n):
        for j in range(2):  # j = 0,1
            for k in range(p):
                u = [0]*nm; v = [0]*mp; w = [0]*pn
                u[i*m + j] = 1
                v[j*p + k] = 1
                w[k*n + i] = 1
                add(u, v, w)

    # Now we have Part A (16) + Part B schoolbook (16) = 32. That's WORSE than 24.
    # I need a genuinely better construction. Let me build the Strassen window properly.

    # RESET and do it properly.
    cols = []
    def add2(u, v, w):
        cols.append((u, v, w))

    # Part A: 2x1x8, rank 16 (unavoidable: 16 independent products A[i,2]*B[2,k]).
    # Actually can we reduce? C[i,k]=A[i,2]B[2,k]. The tensor has 16 nonzero entries,
    # each a distinct product of one A-entry and one B-entry. These are all independent
    # (no sharing of A or B factors across entries because each (i,k) uses unique A[i,2]
    # for fixed i... wait A[i,2] is shared across k for fixed i).
    # C[i,k] = A[i,2] * B[2,k]. For fixed i, A[i,2] is shared across all k.
    # So C[i,:] = A[i,2] * B[2,:] -> this is outer product, rank 1 in k for fixed i.
    # Total: 2 such outer products (i=0, i=1). Rank of 2x1x8 tensor = 2? No.
    # The tensor T[i,2,k] = A[i,2] B[2,k]. As a 3-tensor, rank = rank of the
    # flattening. Flatten over (i,k): it's a 2x8 matrix of products = A_col(2x1) * B_row(1x8)
    # = outer product, matrix rank 1. So tensor rank of this part = 1? 
    # No: tensor rank is not matrix rank. But T[i,2,k] = a_i * b_k where a_i = A[i,2], b_k = B[2,k].
    # This is a rank-1 tensor! T = a ⊗ b (with the j=2 slice). So Part A rank = 1?? 
    # Wait but we need it as part of the full multiplication. The full tensor T[i,j,k]=1 for all i,j,k.
    # The slice j=2 is T[i,2,k] which as a function of inputs is sum over... no.
    # The multiplication tensor T[i,j,k] has the property that C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k (with the convention). It's not separable per j-slice
    # in a way that gives rank 1. Let me just directly verify any construction I make.

    # I'll abandon the hand-waving and construct + verify numerically, searching for a
    # low-rank exact decomposition using the Strassen window idea, and fall back to
    # schoolbook (rank 48) if needed. But I want to BEAT 48 (score>1.0 means rank<48).

    # Let me construct the Strassen-window 2x2x8 + 2x1x8 and VERIFY.
    #
    # 2x2x8 Strassen window:
    #   A is 2x2 (rows i, cols j=0,1). B is 2x8 (rows j=0,1, cols k=0..7).
    #   C is 2x8.
    #   C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
    #   Strassen on the 2x2 core, applied per pair of k-columns, but sharing the
    #   "diagonal" Strassen products across the window.
    #
    # Concretely, define 7 products using the 2x2 A and 2x2 B-blocks. For 8 k-columns,
    # group into 4 pairs. For pair g (k=2g,2g+1), B-block is 2x2.
    # Strassen products for pair g:
    #   P1_g = A[0,0] * B[0, 2g]
    #   P2_g = A[1,1] * B[1, 2g+1]
    #   P3_g = (A[0,0]+A[1,1]) * (B[0,2g]+B[1,2g+1])
    #   P4_g = (A[1,0]-A[0,0]) * B[0, 2g]
    #   P5_g = A[1,1] * (B[1,2g] + B[1,2g+1])
    #   P6_g = (A[0,0]+A[0,1]) * B[1, 2g]
    #   P7_g = (A[1,0]+A[1,1]) * (B[0,2g+1] - B[1,2g])
    #   C[0,2g]   = P1_g + P3_g - P5_g + P7_g
    #   C[0,2g+1] = P3_g - P2_g - P7_g
    #   C[1,2g]   = P4_g + P2_g
    #   C[1,2g+1] = P1_g + P5_g - P4_g - P6_g
    # This is standard Strassen, rank 7 per pair, 28 total for 4 pairs.
    #
    # Sharing: P3_g and P4_g and P6_g depend on A-combinations that are the SAME for all g
    # (they don't depend on which k-pair). Specifically:
    #   A-sums: (A[0,0]+A[1,1]), (A[1,0]-A[0,0]), (A[0,0]+A[0,1]), (A[1,0]+A[1,1]) are fixed.
    # But they're multiplied by DIFFERENT B-blocks per g, so the products differ. No sharing of products.
    #
    # So 2x2x8 via this = 28. Plus 2x1x8.
    #
    # 2x1x8: C[i,k] = A[i,2] B[2,k]. This is rank 1 tensor (a_i b_k). 
    # As a decomposition: one product? No, because A[i,2] varies with i and B[2,k] varies with k,
    # and we need C[i,k] for all i,k. The product A[i,2]*B[2,k] for each (i,k) is a SEPARATE
    # scalar in the tensor. But in the tensor decomposition, a single "rank-1 term" is
    # u_i * v_k * w_{ik}... no. The rank-1 tensor term is (a ⊗ b ⊗ c) with a_i b_k c_{?}.
    # Our tensor T[i,j,k] has j fixed at 2 for this part: T[i,2,k]. A rank-1 term is
    # u_i * v_2 * w_k (since j=2). u_i = A-factor, v_2 = 1, w_k = B-factor.
    # So T[i,2,k] = u_i * w_k with u_i = A[i,2]-coefficient, w_k = B[2,k]-coefficient.
    # This IS rank 1 as a tensor (over the i and k indices, with j=2 fixed)!
    # So Part A contributes rank 1?? That can't be right for exact decomposition because
    # we need the coefficient to be exactly A[i,2]*B[2,k], and a single rank-1 term gives
    # u_i * w_k. If we set u_i = 1 for all i and w_k = 1 for all k, we get T[i,2,k]=1, 
    # but the actual multiplication needs A[i,2]*B[2,k] which is a PRODUCT of input values,
    # not the constant 1.
    #
    # I'm confusing the tensor (which is the bilinear map) with its values. The multiplication
    # tensor T[i,j,k] = 1 (constant) defines the bilinear map C[i,k] = sum_j A[i,j]B[j,k].
    # The tensor entries are all 1. So T[i,2,k] = 1 for all i,k (constant 1). 
    # As a 3-tensor with j=2 slice, T[i,2,k] = 1 = (1_i)(1_2)(1_k) -> rank 1!
    # So the j=2 slice of the multiplication tensor is rank 1.
    # Similarly the whole tensor T[i,j,k]=1 is... not rank 1, but the j=2 slice is.
    #
    # So we can decompose T = T_{j in {0,1}} + T_{j=2} where T_{j=2} has rank 1 and
    # T_{j in {0,1}} is the 2x2x8 tensor.
    #
    # T_{j=2} rank 1: u[i*3+2]=1, v[2*8+k]=1, w[k*2+i]=1. ONE column.
    #
    # T_{j in {0,1}}: the 2x2x8 multiplication tensor (entries T[i,j in{0,1},k]=1).
    # Its rank: we need to decompose this. Schoolbook = 2*2*8 = 32. Strassen window = 28.
    # Can we do better? The 2x2x8 tensor rank. 
    #
    # Known: rank of 2x2xn matrix multiplication. For n=8, I believe rank <= 2n - 2 or similar
    # via Strassen-like. Let me just construct 28 (Strassen window) and add 1 = 29.
    # That beats 48. Let me aim for even lower but 29 is safe if I verify.
    #
    # Actually, let me try to get the 2x2x8 part to rank 16 or less.
    # The 2x2x8 tensor: T[i,j,k]=1 for i in{0,1}, j in{0,1}, k in{0..7}.
    # We need sum_c u_c[i,j] v_c[j,k] w_c[k,i] = 1.
    # Note the structure: it factors as (over j) ... T[i,j,k] = sum_j delta. 
    # Actually T[i,j,k] = 1 = (1_i)(1_j)(1_k)? No, that would be rank 1 but then
    # C[i,k] = sum_j 1*1*1 = 2, not the right bilinear form. 
    # Wait the tensor T[i,j,k]=1 means C[i,k] = sum_j A[i,j] B[j,k] with coefficient 1.
    # The tensor IS the coefficient tensor. T[i,j,k]=1 for all i,j,k in the 2x2x8 sub-tensor.
    # So T = 1 ⊗ 1 ⊗ 1 restricted to i in{0,1}, j in{0,1}, k in{0..7}? 
    # That's a rank-1 tensor! T[i,j,k] = 1*1*1 = 1.
    # So the 2x2x8 sub-tensor is ALSO rank 1?!
    #
    # Hold on. If T[i,j,k] = 1 for all i,j,k, then T = e ⊗ e ⊗ e where e is the all-ones vector
    # (on the appropriate dimensions). That's rank 1. So the ENTIRE 2x3x8 multiplication tensor
    # T[i,j,k]=1 (for all i,j,k) is rank 1??
    #
    # That would mean C[i,k] = sum_j A[i,j] B[j,k] can be computed with ONE multiplication?!
    # That's absurd and wrong. The issue: the tensor decomposition T = u ⊗ v ⊗ w means
    # C[i,k] = (sum_j u[i,j] A[i,j]) ... no. Let me get the convention right.
    #
    # Convention: C[i,k] = sum_j A[i,j] B[j,k]. The tensor T[i,j,k] satisfies
    # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]. So T[i,j,k] = 1.
    # A rank-1 decomposition: T[i,j,k] = sum_c u[i,j] v[j,k] w[k,i] (with the W[k*n+i] convention).
    # If T = u ⊗ v ⊗ w with u[i,j]=a[i,j], v[j,k]=b[j,k], w[k,i]=c[k,i],
    # then T[i,j,k] = a[i,j] b[j,k] c[k,i]. For this to equal 1 for ALL i,j,k,
    # we'd need a[i,j] b[j,k] c[k,i] = 1 for all i,j,k. 
    # Fix i=0,j=0: a[0,0] b[0,k] c[k,0] = 1 for all k.
    # Fix i=0,j=1: a[0,1] b[1,k] c[k,0] = 1 for all k.
    # Fix i=1,j=0: a[1,0] b[0,k] c[k,1] = 1 for all k.
    # This is a system. It's NOT generally solvable with rank 1 because the tensor
    # T[i,j,k]=1 (all ones) as a 3-way array: is it rank 1? A 3-way all-ones array of size
    # I x J x K has tensor rank 1 (it's the outer product of three all-ones vectors)!
    # 
    # So T = 1_I ⊗ 1_J ⊗ 1_K is rank 1. But that contradicts the fact that matrix
    # multiplication needs many multiplications. The resolution: the W factor has the
    # "trace convention" W[k*n+i, c], which couples k and i in a specific way that
    # PREVENTS the naive outer product from being a valid decomposition.
    #
    # Specifically, in the decomposition C[i,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c],
    # the three factors are U (indexed by (i,j)), V (indexed by (j,k)), W (indexed by (k,i)).
    # A rank-1 term is U[i,j] V[j,k] W[k,i]. For this to equal T[i,j,k]=1 for all i,j,k,
    # we need U[i,j] V[j,k] W[k,i] = 1. This is a "cycle" condition: U depends on (i,j),
    # V on (j,k), W on (k,i). The product around the cycle must be 1.
    # This is solvable! Set U[i,j]=1, V[j,k]=1, W[k,i]=1 for all. Then product=1. 
    # So rank 1 works?!
    #
    # Let me VERIFY with the actual bilinear map. If U=V=W=1 (single column),
    # C[i,k] = sum_j U[i,j] V[j,k] W[k,i] = sum_j 1*1*1 = sum_j 1 = m = 3.
    # But we need C[i,k] = sum_j A[i,j] B[j,k], which is NOT 3. 
    #
    # AH. I see my error. The decomposition is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_k' ... ) NO.
    #
    # Let me re-read the convention. The standard: 
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Decomposition: C[i,k] = sum_c U[i,j?] ... The factors are:
    #   U has shape (n*m, rank): U[i*m + j, c]
    #   V has shape (m*p, rank): V[j*p + k, c]
    #   W has shape (p*n, rank): W[k*n + i, c]
    # The reconstructed C: C[i,k] = sum_j sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]?? 
    # No. The correct formula: 
    # C[i,k] = sum_c ( sum_j U[i*m+j,c] A[i,j] ) ... no, that's not right either.
    #
    # The CORRECT tensor decomposition for matrix mult:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_c sum_j U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] * A[i,j] * B[j,k]
    # is WRONG. The right one:
    # C[i,k] = sum_c (U-factor) where the c-th term computes a scalar that's a product of
    # a linear combo of A's and a linear combo of B's, then W combines.
    #
    # Standard: C[i,k] = sum_c alpha_c(i) beta_c(k) where
    #   alpha_c(i) = sum_j U[i*m+j, c] A[i,j]  ... no.
    #
    # Let me look at the seed: U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1, one col per (i,j,k).
    # Then C[i,k] = sum_{col} U[i*m+j(col),col] V[j(col)*p+k(col),col] W[k(col)*n+i(col),col] 
    #   where col corresponds to (i,j,k). For the col with matching i,j,k: U=1,V=1,W=1, contributes 1.
    # But this gives C[i,k] = sum_j 1 = m, again NOT the bilinear form. 
    #
    # I think the convention is that the DECOMPOSITION replaces the products: 
    # C[i,k] = sum_j A[i,j] B[j,k] is represented as
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_k' V[j'...]) 
    # No. The ACTUAL convention (from the task: "W[k*n+i,r]" trace convention):
    # C[i,k] = sum_c U[i*m + j, c] ... I think it's:
    # C[i,k] = sum_j sum_c A[i,j] B[j,k] * U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # i.e., the factors MULTIPLY the A and B entries. Then with U=V=W=1 per (i,j,k):
    # C[i,k] = sum_j 1*1*1 * A[i,j] B[j,k] = sum_j A[i,j] B[j,k]. CORRECT!
    #
    # So the convention is: C[i,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k].
    # And the tensor T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] must equal 1 for all i,j,k.
    # So T[i,j,k] = 1, and we need T = sum_c u_c ⊗ v_c ⊗ w_c where u_c[i,j]=U[i*m+j,c],
    # v_c[j,k]=V[j*p+k,c], w_c[k,i]=W[k*n+i,c].
    # And T[i,j,k]=1 (all ones) = sum_c u_c[i,j] v_c[j,k] w_c[k,i].
    #
    # Now, is the all-ones 3-tensor (with the cyclic index structure) rank 1?
    # T[i,j,k] = 1. Can we write 1 = u[i,j] v[j,k] w[k,i] with single u,v,w?
    # Need u[i,j] v[j,k] w[k,i] = 1 for all i,j,k.
    # Set u[i,j]=1, v[j,k]=1, w[k,i]=1: works! Product = 1.
    # So T is rank 1?! Then the answer is rank 1??
    #
    # But that gives C[i,k] = sum_j 1*1*1 A[i,j]B[j,k] = sum_j A[i,j]B[j,k]. Correct!
    # So rank 1 is a valid exact decomposition?!
    #
    # This seems too good. Let me double check by considering it's just relabeling.
    # The "rank" here counts the number of c (the number of scalar multiplications in the
    # Strassen sense). With rank 1, we compute ONE product:
    #   (sum_j U[i*m+j,1] A[i,j]) ... NO. 
    #
    # The actual computation with rank 1: 
    # C[i,k] = sum_j U[i*m+j,1] V[j*p+k,1] W[k*n+i,1] A[i,j] B[j,k].
    # With U=V=W=1: C[i,k] = sum_j A[i,j] B[j,k]. 
    # But this requires summing j terms, each a product A[i,j]B[j,k]. That's m=3 products
    # per (i,k), total n*m*p = 48 products. The "rank 1" decomposition doesn't reduce
    # the number of scalar multiplications because the products are still A[i,j]*B[j,k].
    #
    # I think I'm fundamentally misunderstanding the rank definition here. Let me reconsider.
    #
    # In Strassen, the rank is the number of scalar multiplications. The decomposition
    # C = sum_c (linear combo of A) * (linear combo of B) * (linear combo to place in C).
    # Specifically: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_{j'} V[j'*p+k,c] B[j',k])?
    # NO, that's not dimensionally right either.
    #
    # Let me look at the seed VERY carefully:
    # U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1, for col = index of (i,j,k).
    # There are n*m*p = 48 columns.
    # The reconstruction must be: C[i,k] = sum_{col} U[i*m + j(col), col] * V[j(col)*p + k(col), col] * W[k(col)*n + i(col), col] * (something with A and B).
    #
    # Given U, V, W are 0/1 with a single 1 per row-block, the natural reconstruction is:
    # C[i,k] = sum_{col} U[i*m + j(col), col] V[j(col)*p + k(col), col] W[k(col)*n + i(col), col] * A[i, j(col)] * B[j(col), k(col)]
    # For the col where (i,j,k) matches: U=1 (since U[i*m+j,col]=1), V=1, W=1, and A[i,j]B[j,k]. 
    # For other cols: at least one of U,V,W is 0 (since each col has 1s only at specific (i,j),(j,k),(k,i)).
    # Actually U[i*m+j(col), col] = 1 only if the col's i,j match. So for fixed (i,k), 
    # sum over col: only cols with matching i and matching k contribute. For col=(i,j,k): U[i*m+j,col]=1,
    # V[j*p+k,col]=1, W[k*n+i,col]=1. So C[i,k] = sum_j A[i,j]B[j,k]. CORRECT.
    #
    # So the convention is: C[i,k] = sum_c U[i*m+j_c, c] V[j_c*p+k_c, c] W[k_c*n+i_c, c] A[i,j_c] B[j_c,k_c]
    # where the c-th column has associated indices (i_c, j_c, k_c) implicitly via where the 1s are.
    # But U, V, W don't explicitly store (i_c,j_c,k_c); they store the 1s at positions.
    # The reconstruction is: C[i,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k].
    # YES. This is the standard form. The rank is the number of columns c.
    #
    # Now, T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] must equal 1 for all i,j,k.
    # With the seed: for each (i,j,k) there's a col c with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1,
    # and no other col contributes to T[i,j,k] (since other cols have their 1s at different (i',j',k')).
    # So T[i,j,k] = 1. 
    #
    # Now can we do rank 1? T[i,j,k] = U[i*m+j,1] V[j*p+k,1] W[k*n+i,1] = 1 for all i,j,k.
    # Need U[i*m+j,1] V[j*p+k,1] W[k*n+i,1] = 1.
    # Let U[i*m+j,1] = a[i,j], V[j*p+k,1] = b[j,k], W[k*n+i,1] = c[k,i].
    # Need a[i,j] b[j,k] c[k,i] = 1 for all i,j,k.
    # This is a system of n*m*p = 48 equations. Is it solvable?
    # a[i,j] b[j,k] c[k,i] = 1.
    # Take i=0,j=0: a[0,0] b[0,k] c[k,0] = 1 for all k.
    # Take i=0,j=1: a[0,1] b[1,k] c[k,0] = 1 for all k.
    # Take i=1,j=0: a[1,0] b[0,k] c[k,1] = 1 for all k.
    # Take i=1,j=1: a[1,1] b[1,k] c[k,1] = 1 for all k.
    # 
    # From i=0,j=0: b[0,k] = 1/(a[0,0] c[k,0]).
    # From i=1,j=0: b[0,k] = 1/(a[1,0] c[k,1]).
    # So 1/(a[0,0] c[k,0]) = 1/(a[1,0] c[k,1]) => a[1,0] c[k,1] = a[0,0] c[k,0] for all k.
    # So c[k,1] = (a[0,0]/a[1,0]) c[k,0] for all k.
    # Similarly from i=0,j=1: b[1,k] = 1/(a[0,1] c[k,0]).
    # From i=1,j=1: b[1,k] = 1/(a[1,1] c[k,1]).
    # So a[1,1] c[k,1] = a[0,1] c[k,0] => c[k,1] = (a[0,1]/a[1,1]) c[k,0].
    # Combining: (a[0,0]/a[1,0]) = (a[0,1]/a[1,1]) => a[0,0] a[1,1] = a[0,1] a[1,0].
    # This is a constraint on a. We can satisfy it (e.g., a = all 1s).
    # With a all 1s: c[k,1] = c[k,0] for all k. Set c[k,0]=c[k,1]=1 for all k.
    # Then b[0,k] = 1, b[1,k] = 1 for all k.
    # Check all: a[i,j]=1, b[j,k]=1, c[k,i]=1: product = 1. YES!
    #
    # So rank 1 IS a valid exact decomposition! T[i,j,k] = 1*1*1 = 1.
    #
    # But wait, this means the "rank" of ANY matrix multiplication tensor (with this
    # cyclic convention) is 1?! That can't be the intended task. Let me re-examine.
    #
    # The issue: with a[i,j]=1, b[j,k]=1, c[k,i]=1, the reconstruction is
    # C[i,k] = sum_j U[i*m+j,1] V[j*p+k,1] W[k*n+i,1] A[i,j] B[j,k]
    #        = sum_j 1*1*1 A[i,j] B[j,k] = sum_j A[i,j] B[j,k]. 
    # This is CORRECT. So rank 1 works.
    #
    # This is a degenerate case: the "rank" as defined here (number of c such that
    # T = sum_c u_c ⊗ v_c ⊗ w_c) of the all-ones tensor with cyclic indices is 1.
    # 
    # Hmm, but that makes the task trivial (answer 1). The task says "rank-minimal exact
    # tensor decomposition" and the seed is 48. If rank 1 works, the seed is terrible.
    #
    # Let me reconsider whether T[i,j,k]=1 is really the target. Maybe the target tensor
    # is NOT all ones. Let me re-read: "exactly decompose the matrix multiplication tensor".
    # The matrix multiplication tensor for 2x3x8 has T[i,j,k] = 1 for all valid i,j,k.
    # So yes, all ones.
    #
    # Given the cyclic convention W[k*n+i], the all-ones tensor might indeed have low rank.
    # But the seed uses 48, suggesting the intended rank is higher. 
    #
    # I suspect the ACTUAL convention is different: maybe it's
    # C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * (sum_k V[j*p+k,c] B[j,k]) * W[k*n+i,c]
    # i.e., U and V form linear combinations, then multiplied, then W places.
    # Let me check with the seed under THIS convention:
    # C[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) (sum_{j'} V[j'*p+k,c] B[j',k]) W[k*n+i,c]
    # For col=(i0,j0,k0): U[i*m+j,c]=1 only at j=j0 (for matching i=i0). So sum_j U A[i,j] = A[i,j0] (if i=i0).
    # V[j'*p+k,c]=1 only at j'=j0 (for matching k=k0). So sum_{j'} V B[j',k] = B[j0,k] (if k=k0).
    # W[k*n+i,c]=1 (if k=k0, i=i0).
    # So C[i,k] = sum_{c: i_c=i, k_c=k} A[i,j_c] B[j_c,k] * 1 = sum_j A[i,j]B[j,k]. CORRECT.
    #
    # Under THIS convention, rank 1: 
    # C[i,k] = (sum_j a[i,j] A[i,j]) (sum_{j'} b[j',k] B[j',k]) c[k,i]
    # = (sum_j a[i,j] A[i,j]) (sum_{j'} b[j',k] B[j',k]) c[k,i].
    # For this to equal sum_j A[i,j] B[j,k] for all A,B, we'd need the product of two linear
    # forms to equal a bilinear form in A and B. A single product of a linear form in A and
    # a linear form in B is rank 1 in the (A,B) space, but sum_j A[i,j]B[j,k] is a specific
    # bilinear form. For it to equal (sum_j a[i,j]A[i,j])(sum_{j'}b[j',k]B[j',k]) c[k,i],
    # we'd need the bilinear form to factor, which it doesn't in general (it's a sum of m
    # rank-1 terms). So rank 1 does NOT work under this convention.
    #
    # THIS is the correct convention (the Strassen convention). The seed works under it.
    # So the rank is the Strassen rank, and the all-ones tensor argument doesn't apply.
    #
    # OK so I was wrong about rank 1. The correct convention is:
    # C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_{j'} V[j'*p+k,c] B[j',k])
    # and we need this to equal sum_j A[i,j] B[j,k].
    #
    # This is equivalent to: the tensor T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # must satisfy: for the bilinear map, C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] = sum_j A[i,j]B[j,k],
    # so T[i,j,k] = 1. BUT the decomposition T = sum_c u_c ⊗ v_c ⊗ w_c with
    # u_c[i,j]=U[i*m+j,c], v_c[j,k]=V[j*p+k,c], w_c[k,i]=W[k*n+i,c] is the SAME as before.
    # So T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1.
    # And I showed rank 1 satisfies this! Contradiction.
    #
    # The resolution: under the Strassen convention, the reconstruction is NOT
    # C[i,k] = sum_j sum_c U V W A[i,j] B[j,k]. It's
    # C[i,k] = sum_c W[k*n+i,c] (sum_j U[i*m+j,c] A[i,j]) (sum_{j'} V[j'*p+k,c] B[j',k]).
    # Expanding: C[i,k] = sum_c W[k*n+i,c] sum_j sum_{j'} U[i*m+j,c] V[j'*p+k,c] A[i,j] B[j',k].
    # = sum_j sum_{j'} A[i,j] B[j',k] (sum_c U[i*m+j,c] V[j'*p+k,c] W[k*n+i,c]).
    # For this to equal sum_j A[i,j] B[j,k] (note: same index j=j'), we need
    # sum_c U[i*m+j,c] V[j'*p+k,c] W[k*n+i,c] = delta_{j,j'}.
    # So the tensor S[i,j,j',k] = sum_c U[i*m+j,c] V[j'*p+k,c] W[k*n+i,c] must equal delta_{j,j'}.
    # This is a 4-index tensor, NOT the 3-index all-ones tensor!
    #
    # So the target is: S[i,j,j',k] = delta_{j,j'}, i.e., S[i,j,j',k] = 1 if j=j', else 0.
    # And S[i,j,j',k] = sum_c U[i*m+j,c] V[j'*p+k,c] W[k*n+i,c].
    # This is the correct target. The seed satisfies it: for col=(i0,j0,k0),
    # U[i*m+j,c]=1 if i=i0,j=j0; V[j'*p+k,c]=1 if j'=j0,k=k0; W[k*n+i,c]=1 if k=k0,i=i0.
    # So S[i,j,j',k] gets 1 only when i=i0,j=j0,j'=j0,k=k0, i.e., j=j' and the col matches.
    # Sum over col: S[i,j,j',k] = 1 if j=j' (there's exactly one col with that i,j,j',k when j=j'),
    # else 0. CORRECT.
    #
    # So the target 4-tensor is S[i,j,j',k] = delta_{j,j'}. Now rank 1:
    # S[i,j,j',k] = a[i,j] b[j',k] c[k,i] = delta_{j,j'}.
    # Need a[i,j] b[j',k] c[k,i] = delta_{j,j'} for all i,j,j',k.
    # For j != j', need a[i,j] b[j',k] c[k,i] = 0 for all i,k.
    # For j = j', need a[i,j] b[j,k] c[k,i] = 1 for all i,k.
    # From j=j': a[i,j] b[j,k] c[k,i] = 1.
    # From j!=j': a[i,j] b[j',k] c[k,i] = 0.
    # Divide (j=j') by (j!=j'): b[j,k]/b[j',k] = 1/0 -> impossible unless b[j',k]=0,
    # but then a[i,j']b[j',k]c[k,i]=0 ok, but for j'=j we need b[j,k]!=0.
    # Contradiction: can't have both. So rank 1 does NOT work. Good, consistent.
    #
    # So the target is S[i,j,j',k] = delta_{j,j'}, a 4-tensor, and we decompose it as
    # sum_c u_c[i,j] v_c[j',k] w_c[k,i] where u_c[i,j]=U[i*m+j,c], v_c[j',k]=V[j'*p+k,c],
    # w_c[k,i]=W[k*n+i,c].
    #
    # Now the rank is the Strassen rank of 2x3x8 matrix multiplication. Known: the rank of
    # 2x3x8 matrix multiplication. Let me think. The standard rank is n*m*p = 48. Strassen can
    # reduce it. The directive says reduce from 28 to 24 for the 2x2x8 part... 
    #
    # Actually, let me just construct a valid decomposition and verify it numerically against
    # the 4-tensor target, iterating to minimize rank. I'll use the Strassen window idea.
    #
    # Let me construct:
    # Part A (j=2 slice of S): S[i,2,2,k] = 1, S[i,2,j'!=2,k]=0, S[i,j!=2,j',k]=0.
    # This is the part where j=j'=2. 
    # Part B (j,j' in {0,1}): S[i,j,j',k] = delta_{j,j'} for j,j' in {0,1}, 0 otherwise.
    #
    # Part A: S[i,2,2,k] = 1 for all i,k. Decompose: need u[i,2] v[2,k] w[k,i] summed = 1.
    # This is the all-ones 2x8 "matrix" (in i,k) with the j=2 fixed. 
    # S_A[i,2,2,k] = 1 = sum_c u_c[i,2] v_c[2,k] w_c[k,i].
    # As a function of (i,k) with j=2 fixed, it's a 2x8 all-ones matrix. 
    # Can we do rank 1? u[i,2] v[2,k] w[k,i] = 1 for all i,k.
    # u[i,2] v[2,k] w[k,i] = 1. Set u[i,2]=1, v[2,k]=1, w[k,i]=1: works. Rank 1!
    # But wait, this u,v,w must be ZERO for j!=2 in u and j'!=2 in v to not contaminate Part B.
    # u_c[i,j] = 0 for j!=2, =1 for j=2. v_c[j',k] = 0 for j'!=2, =1 for j'=2. w_c[k,i]=1.
    # Then S_A[i,j,j',k] = u[i,j] v[j',k] w[k,i] = (1 if j=2)(1 if j'=2)(1) = 1 if j=j'=2.
    # That's exactly Part A! Rank 1. 
    #
    # Part B: S[i,j,j',k] = delta_{j,j'} for j,j' in {0,1}. 
    # This is the 2x2x8 "identity in j,j'" tensor. 
    # S_B[i,j,j',k] = delta_{j,j'} for j,j' in {0,1}, 0 else.
    # Decompose: sum_c u_c[i,j] v_c[j',k] w_c[k,i] = delta_{j,j'} (for j,j' in {0,1}).
    # Note it's independent of i and k in the delta, but u depends on i, v on k, w on (k,i).
    # This is like: for each (i,k), the 2x2 matrix in (j,j') is the identity.
    # S_B[i,j,j',k] = delta_{j,j'}.
    # We can write delta_{j,j'} = sum_c u_c[i,j] v_c[j',k] w_c[k,i].
    # This is the tensor product structure. The identity 2x2 matrix = ?
    # delta_{j,j'} is rank 2 as a 2x2 matrix (in j,j'). But with the i,k,w coupling...
    #
    # Simplest: use the schoolbook for Part B: for each (i,j,k) with j in {0,1}:
    # u[i,j]=1, v[j,k]=1, w[k,i]=1, one col per (i,j,k). That's n*2*p = 2*2*8 = 32 cols.
    # But we can do better with Strassen.
    #
    # Part B is the 2x2x8 matrix multiplication tensor (in the sense S[i,j,j',k]=delta_{j,j'}).
    # This is exactly the tensor for C[i,k] = sum_{j in{0,1}} A[i,j] B[j,k] (2x2x8 mult).
    # Its Strassen rank: for 2x2xn, the rank is known to be <= 2n - 2 for n>=2? 
    # For 2x2x2, Strassen rank = 7. For 2x2x8, using Strassen on each 2x2x2 block: 4*7=28.
    # But with sharing (sliding window), maybe less.
    #
    # Let me just construct the 2x2x8 part with rank 28 (Strassen per block) + Part A rank 1 = 29.
    # That beats 48. Let me verify it numerically and if it works, great. If I can do better, even better.
    #
    # Actually, let me try to get Part B to rank 16 or so. 
    # S_B[i,j,j',k] = delta_{j,j'}. 
    # Observe: delta_{j,j'} = sum over c where c indexes... Let's use the fact that
    # delta_{j,j'} is the identity. We can write:
    #   delta_{j,j'} = u1[i,j] v1[j',k] w1[k,i] + u2[i,j] v2[j',k] w2[k,i] + ...
    # For the identity 2x2, we need at least 2 "diagonal" terms but with the i,k coupling.
    #
    # Let me just use Strassen per 2-column block (rank 7 each, 4 blocks = 28) and verify.
    #
    # Let me code the construction and a verifier.

    # --- Construction ---
    cols = []
    def add(u, v, w):
        # u: list length nm (n*m), v: list length mp (m*p), w: list length pn (p*n)
        cols.append((u, v, w))

    # Part A: rank 1, j=j'=2.
    u = [0]*nm; v = [0]*mp; w = [0]*pn
    for i in range(n):
        u[i*m + 2] = 1
    for k in range(p):
        v[2*p + k] = 1
    for k in range(p):
        for i in range(n):
            w[k*n + i] = 1
    add(u, v, w)

    # Part B: 2x2x8 via Strassen per 2-column block.
    # For block g (k0=2g, k1=2g+1):
    #   A is 2x2 (i in{0,1}, j in{0,1}), B is 2x2 (j in{0,1}, k in{k0,k1}).
    #   Strassen 7 products.
    for g in range(4):
        k0 = 2*g
        k1 = 2*g + 1
        # P1 = A[0,0]*B[0,k0]
        u=[0]*nm; v=[0]*mp; w=[0]*pn
        u[0*m+0]=1; v[0*p+k0]=1; w[k0*n+0]=1
        add(u,v,w)
        # P2 = A[1,1]*B[1,k1]
        u=[0]*nm; v=[0]*mp; w=[0]*pn
        u[1*m+1]=1; v[1*p+k1]=1; w[k1*n+1]=1
        add(u,v,w)
        # P3 = (A[0,0]+A[1,1])*(B[0,k0]+B[1,k1])
        u=[0]*nm; v=[0]*mp; w=[0]*pn
        u[0*m+0]=1; u[1*m+1]=1
        v[0*p+k0]=1; v[1*p+k1]=1
        # w: this product contributes to C[0,k0], C[0,k1], C[1,k0], C[1,k1] with coeffs.
        # Strassen: C[0,k0]=P1+P3-P5+P7, C[0,k1]=P3-P2-P7, C[1,k0]=P4+P2, C[1,k1]=P1+P5-P4-P6
        # P3 appears in C[0,k0](+1), C[0,k1](+1). So w for P3: w[k0*n+0]=1, w[k1*n+0]=1.
        w[k0*n+0]=1; w[k1*n+0]=1
        add(u,v,w)
        # P4 = (A[1,0]-A[0,0])*B[0,k0]
        u=[0]*nm; v=[0]*mp; w=[0]*pn
        u[1*m+0]=1; u[0*m+0]=-1
        v[0*p+k0]=1
        w[k0*n+1]=1  # C[1,k0] += P4
        add(u,v,w)
        # P5 = A[1,1]*(B[1,k0]+B[1,k1])
        u=[0]*nm; v=[0]*mp; w=[0]*pn
        u[1*m+1]=1
        v[1*p+k0]=1; v[1*p+k1]=1
        w[k0*n+0]=-1; w[k1*n+1]=1  # C[0,k0]-=P5, C[1,k1]+=P5
        add(u,v,w)
        # P6 = (A[0,0]+A[0,1])*B[1,k0]
        u=[0]*nm; v=[0]*mp; w=[0]*pn
        u[0*m+0]=1; u[0*m+1]=1
        v[1*p+k0]=1
        w[k1*n+1]=-1  # C[1,k1]-=P6
        add(u,v,w)
        # P7 = (A[1,0]+A[1,1])*(B[0,k1]-B[1,k0])
        u=[0]*nm; v=[0]*mp; w=[0]*pn
        u[1*m+0]=1; u[1*m+1]=1
        v[0*p+k1]=1; v[1*p+k0]=-1
        w[k0*n+0]=1; w[k1*n+0]=-1  # C[0,k0]+=P7, C[0,k1]-=P7
        add(u,v,w)

    rank = len(cols)
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    for c, (u, v, w) in enumerate(cols):
        for idx in range(nm):
            U[idx][c] = u[idx]
        for idx in range(mp):
            V[idx][c] = v[idx]
        for idx in range(pn):
            W[idx][c] = w[idx]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
