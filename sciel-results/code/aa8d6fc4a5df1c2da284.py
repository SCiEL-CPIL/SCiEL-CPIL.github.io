"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Construct a low-rank exact decomposition for 4x5 * 5x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll use a decomposition based on explicit rank-1 terms.
    # The idea: use a structured decomposition that exploits the tensor symmetry.
    # For (4,5,5), we can use a construction where we decompose the 5x5 core
    # using known low-rank schemes and project onto the 4-row space.
    
    # Strategy: Use a decomposition with rank < 100 by combining terms.
    # We'll use a construction based on the following insight:
    # Each output entry C[i,k] = sum_j A[i,j]*B[j,k]
    # We can group the j-sums and use shared bilinear forms.
    
    # For now, let's use a construction that achieves rank 75 by
    # sharing 5 of the 25 column multipliers across the 4 rows.
    # Actually, let's use a more principled approach.
    
    # Use the fact that for the (4,5,5) case, we can use a decomposition
    # where we have 4*5 = 20 terms for the "diagonal" and additional
    # terms for cross-products, but optimized.
    
    # Let's use a known construction: the (n,m,p) tensor can be decomposed
    # with rank at most n*m + m*p + n*p - something.
    
    # Actually, let me use a direct construction with rank 70.
    # We'll use 5 groups of 14 rank-1 terms each.
    
    # Better: use the following explicit decomposition.
    # For each pair (i, k) in the output, C[i,k] = sum_j A[i,j] B[j,k]
    # We need rank-1 tensors u_i * v_j * w_k such that sum u_i v_j w_k = delta_ik
    
    # Let's use a construction with rank = n*p + (n-1)*m*p = 20 + 3*25 = 95. No, that's worse.
    
    # Let me try: rank = m*p + (n-1)*m = 25 + 15 = 40? Let's see if that works.
    
    # Construction with rank 40:
    # For j=0: use 5 terms (one for each k) with U[i*5+0]=1 for all i, V[0*5+k]=1, W[k*4+i] = delta_{i,0}... 
    # This doesn't quite work directly.
    
    # Let me use a different approach: the "border rank" style construction.
    
    # Actually, let's use a simple but effective construction:
    # Rank = n*p + m*(p-1) = 20 + 20 = 40
    # Terms 1-20: for each (i,k), a term that captures A[i,*] dot B[*,k] partially
    # Terms 21-40: correction terms
    
    # Let me just use a verified construction. I'll use rank 75 with a clear structure.
    
    # Construction: For each i in 0..3, j in 0..4:
    #   Term: U[i*m+j] = 1, V[j*p + (j%p)] = 1, W[(j%p)*n + i] = 1
    # This gives 20 terms but doesn't capture all cross terms.
    
    # Let me use the following explicit rank-75 decomposition:
    # We'll have 75 rank-1 terms.
    
    rank = 75
    U = [[0]*rank for _ in range(n*m)]  # 20 x 75
    V = [[0]*rank for _ in range(m*p)]  # 25 x 75
    W = [[0]*rank for _ in range(p*n)]  # 20 x 75
    
    col = 0
    # Group 1: 25 terms for the "main" diagonal-like structure
    # For each j in 0..4, k in 0..4:
    for j in range(m):
        for k in range(p):
            # This term contributes to all i: A[i,j] * B[j,k]
            for i in range(n):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
            col += 1
    
    # Wait, that's just the schoolbook for 5*5=25 terms but with all 4 i's.
    # That gives rank 25*... no, each term has 4 non-zero entries in U and W.
    # That's not a valid rank-1 decomposition because U, V, W must be vectors.
    
    # Let me restart with a proper construction.
    
    # Proper rank-1: each term c has vectors u_c (length 20), v_c (length 25), w_c (length 20)
    # such that T = sum_c u_c ⊗ v_c ⊗ w_c
    
    # Schoolbook: T[i,j,k] = 1 iff output (i,k) uses input (i,j) and (j,k)
    # i.e., T[i,j,k] = 1 for all valid i,j,k (it's the full multiplication tensor)
    
    # Actually T[i,j,k] = 1 if the (i,k) entry of the product involves A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i in 0..3, j in 0..4, k in 0..4.
    # That's a 4x5x5 all-ones tensor? No!
    
    # Wait. The matrix multiplication tensor T_{i,j,k} is defined such that:
    # (A*B)[i,k] = sum_j T_{i,j,k} * A[i,j] * B[j,k]
    # So T_{i,j,k} = 1 if the (i,k) output depends on A[i,j] and B[j,k], which is always
    # the case for the standard multiplication. So T_{i,j,k} = 1 for all i,j,k?
    
    # No! That's wrong. T_{i,j,k} should be 1 only when the j-th element of row i of A
    # multiplies with the j-th element of column k of B. So for each output (i,k),
    # we sum over j: A[i,j]*B[j,k]. So T_{i,j,k} = 1 for all i,j,k in valid range.
    
    # Hmm, but that would make it an all-ones tensor, which has rank 1. That can't be right.
    
    # Let me reconsider. The tensor is T ∈ R^{(n*m) × (m*p) × (p*n)} where
    # T[a,b,c] corresponds to the coefficient in the bilinear form.
    # a = i*m + j (index into A), b = j*p + k (index into B), c = k*n + i (index into C)
    # T[a,b,c] = 1 if a corresponds to (i,j), b to (j,k), and c to (k,i) with consistent i,j,k.
    
    # So T[(i*m+j), (j*p+k), (k*n+i)] = 1 for valid i,j,k, and 0 otherwise.
    # This is the standard matrix multiplication tensor.
    
    # OK so my original understanding was correct. Let me use a known low-rank construction.
    
    # For (4,5,5), let me use a construction with rank 60.
    # I'll use the following approach based on explicit formulas:
    
    # Reset
    rank = 60
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # Use 4 groups of 15 terms each.
    # For each i in 0..3:
    for i in range(n):
        for t in range(15):
            # Define u, v, w vectors for this term
            # u has non-zero at position i*m + j for various j
            # v has non-zero at position j*p + k for various j,k
            # w has non-zero at position k*n + i for various k
            
            # Let's use a structured pattern
            # Term t in group i:
            # u[i*m + (t % 5)] = 1  (one j per term)
            # v[(t % 5)*p + (t // 5)] = 1  (one (j,k) per term, j=t%5, k=t//5)
            # w[(t // 5)*n + i] = 1  (one k per term)
            
            j = t % 5
            k = t // 5
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    # Verify: does this give rank 60 and is it correct?
    # For output (i,k): sum over j of A[i,j]*B[j,k]
    # In our decomposition, for fixed i,k:
    #   Sum over terms where w[k*n+i] != 0 and u[i*m+j'] != 0 and v[j'*p+k'] != 0
    #   with j'=j and k'=k
    #   For term in group i with t = j*5+k: u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1
    #   So the contribution is A[i,j]*B[j,k] for each j,k.
    #   But we only have one term per (j,k) in group i, so we get exactly sum_j A[i,j]*B[j,k].
    #   Wait, we have 15 terms per group i, and (j,k) ranges over 5*5=25 combinations.
    #   But t ranges 0..14, so j=t%5, k=t//5 only covers k=0,1,2 (since t//5 for t=0..14 gives 0,0,0,0,0,1,1,1,1,1,2,2,2,2,2).
    #   So we only cover k=0,1,2! That's wrong.
    
    # Fix: use 25 terms per group, or restructure.
    
    # Let me use rank = 4*25 = 100 (schoolbook) but that's not better.
    
    # Let me use a different construction. Rank = 5*12 = 60.
    # For each k in 0..4: 12 terms
    for _ in range(rank):
        pass  # Reset approach
    
    # New construction: rank 50
    # Use the following: for each j in 0..4, have 10 terms that cover all (i,k) pairs
    # Term for (j, i, k) where we share across i's.
    
    # Actually, let me just use rank 75 with a correct construction.
    rank = 75
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # For each j in 0..4 (5 groups), for each (i,k) in 0..3 x 0..4 (20 per group):
    #   Term: u[i*m+j] = 1, v[j*p+k] = 1, w[k*n+i] = 1
    for j in range(m):
        for i in range(n):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # This is rank 5*4*5 = 100. Still schoolbook.
    
    # I need a genuinely different construction. Let me use the following:
    # The key is to share terms across different (i,k) pairs.
    
    # Construction with rank 55:
    # Part 1: 25 terms for j=0..4, k=0..4, with u[i*m+j]=1 for all i (shared across i)
    #   But then w[k*n+i] must be 1 for all i, which means w is all-ones in the i-block.
    #   That would give C[i,k] += A[i,j]*B[j,k] for all i simultaneously.
    #   So: u_c[i*m+j] = 1 for all i, v_c[j*p+k] = 1, w_c[k*n+i] = 1 for all i.
    #   This is valid! Each such term contributes to all 4 rows.
    #   25 such terms.
    
    # Part 2: We need to verify this is correct.
    # For output (i,k): sum over terms where w[k*n+i]!=0.
    # In Part 1, for term (j,k): w[k*n+i]=1 for all i, u[i*m+j]=1, v[j*p+k]=1.
    # Contribution: A[i,j]*B[j,k] for each j. Sum over j: exactly what we want!
    
    # So Part 1 alone gives rank 25 and is correct!
    
    # Wait, is that right? Let me double-check.
    # Term c = (j,k): u_c is a vector of length 20 with u_c[i*m+j] = 1 for i=0..3, 0 otherwise.
    # v_c is a vector of length 25 with v_c[j*p+k] = 1, 0 otherwise.
    # w_c is a vector of length 20 with w_c[k*n+i] = 1 for i=0..3, 0 otherwise.
    
    # The reconstruction: C[k*n+i] = sum_c w_c[k*n+i] * (sum_a u_c[a]*A[a]) * (sum_b v_c[b]*B[b])
    # = sum_c w_c[k*n+i] * A[i*m+j_c] * B[j_c*p+k_c]
    
    # For output (i,k): sum over c where w_c[k*n+i] != 0.
    # w_c[k*n+i] = 1 iff c = (j,k) for some j (since w_c has 1s at positions k_c*n + i for all i).
    # So for fixed (i,k), the terms with w_c[k*n+i]=1 are those with k_c=k, any j_c.
    # Contribution: sum_{j_c} A[i*m+j_c] * B[j_c*p+k] = sum_j A[i,j]*B[j,k]. Correct!
    
    # So rank 25 works! But wait, that seems too good. Let me re-examine.
    
    # The issue: u_c[i*m+j] = 1 for ALL i. But u_c is a single vector. So u_c has 1s at
    # positions 0*m+j, 1*m+j, 2*m+j, 3*m+j. That's 4 non-zero entries.
    # Similarly w_c has 4 non-zero entries.
    # v_c has 1 non-zero entry.
    # This is a valid rank-1 tensor (outer product of three vectors).
    
    # And the sum over all 25 such terms gives the correct multiplication!
    
    # So the rank is 25, not 100!
    
    # Let me verify with a small example: n=2, m=2, p=2.
    # Rank should be m*p = 4.
    # Terms: (j,k) for j=0,1 and k=0,1.
    # Term (0,0): u = [1,0,1,0] (positions 0*2+0, 1*2+0), v=[1,0,0,0], w=[1,0,1,0]
    # Term (0,1): u = [1,0,1,0], v=[0,1,0,0], w=[0,1,0,1]
    # Term (1,0): u = [0,1,0,1], v=[0,0,1,0], w=[1,0,1,0]
    # Term (1,1): u = [0,1,0,1], v=[0,0,0,1], w=[0,1,0,1]
    
    # Output C[0,0] (position 0*2+0=0): sum_c w_c[0] * A_part * B_part
    #   c=(0,0): w[0]=1, A_part=A[0,0], B_part=B[0,0] -> A[0,0]*B[0,0]
    #   c=(0,1): w[0]=0
    #   c=(1,0): w[0]=1, A_part=A[0,1], B_part=B[1,0] -> A[0,1]*B[1,0]
    #   c=(1,1): w[0]=0
    #   Total: A[0,0]*B[0,0] + A[0,1]*B[1,0]. Correct!
    
    # Output C[1,0] (position 0*2+1=1): 
    #   c=(0,0): w[1]=0
    #   c=(0,1): w[1]=1, A_part=A[1,0], B_part=B[0,1] -> A[1,0]*B[0,1]
    #   c=(1,0): w[1]=1, A_part=A[1,1], B_part=B[1,0] -> A[1,1]*B[1,0]
    #   c=(1,1): w[1]=0
    #   Total: A[1,0]*B[0,1] + A[1,1]*B[1,0]. But we want A[1,0]*B[0,0] + A[1,1]*B[1,0]!
    
    # Oops, it's wrong! The issue is that w_c[k*n+i] for c=(0,1) has k=1, so w[1*2+i] = w[2+i].
    # For i=1: w[3], not w[1]. Let me re-index.
    
    # W index: k*n + i. For n=2:
    # k=0: positions 0,1 (i=0,1)
    # k=1: positions 2,3 (i=0,1)
    
    # So w for term (j=0,k=1): non-zero at positions 1*2+0=2 and 1*2+1=3.
    # Output C[1,0] is at position k=0,i=1 -> index 0*2+1=1.
    # w_c[1] for c=(0,1): w has non-zeros at 2,3, so w[1]=0. Correct, no contribution.
    
    # Let me redo: Output C[i,k] is at position k*n+i.
    # C[0,0]: pos 0. C[1,0]: pos 1. C[0,1]: pos 2. C[1,1]: pos 3.
    
    # C[1,0] (pos 1): 
    #   c=(0,0): w[1]=1 (k=0, i=1). A_part: u has 1s at 0,2 -> A[0,0] and A[1,0]. 
    #     But u is a vector, so sum_a u_c[a]*A[a] = A[0,0] + A[1,0]. That's wrong!
    
    # The problem: u_c has multiple non-zero entries, so it's not selecting a single A[i,j].
    
    # I see the error. The rank-1 term u⊗v⊗w means:
    # C[c] += w[c] * (sum_a u[a]*A[a]) * (sum_b v[b]*B[b])
    
    # If u has 1s at multiple positions, it's summing multiple A entries, which is not what we want
    # for a single rank-1 term in the standard sense.
    
    # Actually, this IS a valid rank-1 tensor decomposition. The "rank" here is the tensor rank,
    # and each term is a simple tensor (outer product of three vectors). The vectors can have
    # multiple non-zero entries.
    
    # But the issue is correctness. Let me re-examine.
    
    # For the (2,2,2) case with my construction:
    # Term (j=0,k=0): u=[1,0,1,0], v=[1,0,0,0], w=[1,0,1,0]
    #   Contribution to C: w * (u·A) * (v·B) = [1,0,1,0] * (A[0,0]+A[1,0]) * B[0,0]
    #   = [A[0,0]+A[1,0], 0, A[0,0]+A[1,0], 0] * B[0,0]
    
    # This is clearly wrong because it adds A[1,0] to C[0,0] and C[0,1].
    
    # So my construction is invalid. The vectors must be such that the decomposition is exact.
    
    # The correct schoolbook has each term with a single non-zero in each of u, v, w.
    # To get lower rank, we need to be more clever.
    
    # Let me just use the schoolbook rank 100 for now, since I don't have a verified low-rank
    # construction. But that gives score 1.0.
    
    # Actually, let me try a different approach: use the known result that the (n,m,p) tensor
    # has rank at most n*m*p - (n-1)*(m-1)*(p-1) or similar bounds.
    
    # For now, let me implement the schoolbook correctly and return it.
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

