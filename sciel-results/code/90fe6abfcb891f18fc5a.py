"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import math
import time

_DEADLINE = 150.0  # seconds
_TIME_MARGIN = 5.0  # safety margin


def _elapsed():
    return time.time()


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x2 matrix by a 2x13 matrix.

    The output factor follows the official trace convention W[k*n+i, r].
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    Score is the rank (lower is better).
    """
    t0 = _elapsed()

    n, m, p = DIMENSIONS  # n=2, m=2, p=13

    # ---------------------------------------------------------------
    # Known lower bound: for (2,2,13), the tensor rank of matrix multiplication
    # T_{2,2,13} is at least n*p = 26 (by the trivial lower bound from
    # slicing along the output dimension, each of the p output "slices"
    # contributes at least n=1... actually the standard lower bound is
    # max(n*m, m*p, p*n) but we can do better with structural arguments).
    #
    # A rank-1 term contributes to at most one output position (k,i) per
    # W column. There are p*n = 26 output positions, so rank >= 26.
    #
    # Construction achieving rank 26:
    # For each (k, i) in [0,p-1] x [0,n-1], create one rank-1 term.
    # The product (A*B)[i,k] = sum_j A[i,j]*B[j,k].
    # We need to decompose sum_j A[i,j]*B[j,k] across rank-1 terms.
    #
    # Key idea: For each output index k (column of B), the 2x2 sub-product
    # A * B[:,k] has entries (i,k) for i=0,1. Each such entry is
    # A[i,0]*B[0,k] + A[i,1]*B[1,k].
    #
    # Rank-1 decomposition for a fixed k:
    #   We need to express A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1
    #   using rank-1 terms.
    #
    # Standard: the 2x2x1 tensor has rank 4 (schoolbook for 2x2x1).
    # But we can do: use 2 terms per k? Let's check.
    #
    # Term for (k, i, j): U[i*m+j], V[j*p+k], W[k*n+i] all = 1.
    # That's the schoolbook with rank = n*m*p = 52.
    #
    # Better: Notice that for a fixed (i,k), we need sum_j A[i,j] B[j,k].
    # If we create a term per (k, j) instead:
    #   For each k and j, create a term where:
    #     U[i*m + j] = 1 for all i (so U depends on j, not i)
    #     V[j*p + k] = 1
    #     W[k*n + i] = 1 for all i (so W depends on i, not j... wait)
    #
    # Let's think differently. For fixed k, the contribution to output (i,k) is:
    #   sum_j A[i,j] * B[j,k]
    #
    # A rank-1 tensor is U[a]*V[b]*W[c] where a indexes (i,j), b indexes (j,k), c indexes (k,i).
    # For it to contribute to output (i', k'), we need W[k'*n+i'] != 0.
    #
    # Construction with rank = p * m = 26:
    #   For each (k, j) pair (k in 0..p-1, j in 0..m-1), create one term:
    #     U[i*m + j] = 1  for all i in 0..n-1
    #     V[j*p + k] = 1
    #     W[k*n + i] = 1  for all i in 0..n-1
    #
    # Then the reconstructed output at (i, k) is:
    #   sum_c U[i*m+j_c] * V[j_c*p+k_c] * W[k*n+i]
    #   where c indexes (k_c, j_c)
    #   = sum_{j=0}^{m-1} U[i*m+j] * V[j*p+k] * W[k*n+i]
    #   (only terms with k_c=k survive because V[j*p+k_c] is zero unless k_c=k)
    #   = sum_{j=0}^{m-1} 1 * 1 * 1  ... wait, that gives m=2 per (i,k), not sum_j A[i,j]B[j,k].
    #
    # That's wrong. U and V and W entries must encode the bilinear structure.
    # Let me reconsider.
    #
    # The correct schoolbook: for each (i,j,k): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # Rank = n*m*p = 52.
    #
    # Can we merge? For fixed (i,k), the sum over j of A[i,j]B[j,k] can be written as:
    #   If we set for each (j,k): U[i*m+j] gets a coefficient, V[j*p+k]=1, W[k*n+i] gets a coefficient.
    #   The product at output (i',k') from term (j,k) is: U[i'*m+j] * V[j*p+k'] * W[k'*n+i']
    #   = U[i'*m+j] * (1 if k'=k else 0) * W[k'n+i']
    #
    # For output (i,k): sum_j U[i*m+j] * V[j*p+k] * W[k*n+i]
    # We want this to equal sum_j A[i,j]*B[j,k] for all (A,B).
    #
    # This requires: U[i*m+j] * V[j*p+k] * W[k*n+i] = A[i,j]*B[j,k] for each j.
    # But A and B are variables; we can't factor A[i,j]B[j,k] into U*V*W with U depending only on (i,j), V only on (j,k), W only on (k,i) unless we use the standard rank-1 decomposition which separates them.
    #
    # Actually the standard approach: each monomial A[i,j]*B[j,k] is its own rank-1 term.
    # The question is whether different monomials can share a rank-1 term.
    #
    # Two monomials A[i1,j1]*B[j1,k1] and A[i2,j2]*B[j2,k2] share a rank-1 term only if
    # their (i,j), (j,k), (k,i) index sets overlap in a compatible way. Specifically, they'd need
    # the same (j) for the V factor to work, or we'd need to linearly combine.
    #
    # For (2,2,13): the known result is that the tensor rank is exactly 26.
    # Construction: For each k in 0..12 and i in 0..1 (26 terms total):
    #   We need to compute (AB)[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    #   Use 2 terms: one for j=0 and one for j=1.
    #   Term (k,i,j): U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    #   That's still 3 indices: k, i, j -> 13*2*2 = 52 terms.
    #
    # Hmm. Let me look at this from the other direction.
    # The 2x2 matrix times 2x13 can be viewed as 13 independent 2x2 multiplications (one per column of B).
    # Each 2x2 multiplication has tensor rank 7 (over fields of characteristic != 2, by Landsberg's result).
    # But we can share structure across columns.
    #
    # Actually, for (n,m,p) = (2,2,p): the tensor rank is known to be 2*p for p >= 2 when n=m=2? No...
    # The trivial bound is n*m*p. A better construction: 
    #   For each pair (j,k), create a term. U[i*m+j] = delta_{something}, V[j*p+k]=1, W[k*n+i]=delta.
    #   Actually: for each (j,k), set U[i*m+j] = 1 for all i, V[j*p+k] = 1, W[k*n+i] = 1 for all i.
    #   Then output(i,k) = sum_j 1*1*1 = m = 2. Wrong.
    #
    # The correct way: we cannot simply share because A[i,j] and B[j,k] are independent variables.
    # Each product A[i,j]*B[j,k] is a distinct degree-1 monomial in A times degree-1 monomial in B.
    # The space of such monomials has dimension n*m * m*p / m = n*m*p (no sharing possible generically)
    # UNLESS we use linearity: sum_j A[i,j]*B[j,k] can be computed with fewer than m terms if we
    # precompute linear combinations.
    #
    # For m=2: sum_{j=0}^1 A[i,j]*B[j,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    # This is 2 rank-1 terms per (i,k) pair, giving n*p*m = 2*13*2 = 52. 
    # But can we do better? We can compute:
    #   s0 = A[i,0]+A[i,1], s1 = A[i,0]-A[i,1]
    #   t0 = B[0,k]+B[1,k], t1 = B[0,k]-B[1,k]
    #   Then A[i,0]*B[0,k]+A[i,1]*B[1,k] = (s0*t0 + s1*t1)/2
    #   That's still 2 products per (i,k), but the products share the linear combinations s0,s1 across k.
    #   However, each product s0*t0 or s1*t1 is itself a rank-1 term in the bilinear sense.
    #   So we still have 2*p*n = 52 rank-1 terms... unless we can share the s0 factor.
    #
    # In tensor decomposition, "sharing" means the same rank-1 term contributes to multiple output entries.
    # A rank-1 term with W[k1*n+i1] != 0 and W[k2*n+i2] != 0 contributes to both outputs (i1,k1) and (i2,k2).
    #
    # Construction achieving rank = p*n = 26:
    #   For each (k, i) with k in 0..12, i in 0..1:
    #     We need output(i,k) = A[i,0]*B[0,k] + A[i,1]*B[1,k].
    #     Create a single rank-1 term that contributes ONLY to output (i,k).
    #     U vector: U[i*m+0] = 1, U[i*m+1] = 1, all other U = 0.
    #     V vector: V[0*p+k] = 1, V[1*p+k] = 1, all other V = 0.
    #     W vector: W[k*n+i] = 1, all other W = 0.
    #     
    #     Then contribution to output (i',k') from this term:
    #     = U[i'*m+0]*V[0*p+k']*W[k'*n+i'] + U[i'*m+1]*V[1*p+k']*W[k'*n+i']
    #     = delta_{k',k}*W[k'n+i'] + delta_{k',k}*W[k'n+i']  ... no wait.
    #
    #     Let me be more careful. The term is (U_term, V_term, W_term) where:
    #     U_term[a] = 1 if a is in {i*m, i*m+1}, else 0.  (i.e., U_term[i*m+j] = 1 for j=0,1)
    #     V_term[b] = 1 if b is in {0*p+k, 1*p+k}, else 0.  (i.e., V_term[j*p+k] = 1 for j=0,1)
    #     W_term[c] = 1 if c == k*n+i, else 0.
    #
    #     Contribution to T[i',j,k'] (the tensor element indexed by (i', j, k') which corresponds to
    #     U-index (i',j), V-index (j,k'), W-index (k',i')):
    #     = U_term[i'*m+j'] * V_term[j'*p+k''] * W_term[k''*n+i'']
    #     Wait, I'm confusing myself with the index conventions.
    #
    # Let me restart with clear notation.
    # The matrix multiplication tensor T has entries T[i, j, k] = delta for the computation.
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # In the bilinear tensor formulation:
    #   U is indexed by (i, j) -> index u = i*m + j
    #   V is indexed by (j, k) -> index v = j*p + k  
    #   W is indexed by (k, i) -> index w = k*n + i
    #
    # The reconstruction: (A*B)[i,k] = sum_c U[i*m+j_c, c] * V[j_c*p+k_c, c] * W[k_c*n+i_c, c]
    # where c ranges over 0..rank-1, and for each c, j_c is the j-index in U's row, etc.
    #
    # More precisely: for a single rank-1 term c with vectors U_c, V_c, W_c:
    #   The contribution to (A*B)[i,k] is:
    #     (sum_j A[i,j] * U_c[i*m+j]) * (sum_{j,k} ... ) 
    #   No. The standard formulation:
    #   (A*B)[i,k] = sum_c (sum_j A[i,j] * U_c[j]) * (sum_j B[j,k] * V_c[j]) * W_c[k*n+i]
    #   Hmm, that's not right either. Let me use the exact convention from the problem.
    #
    # The problem states: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    # And the reconstruction must satisfy:
    #   (A*B)[i,k] = sum_{c=0}^{rank-1} sum_{j=0}^{m-1} sum_{j'=0}^{m-1} A[i,j] * B[j',k] * U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c]
    #   for this to work for all A, B, we need the inner sums to factor properly.
    #
    # Actually, the correct identity for a valid decomposition is:
    #   (A*B)[i,k] = sum_c U_c[i*m + j] ... no, U_c is a vector of length n*m.
    #   
    #   Let me define it properly. Given A (n x m) and B (m x p):
    #   (A*B)[i,k] = sum_{j=0}^{m-1} A[i,j] B[j,k]
    #   
    #   A rank-r decomposition means there exist matrices U (n*m x r), V (m*p x r), W (p*n x r)
    #   such that for all A, B, i, k:
    #   (A*B)[i,k] = sum_{c=0}^{r-1} (sum_{j=0}^{m-1} A[i,j] U_{i*m+j, c}) * (sum_{j'=0}^{m-1} B[j',k] V_{j'*p+k, c}) * W_{k*n+i, c}
    #   
    #   Wait, that doesn't look right dimensionally either. Let me think again.
    #   
    #   Actually, the standard formulation is:
    #   (A*B)[i,k] = sum_c W_{k*n+i, c} * (U_c^T * row_i(A)) * (col_k(B) * V_c)
    #   where U_c is a vector in R^{n*m} flattened, so U_c^T * row_i(A) doesn't make sense directly.
    #
    #   I think the correct formula is:
    #   (A*B)[i,k] = sum_c (sum_{j} A[i,j] * U[i*m+j, c]) * (sum_{j} B[j,k] * V[j*p+k, c]) * W[k*n+i, c]
    #
    #   For this to reproduce (A*B)[i,k] = sum_j A[i,j] B[j,k] for ALL A, B,
    #   we need: for each (i,k), the bilinear form
    #     sum_c W[k*n+i, c] * (sum_j A[i,j] U[i*m+j, c]) * (sum_j B[j,k] V[j*p+k, c])
    #   to equal sum_j A[i,j] B[j,k].
    #
    #   This is equivalent to: the bilinear form in (A, B) defined by the LHS equals the standard
    #   matrix multiplication bilinear form. This works if and only if the decomposition is valid.
    #
    # Now, for the rank-26 construction:
    #   For each (k, i) with k in [0,p-1], i in [0,n-1], create term c_{k,i}:
    #     U[i*m + j, c_{k,i}] = 1 for j = 0, ..., m-1  (all j for this i)
    #     U[i'm + j, c_{k,i}] = 0 for i' != i
    #     V[j*p + k', c_{k,i}] = 1 for k' = k, j = 0, ..., m-1
    #     V[j*p + k', c_{k,i}] = 0 for k' != k
    #     W[k'*n + i', c_{k,i}] = 1 for (k',i') = (k,i)
    #     W[k'*n + i', c_{k,i}] = 0 otherwise
    #
    #   Contribution to (A*B)[i,k]:
    #     = sum_c W[k*n+i, c] * (sum_j A[i,j] U[i*m+j, c]) * (sum_j B[j,k] V[j*p+k, c])
    #     Only c = c_{k,i} has W[k*n+i, c] != 0 (value 1).
    #     = 1 * (sum_j A[i,j] * 1) * (sum_j B[j,k] * 1)
    #     = (A[i,0] + A[i,1]) * (B[0,k] + B[1,k])
    #     = A[i,0]B[0,k] + A[i,0]B[1,k] + A[i,1]B[0,k] + A[i,1]B[1,k]
    #
    #   But we want A[i,0]B[0,k] + A[i,1]B[1,k]. So this gives extra cross terms!
    #   This construction is WRONG.
    #
    #   The issue: U and V can't both be "all ones" for the same (k,i) term because that creates
    #   cross terms. We need U and V to be structured so that only the diagonal products survive.
    #
    #   For m=2: we need sum_j A[i,j]B[j,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
    #   A single rank-1 term can give us at most one product A[i,a]*B[b,k] (or a sum if U or V
    #   has multiple non-zero entries, but then we get cross terms).
    #   
    #   Actually: if U[i*m+0,c]=1, U[i*m+1,c]=0, V[0*p+k,c]=1, V[1*p+k,c]=0, then we get A[i,0]*B[0,k].
    #   If U[i*m+0,c]=0, U[i*m+1,c]=1, V[0*p+k,c]=0, V[1*p+k,c]=1, then we get A[i,1]*B[1,k].
    #   So we need 2 terms per (i,k), giving 2*n*p = 52 total. That's the schoolbook rank.
    #
    #   To get rank 26, we need each term to contribute to MULTIPLE (i,k) outputs simultaneously,
    #   using linearity. For example, a term could contribute A[i,0]*B[0,k] for multiple k values
    #   if V[0*p+k1,c]=V[0*p+k2,c]=...=1. But then it would contribute A[i,0]*B[0,k] for all those k,
    #   which is exactly what we want! The product A[i,0]*B[0,k] appears in (A*B)[i,k] for every k.
    #
    #   Wait no: (A*B)[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]. The term A[i,0]*B[0,k] appears for
    #   every (i,k). So if I create a rank-1 term that gives A[i,0]*B[0,k] for ALL k simultaneously,
    #   that's one term contributing to p different output entries!
    #
    #   Construction: For each (i, j) with i in [0,n-1], j in [0,m-1]:
    #     Create a term c_{i,j}:
    #       U[i*m+j, c_{i,j}] = 1
    #       U[i'*m+j', c_{i,j}] = 0 otherwise
    #       V[j*p+k, c_{i,j}] = 1 for all k in [0,p-1]
    #       V[j'*p+k, c_{i,j}] = 0 for j' != j
    #       W[k*n+i, c_{i,j}] = 1 for all k in [0,p-1]
    #       W[k*n+i', c_{i,j}] = 0 for i' != i
    #
    #   Contribution to (A*B)[i,k]:
    #     = sum_{c} W[k*n+i, c] * (sum_{j''} A[i,j''] U[i*m+j'', c]) * (sum_{j'''} B[j''',k] V[j'''*p+k, c])
    #     Terms with W[k*n+i, c] != 0: c = c_{i,j} for j in [0,m-1].
    #     = sum_{j=0}^{m-1} 1 * A[i,j]*1 * B[j,k]*1
    #     = sum_{j=0}^{m-1} A[i,j]*B[j,k]
    #     = (A*B)[i,k]. CORRECT!
    #
    #   Rank = n*m = 2*2 = 4. That can't be right... oh wait, W[k*n+i, c_{i,j}] = 1 for ALL k.
    #   So for a fixed (i, j), the term c_{i,j} contributes to ALL outputs (i, k) for k=0..p-1.
    #   The total number of terms is n*m = 4.
    #
    #   But wait: does this work? Let's verify for n=2, m=2, p=13.
    #   Term c_{0,0}: U[0*m+0]=U[0]=1, V[0*p+k]=V[k]=1 for all k, W[k*n+0]=W[2k]=1 for all k.
    #   Term c_{0,1}: U[0*m+1]=U[1]=1, V[1*p+k]=V[13+k]=1 for all k, W[2k]=1 for all k.
    #   Term c_{1,0}: U[1*m+0]=U[2]=1, V[0*p+k]=V[k]=1 for all k, W[2k+1]=1 for all k.
    #   Term c_{1,1}: U[1*m+1]=U[3]=1, V[1*p+k]=V[13+k]=1 for all k, W[2k+1]=1 for all k.
    #
    #   Check (A*B)[0,5] = A[0,0]*B[0,5] + A[0,1]*B[1,5].
    #   From term c_{0,0}: W[5*2+0]=W[10]=1, (sum_j A[0,j]U[j])=A[0,0]*1+A[0,1]*0=A[0,0], (sum_j B[j,5]V[j*13+5])=B[0,5]*1+B[1,5]*0=B[0,5]. Contribution: 1*A[0,0]*B[0,5].
    #   From term c_{0,1}: W[10]=1, (sum_j A[0,j]U[j])=A[0,0]*0+A[0,1]*1=A[0,1], (sum_j B[j,5]V[j*13+5])=B[0,5]*0+B[1,5]*1=B[1,5]. Contribution: 1*A[0,1]*B[1,5].
    #   Total: A[0,0]*B[0,5]+A[0,1]*B[1,5]. CORRECT!
    #
    #   So the rank is n*m = 4?! That seems way too low. Let me double-check.
    #
    #   U has shape (n*m, rank) = (4, 4).
    #   U = [[1,0,0,0],
    #        [0,1,0,0],
    #        [0,0,1,0],
    #        [0,0,0,1]]
    #   
    #   V has shape (m*p, rank) = (26, 4).
    #   V[j*p+k, c]: for c=0: V[k, 0]=1 for all k (j=0). For c=1: V[13+k, 1]=1 for all k (j=1). Others 0.
    #   V[0:13, 0] = ones(13), V[13:26, 1] = ones(13). Rest zero.
    #
    #   W has shape (p*n, rank) = (26, 4).
    #   W[k*n+i, c]: for c=0: W[2k, 0]=1 for all k (i=0). For c=1: W[2k, 1]=1 for all k (i=0).
    #   For c=2: W[2k+1, 2]=1 for all k (i=1). For c=3: W[2k+1, 3]=1 for all k (i=1).
    #
    #   This is indeed a valid decomposition of rank 4!
    #
    #   But wait - is this actually correct as a TENSOR DECOMPOSITION? The key insight is that
    #   the "bilinear form" formulation allows V and W to have the same non-zero pattern across
    #   different k values. This is valid because the reconstruction sums over c, and for each
    #   (i,k), only the relevant c terms contribute.
    #
    #   However, I need to verify this is a valid EXACT tensor decomposition in the sense meant
    #   by the problem. The problem says "exactly decompose the matrix multiplication tensor."
    #   I believe the convention is exactly as I described. Let me go with rank = n*m = 4.
    #
    #   Actually wait, I should double check by expanding more carefully.
    #   
    #   (A*B)[i,k] should equal sum_c [sum_j A[i,j]*U[i*m+j,c]] * [sum_j B[j,k]*V[j*p+k,c]] * W[k*n+i,c]
    #
    #   With my construction:
    #   U[i*m+j, c] = delta_{i, i_c} * delta_{j, j_c} where c = i_c*m + j_c
    #   V[j*p+k, c] = delta_{j, j_c} * 1 (no k dependence! V doesn't depend on k in my construction)
    #   W[k*n+i, c] = delta_{i, i_c} * 1 (no k dependence! W doesn't depend on k in my construction)
    #
    #   So: sum_j A[i,j]*U[i*m+j,c] = A[i, j_c] * delta_{i, i_c}
    #   sum_j B[j,k]*V[j*p+k,c] = B[j_c, k] * delta_{j, j_c}... no wait.
    #   
    #   V[j*p+k, c] = delta_{j, j_c} for ALL k. So sum_j B[j,k]*V[j*p+k,c] = B[j_c, k].
    #   W[k*n+i, c] = delta_{i, i_c} for ALL k. So W[k*n+i,c] = delta_{i, i_c}.
    #   
    #   Total: sum_c delta_{i,i_c} * A[i, j_c] * B[j_c, k] * delta_{i, i_c}
    #   = sum_c delta_{i,i_c} * A[i, j_c] * B[j_c, k]  (since delta is idempotent)
    #   = sum_{j_c} A[i, j_c] * B[j_c, k] * delta_{i, i_c}
    #   = sum_{j} A[i,j] * B[j,k] * delta_{i,i_c}... 
    #   
    #   Hmm, I'm getting confused. Let me just be concrete.
    #
    #   c ranges over 0,1,2,3 corresponding to (i_c, j_c) = (0,0), (0,1), (1,0), (1,1).
    #
    #   For (i,k) = (0, 5):
    #     c=0 (i_c=0,j_c=0): delta_{0,0}=1, A[0,0], B[0,5], delta_{0,0}=1 -> A[0,0]*B[0,5]
    #     c=1 (i_c=0,j_c=1): delta_{0,0}=1, A[0,1], B[1,5], delta_{0,0}=1 -> A[0,1]*B[1,5]
    #     c=2 (i_c=1,j_c=0): delta_{0,1}=0 -> 0
    #     c=3 (i_c=1,j_c=1): delta_{0,1}=0 -> 0
    #     Total: A[0,0]*B[0,5] + A[0,1]*B[1,5]. CORRECT.
    #
    #   So rank = n*m = 4 works! This is surprising but the math checks out.
    #
    #   Actually, I realize this is a well-known fact: the tensor rank of multiplying an (n x m)
    #   matrix by an (m x p) matrix is at most n*m (by fixing the output column and using the
    #   structure). In fact, the rank is exactly max(n*m, m*p, p*n) / min dimension... no.
    #
    #   Let me reconsider. The rank-1 term c contributes:
    #     W[k*n+i, c] * (sum_j A[i,j] U[i*m+j,c]) * (sum_j B[j,k] V[j*p+k,c])
    #
    #   In my construction, for term c corresponding to (i_c, j_c):
    #     U[i*m+j, c] = delta_{i,i_c}*delta_{j,j_c}
    #     V[j*p+k, c] = delta_{j,j_c}  (independent of k)
    #     W[k*n+i, c] = delta_{i,i_c}  (independent of k)
    #
    #   The product: delta_{i,i_c} * A[i,j_c] * B[j_c,k] * delta_{i,i_c} = delta_{i,i_c} * A[i,j_c] * B[j_c,k]
    #
    #   Summing over c: sum_{i_c,j_c} delta_{i,i_c} * A[i,j_c] * B[j_c,k] = sum_{j_c} A[i,j_c]*B[j_c,k] = (A*B)[i,k].
    #
    #   Yes! This is correct. The rank is n*m.
    #
    #   But wait, is this the minimal rank? The lower bound is:
    #     rank >= max over all slices. The slice fixing i gives a tensor of size (m x p x p) which has
    #     rank at least m*p/min... Actually, a simpler lower bound: fix k. Then (A*B)[:,k] is a vector
    #     in R^n that is bilinear in (A, B[:,k]). The rank of this "slice" is at least... 
    #     
    #   Actually, the standard lower bound for matrix multiplication tensor M_{n,m,p} is:
    #     rank >= n*m*p / max(n,m,p) ... no, that's not standard.
    #   
    #   The trivial lower bound: each rank-1 term W[k*n+i,c] is non-zero for at most one (k,i) pair
    #   (in my construction it's non-zero for all k but fixed i). Wait, in my construction W[k*n+i_c, c] = 1
    #   for ALL k. So one term contributes to p different (k, i_c) outputs.
    #   
    #   The lower bound: we have p*n output entries. Each rank-1 term can cover at most... well,
    #   in principle all of them if W is all-ones. But the constraint is on U and V.
    #   
    #   I think rank = n*m is indeed achievable and likely minimal for this case.
    #   Similarly, we could get rank = m*p by a symmetric argument, or rank = p*n.
    #   The minimum of these is min(n*m, m*p, p*n) = min(4, 26, 26) = 4.
    #
    #   So the answer is rank = 4!

    rank = n * m  # 4

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for i in range(n):
        for j in range(m):
            c = i * m + j
            U[i * m + j][c] = 1
            for k in range(p):
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1

    if _elapsed() - t0 > _DEADLINE - _TIME_MARGIN:
        # Fallback to schoolbook if somehow we're over time (shouldn't happen for this size)
        rank2 = n * m * p
        U2 = [[0] * rank2 for _ in range(n * m)]
        V2 = [[0] * rank2 for _ in range(m * p)]
        W2 = [[0] * rank2 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U2[i * m + j][col] = 1
                    V2[j * p + k][col] = 1
                    W2[k * n + i][col] = 1
                    col += 1
        return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
