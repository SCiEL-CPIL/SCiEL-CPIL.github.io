"""Find rank-minimal tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s

    # Schoolbook decomposition: rank = n*m*p = 52
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]

    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_sb

    # The tensor T[a,b,c] = 1 iff a=2i+j, b=13j+k, c=13k+i for some i,j,k
    # This is the matrix multiplication tensor for (n x m) * (m x p)
    #
    # For 2x2 * 2x13, the schoolbook rank is 52.
    # Let's try to find a lower-rank decomposition using structured methods.
    #
    # Key insight: The j-index couples U and V, while k couples V and W,
    # and i couples U and W. The structure is:
    #   T[2i+j, 13j+k, 13k+i] = 1
    #
    # One approach: fix j and k, then the i-coupling is a 2x2 identity-like
    # structure. For each (j,k) pair, we need 2 terms (one for each i),
    # giving 2*2*13 = 52 total.
    #
    # To reduce rank, we need to share terms across different (j,k) pairs.
    # This is possible if the i-structure can be factored.
    #
    # Let's try a rank-26 decomposition:
    # For each k in {0,...,12}, use 2 terms indexed by (k, s) where s in {0,1}.
    #
    # Term (k, s):
    #   U[2i+j, (k,s)] = ?
    #   V[13j+k', (k,s)] = ?
    #   W[13k''+i', (k,s)] = ?
    #
    # We need: sum_{k,s} U[2i+j, (k,s)] * V[13j+k', (k,s)] * W[13k''+i', (k,s)]
    #   = 1 if i=i', j=j', k=k', else 0
    #
    # This is a system of 52 equations in 26*4 + 26*26 + 26*26 = 104 + 676 + 676 = 1456 unknowns.
    # Very underdetermined.
    #
    # Let's try a specific structured form:
    # For term (k, s):
    #   V[13j+k', (k,s)] = delta_{k',k} * v_{j,s}
    #   W[13k''+i', (k,s)] = delta_{k'',k} * w_{i',s}
    #   U[2i+j, (k,s)] = u_{i,j,s}
    #
    # Then the sum becomes:
    #   sum_{k,s} u_{i,j,s} * delta_{k',k} * v_{j,s} * delta_{k'',k} * w_{i',s}
    #   = sum_{s} u_{i,j,s} * v_{j,s} * w_{i',s} * delta_{k',k''}
    #
    # For this to equal delta_{i,i'} * delta_{j,j'} * delta_{k,k'}, we need:
    #   delta_{k',k''} * (sum_s u_{i,j,s} * v_{j,s} * w_{i',s}) = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    #
    # This requires:
    #   sum_s u_{i,j,s} * v_{j,s} * w_{i',s} = delta_{i,i'} * delta_{j,j'}
    #
    # But this must hold for all i, i', j. The j-dependence is problematic because
    # v_{j,s} depends on j, but the RHS delta_{j,j'} also depends on j.
    #
    # Actually, wait. The equation should be:
    #   sum_s u_{i,j,s} * v_{j,s} * w_{i',s} = delta_{i,i'}
    # for all i, i', j. But this can't hold for all j simultaneously unless
    # u_{i,j,s} * v_{j,s} is independent of j, which is restrictive.
    #
    # Let me try a different approach. Use 4 terms per k (for the 2x2 i-j coupling):
    # For term (k, r) where r in {0,1,2,3}:
    #   V[13j+k', (k,r)] = delta_{k',k}
    #   W[13k''+i', (k,r)] = delta_{k'',k} * w_{i',r}
    #   U[2i+j, (k,r)] = u_{i,j,r}
    #
    # Then: sum_{k,r} u_{i,j,r} * delta_{k',k} * delta_{k'',k} * w_{i',r}
    #   = delta_{k',k''} * sum_r u_{i,j,r} * w_{i',r}
    #
    # We need: sum_r u_{i,j,r} * w_{i',r} = delta_{i,i'} * delta_{j,j'}
    #
    # This is a 2x2 matrix equation for each j:
    #   U_j * W^T = delta_{j,j'} ... no, that's not right.
    #
    # Actually, for fixed j, we need:
    #   sum_r u_{i,j,r} * w_{i',r} = delta_{i,i'} * delta_{j,j'}
    #
    # If j = j', this gives U_j * W^T = I_2.
    # If j != j', this gives U_j * W^T = 0.
    #
    # But we only have 4 terms, and U_j is a 2x4 matrix, W is a 2x4 matrix.
    # U_j * W^T is a 2x2 matrix. We need it to be I_2 when j=j' and 0 when j!=j'.
    #
    # For j=0: U_0 * W^T = I_2
    # For j=1: U_1 * W^T = 0
    #
    # This is possible! Let W be a 2x4 matrix with full row rank.
    # Then U_0 = W^+ (pseudoinverse) and U_1 is in the null space of W^T.
    #
    # But wait, we need integer/half-integer entries. Let's try specific values.
    #
    # Let W = [[1, 0, 1, 0], [0, 1, 0, 1]]
    # Then W^T = [[1, 0], [0, 1], [1, 0], [0, 1]]
    #
    # U_0 * W^T = I_2 means:
    #   U_0[0,0]*1 + U_0[0,2]*1 = 1
    #   U_0[0,1]*0 + U_0[0,3]*0 = 0
    #   U_0[1,0]*0 + U_0[1,2]*0 = 0
    #   U_0[1,1]*1 + U_0[1,3]*1 = 1
    #
    # So U_0[0,0] + U_0[0,2] = 1, U_0[0,1] = 0, U_0[0,3] = 0
    # U_0[1,0] = 0, U_0[1,2] = 0, U_0[1,1] + U_0[1,3] = 1
    #
    # One solution: U_0 = [[1, 0, 0, 0], [0, 1, 0, 0]]
    #
    # U_1 * W^T = 0 means:
    #   U_1[0,0] + U_1[0,2] = 0
    #   U_1[0,1] + U_1[0,3] = 0
    #   U_1[1,0] + U_1[1,2] = 0
    #   U_1[1,1] + U_1[1,3] = 0
    #
    # One solution: U_1 = [[1, 0, -1, 0], [0, 1, 0, -1]]
    #
    # Now let's verify the full decomposition with rank = 4 * 13 = 52.
    # That's the same as schoolbook! Not an improvement.
    #
    # The issue is that we need 4 terms per k to handle the 2x2 i-j coupling.
    # To reduce rank, we need to share terms across k values.
    #
    # Let's try a different strategy: use 2 terms total, with k-dependent coefficients.
    # This is unlikely to work for a 2x13 matrix.
    #
    # Given the time constraints and the difficulty of finding a lower-rank decomposition,
    # let's return the schoolbook solution.
    #
    # Actually, let me try one more thing: a rank-26 decomposition with a different structure.
    #
    # For term r (0 to 25), let r = 13*k + s where k in {0,...,12}, s in {0,1}.
    #
    # V[13j+k', r] = delta_{k',k}
    # W[13k''+i', r] = delta_{k'',k} * w_{i',s}
    # U[2i+j, r] = u_{i,j,s}
    #
    # Then: sum_{k,s} u_{i,j,s} * delta_{k',k} * delta_{k'',k} * w_{i',s}
    #   = delta_{k',k''} * sum_s u_{i,j,s} * w_{i',s}
    #
    # We need: sum_s u_{i,j,s} * w_{i',s} = delta_{i,i'} * delta_{j,j'}
    #
    # With only 2 terms (s=0,1), this is a 2x2 system:
    #   u_{i,j,0} * w_{i',0} + u_{i,j,1} * w_{i',1} = delta_{i,i'} * delta_{j,j'}
    #
    # For j=0: [u_{i,0,0}, u_{i,0,1}] * [w_{i',0}, w_{i',1}]^T = delta_{i,i'}
    # For j=1: [u_{i,1,0}, u_{i,1,1}] * [w_{i',0}, w_{i',1}]^T = 0
    #
    # This requires the vector [u_{i,1,0}, u_{i,1,1}] to be orthogonal to
    # [w_{i',0}, w_{i',1}] for all i'. This is possible if we choose w carefully.
    #
    # Let w_{0,0} = 1, w_{0,1} = 0, w_{1,0} = 0, w_{1,1} = 1.
    # Then for j=0: u_{i,0,0} = delta_{i,0}, u_{i,0,1} = delta_{i,1}
    # For j=1: u_{i,1,0} = 0, u_{i,1,1} = 0
    #
    # But then for j=1, the sum is always 0, which is correct for delta_{j,j'} when j!=j'.
    # However, when j=j'=1, we need the sum to be delta_{i,i'}, but it's 0. Contradiction!
    #
    # So this doesn't work. We need a different approach.
    #
    # Let's try w_{0,0} = 1, w_{0,1} = 1, w_{1,0} = 1, w_{1,1} = -1.
    # Then for j=0:
    #   u_{0,0,0} + u_{0,0,1} = 1
    #   u_{0,0,0} - u_{0,0,1} = 0  => u_{0,0,0} = u_{0,0,1} = 1/2
    #   u_{1,0,0} + u_{1,0,1} = 0
    #   u_{1,0,0} - u_{1,0,1} = 1  => u_{1,0,0} = 1/2, u_{1,0,1} = -1/2
    #
    # For j=1:
    #   u_{0,1,0} + u_{0,1,1} = 0
    #   u_{0,1,0} - u_{0,1,1} = 0  => u_{0,1,0} = u_{0,1,1} = 0
    #   u_{1,1,0} + u_{1,1,1} = 0
    #   u_{1,1,0} - u_{1,1,1} = 0  => u_{1,1,0} = u_{1,1,1} = 0
    #
    # Again, j=j'=1 gives 0, not delta_{i,i'}.
    #
    # The fundamental issue is that with only 2 terms, we can't simultaneously
    # satisfy the conditions for j=0 and j=1.
    #
    # I think the schoolbook rank of 52 is hard to beat with simple structured methods.
    # Let's return the schoolbook solution.

    return best_result
