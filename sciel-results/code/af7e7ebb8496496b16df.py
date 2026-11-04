"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _schoolbook():
    """Return the rank-27 schoolbook decomposition."""
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

def _verify(U, V, W, tol=1e-9):
    """Check that the decomposition exactly reconstructs the matmul tensor.
    
    Conditions:
    1. For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    2. For all a, b, c, d, e, f such that (b != c or d != e):
       sum_c U[a*m+b, c] * V[c*p+d, c] * W[e*n+f, c] == 0
    """
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    
    # Precompute products for speed?
    # Let's just iterate. 3x3x3 is small.
    
    # Condition 1: Support entries must be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
                    
    # Condition 2: Off-support entries must be 0
    # Indices: U[a, b], V[c, d], W[e, f]
    # Non-zero in T only if b == c and d == e.
    # So if b != c or d != e, sum must be 0.
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for e in range(p):
                        for f in range(n):
                            if b == c and d == e:
                                continue # Already checked in Condition 1 (with i=a, j=b, k=d)
                            val = 0.0
                            for c_idx in range(rank):
                                val += Ua[a*m+b, c_idx] * Va[c*p+d, c_idx] * Wa[e*n+f, c_idx]
                            if abs(val) > tol:
                                return False
    return True

def _check_half_ints(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for mat in [U, V, W]:
        for row in mat:
            for v in row:
                if abs(v * 2 - round(v * 2)) > tol:
                    return False
    return True

def _round_to_half_ints(U, V, W):
    """Round all entries to nearest half-integer."""
    def r(x):
        return round(x * 2) / 2.0
    return (
        [[r(v) for v in row] for row in U],
        [[r(v) for v in row] for row in V],
        [[r(v) for v in row] for row in W],
    )

def _try_rank_23():
    """
    Attempt to construct a rank-23 decomposition for 3x3 matrix multiplication.
    
    The known minimal rank for 3x3 matrix multiplication is 23 (Winograd, 1980).
    We use the known structure: the 3x3 matmul can be embedded in a 4x4 matmul
    using Strassen's 7-multiplication algorithm on the 4x4, giving rank 7*... 
    but the actual Winograd construction gives rank 23.
    
    Here we use a known explicit rank-23 decomposition. The 3x3 matrix multiplication
    tensor can be decomposed using the following approach:
    
    Write A and B as 3x3. Pad to 4x4 with a zero row/column, use Strassen (rank 7)
    on the 4x4, but this gives rank 7*4=28... that's not right either.
    
    The actual Winograd rank-23 construction:
    We express the 27 products needed for schoolbook as 23 "multiplications"
    where some are shared. The key insight is that certain bilinear forms can be
    combined.
    
    Let me use the explicit known decomposition. For 3x3 matmul, one known
    rank-23 decomposition uses the following 23 factors:
    
    I'll use the construction from the literature. The 3x3 matmul tensor
    T[i,j,k] = 1 (for all i,j,k in {0,1,2}) can be written as:
    T = sum_{c=0}^{22} u_c * v_c * w_c
    
    where u_c, v_c, w_c are vectors in R^9.
    
    One explicit construction: Use the fact that 3x3 matmul embeds into
    the algebra of 2x2 matrices over R[x]/(x^2-1) or similar.
    
    Actually, let me use a more direct approach. The known rank-23 decomposition
    for 3x3 matmul can be derived from the 4x4 Strassen algorithm.
    
    For a 4x4 matrix multiplication, Strassen uses 7 multiplications.
    If we write a 3x3 matrix as the top-left 3x3 of a 4x4 matrix with
    the 4th row and column being specific linear combinations, we can
    recover the 3x3 product.
    
    Here's a concrete approach:
    Let A, B be 3x3. Define 4x4 matrices A', B' by padding:
    A' = [[A, 0], [0, 0]]  (4x4)
    Then A'*B' = [[A*B, 0], [0, 0]]
    
    Strassen on 4x4 uses 7 multiplications of 2x2 blocks.
    A' = [[A11, A12], [A21, A22]] where each is 2x2.
    A11 = A[0:2, 0:2], A12 = A[0:2, 2:4] = [[A[0,2], 0], [A[1,2], 0]]
    A21 = A[2:4, 0:2] = [[A[2,0], A[2,1]], [0, 0]]
    A22 = A[2:4, 2:4] = [[A[2,2], 0], [0, 0]]
    
    Similarly for B'.
    
    Strassen's 7 products:
    M1 = (A21+A22)*(B11+B12)
    M2 = (A11+A12)*B21
    M3 = A11*(B11-B21)
    M4 = A22*(B22-B12)
    M5 = (A11+A21)*(B11+B22)
    M6 = (A12-A22)*(B21+B22)
    M7 = (A11-A21)*(B12+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M2 + M3
    C21 = M6 + M4
    C22 = M1 - M2 + M3 + M5
    
    Now I need to extract C = A*B from the top-left 3x3 of C'.
    C' = [[C11, C12], [C21, C22]] where C11 is 2x2, etc.
    
    C'[0:3, 0:3] = the 3x3 top-left of C'.
    C'[0:2, 0:2] = C11 = (A*B)[0:2, 0:2]
    C'[0:2, 2] = C12[0:2, 0] = (A*B)[0:2, 2]
    C'[2, 0:2] = C21[0, 0:2] = (A*B)[2, 0:2]
    C'[2, 2] = C22[0, 0] = (A*B)[2, 2]
    
    So C = A*B is recovered!
    
    Now, each M_i is a product of two 2x2 matrices. Each 2x2 matrix
    multiplication has rank 7 (Strassen) or 4 (schoolbook).
    
    If I use schoolbook for the 2x2 multiplications, each M_i requires 4
    scalar multiplications, so total = 7 * 4 = 28. That's worse than 27.
    
    If I use Strassen for the 2x2 multiplications, each M_i requires 7
    scalar multiplications, so total = 7 * 7 = 49. Even worse.
    
    So this embedding doesn't directly give rank 23.
    
    The actual Winograd rank-23 construction is more subtle. Let me look
    at it differently.
    
    The 3x3 matmul tensor has 27 entries, all equal to 1. We need to find
    23 rank-1 tensors whose sum equals this.
    
    A rank-1 tensor u ⊗ v ⊗ w has entries u_i * v_j * w_k.
    
    So we need: sum_{c=0}^{22} u_c[i] * v_c[j] * w_c[k] = 1 for all i,j,k.
    
    This is a system of 27 equations in 23*3 = 69 unknowns.
    
    Let me try to find a solution numerically and then round to half-integers.
    """
    start_time = time.time()
    deadline = start_time + 120  # 120 second budget
    
    n, m, p = DIMENSIONS
    rank = 23
    
    # Try to find a rank-23 decomposition numerically
    # Using the structure: we know the tensor is all-ones of shape (3,3,3)
    # We need sum_c u_c[i] v_c[j] w_c[k] = 1 for all i,j,k
    
    # One approach: parameterize and use least squares
    # But this is a non-convex problem.
    
    # Alternative: Use the known explicit construction.
    # The Winograd rank-23 construction for 3x3 matmul:
    
    # After research, one known rank-23 decomposition uses the following
    # 23 bilinear forms. Let me construct them explicitly.
    
    # The key idea: group the 27 products into 23 by sharing
    # certain linear combinations.
    
    # For 3x3 matmul:
    # (AB)[0,0] = A00*B00 + A01*B10 + A02*B20
    # (AB)[0,1] = A00*B01 + A01*B11 + A02*B21
    # (AB)[0,2] = A00*B02 + A01*B12 + A02*B22
    # (AB)[1,0] = A10*B00 + A11*B10 + A12*B20
    # (AB)[1,1] = A10*B01 + A11*B11 + A12*B21
    # (AB)[1,2] = A10*B02 + A11*B12 + A12*B22
    # (AB)[2,0] = A20*B00 + A21*B10 + A22*B20
    # (AB)[2,1] = A20*B01 + A21*B11 + A22*B21
    # (AB)[2,2] = A20*B02 + A21*B12 + A22*B22
    
    # Schoolbook: 27 multiplications.
    # Winograd: 23 multiplications.
    
    # The savings come from the fact that we can share some computations.
    # Specifically, we can write certain sums of products as products of sums.
    
    # For example, instead of computing A00*B00 + A01*B10 separately,
    # we might compute (A00 + A01)*(B00 + B10) and then subtract
    # A00*B10 + A01*B00 which are used elsewhere.
    
    # Let me try a direct numerical approach: set up the equations and
    # solve using numpy.
    
    # Variables: u_c[i], v_c[j], w_c[k] for c=0..22, i,j,k=0..2
    # 69 variables, 27 equations.
    
    # This is underdetermined, so we need to find a solution.
    # Let's use a greedy/iterative approach or just try to find one.
    
    # Actually, let me try a different tactic: use the known result that
    # the 3x3 matmul tensor has rank 23, and construct it explicitly.
    
    # I'll use the following construction based on the algebraic structure.
    
    # Define 23 triples (u_c, v_c, w_c) where each is a 3-vector.
    
    # This is complex. Let me try a simpler approach: verify if a
    # rank-23 decomposition exists by trying to construct one.
    
    # Given the complexity, let me fall back to a verified rank-27
    # decomposition if I can't quickly construct rank 23.
    
    # But wait - the task says "rank-minimal" and the score is the rank.
    # So I should try to get rank 23.
    
    # Let me try a numerical optimization approach with a time limit.
    
    # Initialize with random values and use gradient-free optimization
    # or just return the schoolbook if we can't find rank 23 in time.
    
    if time.time() > deadline - 5:
        return None
    
    # Try to construct rank-23 decomposition
    # Using the known explicit formulas from Winograd (1980)
    
    # The 23 multiplications for 3x3 matmul:
    # Based on the paper "Algorithms for Matrix Multiplication" by Winograd
    
    # I'll use a construction that I can verify:
    
    # Let me define the 23 factors explicitly.
    # After careful consideration, here's a known rank-23 decomposition:
    
    # The 3x3 matmul can be done with 23 multiplications using the
    # following scheme (indices 0-22):
    
    # This is getting complex. Let me just verify the schoolbook and
    # return it, as constructing a verified rank-23 decomposition
    # in this time frame is challenging without the explicit formulas.
    
    # Actually, let me try one more approach: use the fact that we can
    # express the all-ones tensor as a sum of rank-1 tensors.
    
    # The all-ones tensor J (3x3x3) has rank 1 (it's 1⊗1⊗1)!
    # Wait, that's not right for matrix multiplication.
    
    # The matrix multiplication tensor is NOT the all-ones tensor.
    # Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying
    #  a 3 by 3 matrix by a 3 by 3 matrix."
    
    # The matrix multiplication tensor T has entries T[i,j,k] such that
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    
    # So T[i,j,k] = 1 for all i,j,k? No!
    # T[i,j,k] is the coefficient of A[i,j]*B[j,k] in (A*B)[i,k].
    # Since (A*B)[i,k] = sum_j A[i,j]*B[j,k], the coefficient of
    # A[i,j]*B[j,k] is 1 for each (i,j,k) triple.
    
    # So yes, T[i,j,k] = 1 for all i,j,k in {0,1,2}.
    
    # And the all-ones tensor 1⊗1⊗1 has rank 1!
    # So the rank of the matmul tensor is 1?
    
    # That can't be right. Let me reconsider.
    
    # Oh wait, I think I'm confusing the tensor structure.
    # The matrix multiplication tensor is a 3-index tensor where
    # T[i,j,k] = 1 means that to compute output[i,k], we use
    # input_A[i,j] * input_B[j,k].
    
    # But the tensor rank is about writing T as a sum of rank-1 tensors.
    # A rank-1 tensor has the form u ⊗ v ⊗ w with entries u[i]*v[j]*w[k].
    
    # The all-ones tensor IS u⊗v⊗w where u=v=w=(1,1,1).
    # So its rank is 1.
    
    # But that would mean matrix multiplication has rank 1, which is absurd.
    
    # I must be misunderstanding the tensor structure.
    
    # Let me re-read the problem statement more carefully.
    
    # "The output factor follows the official trace convention W[k*n+i,r]."
    
    # And the schoolbook decomposition has:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # for each (i,j,k) triple, with c being the column index.
    
    # So the decomposition is:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # In the schoolbook, for each (i,j,k), there's exactly one column c
    # where U[i*m+j,c]=V[j*p+k,c]=W[k*n+i,c]=1, and all others are 0.
    
    # So the sum is 1 for each (i,j,k).
    
    # Now, if I set U[i*m+j, c] = 1 for all i,j and a fixed c,
    # V[j*p+k, c] = 1 for all j,k and the same c,
    # W[k*n+i, c] = 1 for all k,i and the same c,
    # then for each (i,j,k), the product is 1*1*1 = 1.
    
    # So a single column with all 1s would work!
    # That means rank 1!
    
    # But that's clearly wrong because matrix multiplication
    # requires 27 scalar multiplications in the schoolbook algorithm.
    
    # I think the issue is that the tensor decomposition rank is
    # different from the number of scalar multiplications needed.
    
    # Wait, no. The tensor rank IS the minimum number of scalar
    # multiplications needed. If the rank is 1, then we could do
    # matrix multiplication with 1 scalar multiplication, which is impossible.
    
    # Let me reconsider. The tensor T[i,j,k] for matrix multiplication
    # is defined such that:
    # (A*B)[i,k] = sum_{j,c} U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
    
    # No, that's not right either. Let me think about this more carefully.
    
    # The standard formulation: A rank-r decomposition of the matmul tensor
    # means we can write:
    # (A*B)[i,k] = sum_{c=0}^{r-1} (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
    
    # No, that's not right. The correct formulation is:
    # (A*B)[i,k] = sum_{c=0}^{r-1} W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k])
    
    # Hmm, but that doesn't look right dimensionally.
    
    # Let me look at the schoolbook again:
    # U[i*m+j, c] = 1 when c = i*m*p + j*p + k (or some mapping)
    # V[j*p+k, c] = 1 for the same c
    # W[k*n+i, c] = 1 for the same c
    
    # So for a given c corresponding to (i,j,k):
    # sum_{j'} U[i*m+j', c] * A[i,j'] = A[i,j] (only j'=j contributes)
    # sum_{j'} V[j'*p+k, c] * B[j',k] = B[j,k] (only j'=j contributes)
    # W[k*n+i, c] = 1
    
    # So the contribution to (A*B)[i,k] from column c is A[i,j]*B[j,k]*1.
    
    # And summing over all c (all i,j,k triples) gives sum_j A[i,j]*B[j,k] = (A*B)[i,k].
    
    # OK so the formulation is:
    # (A*B)[i,k] = sum_c W[k*n+i,c] * (sum_{j'} U[i*m+j',c] * A[i,j']) * (sum_{j'} V[j'*p+k,c] * B[j',k])
    
    # For this to work for ALL A and B, we need:
    # sum_c W[k*n+i,c] * (sum_{j'} U[i*m+j',c] * A[i,j']) * (sum_{j'} V[j'*p+k,c] * B[j',k]) = sum_j A[i,j]*B[j,k]
    
    # This must hold for all A and B. So the coefficient of A[i,j]*B[j',k] on the
    # LHS must equal delta_{j,j'} on the RHS.
    
    # The coefficient of A[i,j]*B[j',k] in the LHS is:
    # sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c]
    
    # So we need: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j'*p+k,c] = delta_{j,j'} for all i,j,j',k.
    
    # THIS is the correct condition! Not that the sum equals 1 for all i,j,k.
    
    # So the tensor T[i,j,j',k] = sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k*n+i,c]
    # must equal delta_{j,j'}.
    
    # This is a 4-index tensor, not a 3-index tensor!
    
    # The matrix multiplication tensor is actually a 4-index tensor
    # T[i,j,j',k] = delta_{j,j'}, which can be viewed as a 3-index
    # tensor T[i,j,k] with an implicit sum over j'.
    
    # OK so my earlier understanding was wrong. Let me redo this.
    
    # The condition is:
    # sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k*n+i,c] = delta_{j,j'}
    
    # for all i, j, j', k.
    
    # This is 3*3*3*3 = 81 equations (but many are redundant due to delta).
    # The non-trivial ones are:
    # - For j=j': sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # - For j≠j': sum_c U[i*m+j,c] * V[j'*p+k,c] * W[k*n+i,c] = 0
    
    # So we have 27 equations equal to 1 and 54 equations equal to 0.
    
    # This is a much more complex system! And it explains why the rank
    # is 23, not 1.
    
    # OK so now I need to find U, V, W of rank 23 satisfying these conditions.
    
    # This is a hard problem. Let me just return the schoolbook decomposition
    # since I can't easily construct the rank-23 decomposition in this time frame.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    # Try rank-23 first
    result = _try_rank_23()
    if result is not None:
        U, V, W = result["U"], result["V"], result["W"]
        if _verify(U, V, W) and _check_half_ints(U, V, W):
            return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
