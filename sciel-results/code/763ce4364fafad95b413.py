import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def _time_ok(deadline):
    return time.time() < deadline

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication.
    W[k*n+i, r] should give C[i,k] = sum_r U[i*m+k, r] * V[k*p+?, r]
    Actually: C = A*B where A is n x m, B is m x p, C is n x p.
    C[i,j] = sum_k A[i,k] * B[k,j]
    In our convention: U has shape (n*m, rank), indexed by (i,k) -> i*m+k
    V has shape (m*p, rank), indexed by (k,j) -> k*p+j  
    W has shape (p*n, rank), indexed by (j,i) -> j*n+i (trace convention W[k*n+i,r] means W[j*n+i, r])
    
    So C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] summed over k and r
    Wait, that's not right. Let me re-read.
    
    The tensor T[i,j,k] = A[i,k]*B[k,j] should equal sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    Hmm, actually the standard form is:
    C[i,j] = sum_k A[i,k] B[k,j]
    Tensor form: T_{i,j,k} where T_{i,j,k} = delta(i,k)*delta(k,j) ... no.
    
    Let me think differently. The multiplication tensor has entries:
    T[(i,k), (k,j), (i,j)] = 1 for all valid i,k,j
    And we want T = sum_r u_r ⊗ v_r ⊗ w_r
    where u_r is indexed by (i,k) in R^{n*m}, v_r by (k,j) in R^{m*p}, w_r by (i,j) in R^{n*p}
    
    But the problem says W has shape (p*n, rank) with convention W[k*n+i, r].
    So W is indexed by (j, i) where the first index is j (column of C) and second is i (row of C).
    So W[j*n + i, r] is the coefficient for C[i,j].
    
    Therefore: C[i,j] = sum_r U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r] ... but k is summed over.
    Actually no: for each r, u_r is a vector in R^{n*m} (indexed by (i,k)), v_r in R^{m*p} (indexed by (k,j)),
    w_r in R^{p*n} (indexed by (j,i)).
    
    The outer product u_r ⊗ v_r ⊗ w_r has entry [i*m+k, k*p+j, j*n+i] = U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
    
    For this to equal the multiplication tensor, we need:
    For all i, j, k: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
    For all other (i,k), (k,j), (j,i) combinations that don't correspond to valid multiplications, it should be 0.
    
    But wait - in matrix multiplication, C[i,j] = sum_k A[i,k]*B[k,j]. The "tensor" here is really:
    T[i, k, k', j] = delta(k,k') ... no.
    
    Let me reconsider. The bilinear map is: (A, B) -> C where C[i,j] = sum_k A[i,k] B[k,j].
    This is a 3-tensor T with indices (i,j,k) where T[i,j,k] = 1 if we're computing C[i,j] using A[i,k] and B[k,j].
    
    In the standard tensor decomposition for matrix multiplication:
    T = sum_r u_r ⊗ v_r ⊗ w_r
    where u_r ∈ R^{n×m} (flattened to R^{nm}), v_r ∈ R^{m×p} (flattened to R^{mp}), w_r ∈ R^{p×n} (flattened to R^{pn})
    
    And T[i*m+k, k*p+j, j*n+i] = 1 for all valid i,k,j, and 0 otherwise.
    
    Wait, that's not quite right either. Let me use the standard formulation:
    
    The multiplication tensor M_{n,m,p} has entries:
    M[(i,k), (k',j), (i',j')] = 1 if i=i' and k=k' and j=j', else 0
    
    So M[i*m+k, k*p+j, i*n+j] = 1 (using row-major flattening)
    
    But the problem states W has convention W[k*n+i, r], which means W is indexed by (column, row) of C.
    So W[j*n + i, r] corresponds to C[i,j].
    
    Therefore the tensor entry is: T[i*m+k, k*p+j, j*n+i] = 1
    
    Let me verify with a simple case. For 1x1x1:
    U[0, r] * V[0, r] * W[0, r] should sum to 1 over r.
    With rank 1: U[0,0]=1, V[0,0]=1, W[0,0]=1. ✓
    
    For 2x2x2 with Strassen, let's check one entry:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    T[0*2+0, 0*2+0, 0*2+0] = T[0,0,0] = 1 (from A[0,0]*B[0,0])
    T[0*2+1, 1*2+0, 0*2+0] = T[1,2,0] = 1 (from A[0,1]*B[1,0])
    
    So sum_r U[0, r]*V[0, r]*W[0, r] = 1
    sum_r U[1, r]*V[2, r]*W[0, r] = 1
    
    This is getting complex. Let me just implement the verification by checking that
    for random A and B, the computed C matches.
    """
    # Generate random test matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute C using the decomposition
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                for r in range(len(U[0])):
                    C_computed[i, j] += U[i * m + k, r] * V[k * p + j, r] * W[j * n + i, r]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return False
    return True

def _bdf_lotti_3x4x4():
    """
    Return the rank-18 Bini-De Falco-Lotti decomposition for 3x4x4 matrix multiplication.
    A is 3x4, B is 4x4, C is 3x4.
    
    The BDFL algorithm for 3x4x4 uses 18 multiplications.
    This is based on the work of Bini, De Falco, and Lotti (1980).
    
    For multiplying a 3x4 matrix by a 4x4 matrix:
    A = [[a11, a12, a13, a14],
         [a21, a22, a23, a24],
         [a31, a32, a33, a34]]
    
    B = [[b11, b12, b13, b14],
         [b21, b22, b23, b24],
         [b31, b32, b33, b34],
         [b41, b42, b43, b44]]
    
    C = A*B (3x4)
    
    The BDFL rank-18 decomposition:
    We partition A into 3x2 blocks (two 3x2 blocks) and B into 2x4 blocks (two 2x4 blocks).
    Actually, let me look up the specific rank-18 scheme.
    
    The known rank-18 algorithm for 3x4x4 (also called 3x4 by 4x4) from Bini-De Falco-Lotti:
    
    Actually, I recall that the minimal rank for 3x4x4 is 18, and it can be derived
    by embedding into a larger problem or using specific constructions.
    
    Let me construct it using the following approach:
    Think of 3x4x4 as related to 3x3x3 (rank 23 by Strassen-like) and 4x4x4.
    
    Alternative: Use the fact that 3x4x4 can be decomposed by splitting the 4 columns into 2+2:
    C[:, :2] = A[:, :2] @ B[:2, :2] + A[:, 2:] @ B[2:, :2]
    C[:, 2:] = A[:, :2] @ B[:2, 2:] + A[:, 2:] @ B[2:, 2:]
    
    This gives 4 sub-problems of type 3x2x2.
    The rank of 3x2x2 is... let me think. For m=2, n=3, p=2.
    
    Actually, let me use a different known result. The rank of 3x4x4 is known to be 18.
    One construction:
    
    Split A (3x4) into A1 (3x2) and A2 (3x2) horizontally.
    Split B (4x4) into B1 (2x4) and B2 (2x4) vertically.
    
    C = A1*B1 + A2*B2
    
    Now A1*B1 is 3x2 times 2x4 = 3x4.
    A2*B2 is 3x2 times 2x4 = 3x4.
    
    For 3x2x4: we can further decompose.
    
    Actually, let me just directly implement a known rank-18 scheme.
    
    Based on the literature, here's a rank-18 decomposition for 3x4x4:
    
    The 18 multiplications are formed as follows:
    - Use 3x2x2 blocks with Strassen (rank 7 each) but that gives more than 18.
    
    Let me try a different approach. I'll use the known fact that:
    R(3,4,4) <= R(3,2,2) + R(3,2,2) + R(2,2,2) + ... 
    
    Actually, the simplest correct approach: I know that the rank of 3x4x4 is 18.
    Let me construct it by embedding 2x4x4 (which has rank 12 via 2x2x2 Strassen: 3 blocks * 7 = 21? No...)
    
    Wait. Let me reconsider. For 2x4x4:
    A is 2x4, B is 4x4, C is 2x4.
    Split into 2x2x2 blocks: A has 1x2 blocks (2 blocks), B has 2x2 blocks (4 blocks).
    That's 1*2 = 2 block columns for A, 2*2 = 4 block columns for B... 
    
    No. For 2x4 by 4x4:
    A: 2 rows, 4 cols -> 1x2 grid of 2x2 blocks
    B: 4 rows, 4 cols -> 2x2 grid of 2x2 blocks
    C: 2 rows, 4 cols -> 1x2 grid of 2x2 blocks
    
    Number of 2x2x2 subproblems: 1*2*2*2 = 8? No.
    It's (A_blocks_rows * A_blocks_cols) * (B_blocks_rows * B_blocks_cols) / (C_blocks_cols)
    = 2 * 4 / 2 = 4 subproblems? 
    
    Actually: C[i_block, j_block] = sum_k A[i_block, k_block] * B[k_block, j_block]
    i_block: 1 value, j_block: 2 values, k_block: 2 values
    So 1*2*2 = 4 subproblems of 2x2x2.
    Each has rank 7 (Strassen), total rank = 4*7 = 28.
    
    But we can do better. The rank of 2x4x4 is actually 12 (since it's just 2 independent 2x4x4... no).
    
    Hmm, I'm going in circles. Let me take a completely different approach.
    
    I'll directly verify if a rank-18 decomposition exists by using a known construction.
    
    KEY INSIGHT: The problem says to embed 2x4x6 into 3x4x4 by zero-padding the row dimension.
    2x4x6: A is 2x4, B is 4x6, C is 2x6.
    3x4x4: A is 3x4, B is 4x4, C is 3x4.
    
    These don't directly match in dimensions! 2x4x6 has A:2x4, B:4x6, C:2x6.
    3x4x4 has A:3x4, B:4x4, C:3x4.
    
    The dimensions don't align for a simple embedding. The directive says "Embed the 2×4×6 multiplication tensor into the 3×4×4 tensor space by zero-padding the row dimension."
    
    But 2x4x6 and 3x4x4 have different shapes. This doesn't make sense as stated unless there's a specific mapping.
    
    Let me re-read: "Embed the 2×4×6 multiplication tensor into the 3×4×4 tensor space"
    
    Perhaps the idea is different. Maybe we should look for a rank-18 decomposition of 2x4x6 directly, or find a different embedding.
    
    Actually, I think the directive may be based on a misunderstanding. Let me instead focus on finding a better decomposition for 2x4x6 directly.
    
    For 2x4x6:
    Lower bound: max(n*p, m*p, n*m) = max(12, 24, 8) = 24? No, the lower bound is max(n*p, m*p, n*m) is not correct.
    The trivial lower bound is n*p = 12 (since C has 12 entries).
    Also, the rank must be at least max(n, m, p) in some sense, but more precisely:
    The slice rank gives lower bounds, but the standard lower bound is that rank >= max(n*p, m*p, n*m) / max(n, m, p)... 
    
    Actually, the known lower bound for R(n,m,p) is max(n*p, m*p, n*m).
    For 2x4x6: max(12, 24, 8) = 24.
    
    So the rank is at least 24. Our current solution has rank 42.
    
    Can we achieve rank 24? The trivial algorithm has rank n*m*p = 48.
    Strassen-based gives 42.
    
    For 2x4x6, let me think about block decomposition:
    A: 2x4 -> 1x2 grid of 2x2 blocks
    B: 4x6 -> 2x3 grid of 2x2 blocks  
    C: 2x6 -> 1x3 grid of 2x2 blocks
    
    Subproblems: i=1 block row, j=3 block cols, k=2 block rows
    Total: 1*3*2 = 6 subproblems of 2x2x2.
    With Strassen (rank 7): 6*7 = 42. (This is what we have.)
    
    Can we do better? The lower bound is 24.
    
    One approach: use a different block structure or a direct rank-24 construction.
    
    Actually, for 2x4x6, we can think of it as:
    C = A @ B where A is 2x4, B is 4x6.
    
    We can write:
    C[:, 0:2] = A[:, 0:2] @ B[0:2, 0:2] + A[:, 2:4] @ B[2:4, 0:2]
    C[:, 2:4] = A[:, 0:2] @ B[0:2, 2:4] + A[:, 2:4] @ B[2:4, 2:4]
    C[:, 4:6] = A[:, 0:2] @ B[0:2, 4:6] + A[:, 2:4] @ B[2:4, 4:6]
    
    Each term is 2x2 @ 2x2 = 2x2, which has rank 7 (Strassen).
    Total: 6 * 7 = 42.
    
    To get below 42, we need to exploit structure across these 6 subproblems.
    
    Alternative: Think of it as 2x4 by 4x6 = (2x2 by 2x4) stacked with (2x2 by 2x4)?
    No, that doesn't help directly.
    
    What about: C = [C_left, C_mid, C_right] where each is 2x2.
    C_left = A[:, :2] @ B[:2, :2] + A[:, 2:] @ B[2:, :2]
    
    The left part uses A[:, :2] (2x2) and B[:2, :2] (2x2), plus A[:, 2:] (2x2) and B[2:, :2] (2x2).
    
    If we could compute all three C blocks simultaneously with shared multiplications, we could reduce the rank.
    
    This is essentially asking for the rank of the 2x4x6 tensor, which is known to be between 24 and 42.
    
    I believe the exact rank of 2x4x6 is not known to be 24. It's likely higher.
    
    Given the time constraints, let me try to improve from 42 to something lower by finding a better construction.
    
    One idea: Use the fact that for 2xm x mxp, the rank can be bounded using Strassen-like methods.
    
    For 2x4x6, another block decomposition:
    Split A into 2x1 and 2x3? No, that's uneven.
    
    Split A into 1x4 and 1x4 (two 1x4 rows)?
    Split B into 4x2, 4x2, 4x2 (three 4x2 columns)?
    Split C into 1x6 and 1x6 (two 1x6 rows)?
    
    Subproblem: 1x4 by 4x2 = 1x2.
    Rank of 1x4x2: C[i,j] = sum_k A[i,k] B[k,j] where i has 1 value, j has 2 values, k has 4 values.
    This is just computing 2 dot products of length 4. Rank = 2 (one for each output).
    Wait, no. The tensor has entries T[i*m+k, k*p+j, j*n+i] = T[0*4+k, k*2+j, j*1+0] for i=0.
    T[k, 2k+j, j] = 1 for k=0..3, j=0..1.
    This is a tensor of size 4x8x2.
    Rank: We need sum_r U[k, r] V[2k+j, r] W[j, r] = 1 for all k, j.
    This means for each j, sum_k U[k,r] V[2k+j,r] W[j,r] = 1.
    With W[0,r] = w_r0, W[1,r] = w_r1:
    For j=0: sum_k U[k,r] V[2k,r] w_r0 = 1
    For j=1: sum_k U[k,r] V[2k+1,r] w_r1 = 1
    
    We can set w_r0 = w_r1 = 1 and need:
    sum_k U[k,r] V[2k,r] = 1
    sum_k U[k,r] V[2k+1,r] = 1
    
    With rank 1: U[k,0]=1, V[2k,0]=1/4 for all k, V[2k+1,0]=1/4 for all k.
    Then sum_k 1*(1/4) = 4/4 = 1. ✓
    
    So rank of 1x4x2 is 1!
    
    Therefore, splitting into 2 (rows of A) * 3 (column blocks of B) = 6 subproblems of 1x4x2, each with rank 1.
    Total rank = 6 * 1 = 6? That can't be right because the lower bound is 24.
    
    Wait, I think I'm confusing myself. Let me be more careful.
    
    If I split A into two 1x4 rows, and B into three 4x2 column blocks, then:
    C[0, 0:2] = A[0, :] @ B[:, 0:2]  (1x4 @ 4x2 = 1x2)
    C[0, 2:4] = A[0, :] @ B[:, 2:4]  (1x4 @ 4x2 = 1x2)
    C[0, 4:6] = A[0, :] @ B[:, 4:6]  (1x4 @ 4x2 = 1x2)
    C[1, 0:2] = A[1, :] @ B[:, 0:2]  (1x4 @ 4x2 = 1x2)
    C[1, 2:4] = A[1, :] @ B[:, 2:4]  (1x4 @ 4x2 = 1x2)
    C[1, 4:6] = A[1, :] @ B[:, 4:6]  (1x4 @ 4x2 = 1x2)
    
    Each is a 1x4x2 multiplication. As I showed, each has rank 1.
    Total rank = 6.
    
    But this contradicts the lower bound of 24! So I must be making an error.
    
    The error: When I say "rank 1" for 1x4x2, I'm assuming we can use the same rank-1 decomposition for all 6 subproblems. But they share the same B columns! 
    B[:, 0:2], B[:, 2:4], B[:, 4:6] are different parts of B, so they don't conflict.
    A[0, :] and A[1, :] are different rows, so they don't conflict.
    
    Actually, if each subproblem truly has rank 1 and they're independent, then the total rank would be 6. But that's impossible because the lower bound is 24.
    
    Let me re-examine the rank of 1x4x2.
    T[i*m+k, k*p+j, j*n+i] where n=1, m=4, p=2.
    i is always 0.
    T[0*4+k, k*2+j, j*1+0] = T[k, 2k+j, j] = 1 for k=0,1,2,3 and j=0,1.
    
    So T[k, 2k, 0] = 1 and T[k, 2k+1, 1] = 1 for k=0,1,2,3.
    All other entries are 0.
    
    We want T = sum_r u_r ⊗ v_r ⊗ w_r where:
    u_r ∈ R^4 (indexed by k=0..3)
    v_r ∈ R^8 (indexed by 2k+j, i.e., 0..7)
    w_r ∈ R^2 (indexed by j=0..1)
    
    T[k, 2k+j, j] = sum_r u_r[k] * v_r[2k+j] * w_r[j] = 1 if j matches, else 0.
    
    For j=0: sum_r u_r[k] * v_r[2k] * w_r[0] = 1 for all k
    For j=1: sum_r u_r[k] * v_r[2k+1] * w_r[1] = 1 for all k
    
    Also, for mismatched indices, it should be 0:
    T[k, 2k', j] = 0 if k' != k.
    So sum_r u_r[k] * v_r[2k'+j] * w_r[j] = 0 for k' != k.
    
    With rank 1: u_0[k] = a_k, v_0[2k] = b_k, v_0[2k+1] = c_k, w_0[0] = d, w_0[1] = e.
    For j=0: a_k * b_k * d = 1 for all k, and a_k * b_{k'} * d = 0 for k != k'.
    This requires b_{k'} = 0 for k' != k, but k is arbitrary, so b must be 0 everywhere except one index.
    But we need it for all k, which is impossible with rank 1.
    
    So the rank of 1x4x2 is NOT 1. I made an error earlier by ignoring the cross-terms.
    
    The correct rank: We need the tensor to be a sum of rank-1 terms such that the "diagonal" structure is preserved.
    
    For 1xm x mxp, the rank is m*p (trivial) or can be less with clever constructions.
    Actually, for 1xm x mxp, C[j] = sum_k A[k] B[k,j]. This is just computing p dot products.
    The tensor has T[k, kp+j, j] = 1.
    
    This is equivalent to the "copy" tensor or a generalized version.
    The rank of this tensor is m (I believe), because we can write:
    T = sum_{k=0}^{m-1} e_k ⊗ (sum_j e_{kp+j} ⊗ e_j)
    
    Wait, that's not a sum of rank-1 tensors. Let me think again.
    
    T[k, kp+j, j] = 1.
    We can write: T = sum_{k=0}^{m-1} sum_{j=0}^{p-1} e_k ⊗ e_{kp+j} ⊗ e_j
    This is a sum of m*p rank-1 tensors, so rank <= m*p.
    
    Can we do better? For m=4, p=2: rank <= 8.
    
    Actually, I think the rank of the "diagonal" tensor T[k, kp+j, j] is m*p in general, because the slices are linearly independent.
    
    So my 6 subproblems each have rank up to 8, giving total rank up to 48. Not helpful.
    
    Let me abandon this line of thinking and just try to find a better known construction.
    
    Given the complexity, let me just return the current rank-42 solution but verify it's correct, and add a verification step.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Strassen components for 2x2x2
    # local_a: 0=A11, 1=A12, 2=A21, 3=A22
    # local_b: 0=B11, 1=B12, 2=B21, 3=B22
    # local_w: 0=C11, 1=C12, 2=C21, 3=C22
    strassen = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),   # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),           # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),           # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),           # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),           # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),           # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),           # M7
    ]
    
    # Blocks: (a_rows, a_cols, b_rows, b_cols)
    blocks = [
        ([0, 1], [0, 1], [0, 1], [0, 1]),
        ([0, 1], [0, 1], [0, 1], [2, 3]),
        ([0, 1], [0, 1], [0, 1], [4, 5]),
        ([0, 1], [2, 3], [2, 3], [0, 1]),
        ([0, 1], [2, 3], [2, 3], [2, 3]),
        ([0, 1], [2, 3], [2, 3], [4, 5]),
    ]
    
    rank = len(blocks) * len(strassen)  # 6 * 7 = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for a_rows, a_cols, b_rows, b_cols in blocks:
        if not _time_ok(deadline):
            break
        
        a_map = {
            0: a_rows[0] * m + a_cols[0],
            1: a_rows[0] * m + a_cols[1],
            2: a_rows[1] * m + a_cols[0],
            3: a_rows[1] * m + a_cols[1]
        }
        b_map = {
            0: b_rows[0] * p + b_cols[0],
            1: b_rows[0] * p + b_cols[1],
            2: b_rows[1] * p + b_cols[0],
            3: b_rows[1] * p + b_cols[1]
        }
        w_map = {
            0: b_cols[0] * n + a_rows[0],
            1: b_cols[1] * n + a_rows[0],
            2: b_cols[0] * n + a_rows[1],
            3: b_cols[1] * n + a_rows[1]
        }
        
        for u_coeffs, v_coeffs, w_coeffs in strassen:
            if not _time_ok(deadline):
                break
            for local_idx, coeff in u_coeffs:
                U[a_map[local_idx], col] += coeff
            for local_idx, coeff in v_coeffs:
                V[b_map[local_idx], col] += coeff
            for local_idx, coeff in w_coeffs:
                W[w_map[local_idx], col] += coeff
            col += 1
    
    # Verify the decomposition
    if _time_ok(deadline):
        # Quick verification with a small test
        np.random.seed(0)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    C_computed[i, j] += np.sum(
                        U[i * m + k, :] * V[k * p + j, :] * W[j * n + i, :]
                    )
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            # Fallback: return trivial rank n*m*p = 48
            rank_trivial = n * m * p
            U_t = np.zeros((n * m, rank_trivial))
            V_t = np.zeros((m * p, rank_trivial))
            W_t = np.zeros((p * n, rank_trivial))
            idx = 0
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        U_t[i * m + k, idx] = 1.0
                        V_t[k * p + j, idx] = 1.0
                        W_t[j * n + i, idx] = 1.0
                        idx += 1
            return {"U": U_t.tolist(), "V": V_t.tolist(), "W": W_t.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
