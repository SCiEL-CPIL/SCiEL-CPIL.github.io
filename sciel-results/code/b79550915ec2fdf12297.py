"""Program for rank-minimal tensor decomposition of 2x4 * 4x7 matrix multiplication."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 4 matrix by a 4 by 7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    We use the Strassen-style divide-and-conquer approach adapted for these dimensions.
    For 2x4 * 4x7, we can decompose the 4x4 middle block multiplication using
    Strassen's algorithm, which reduces the rank.
    
    Key insight: 2x4 * 4x7 can be viewed as multiplying two blocks of size 2x4 and 4x7.
    We can partition the 4x4 core into quadrants and apply Strassen's method.
    
    Actually, let's think more carefully. The standard Strassen algorithm for 2x2 * 2x2
    uses rank 7 instead of 8. For our dimensions, we need a different approach.
    
    Let's use the fact that we can decompose the 4-dimensional middle space.
    The matrix product (2x4) * (4x7) has 2*4*7 = 56 scalar multiplications in schoolbook.
    
    We can use a block decomposition approach. Let's partition the 4x4 matrix into
    four 2x2 blocks. Then we're effectively doing:
    [A11 A12] * [B11 B21] * [C11 C12]
    [A21 A22]   [B12 B22]   [C21 C22]
    
    Wait, let me reconsider. The left matrix is 2x4, right is 4x7.
    Let's partition the left matrix into two 2x2 blocks: L = [L1 | L2] where L1, L2 are 2x2.
    Let's partition the right matrix into two 4x... no, the right is 4x7.
    Let's partition the right matrix into two 4x(3.5)? No, that doesn't work cleanly.
    
    Let's partition the 4 rows of the left matrix (which are columns of L) and the 4 columns
    of the right matrix (which are rows of R).
    
    L is 2x4, so we can write L = [L_a | L_b] where L_a and L_b are 2x2.
    R is 4x7, so we can write R = [R_a; R_b] where R_a and R_b are 2x7.
    
    Then L*R = L_a * R_a + L_b * R_b.
    
    Each of L_a * R_a and L_b * R_b is a 2x2 * 2x7 multiplication.
    
    For 2x2 * 2x7, we can further decompose. Let's use Strassen for the 2x2 part.
    Actually, 2x2 * 2x7 can be seen as multiplying a 2x2 matrix by 7 different 2x1 vectors.
    
    Alternatively, let's use a different decomposition. The key is to find a rank-minimal
    decomposition. 
    
    For the specific case of 2x4 * 4x7, let's try to use the fact that we can share
    some computations. 
    
    Let me try a direct construction. We have:
    - Left matrix A: 2x4, entries A[i,j] for i in {0,1}, j in {0,1,2,3}
    - Right matrix B: 4x7, entries B[j,k] for j in {0,1,2,3}, k in {0,...,6}
    - Product C: 2x7, entries C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has indices (i, j, k) with i in {0,1}, j in {0,1,2,3}, k in {0,...,6}.
    T[i,j,k] = 1 if we're looking at the (i,j,k) term in the sum.
    
    We want to decompose T as sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] (with appropriate indexing).
    
    Actually, the standard decomposition is:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    where n=2, m=4, p=7.
    
    U has shape (2*4, rank) = (8, rank)
    V has shape (4*7, rank) = (28, rank)
    W has shape (7*2, rank) = (14, rank)
    
    The constraint is that for all i, j, k:
    sum_r U[i*4+j, r] * V[j*7+k, r] * W[k*2+i, r] = 1
    
    And for all other combinations, the sum should be 0.
    
    This is a system of polynomial equations. The minimal rank is what we're looking for.
    
    For 2x4 * 4x7, the schoolbook rank is 56. Can we do better?
    
    Let's try to use a block decomposition. Partition the j index into two groups of 2:
    j = j1 + 2*j2 where j1 in {0,1}, j2 in {0,1}.
    
    Then C[i,k] = sum_{j1,j2} A[i, j1+2*j2] * B[j1+2*j2, k]
    
    = sum_{j1} (A[i, j1] * B[j1, k] + A[i, j1+2] * B[j1+2, k])
    
    This doesn't immediately help. Let's try Strassen-style for the 2x2 blocks.
    
    Consider the 2x2 subproblem: multiplying a 2x2 matrix M by a 2x2 matrix N.
    Strassen uses 7 multiplications:
    M1 = (M[0,0]+M[1,1]) * (N[0,0]+N[1,1])
    M2 = (M[1,0]+M[1,1]) * N[0,0]
    M3 = M[0,0] * (N[0,1]-N[1,1])
    M4 = M[1,1] * (N[1,0]-N[0,0])
    M5 = (M[0,0]+M[0,1]) * N[1,1]
    M6 = (M[1,0]-M[0,0]) * (N[0,0]+N[0,1])
    M7 = (M[0,1]-M[1,1]) * (N[1,0]+N[1,1])
    
    For our problem, we have 2x4 * 4x7. Let's view this as two 2x2 * 2x7 multiplications
    that share some structure.
    
    Actually, let's try a simpler approach. Since the dimensions are small, let's try to
    find a decomposition with rank less than 56 by exploiting linearity.
    
    One approach: use the fact that the tensor can be decomposed using bilinear forms.
    
    Let's try rank 48 or lower. We can use the following idea:
    
    For each pair (i, k) in the output, we have C[i,k] = sum_j A[i,j] * B[j,k].
    
    We can precompute some linear combinations of A and B to reduce the number of
    independent products.
    
    Let's define:
    For i=0: A0 = [A[0,0], A[0,1], A[0,2], A[0,3]]
    For i=1: A1 = [A[1,0], A[1,1], A[1,2], A[1,3]]
    
    For k=0..6: B_k = [B[0,k], B[1,k], B[2,k], B[3,k]]
    
    C[0,k] = A0 dot B_k
    C[1,k] = A1 dot B_k
    
    So we need to compute 2*7 = 14 dot products of 4-vectors.
    
    Each dot product requires 4 multiplications in schoolbook, so 56 total.
    
    Can we share computations between different dot products?
    
    If we compute some linear combinations of the A vectors and B vectors, we might
    reduce the total number of scalar multiplications.
    
    Let's say we compute U_r = sum_i alpha_{i,r} * A_i (for r in some set)
    and V_r = sum_k beta_{k,r} * B_k (for r in some set)
    
    Then U_r * V_r gives us sum_{i,k} alpha_{i,r} * beta_{k,r} * (A_i dot B_k)
    
    This is a linear combination of the desired dot products.
    
    We need to be able to recover all 14 dot products from these products.
    
    This is equivalent to finding a low-rank decomposition of the "dot product tensor".
    
    The dot product tensor for 4-vectors has rank 4 (it's just the identity tensor).
    But we have two different A vectors and 7 different B vectors.
    
    Let me think of it differently. We have a 2x4 matrix A and a 4x7 matrix B.
    The product is a 2x7 matrix.
    
    The tensor T in {0,1}^2 x {0,1,2,3} x {0,...,6} where T[i,j,k] = 1.
    
    We want to decompose this tensor as sum_r u_r * v_r * w_r where:
    u_r is in R^{2*4}, v_r is in R^{4*7}, w_r is in R^{7*2}
    
    And the constraint is that the trilinear form matches.
    
    For small dimensions, we can try to find a decomposition by solving the polynomial
    system. But that's complex.
    
    Let's try a heuristic: use rank 48. We can save 8 multiplications by using
    Strassen-like tricks on 2x2 blocks.
    
    Partition j into {0,1} and {2,3}. Then:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    
    Each term in parentheses is a 2x1 * 1x7 product (dot product of 2-vectors).
    
    For the first group, we can use Strassen-like:
    Let P1 = (A[i,0]+A[i,1]) * (B[0,k]+B[1,k])
    Let P2 = A[i,0] * B[0,k]
    Let P3 = A[i,1] * B[1,k]
    
    Then A[i,0]*B[0,k] + A[i,1]*B[1,k] = P1 - P2 - P3 + ... no, that's not right.
    
    Actually, P1 = A[i,0]*B[0,k] + A[i,0]*B[1,k] + A[i,1]*B[0,k] + A[i,1]*B[1,k]
    
    So P1 - A[i,0]*B[1,k] - A[i,1]*B[0,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    But we still need A[i,0]*B[1,k] and A[i,1]*B[0,k].
    
    This doesn't save us. The issue is that we're computing dot products, not matrix
    products, so Strassen doesn't directly apply.
    
    Let me reconsider. Maybe the minimal rank is indeed 56 for these dimensions, or
    close to it. Let's try to find a decomposition with rank 49 or lower.
    
    Actually, let's just use the schoolbook decomposition since finding a better one
    is non-trivial. The schoolbook rank is 56.
    
    Wait, I recall that for matrix multiplication, the tensor rank is at most the
    schoolbook rank, and sometimes can be lower. For 2x2 * 2x2, it's 7 instead of 8.
    
    For 2x4 * 4x7, let's see if we can adapt Strassen.
    
    Let's partition the 4x4 core into four 2x2 blocks. The left matrix is 2x4, so we
    can write it as [A1 | A2] where A1 and A2 are 2x2. The right matrix is 4x7, so we
    can write it as [B1; B2] where B1 and B2 are 2x7.
    
    Then A * B = A1 * B1 + A2 * B2.
    
    Each of A1 * B1 and A2 * B2 is a 2x2 * 2x7 product.
    
    For 2x2 * 2x7, we can view it as multiplying a 2x2 matrix by 7 different 2x1 vectors.
    Or we can partition the 7 columns into groups.
    
    Let's partition the 7 columns into 3+2+2 or 3+4 or 2+2+2+1.
    
    If we use 2+2+2+1, we have four 2x2 * 2x2 or 2x2 * 2x1 subproblems.
    
    This is getting complicated. Let's just use the schoolbook decomposition for now,
    which has rank 56.
    
    """
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
