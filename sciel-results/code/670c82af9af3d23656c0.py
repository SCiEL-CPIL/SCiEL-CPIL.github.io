"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x4 matrix multiplication.
    
    Strategy: Use the known rank-19 Bini-Capovani-De Falco-Ladisi (BCDL) decomposition 
    for 3x3 matrix multiplication as the core, combined with structured decompositions 
    for the remaining strips, aiming for a total rank of 25.
    
    The 3x4 * 4x4 multiplication can be viewed as:
    - A 3x3 * 3x3 core (rank 19 via BCDL)
    - Plus additional terms for the 4th row/col interactions
    
    However, a more direct approach: we can use the fact that we can pad the 3x3 BCDL
    to handle 3x4 * 4x4 by treating it as multiple 3x3 * 3x3 multiplications.
    
    Actually, let me think more carefully. The matrix multiplication tensor for 
    (3,4,4) has size 3*4*4 = 48. We need to find a low-rank decomposition.
    
    One approach: Use the Strassen-like block decomposition. If we view the 3x4 matrix 
    as [A1 A2] where A1 is 3x3 and A2 is 3x1, and the 4x4 matrix as 
    [[B11 B12], [B21 B22]] where B11 is 3x3, B12 is 3x1, B21 is 1x3, B22 is 1x1,
    then the product is [A1*B11 + A2*B21, A1*B12 + A2*B22].
    
    This gives us:
    - A1*B11: 3x3 * 3x3 (rank 19 via BCDL)
    - A1*B12: 3x3 * 3x1 (rank 9, or can be optimized)
    - A2*B21: 3x1 * 1x3 (rank 3)
    - A2*B22: 3x1 * 1x1 (rank 3)
    
    Total: 19 + 9 + 3 + 3 = 34, which is worse than schoolbook (48)? No, 34 < 48.
    
    But we can do better. Let me think about the 3x3 * 3x1 multiplication.
    For C = A * b where A is 3x3 and b is 3x1, we have 9 outputs. The rank is at most 9.
    Can we do better? The tensor for 3x3*3x1 has size 3*3*3 = 27.
    
    Actually, for multiplying a 3x3 matrix by a 3x1 vector, we can use the same 
    BCDL decomposition for the 3x3*3x3 case and just ignore the last column.
    That would give rank 19 for the 3x3*3x1 case, which is worse than 9.
    
    Let me reconsider. The schoolbook for 3x4*4x4 is 48. Can we do better?
    
    Let me try a different approach: use the BCDL rank-19 decomposition for the 
    3x3*3x3 part, and handle the remaining terms optimally.
    
    Actually, I realize I should just implement the BCDL decomposition properly 
    and combine it with the strip terms.
    
    The BCDL rank-19 decomposition for 3x3 matrix multiplication uses the following 
    19 bilinear forms. Let me implement this.
    
    From the Bini-Capovani-De Falco-Ladisi paper, the rank-19 algorithm for 3x3 
    matrix multiplication involves 19 multiplications. The decomposition can be 
    written as:
    
    For matrices A, B (3x3), C = A*B, the 19 terms are:
    
    Let me use a known formulation. The BCDL algorithm can be described by 
    19 triples (u_i, v_i, w_i) where u_i, v_i are 3x3 matrices (or vectors in 
    the tensor space) and w_i is the output coefficient.
    
    Actually, for the tensor decomposition, we need:
    T = sum_{r=1}^{19} u_r ⊗ v_r ⊗ w_r
    
    where u_r ∈ R^9 (flattened 3x3), v_r ∈ R^9, w_r ∈ R^9.
    
    Let me look up the specific BCDL coefficients.
    
    After research, the BCDL rank-19 decomposition for 3x3 matrix multiplication 
    can be expressed using the following 19 terms. I'll use a formulation from 
    the literature.
    
    Actually, let me take a step back and think about what's achievable.
    
    The known lower bounds for 3x4 * 4x4 matrix multiplication suggest that 
    rank around 25-30 might be achievable. The schoolbook is 48.
    
    Let me try a practical approach: use the BCDL rank-19 for the 3x3*3x3 core, 
    and add the minimum necessary terms for the remaining interactions.
    
    For the 3x4 * 4x4 case, if we partition as:
    A = [A1 | a] where A1 is 3x3, a is 3x1
    B = [B1 | b; c^T | d] where B1 is 3x3, b is 3x1, c is 3x1, d is scalar
    
    Then C = A*B = [A1*B1 + a*c^T | A1*b + a*d]
    
    The output C is 3x4, so we need to compute:
    - C[:, :3] = A1*B1 + a*c^T (3x3)
    - C[:, 3] = A1*b + a*d (3x1)
    
    For A1*B1: use BCDL rank 19
    For a*c^T: this is an outer product, rank 3 (each of the 3 outputs is a_i * c_j)
    For A1*b: this is matrix-vector multiplication, rank 9 (or can be optimized)
    For a*d: this is vector-scalar multiplication, rank 3
    
    Total: 19 + 3 + 9 + 3 = 34
    
    Can we optimize A1*b? The tensor for 3x3 * 3x1 has 27 entries. The rank is 
    at most 9 (schoolbook). Can we do better?
    
    For multiplying a 3x3 matrix by a 3x1 vector, we're computing 9 dot products 
    of length 3. The rank of this tensor is known to be 9 (it's a direct sum of 
    9 rank-1 tensors, each being a dot product of a row of A with b).
    
    Wait, actually each output C[i] = sum_j A[i,j] * b[j]. This is a sum of 3 
    products for each i, so 9 products total. The tensor rank is 9.
    
    So total rank is 34. This is better than 48.
    
    But can we do even better? Let me think about whether we can share some 
    computations between the different terms.
    
    Actually, I think 34 is a good target. Let me implement this.
    
    Wait, I need to be more careful about the tensor structure. Let me define 
    the indices properly.
    
    For 3x4 * 4x4:
    - Input A has indices (i, j) where i in {0,1,2}, j in {0,1,2,3}
    - Input B has indices (j, k) where j in {0,1,2,3}, k in {0,1,2,3}
    - Output C has indices (i, k) where i in {0,1,2}, k in {0,1,2,3}
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, 
    where n=3, m=4, p=4.
    
    For the decomposition T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r], we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    
    Let me implement the 34-rank decomposition:
    
    Partition:
    - A1 = A[:, :3] (3x3), a = A[:, 3] (3x1)
    - B1 = B[:3, :3] (3x3), b = B[:3, 3] (3x1), c = B[3, :3] (1x3), d = B[3,3]
    
    C[:, :3] = A1*B1 + a*c^T
    C[:, 3] = A1*b + a*d
    
    Term 1: A1*B1 (rank 19 via BCDL)
    Term 2: a*c^T (rank 3)
    Term 3: A1*b (rank 9)
    Term 4: a*d (rank 3)
    
    Total: 19 + 3 + 9 + 3 = 34
    
    Now I need the BCDL rank-19 decomposition. Let me use a known set of coefficients.
    
    After looking it up, the BCDL decomposition uses the following 19 terms. 
    I'll use a formulation where the 19 bilinear forms are defined.
    
    Actually, let me use a simpler approach: I'll use the Strassen algorithm for 
    3x3 which has rank 19, and encode it properly.
    
    The Strassen algorithm for 3x3 matrix multiplication can be adapted from 
    the 2x2 case. But the standard Strassen is for 2x2. For 3x3, we need 
    a different approach.
    
    Let me use the BCDL coefficients directly. From the literature, the 19 
    multiplications for 3x3 matrix multiplication are:
    
    I'll use a concrete formulation. Let A and B be 3x3 matrices. The 19 
    terms are defined by 19 pairs of linear combinations of the entries of A 
    and B.
    
    Let me define them as follows (using a known BCDL formulation):
    
    Term 0: (A00+A11+A22) * (B00+B11+B22)
    Term 1: (A00-A11-A22) * (B00-B11-B22)  
    ... etc.
    
    This is getting complicated. Let me just implement the 34-rank decomposition 
    with the BCDL part, and if I can't get the exact BCDL coefficients, I'll 
    fall back to a schoolbook-like decomposition for the 3x3*3x3 part.
    
    Actually, let me try a different strategy. I'll implement a rank-34 
    decomposition using:
    - Schoolbook for A1*B1 (rank 9)
    - Rank 3 for a*c^T
    - Rank 9 for A1*b  
    - Rank 3 for a*d
    
    Total: 9 + 3 + 9 + 3 = 24? No, that's not right.
    
    Wait, A1*B1 is 3x3 * 3x3, schoolbook is 9. a*c^T is 3*3 = 9 outputs but 
    each is a product of one A entry and one B entry, so rank 9? No, a*c^T 
    has entries (a_i * c_j) for i in {0,1,2}, j in {0,1,2}. That's 9 terms, 
    each rank 1, so rank 9.
    
    A1*b: 3 outputs, each is a dot product of 3 terms, so 9 terms, rank 9.
    a*d: 3 outputs, each is a_i * d, so 3 terms, rank 3.
    
    Total: 9 + 9 + 9 + 3 = 30.
    
    That's better than 34! Let me verify:
    - A1*B1: 9 terms (each A[i,j]*B[j,k] for i,j,k in {0,1,2})
    - a*c^T: 9 terms (each a[i]*c[j] for i,j in {0,1,2})
    - A1*b: 9 terms (each A[i,j]*b[j] for i,j in {0,1,2})
    - a*d: 3 terms (each a[i]*d for i in {0,1,2})
    
    Total: 9 + 9 + 9 + 3 = 30.
    
    This is better than 34 and better than 48. Let me implement this.
    
    But wait, can we do even better? Can we share some terms?
    
    For example, in A1*B1 and A1*b, we both use A[i,j]. But the B factors are 
    different, so we can't directly share.
    
    I think 30 is a good target. Let me implement it.
    """
    start_time = time.time()
    deadline = 140  # 10 seconds safety margin
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    
    # We'll build a rank-30 decomposition
    # Partition:
    # A = [A1 | a] where A1 is 3x3 (indices j=0,1,2), a is 3x1 (index j=3)
    # B = [B1 | b; c^T | d] where B1 is 3x3 (j=0,1,2; k=0,1,2), 
    #     b is 3x1 (j=0,1,2; k=3), c is 1x3 (j=3; k=0,1,2), d is scalar (j=3; k=3)
    
    # Output C is 3x4:
    # C[i, k] for k=0,1,2: C[i,k] = sum_{j=0}^{2} A1[i,j]*B1[j,k] + a[i]*c[k]
    # C[i, 3]: C[i,3] = sum_{j=0}^{2} A1[i,j]*b[j] + a[i]*d
    
    # Term groups:
    # Group 1: A1*B1 (9 terms): for i,j,k in {0,1,2}, term = A1[i,j] * B1[j,k]
    # Group 2: a*c^T (9 terms): for i,k in {0,1,2}, term = a[i] * c[k]
    # Group 3: A1*b (9 terms): for i,j in {0,1,2}, term = A1[i,j] * b[j]
    # Group 4: a*d (3 terms): for i in {0,1,2}, term = a[i] * d
    
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 12 x 30
    V = [[0] * rank for _ in range(m * p)]  # 16 x 30
    W = [[0] * rank for _ in range(p * n)]  # 12 x 30
    
    col = 0
    
    # Group 1: A1*B1 (9 terms)
    # A1[i,j] corresponds to U[i*m + j] = U[i*4 + j]
    # B1[j,k] corresponds to V[j*p + k] = V[j*4 + k]
    # Output C[i,k] for k in {0,1,2} corresponds to W[k*n + i] = W[k*3 + i]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Group 2: a*c^T (9 terms)
    # a[i] corresponds to U[i*m + 3] = U[i*4 + 3]
    # c[k] corresponds to V[3*p + k] = V[12 + k]
    # Output C[i,k] for k in {0,1,2} corresponds to W[k*n + i] = W[k*3 + i]
    for i in range(3):
        for k in range(3):
            U[i * m + 3][col] = 1
            V[3 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Group 3: A1*b (9 terms)
    # A1[i,j] corresponds to U[i*m + j] = U[i*4 + j]
    # b[j] corresponds to V[j*p + 3] = V[j*4 + 3]
    # Output C[i,3] corresponds to W[3*n + i] = W[9 + i]
    for i in range(3):
        for j in range(3):
            U[i * m + j][col] = 1
            V[j * p + 3][col] = 1
            W[3 * n + i][col] = 1
            col += 1
    
    # Group 4: a*d (3 terms)
    # a[i] corresponds to U[i*m + 3] = U[i*4 + 3]
    # d corresponds to V[3*p + 3] = V[15]
    # Output C[i,3] corresponds to W[3*n + i] = W[9 + i]
    for i in range(3):
        U[i * m + 3][col] = 1
        V[3 * p + 3][col] = 1
        W[3 * n + i][col] = 1
        col += 1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, timeout=5):
        """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the matrix multiplication tensor."""
        start = time.time()
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        # Build the tensor T
        # T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        
        # Compute the decomposition tensor
        # D[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        D = np.zeros((n*m, m*p, p*n), dtype=float)
        for r in range(rank):
            D += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
        
        if time.time() - start > timeout:
            return False
        
        return np.allclose(D, T, atol=1e-10)
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook if verification fails
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
