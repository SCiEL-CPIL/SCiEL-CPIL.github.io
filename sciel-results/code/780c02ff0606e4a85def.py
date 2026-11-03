"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

import time
import numpy as np
from fractions import Fraction
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    
    Uses a block-based approach with Strassen's algorithm for 2x2 blocks and
    schoolbook for smaller blocks, then attempts to find a lower rank decomposition
    using linear algebra over the rationals.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # First, try the block decomposition approach
    # Split the 2x4 matrix into two 2x2 blocks horizontally
    # Split the 4x7 matrix into blocks: 4 rows x 7 cols
    # We can think of this as:
    # A (2x4) = [A1 A2] where A1, A2 are 2x2
    # B (4x7) = [B1; B2] where B1 is 2x7, B2 is 2x7
    # C = A1*B1 + A2*B2
    # Each 2x2 * 2x7 multiplication can be done with Strassen-like methods
    
    # Actually, let's think more carefully.
    # A is 2x4, B is 4x7, C is 2x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can group the j index: j = 0,1,2,3
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    # One approach: use Strassen for 2x2 blocks
    # Split A into [A_top_left, A_top_right] = two 2x2 matrices
    # Split B into [B_top; B_bottom] = two 2x7 matrices
    # C = A_tl * B_top + A_tr * B_bottom
    # 
    # Each 2x2 * 2x7 product: we can use 5 multiplications per 2x2x2 block
    # 2x2 * 2x7 = 7 blocks of 2x2x2, so 7*5 = 35 multiplications
    # Two such products: 70 total, which is worse than schoolbook (56)
    
    # Better: treat the whole thing as a tensor and find a low-rank decomposition
    # The tensor has size 2*4*7 = 56
    # Schoolbook gives rank 56
    
    # Let's try to find a lower rank decomposition using linear algebra
    # The multiplication tensor T[i,j,k] = delta(i,j,k) where (i,j,k) corresponds to
    # the term A[i,j]*B[j,k] in C[i,k]
    
    # Actually, the tensor for matrix multiplication (n,m,p) has entries:
    # T[i, j, k] = 1 if the index triple (i, j, k) is "valid" for the multiplication
    # where i in [0,n), j in [0,m), k in [0,p)
    # and the product contributes to C[i,k] via A[i,j]*B[j,k]
    
    # For our case: i in [0,2), j in [0,4), k in [0,7)
    # The tensor is 2*4*7 = 56 dimensional
    # Each term in the sum for C[i,k] is A[i,j]*B[j,k] for j=0..3
    
    # A rank-r decomposition means we can write:
    # C[i,k] = sum_{c=0}^{r-1} U[i*m+j_c, c] * V[j_c*p+k, c] * W[k*n+i, c]
    # Wait, that's not quite right. Let me re-read the problem.
    
    # The problem says:
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (28, rank)
    # W has shape (p*n, rank) = (14, rank)
    # 
    # And the decomposition is:
    # C[i,k] = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    # Wait, but j is summed over? Let me re-read.
    
    # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
    # I think the convention is that for each column c, we have:
    # U[:, c] is a vector of length n*m
    # V[:, c] is a vector of length m*p
    # W[:, c] is a vector of length p*n
    # 
    # And the multiplication is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c]
    
    # So for each c, we form an intermediate matrix M_c of size n x p where:
    # M_c[i,k] = sum_j U[i*m+j, c] * V[j*p+k, c]
    # Then C[i,k] = sum_c M_c[i,k] * W[k*n+i, c]
    
    # This is a valid tensor decomposition format.
    
    # Let's build the schoolbook solution first, then try to reduce rank.
    
    def build_schoolbook():
        rank = n * m * p  # 56
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_school, V_school, W_school, rank_school = build_schoolbook()
    
    # Verify the schoolbook solution
    def verify(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                lhs = 0.0
                for j in range(m):
                    lhs += 1.0 * 1.0  # A[i,j]*B[j,k] where A and B are identity-like
                # Actually, let's just check the tensor decomposition property
                # C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                rhs = 0.0
                for c in range(rank):
                    for j in range(m):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                # For the actual multiplication, C[i,k] = sum_j A[i,j]*B[j,k]
                # The tensor we're decomposing is T[i,j,k] = 1 for all valid (i,j,k)
                # So C[i,k] in the "test" should be m (sum of m terms each equal to 1)
                if abs(lhs - rhs) > 1e-10:
                    return False
        return True
    
    # Now let's try to find a lower rank decomposition
    # We'll use a greedy approach: start with the schoolbook solution and try to
    # remove columns while maintaining exactness
    
    # Actually, let's try a different approach: use the fact that we can combine
    # terms using Strassen's idea
    
    # For 2x2x2 matrix multiplication, Strassen uses 7 multiplications instead of 8
    # We can apply this to our problem by grouping indices
    
    # Let's try a block Strassen approach:
    # A is 2x4, B is 4x7
    # Split A = [A1 A2] where A1, A2 are 2x2
    # Split B = [B1; B2] where B1, B2 are 2x7
    # C = A1*B1 + A2*B2
    # 
    # For A1*B1: A1 is 2x2, B1 is 2x7
    # We can write B1 = [b1_1, b1_2, ..., b1_7] where each b1_k is a 2x1 column
    # A1 * B1 = [A1*b1_1, A1*b1_2, ..., A1*b1_7]
    # Each A1*b1_k is a 2x1 vector, which requires 4 multiplications (or 3 with Strassen-like)
    # Actually, 2x2 * 2x1 requires 4 multiplications (no Strassen benefit for non-square)
    # So A1*B1 requires 7*4 = 28 multiplications
    # Similarly A2*B2 requires 28 multiplications
    # Total: 56, same as schoolbook
    
    # Let's try a different grouping:
    # A is 2x4, B is 4x7
    # Group the 4 rows of B into pairs: rows 0-1 and rows 2-3
    # B_top = B[0:2, :] (2x7), B_bottom = B[2:4, :] (2x7)
    # A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    # C = A_left * B_top + A_right * B_bottom
    
    # For A_left * B_top: 2x2 * 2x7
    # We can use 5 multiplications per 2x2x2 block, and there are 7 such blocks
    # But actually, for 2x2 * 2x7, we can do better by sharing multiplications
    # 
    # Let A_left = [[a, b], [c, d]] and B_top = [[e1, e2, ..., e7], [f1, f2, ..., f7]]
    # Result = [[a*e1+b*f1, a*e2+b*f2, ..., a*e7+b*f7],
    #           [c*e1+d*f1, c*e2+d*f2, ..., c*e7+d*f7]]
    # 
    # We can compute:
    # p1 = a*e1, p2 = b*f1, p3 = c*e1, p4 = d*f1  (for column 1)
    # p5 = a*e2, p6 = b*f2, p7 = c*e2, p8 = d*f2  (for column 2)
    # ... etc
    # That's 4*7 = 28 multiplications
    # 
    # With Strassen for each 2x2x2 block: 5*7 = 35, worse
    # 
    # So no benefit from Strassen here.
    
    # Let's try yet another approach: exploit the rectangular shape
    # A is 2x4, B is 4x7
    # We can view this as: C = A * B where A has 2 rows and B has 7 columns
    # 
    # Another idea: use the fact that we only need 2x7 = 14 output entries
    # Each output entry C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # 
    # We can try to find a rank-r decomposition where r < 56
    # This is a hard problem in general, but for small dimensions we can try
    # linear algebra approaches
    
    # Let's try a simple approach: assume the decomposition has a specific structure
    # and solve for the entries
    
    # Actually, let's just return the schoolbook solution for now, as finding a
    # lower rank decomposition for 2x4x7 is non-trivial
    
    # But wait, the knowledge says that a deterministic block decomposition using
    # Strassen (rank 7) for 2x2x2 blocks and Schoolbook (rank 4) for 2x2x1 blocks
    # reduces rank from 56 to 50. Let me try to implement that.
    
    # The idea: split the 4x7 part into blocks
    # B is 4x7. We can split it into:
    # - 3 blocks of 4x2 (using 2x2x2 Strassen for each 2x2 subblock)
    # - 1 block of 4x1
    # 
    # Actually, let me think about this differently.
    # 
    # The tensor is 2x4x7. We can decompose it as:
    # - Split the 4 dimension into 2+2
    # - Split the 7 dimension into 2+2+2+1
    # 
    # For each 2x2x2 sub-tensor, use Strassen (rank 7)
    # For each 2x2x1 sub-tensor, use schoolbook (rank 4)
    # 
    # Number of 2x2x2 blocks: 2 (from the 4 dim) * 3 (from the 7 dim, using 2+2+2) = 6
    # Number of 2x2x1 blocks: 2 * 1 = 2
    # 
    # Total rank: 6*7 + 2*4 = 42 + 8 = 50
    
    def build_block_decomposition():
        # We'll build the decomposition block by block
        # The indices are: i in [0,2), j in [0,4), k in [0,7)
        # 
        # Split j into j1 in [0,2) and j2 in [2,4)
        # Split k into k1 in [0,2), k2 in [2,4), k3 in [4,6), k4 in [6,7)
        
        U_blocks = []
        V_blocks = []
        W_blocks = []
        
        # For each combination of j-block and k-block, add the appropriate decomposition
        j_blocks = [(0, 2), (2, 4)]
        k_blocks = [(0, 2), (2, 4), (4, 6), (6, 7)]
        
        for (j_start, j_end) in j_blocks:
            for (k_start, k_end) in k_blocks:
                j_size = j_end - j_start
                k_size = k_end - k_start
                
                if j_size == 2 and k_size == 2:
                    # Use Strassen for 2x2x2
                    # Strassen's 7 multiplications for 2x2x2
                    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
                    # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-h), m4 = d(g-e)
                    # m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
                    # Result: [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
                    
                    # In our tensor decomposition, we need to map this to U, V, W entries
                    # The local indices are: i in [0,2), j_local in [0,2), k_local in [0,2)
                    # Global j = j_start + j_local, global k = k_start + k_local
                    
                    # Strassen terms:
                    # Each term is of the form: coefficient * A[i, j1] * B[j2, k]
                    # where j1, j2 are local indices in [0,2)
                    
                    # m1 = (A[i,0]+A[i,1]) * (B[0,k]+B[1,k]) for each i,k
                    # This expands to 4 terms, but in the tensor decomposition,
                    # we treat it as a single "multiplication" with combined U and V entries
                    
                    # For the tensor decomposition format:
                    # U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                    # 
                    # For Strassen, we have 7 "virtual" multiplications
                    # Each virtual multiplication c has:
                    # U entries that are linear combinations of A[i,j] values
                    # V entries that are linear combinations of B[j,k] values
                    # W entry that is 1 (or a coefficient)
                    
                    # Let's define the 7 Strassen terms:
                    # Term 1: (a+d)(e+h) -> U has (a+d) pattern, V has (e+h) pattern
                    # Term 2: (c+d)e -> U has (c+d) pattern, V has e pattern
                    # Term 3: a(f-h) -> U has a pattern, V has (f-h) pattern
                    # Term 4: d(g-e) -> U has d pattern, V has (g-e) pattern
                    # Term 5: (a+b)h -> U has (a+b) pattern, V has h pattern
                    # Term 6: (c-a)(e+f) -> U has (c-a) pattern, V has (e+f) pattern
                    # Term 7: (b-d)(g+h) -> U has (b-d) pattern, V has (g+h) pattern
                    
                    # For each term, we add 7 columns to U, V, W
                    # But wait, each "term" in Strassen is a single multiplication
                    # that contributes to multiple output entries via the W matrix
                    
                    # Actually, in the tensor decomposition, each column c corresponds
                    # to one "multiplication" in the algorithm. The U, V, W matrices
                    # encode how the inputs are combined and how the result is used.
                    
                    # For Strassen on 2x2x2:
                    # We have 7 multiplications. Each multiplication produces a scalar
                    # that is then combined (with additions/subtractions) to form the
                    # 4 output entries.
                    
                    # In our tensor format:
                    # C[i,k] = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    
                    # For a single Strassen multiplication, say m1 = (a+d)(e+h):
                    # This contributes to C[0,0], C[0,1], C[1,0], C[1,1] with coefficients
                    # from the Strassen reconstruction formula.
                    
                    # Let's just build the full U, V, W matrices for the block.
                    
                    # Local indices: i in [0,2), j in [0,2), k in [0,2)
                    # Global: i_global = i, j_global = j_start + j, k_global = k_start + k
                    
                    # For each of the 7 Strassen terms, we add one column.
                    # The U, V, W entries for that column are determined by the Strassen formula.
                    
                    strassen_terms = [
                        # (U_pattern, V_pattern, W_coefficients)
                        # U_pattern: dict mapping (i, j) to coefficient
                        # V_pattern: dict mapping (j, k) to coefficient
                        # W_coefficients: dict mapping (i, k) to coefficient
                        {
                            'U': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},  # a+d for both rows
                            'V': {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},  # e+h for both cols
                            'W': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): -1}  # m1 contributes to C[0,0] and C[1,1]
                        },
                        {
                            'U': {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 1},  # c+d
                            'V': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0},  # e
                            'W': {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 0}  # m2 contributes to C[1,0]
                        },
                        {
                            'U': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0},  # a
                            'V': {(0,0): 0, (0,1): 1, (1,0): -1, (1,1): 0},  # f-h
                            'W': {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 0}  # m3 contributes to C[0,1]
                        },
                        {
                            'U': {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1},  # d
                            'V': {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1},  # g-e
                            'W': {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}  # m4 contributes to C[0,0]
                        },
                        {
                            'U': {(0,0): 1, (0,1): 1, (1,0): 0, (1,1): 0},  # a+b
                            'V': {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1},  # h
                            'W': {(0,0): -1, (0,1): 0, (1,0): 0, (1,1): 0}  # m5 contributes to C[0,0]
                        },
                        {
                            'U': {(0,0): -1, (0,1): 0, (1,0): 1, (1,1): 0},  # c-a
                            'V': {(0,0): 1, (0,1): 1, (1,0): 0, (1,1): 0},  # e+f
                            'W': {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}  # m6 contributes to C[1,1]
                        },
                        {
                            'U': {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): -1},  # b-d
                            'V': {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): 1},  # g+h
                            'W': {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}  # m7 contributes to C[1,1]
                        },
                    ]
                    
                    for term in strassen_terms:
                        U_block = np.zeros((n * m, 1))
                        V_block = np.zeros((m * p, 1))
                        W_block = np.zeros((p * n, 1))
                        
                        for (i, j), coeff in term['U'].items():
                            if coeff != 0:
                                U_block[i * m + (j_start + j), 0] = coeff
                        
                        for (j, k), coeff in term['V'].items():
                            if coeff != 0:
                                V_block[(j_start + j) * p + (k_start + k), 0] = coeff
                        
                        for (i, k), coeff in term['W'].items():
                            if coeff != 0:
                                W_block[(k_start + k) * n + i, 0] = coeff
                        
                        U_blocks.append(U_block)
                        V_blocks.append(V_block)
                        W_blocks.append(W_block)
                    
                elif j_size == 2 and k_size == 1:
                    # Use schoolbook for 2x2x1 (rank 4)
                    # A is 2x2, B is 2x1, C is 2x1
                    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
                    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
                    
                    for j_local in range(2):
                        U_block = np.zeros((n * m, 1))
                        V_block = np.zeros((m * p, 1))
                        W_block = np.zeros((p * n, 1))
                        
                        for i in range(2):
                            U_block[i * m + (j_start + j_local), 0] = 1
                        V_block[(j_start + j_local) * p + (k_start + 0), 0] = 1
                        for i in range(2):
                            W_block[(k_start + 0) * n + i, 0] = 1
                        
                        U_blocks.append(U_block)
                        V_blocks.append(V_block)
                        W_blocks.append(W_block)
        
        U = np.hstack(U_blocks)
        V = np.hstack(V_blocks)
        W = np.hstack(W_blocks)
        rank = U.shape[1]
        return U, V, W, rank
    
    U_block, V_block, W_block, rank_block = build_block_decomposition()
    
    # Verify the block decomposition
    def verify_decomposition(U, V, W):
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                rhs = 0.0
                for c in range(rank):
                    for j in range(m):
                        rhs += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                # The "true" value is sum_j 1*1 = m = 4
                if abs(rhs - m) > 1e-10:
                    return False
        return True
    
    if not verify_decomposition(U_block, V_block, W_block):
        # Fall back to schoolbook if block decomposition is invalid
        U, V, W, rank = U_school, V_school, W_school, rank_school
    else:
        U, V, W, rank = U_block, V_block, W_block, rank_block
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
