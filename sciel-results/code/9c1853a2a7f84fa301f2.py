"""
Strategy: Algebraic Tensor Decomposition via Block Strassen Recursion.
The current program attempts random local search which fails to find the algebraic structure.
This program explicitly constructs the tensor decomposition for 2x4 * 4x8 using
Strassen's algorithm logic.

Dimensions: n=2, m=4, p=8.
We can view A (2x4) and B (4x8) as blocks of 2x2.
A = [[A11, A12], [A21, A22]] where each is 1x2? No, let's stick to standard Strassen on square blocks.
Actually, 2x4 * 4x8 is not square.
Let's decompose based on the Strassen formula for 2x2 * 2x2, which has rank 7 instead of 8.
For n=2, m=4, p=8:
We can split the middle dimension m=4 into two blocks of 2.
A (2x4) -> A1 (2x2), A2 (2x2)
B (4x8) -> B1 (2x8), B2 (2x8)
Result C (2x8) = A1*B1 + A2*B2.
This doesn't directly apply Strassen because B1 and B2 are 2x8, not 2x2.

However, we can split p=8 into four blocks of 2.
B1 (2x2), B2 (2x2), B3 (2x2), B4 (2x2).
C = A1*B1 + A2*B2 + A1*B3 + A2*B4 ... wait, indices.
A is 2x4. B is 4x8.
Let A = [A_L, A_R] (each 2x2).
Let B = [[B_LL, B_LR], [B_RL, B_RR]] (each 2x2).
C = A_L * B_L + A_R * B_R, where B_L is 2x4 and B_R is 2x4.
This is getting complex. Let's use the general Strassen decomposition for $n \times m$ by $m \times p$.
If n, m, p are powers of 2, we can recursively apply Strassen.
Base case: 1x1 * 1x1 -> rank 1.
Case: 2x2 * 2x2 -> rank 7.
Case: 2x4 * 4x8:
Split A (2x4) into A1 (1x4), A2 (1x4)? No, Strassen splits all dimensions by 2.
Split A (2x4) -> A11(1x2), A12(1x2), A21(1x2), A22(1x2).
Split B (4x8) -> B11(2x4), B12(2x4), B21(2x4), B22(2x4).
This mixes dimensions. Standard Strassen applies to square matrices.
For rectangular, we can pad or use a generalized version.

Alternative: Use the known Strassen-like decomposition for 2x4x8 directly.
Rank lower bound for 2x4x8 is not trivially 32.
However, we can construct a decomposition of rank 32 by treating it as two independent 2x4x4 multiplications?
No, 2x4 * 4x8 = (2x4 * 4x4) + (2x4 * 4x4) if we split p.
Let B = [B1, B2] where B1, B2 are 4x4.
C = A*B1 + A*B2.
If we compute A*B1 (2x4) and A*B2 (2x4) separately using Strassen for 2x4*4x4?
2x4 * 4x4:
Split A (2x4) -> A1(1x4), A2(1x4)?
Let's just implement the recursive Strassen construction for the tensor.

Recursive Construction:
def build_strassen(n, m, p):
    if n==1 and m==1 and p==1:
        return U(1,1)=1, V(1,1)=1, W(1,1)=1
    Split dimensions in half.
    n1=n//2, m1=m//2, p1=p//2.
    
    This requires n,m,p to be even. 2,4,8 are all even.
    
    Let's trace the recursion for 2x4x8.
    Level 0: 2x4x8
    Split A (2x4) into 4 blocks of 1x2: A11, A12, A21, A22.
    Split B (4x8) into 4 blocks of 2x4: B11, B12, B21, B22.
    Split C (2x8) into 4 blocks of 1x4: C11, C12, C21, C22.
    
    This is not standard Strassen. Standard Strassen assumes square blocks.
    For rectangular, we can use "Strassen for rectangular" or just decompose into square sub-problems.
    
    Let's try a different decomposition:
    2x4 * 4x8.
    View as (2x2 * 2x4) + (2x2 * 2x4) by splitting the middle dimension m=4 into two 2s.
    A = [A1, A2] (2x2 each).
    B = [B1; B2] (2x8 each).
    C = A1*B1 + A2*B2.
    Now we have two multiplications of 2x2 * 2x8.
    2x2 * 2x8:
    Split p=8 into four 2s.
    B1 = [B11, B12, B13, B14] (2x2 each).
    C1 = A1*B11 + A1*B12 + A1*B13 + A1*B14.
    This is 4 multiplications of 2x2 * 2x2.
    Each 2x2 * 2x2 can be done with rank 7 (Strassen).
    Total rank = 4 * 7 = 28 for the first half (A1 part).
    Similarly for A2 part: 4 * 7 = 28.
    Total rank = 56.
    
    This matches the "inspiration" finding: Strassen reduces 64 to 56.
    Can we do better?
    Maybe combine the Strassen terms across the split?
    
    Let's stick to the 56 rank construction first, as it is a guaranteed improvement over 64.
    The current program scores 1.0 (rank 64). 56 is a 1.14x improvement.
    
    Implementation Plan:
    1. Define the Strassen decomposition for 2x2 * 2x2 (rank 7).
    2. Construct the 2x4 * 4x8 decomposition by tiling these 2x2 * 2x2 blocks.
    3. Verify and return.
"""

import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-56 decomposition for 2x4 * 4x8 using Strassen's algorithm logic.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    # 1. Define Strassen decomposition for 2x2 * 2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A*B
    # Strassen terms:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # We need to map these to the global tensor indices.
    # Global dimensions: n=2, m=4, p=8.
    # We split m into two blocks of 2 (m1=2, m2=2).
    # We split p into four blocks of 2 (p1=2, p2=2, p3=2, p4=2).
    # A is 2x4. Blocks: A1 (2x2) corresponds to A[:, 0:2], A2 (2x2) corresponds to A[:, 2:4].
    # B is 4x8. Blocks:
    #   B1 (2x8) corresponds to B[0:2, :]. Split into B11(2x2), B12(2x2), B13(2x2), B14(2x2).
    #   B2 (2x8) corresponds to B[2:4, :]. Split into B21(2x2), B22(2x2), B23(2x2), B24(2x2).
    
    # C = A1*B1 + A2*B2.
    # A1*B1 = A1*B11 + A1*B12 + A1*B13 + A1*B14.
    # A2*B2 = A2*B21 + A2*B22 + A2*B23 + A2*B24.
    
    # Each term (e.g., A1*B11) is a 2x2 * 2x2 multiplication.
    # We apply Strassen to each of the 8 such multiplications.
    # Total rank = 8 * 7 = 56.

    # Let's build the U, V, W matrices.
    # U shape: (n*m, rank) = (8, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (16, 56)
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Strassen coefficients for 2x2
    # M1: U coeff on A, V coeff on B, W coeff on C
    # A indices in 2x2 block: 00, 01, 10, 11
    # B indices in 2x2 block: 00, 01, 10, 11
    # C indices in 2x2 block: 00, 01, 10, 11
    
    # We define the 7 Strassen terms as tuples of (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    # Indices are local to the 2x2 block.
    
    strassen_terms = [
        # M1 = (a00+a11)*(b00+b11) -> C00 += M1, C11 += M1
        {
            'u': [(0, 1), (3, 1)],  # a00, a11
            'v': [(0, 1), (3, 1)],  # b00, b11
            'w': [(0, 1), (3, 1)]   # c00, c11
        },
        # M2 = (a10+a11)*b00 -> C10 += M2, C11 -= M2
        {
            'u': [(2, 1), (3, 1)],  # a10, a11
            'v': [(0, 1)],          # b00
            'w': [(2, 1), (3, -1)]  # c10, c11
        },
        # M3 = a00*(b01-b11) -> C01 += M3, C11 += M3
        {
            'u': [(0, 1)],          # a00
            'v': [(1, 1), (3, -1)], # b01, b11
            'w': [(1, 1), (3, 1)]   # c01, c11
        },
        # M4 = a11*(b10-b00) -> C00 += M4, C10 += M4
        {
            'u': [(3, 1)],          # a11
            'v': [(2, 1), (0, -1)], # b10, b00
            'w': [(0, 1), (2, 1)]   # c00, c10
        },
        # M5 = (a00+a01)*b11 -> C00 -= M5, C01 += M5
        {
            'u': [(0, 1), (1, 1)],  # a00, a01
            'v': [(3, 1)],          # b11
            'w': [(0, -1), (1, 1)]  # c00, c01
        },
        # M6 = (a10-a00)*(b00+b01) -> C11 += M6
        {
            'u': [(2, 1), (0, -1)], # a10, a00
            'v': [(0, 1), (1, 1)],  # b00, b01
            'w': [(3, 1)]           # c11
        },
        # M7 = (a01-a11)*(b10+b11) -> C00 += M7
        {
            'u': [(1, 1), (3, -1)], # a01, a11
            'v': [(2, 1), (3, 1)],  # b10, b11
            'w': [(0, 1)]           # c00
        }
    ]

    # Map local 2x2 indices to global indices
    # Global A is 2x4.
    # Block A1 (rows 0-1, cols 0-1 of A):
    #   Local 00 -> Global A[0,0] -> idx 0
    #   Local 01 -> Global A[0,1] -> idx 1
    #   Local 10 -> Global A[1,0] -> idx 4
    #   Local 11 -> Global A[1,1] -> idx 5
    # Block A2 (rows 0-1, cols 2-3 of A):
    #   Local 00 -> Global A[0,2] -> idx 2
    #   Local 01 -> Global A[0,3] -> idx 3
    #   Local 10 -> Global A[1,2] -> idx 6
    #   Local 11 -> Global A[1,3] -> idx 7

    # Global B is 4x8.
    # Block B1 (rows 0-1, cols 0-7 of B):
    #   Sub-block B11 (rows 0-1, cols 0-1):
    #     Local 00 -> Global B[0,0] -> idx 0
    #     Local 01 -> Global B[0,1] -> idx 1
    #     Local 10 -> Global B[1,0] -> idx 8
    #     Local 11 -> Global B[1,1] -> idx 9
    #   Sub-block B12 (rows 0-1, cols 2-3):
    #     Local 00 -> Global B[0,2] -> idx 2
    #     Local 01 -> Global B[0,3] -> idx 3
    #     Local 10 -> Global B[1,2] -> idx 10
    #     Local 11 -> Global B[1,3] -> idx 11
    #   Sub-block B13 (rows 0-1, cols 4-5):
    #     Local 00 -> Global B[0,4] -> idx 4
    #     Local 01 -> Global B[0,5] -> idx 5
    #     Local 10 -> Global B[1,4] -> idx 12
    #     Local 11 -> Global B[1,5] -> idx 13
    #   Sub-block B14 (rows 0-1, cols 6-7):
    #     Local 00 -> Global B[0,6] -> idx 6
    #     Local 01 -> Global B[0,7] -> idx 7
    #     Local 10 -> Global B[1,6] -> idx 14
    #     Local 11 -> Global B[1,7] -> idx 15
    # Block B2 (rows 2-3, cols 0-7 of B):
    #   Sub-block B21 (rows 2-3, cols 0-1):
    #     Local 00 -> Global B[2,0] -> idx 16
    #     Local 01 -> Global B[2,1] -> idx 17
    #     Local 10 -> Global B[3,0] -> idx 24
    #     Local 11 -> Global B[3,1] -> idx 25
    #   ... and so on.

    # Global C is 2x8.
    # Block C1 (rows 0-1, cols 0-7 of C):
    #   Sub-block C11 (rows 0-1, cols 0-1):
    #     Local 00 -> Global C[0,0] -> idx 0
    #     Local 01 -> Global C[0,1] -> idx 1
    #     Local 10 -> Global C[1,0] -> idx 2
    #     Local 11 -> Global C[1,1] -> idx 3
    #   ...

    # Helper to get global index for A
    def a_idx(r, c):
        return r * 4 + c

    # Helper to get global index for B
    def b_idx(r, c):
        return r * 8 + c

    # Helper to get global index for C (W convention: k*n+i, so C[k, i])
    # C is 2x8. C[i, k]. W index is k*2 + i.
    def w_idx(i, k):
        return k * 2 + i

    # Define the 8 sub-problems:
    # (A_block_offset, B_block_offset, C_block_offset)
    # A_block_offset: 0 for A1 (cols 0-1), 2 for A2 (cols 2-3)
    # B_block_offset: 0 for B1 (rows 0-1), 16 for B2 (rows 2-3) -- wait, B index is r*8+c.
    # B1 rows 0-1. B2 rows 2-3.
    # B sub-block col offset: 0, 2, 4, 6.
    
    sub_problems = []
    # A1 part
    for b_col_off in [0, 2, 4, 6]:
        sub_problems.append((0, b_col_off, 0)) # A1, B1_sub, C1_sub
    # A2 part
    for b_col_off in [0, 2, 4, 6]:
        sub_problems.append((2, 16 + b_col_off, 0)) # A2, B2_sub, C2_sub? 
        # Wait, C is 2x8. A1*B1 contributes to C cols 0-7?
        # A1 is 2x2 (rows 0-1, cols 0-1 of A).
        # B1 is 2x8 (rows 0-1, cols 0-7 of B).
        # A1*B1 is 2x8. It contributes to all of C.
        # A2 is 2x2 (rows 0-1, cols 2-3 of A).
        # B2 is 2x8 (rows 2-3, cols 0-7 of B).
        # A2*B2 is 2x8. It contributes to all of C.
        # So both contribute to the same C blocks.
        
    # Let's refine the mapping.
    # We have 8 Strassen decompositions.
    # For each decomposition d in 0..7:
    #   We pick a specific A_sub (A1 or A2), a specific B_sub (B11..B14 or B21..B24), and C_sub (C11..C14 or C21..C24).
    #   Actually, A1*B11 contributes to C11.
    #   A1*B12 contributes to C12.
    #   A1*B13 contributes to C13.
    #   A1*B14 contributes to C14.
    #   A2*B21 contributes to C11.
    #   A2*B22 contributes to C12.
    #   A2*B23 contributes to C13.
    #   A2*B24 contributes to C14.
    
    # So for C11 (cols 0-1 of C), we have contributions from A1*B11 and A2*B21.
    # Both use Strassen.
    # Total 8 Strassen instances.
    
    col = 0
    for a_off in [0, 2]: # 0 for A1, 2 for A2
        for b_col_off in [0, 2, 4, 6]:
            # Determine B row offset
            if a_off == 0:
                b_row_off = 0 # B1
            else:
                b_row_off = 16 # B2 (rows 2-3 start at index 16 in B's flattened 4x8? No.
                # B is 4x8. Flattened index r*8+c.
                # Row 0: 0-7. Row 1: 8-15. Row 2: 16-23. Row 3: 24-31.
                # B1 is rows 0-1. B2 is rows 2-3.
                # So B1 base index 0. B2 base index 16.
                pass
            
            # C column offset is same as B column offset.
            c_col_off = b_col_off
            
            # Apply Strassen terms
            for term in strassen_terms:
                # U part
                for (local_idx, coeff) in term['u']:
                    lr, lc = local_idx // 2, local_idx % 2
                    gr = lr
                    gc = a_off + lc
                    g_idx = a_idx(gr, gc)
                    U[g_idx, col] += coeff
                
                # V part
                for (local_idx, coeff) in term['v']:
                    lr, lc = local_idx // 2, local_idx % 2
                    gr = b_row_off // 8 + lr # b_row_off is 0 or 16. 16//8 = 2.
                    gc = b_col_off + lc
                    g_idx = b_idx(gr, gc)
                    V[g_idx, col] += coeff
                
                # W part
                for (local_idx, coeff) in term['w']:
                    lr, lc = local_idx // 2, local_idx % 2
                    gi = lr
                    gk = c_col_off + lc
                    g_idx = w_idx(gi, gk)
                    W[g_idx, col] += coeff
                
                col += 1
    
    # Verify
    def verify(U_mat, V_mat, W_mat, n, m, p, num_tests=10):
        rng = np.random.default_rng(42)
        for _ in range(num_tests):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((m, p))
            expected = A @ B
            computed = np.zeros((n, p))
            for c in range(U_mat.shape[1]):
                u_vec = U_mat[:, c]
                v_vec = V_mat[:, c]
                w_vec = W_mat[:, c]
                u_reshaped = u_vec.reshape(n, m)
                v_reshaped = v_vec.reshape(m, p)
                w_reshaped = w_vec.reshape(p, n)
                uv = u_reshaped @ v_reshaped
                w_transposed = w_reshaped.T
                computed += uv * w_transposed
            if not np.allclose(computed, expected, atol=1e-10, rtol=1e-10):
                return False
        return True

    if verify(U, V, W, n, m, p):
        def to_half_integer_list(mat):
            result = []
            for row in mat:
                row_list = []
                for val in row:
                    rounded = round(val * 2) / 2
                    row_list.append(rounded)
                result.append(row_list)
            return result
        
        return {
            "U": to_half_integer_list(U),
            "V": to_half_integer_list(V),
            "W": to_half_integer_list(W)
        }
    else:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        def to_half_integer_list(mat):
            result = []
            for row in mat:
                row_list = []
                for val in row:
                    rounded = round(val * 2) / 2
                    row_list.append(rounded)
                result.append(row_list)
            return result
        
        return {
            "U": to_half_integer_list(U),
            "V": to_half_integer_list(V),
            "W": to_half_integer_list(W)
        }
# EVOLVE-BLOCK-END
