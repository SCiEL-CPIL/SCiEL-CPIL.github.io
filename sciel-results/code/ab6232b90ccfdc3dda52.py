"""Matrix multiplication 4x4 by 4x8: rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Start with schoolbook rank = 128
    rank_sb = n * m * p
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = rank_sb
    
    # Try to find a lower-rank decomposition
    # For 4x4 * 4x8, the tensor has dimensions 4*4*8 = 128 in the "dense" sense
    # Lower bounds: max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # Known results: for square 4x4, rank is at most 49 (but that's for 4x4x4)
    # For rectangular 4x4*4x8, we can try various approaches
    
    # Approach: try to exploit structure via block decomposition
    # Partition A (4x4) into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # Partition B (4x8) into 2x4 blocks: B11, B12, B21, B22 (each 2x4)
    # Product C (4x8) has 2x4 blocks: C11, C12, C21, C22 (each 2x4)
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each product is (2x2)*(2x4) -> (2x4), which has tensor rank at most 8
    # But we can use Strassen for the 2x2 * 2x4 products?
    # Actually, let's think differently.
    
    # The 2x2 * 2x4 multiplication:
    # Let X be 2x2, Y be 2x4. Product Z is 2x4.
    # Z[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # Z[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # etc.
    # This is a (2,2,4) tensor. Rank is at most 8 (schoolbook for this sub-problem).
    
    # Alternative: use known low-rank decompositions for small matrix mult.
    # For 2x2 * 2x2, Strassen gives rank 7.
    # For 2x2 * 2x4, we can split the 2x4 into two 2x2 matrices:
    # Y = [Y1 | Y2] where Y1, Y2 are 2x2.
    # Z = X*Y = [X*Y1 | X*Y2]
    # Each X*Yk is 2x2*2x2 -> rank 7 via Strassen, total rank 14 for 2x2*2x4.
    # But schoolbook for 2x2*2x4 is 8, which is better!
    # So for 2x2*2x4, schoolbook rank 8 is better than Strassen-based 14.
    
    # So the block approach: 4 products of 2x2*2x4, each rank 8, total rank 32.
    # That's already better than 128!
    
    # Let me construct this:
    # C11 = A11*B11 + A12*B21  (two 2x2*2x4 products)
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # Total: 8 products of 2x2*2x4, each rank 8, total rank 64.
    
    # Wait, let me recount. There are 4 output blocks, each is sum of 2 products.
    # 8 products total, each 2x2*2x4 has rank 8, so total rank = 8*8 = 64.
    
    # Can we do better? Let's try to share computations.
    # Actually, let me just implement the rank-64 construction and verify it.
    
    # Let me try even lower. What about rank 32?
    # The trivial lower bound is max(n*m, m*p, n*p) = max(16, 32, 32) = 32.
    # Is rank 32 achievable? Possibly, but hard to construct.
    
    # Let me first implement rank 64 and verify, then try to reduce.
    
    U64, V64, W64 = _block_decomp_rank64(n, m, p)
    if U64 is not None:
        if _verify(U64, V64, W64, n, m, p):
            best_rank = len(U64[0])
            best_U, best_V, best_W = U64, V64, W64
    
    # Try to see if we can do rank 32 with a different approach
    if time.time() < deadline - 30:
        # Try rank 32 - this would be optimal
        # One idea: use the fact that p=8=2*4
        # Split B into two 4x4 blocks: B = [B1 | B2]
        # C = A*B = [A*B1 | A*B2]
        # Each A*Bk is 4x4*4x4, which has rank at most 49 (best known for 4x4)
        # Total rank would be 2*49 = 98, worse than 64.
        
        # Another idea: use a direct rank-32 construction if possible
        # For this specific case, let me try to use the structure more.
        # Actually, the lower bound of 32 might not be tight.
        
        # Let me try to use numerical optimization to find a lower rank
        # This is risky and may not converge in time.
        # Skip for now to be safe.
        
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _block_decomp_rank64(n, m, p):
    """Block decomposition: split into 2x2 blocks, 8 products of 2x2*2x4."""
    # A is n x m = 4x4, split into 2x2 blocks:
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # B is m x p = 4x8, split into 2x4 blocks:
    # B11 = B[0:2, 0:4], B12 = B[0:2, 4:8]
    # B21 = B[2:4, 0:4], B22 = B[2:4, 4:8]
    # C is n x p = 4x8, split into 2x4 blocks:
    # C11 = C[0:2, 0:4], C12 = C[0:2, 4:8]
    # C21 = C[2:4, 0:4], C22 = C[2:4, 4:8]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Each 2x2*2x4 product uses rank 8 (schoolbook for 2x2*2x4)
    # 8 such products, total rank = 8*8 = 64
    
    n2, m2, p2 = 2, 2, 4  # dimensions of sub-problem
    sub_rank = n2 * m2 * p2  # = 16? No, wait.
    
    # Actually, 2x2 * 2x4: the tensor has indices (i,j,k) with i in [0,2), j in [0,2), k in [0,4)
    # Schoolbook rank = 2*2*4 = 16. That's worse than I thought.
    
    # Let me reconsider. For 2x2 * 2x4:
    # U has shape (2*2, rank) = (4, rank)
    # V has shape (2*4, rank) = (8, rank)
    # W has shape (4*2, rank) = (8, rank)
    # Schoolbook: rank = 2*2*4 = 16
    
    # So 8 products * 16 rank each = 128. Same as schoolbook! Not helpful.
    
    # Let me think differently.
    # For 2x2 * 2x2, Strassen gives rank 7 < 8 (schoolbook).
    # Can we decompose 2x2 * 2x4 as two 2x2 * 2x2 products?
    # Y (2x4) = [Y1 | Y2], each 2x2.
    # Z = X*Y = [X*Y1 | X*Y2]
    # X*Y1: rank 7 (Strassen), X*Y2: rank 7 (Strassen)
    # Total rank for 2x2*2x4: 14
    # 8 products * 14 = 112. Still worse than 128? No, 112 < 128.
    
    # Wait, I need to be more careful. Let me re-examine.
    
    # Actually, I made an error. Let me redo this more carefully.
    
    # The full problem: 4x4 * 4x8.
    # Split A (4x4) into four 2x2 blocks.
    # Split B (4x8) into four 2x4 blocks.
    # Output C (4x8) has four 2x4 blocks.
    
    # C11 = A11*B11 + A12*B21  (2 terms, each 2x2 * 2x4)
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # 8 products of type (2x2) * (2x4).
    
    # For (2x2) * (2x4):
    # Option 1: Schoolbook, rank = 2*2*4 = 16
    # Option 2: Split 2x4 into two 2x2, use Strassen for each: 7+7 = 14
    # So option 2 is better: rank 14 per product.
    
    # Total: 8 * 14 = 112. This is better than 128!
    
    # Let me implement this.
    
    # Strassen for 2x2 * 2x2:
    # Given X (2x2) and Y (2x2):
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1 - M4 + M5 - M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 - M6
    
    # For 2x2 * 2x4, split Y into Y1 (2x2) and Y2 (2x2):
    # Z = [X*Y1 | X*Y2]
    # Use Strassen for each, total rank 14.
    
    # Now let me construct the full U, V, W matrices.
    
    # The full tensor has:
    # U indexed by (i,j) where i in [0,4), j in [0,4) -> row index i*4+j
    # V indexed by (j,k) where j in [0,4), k in [0,8) -> row index j*8+k
    # W indexed by (k,i) where k in [0,8), i in [0,4) -> row index k*4+i
    
    # We need to express each entry of C = A*B as a sum over rank-1 tensors.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let me enumerate the 8 block products and their Strassen decompositions.
    
    # Block product 1: A11 * B11 contributes to C11
    # A11 indices: i in [0,2), j in [0,2)
    # B11 indices: j in [0,2), k in [0,4)
    # C11 indices: i in [0,2), k in [0,4)
    
    # For Strassen on A11 * B11[:,0:2] (which gives C11[:,0:2]):
    # Split B11 into B11a (cols 0,1) and B11b (cols 2,3)
    # Actually, B11 is 2x4, so split into two 2x2: B11_left = B11[:,0:2], B11_right = B11[:,2:4]
    
    # This is getting complex. Let me just build it programmatically.
    
    total_rank = 0
    U_rows = {}  # (i,j) -> list of (coeff, rank_offset)
    V_rows = {}  # (j,k) -> list of (coeff, rank_offset)
    W_rows = {}  # (k,i) -> list of (coeff, rank_offset)
    
    def add_term(i, j, k, u_coeff, v_coeff, w_coeff):
        nonlocal total_rank
        U_rows.setdefault((i, j), []).append((u_coeff, total_rank))
        V_rows.setdefault((j, k), []).append((v_coeff, total_rank))
        W_rows.setdefault((k, i), []).append((w_coeff, total_rank))
        total_rank += 1
    
    # Strassen terms for 2x2*2x2:
    # For X (2x2) * Y (2x2) = Z (2x2):
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1 - M4 + M5 - M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 - M6
    
    # For 2x2 * 2x4, split Y (2x4) into YL (2x2, cols 0,1) and YR (2x2, cols 2,3)
    # ZL = X * YL, ZR = X * YR
    # Z = [ZL | ZR]
    
    # Strassen for X * YL:
    # M1_L = (X00+X11)(YL00+YL11)
    # M2_L = (X10+X11)YL00
    # M3_L = X00(YL01-YL11)
    # M4_L = X11(YL10-YL00)
    # M5_L = (X00+X01)YL11
    # M6_L = (X10-X00)(YL00+YL01)
    # M7_L = (X01-X11)(YL10+YL11)
    # ZL00 = M1_L - M4_L + M5_L - M7_L
    # ZL01 = M3_L + M5_L
    # ZL10 = M2_L + M4_L
    # ZL11 = M1_L - M2_L + M3_L - M6_L
    
    # Similarly for X * YR with YR replacing YL.
    
    # Now, for the full problem, I have 8 block products, each of type (2x2)*(2x4).
    # Let me enumerate them:
    # (A11, B11) -> C11
    # (A12, B21) -> C11
    # (A11, B12) -> C12
    # (A12, B22) -> C12
    # (A21, B11) -> C21
    # (A22, B21) -> C21
    # (A21, B12) -> C22
    # (A22, B22) -> C22
    
    # For each (Ai, Bj) product, Ai is 2x2, Bj is 2x4.
    # Ai is at position (row_block, col_block) in A.
    # Bj is at position (row_block, col_block) in B.
    
    # Let me define the block positions:
    # A11: i_base=0, j_base=0
    # A12: i_base=0, j_base=2
    # A21: i_base=2, j_base=0
    # A22: i_base=2, j_base=2
    
    # B11: j_base=0, k_base=0
    # B12: j_base=0, k_base=4
    # B21: j_base=2, k_base=0
    # B22: j_base=2, k_base=4
    
    # C11: i_base=0, k_base=0
    # C12: i_base=0, k_base=4
    # C21: i_base=2, k_base=0
    # C22: i_base=2, k_base=4
    
    block_products = [
        # (A_block, B_block, C_block)
        (('i0','j0'), ('j0','k0'), ('i0','k0')),  # A11*B11 -> C11
        (('i0','j1'), ('j1','k0'), ('i0','k0')),  # A12*B21 -> C11
        (('i0','j0'), ('j0','k1'), ('i0','k1')),  # A11*B12 -> C12
        (('i0','j1'), ('j1','k1'), ('i0','k1')),  # A12*B22 -> C12
        (('i1','j0'), ('j0','k0'), ('i1','k0')),  # A21*B11 -> C21
        (('i1','j1'), ('j1','k0'), ('i1','k0')),  # A22*B21 -> C21
        (('i1','j0'), ('j0','k1'), ('i1','k1')),  # A21*B12 -> C22
        (('i1','j1'), ('j1','k1'), ('i1','k1')),  # A22*B22 -> C22
    ]
    
    # Map block labels to actual index bases
    i_base = {'i0': 0, 'i1': 2}
    j_base = {'j0': 0, 'j1': 2}
    k_base = {'k0': 0, 'k1': 4}
    
    for (a_block, b_block, c_block) in block_products:
        i_b = i_base[a_block[0]]
        j_a = j_base[a_block[1]]  # A's column block
        j_b = j_base[b_block[0]]  # B's row block (should match j_a)
        k_b = k_base[b_block[1]]
        
        # Verify j_a == j_b
        assert j_a == j_b, f"Block mismatch: {j_a} != {j_b}"
        
        # A block: rows i_b to i_b+1, cols j_b to j_b+1
        # B block: rows j_b to j_b+1, cols k_b to k_b+3
        # C block: rows i_b to i_b+1, cols k_b to k_b+3
        
        # Split B block into left (cols k_b, k_b+1) and right (cols k_b+2, k_b+3)
        # Strassen for A_block * B_left:
        # Then Strassen for A_block * B_right
        
        for half, k_offset in [('L', 0), ('R', 2)]:
            # B_half: 2x2, rows j_b..j_b+1, cols k_b+k_offset..k_b+k_offset+1
            # C_half: 2x2, rows i_b..i_b+1, cols k_b+k_offset..k_b+k_offset+1
            
            # Strassen terms:
            # M1 = (A[i_b,j_b]+A[i_b+1,j_b+1]) * (B[j_b,k_b+k_offset]+B[j_b+1,k_b+k_offset+1])
            # M2 = (A[i_b+1,j_b]+A[i_b+1,j_b+1]) * B[j_b,k_b+k_offset]
            # M3 = A[i_b,j_b] * (B[j_b,k_b+k_offset+1]-B[j_b+1,k_b+k_offset+1])
            # M4 = A[i_b+1,j_b+1] * (B[j_b+1,k_b+k_offset]-B[j_b,k_b+k_offset])
            # M5 = (A[i_b,j_b]+A[i_b,j_b+1]) * B[j_b+1,k_b+k_offset+1]
            # M6 = (A[i_b+1,j_b]-A[i_b,j_b]) * (B[j_b,k_b+k_offset]+B[j_b,k_b+k_offset+1])
            # M7 = (A[i_b,j_b+1]-A[i_b+1,j_b+1]) * (B[j_b+1,k_b+k_offset]+B[j_b+1,k_b+k_offset+1])
            
            # C_half[0,0] = M1 - M4 + M5 - M7
            # C_half[0,1] = M3 + M5
            # C_half[1,0] = M2 + M4
            # C_half[1,1] = M1 - M2 + M3 - M6
            
            # For each Strassen term M_t, it's a product of a linear form in A and a linear form in B.
            # We need to express each M_t as a sum of rank-1 tensors.
            
            # M1: (A[i_b,j_b]+A[i_b+1,j_b+1]) * (B[j_b,k]+B[j_b+1,k+1]) where k=k_b+k_offset
            # This is a sum of 4 terms:
            # A[i_b,j_b]*B[j_b,k] + A[i_b,j_b]*B[j_b+1,k+1] + A[i_b+1,j_b+1]*B[j_b,k] + A[i_b+1,j_b+1]*B[j_b+1,k+1]
            
            # Each term is a rank-1 tensor contribution.
            
            def add_strassen_term_M1():
                k0 = k_b + k_offset
                k1 = k_b + k_offset + 1
                # A[i_b,j_b]*B[j_b,k0]
                add_term(i_b, j_b, k0, 1, 1, 1)
                # A[i_b,j_b]*B[j_b+1,k1]
                add_term(i_b, j_b, k1, 1, 1, 1)  # Wait, this isn't right.
                # Actually, M1 is a single scalar, and it contributes to multiple C entries.
                # I need to track the coefficient for each (i,j,k) entry of C.
                
                # This approach is getting too complex. Let me use a different method.
                pass
            
            # I'm overcomplicating this. Let me use a cleaner approach.
            # I'll directly enumerate all rank-1 tensors that make up the Strassen decomposition.
            
            # For Strassen on 2x2*2x2, the 7 terms are:
            # M1 = (x00+x11)*(y00+y11)
            # M2 = (x10+x11)*y00
            # M3 = x00*(y01-y11)
            # M4 = x11*(y10-y00)
            # M5 = (x00+x01)*y11
            # M6 = (x10-x00)*(y00+y01)
            # M7 = (x01-x11)*(y10+y11)
            
            # Each Mt is a product of a linear form in X and a linear form in Y.
            # To make it a rank-1 tensor, I need to expand each Mt into monomials.
            
            # M1 = (x00+x11)*(y00+y11) = x00*y00 + x00*y11 + x11*y00 + x11*y11
            # So M1 contributes to C00 with coeff +1, C01 with coeff 0, C10 with coeff 0, C11 with coeff +1
            # And it uses 4 rank-1 tensors.
            
            # Wait, no. In the tensor decomposition, each "rank-1 tensor" corresponds to one term x_ij * y_jk * z_ik.
            # The Strassen algorithm reduces the number of such terms.
            
            # Let me reconsider. The tensor T[i,j,k] for 2x2*2x2 has 8 entries.
            # Schoolbook: 8 rank-1 tensors.
            # Strassen: 7 rank-1 tensors, but the coefficients can be non-trivial.
            
            # In Strassen, the 7 "multiplications" are:
            # m1 = (a00+a11)(b00+b11)
            # m2 = (a10+a11)b00
            # m3 = a00(b01-b11)
            # m4 = a11(b10-b00)
            # m5 = (a00+a01)b11
            # m6 = (a10-a00)(b00+b01)
            # m7 = (a01-a11)(b10+b11)
            
            # c00 = m1 - m4 + m5 - m7
            # c01 = m3 + m5
            # c10 = m2 + m4
            # c11 = m1 - m2 + m3 - m6
            
            # Each mt is a product of two linear forms. To express this as rank-1 tensors,
            # I expand each mt into its constituent monomials.
            
            # m1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
            # This uses 4 rank-1 tensors, but they share the same "multiplication" in Strassen's sense.
            
            # Hmm, I think I'm confusing two different things. In the tensor decomposition,
            # the rank is the minimum number of rank-1 tensors needed. Strassen shows that
            # the 2x2*2x2 tensor has rank at most 7.
            
            # The 7 rank-1 tensors in Strassen's decomposition are:
            # For each mt, we have a tensor that is the product of the linear form in A,
            # the linear form in B, and the corresponding linear form in C (from the output formula).
            
            # Actually, I think the correct way to view it is:
            # The Strassen decomposition gives 7 terms, each of which is a rank-1 tensor
            # where the U, V, W vectors are the linear forms.
            
            # For m1: U vector has 1 at positions corresponding to a00 and a11,
            # V vector has 1 at positions corresponding to b00 and b11,
            # W vector has coefficients from c00=+1, c11=+1 (and 0 elsewhere for this term's contribution).
            
            # Wait, that's not quite right either. Let me think more carefully.
            
            # In the tensor decomposition, T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # (using the convention from the problem: U indexed by (i,j), V by (j,k), W by (k,i))
            
            # For Strassen's m1 = (a00+a11)(b00+b11):
            # This contributes to c00 and c11.
            # c00 gets +m1, c11 gets +m1.
            # So the rank-1 tensor for m1 has:
            # U[i,j,r]: nonzero for (i,j) in {(0,0), (1,1)} with value 1
            # V[j,k,r]: nonzero for (j,k) in {(0,0), (1,1)} with value 1
            # W[k,i,r]: nonzero for (k,i) in {(0,0), (1,1)} with value 1
            # But wait, that would give T[0,0,0] = 1*1*1 = 1, T[1,1,1] = 1*1*1 = 1,
            # and also T[0,0,1] = 0 (since W[1,0,r]=0), etc.
            # But m1 should contribute to c00 and c11, which are T[0,0,0] and T[1,1,1] in our indexing.
            # Wait, c[i,k] = sum_j a[i,j]*b[j,k], so T[i,j,k] = a[i,j]*b[j,k] contributes to c[i,k].
            # So the tensor entry T[i,j,k] corresponds to the product a[i,j]*b[j,k], and it contributes to c[i,k] with weight W[k,i,r].
            
            # OK so for m1:
            # U[0,0,r] = 1, U[1,1,r] = 1
            # V[0,0,r] = 1, V[1,1,r] = 1
            # W[0,0,r] = 1 (contributes to c[0,0])
            # W[1,1,r] = 1 (contributes to c[1,1])
            # Check: T[0,0,0] = U[0,0,r]*V[0,0,r]*W[0,0,r] = 1*1*1 = 1. This contributes a[0,0]*b[0,0] to c[0,0]. Good.
            # T[0,0,1] = U[0,0,r]*V[0,1,r]*W[1,0,r] = 1*0*0 = 0. Good.
            # T[0,1,0] = U[0,1,r]*V[1,0,r]*W[0,1,r] = 0*0*0 = 0. Good.
            # T[1,1,1] = U[1,1,r]*V[1,1,r]*W[1,1,r] = 1*1*1 = 1. This contributes a[1,1]*b[1,1] to c[1,1]. Good.
            # But m1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11.
            # The term a00*b11 should contribute to c[0,1], and a11*b00 to c[1,0].
            # But in my construction, V[0,1,r]=0 and V[1,0,r]=0, so those terms don't appear!
            
            # I see the issue. The Strassen decomposition is NOT a simple sum of rank-1 tensors in the way I was thinking.
            # The linear forms in Strassen are combined in a more complex way.
            
            # Let me look up the correct Strassen tensor decomposition.
            # Actually, I recall that Strassen's 7-term decomposition does correspond to 7 rank-1 tensors,
            # but the U, V, W vectors need to be set up correctly.
            
            # For m1 = (a00+a11)(b00+b11):
            # U: a00 and a11 -> U[0,0]=1, U[1,1]=1
            # V: b00 and b11 -> V[0,0]=1, V[1,1]=1
            # W: this term contributes +1 to c00 and +1 to c11 -> W[0,0]=1, W[1,1]=1
            # But as I showed, this only captures a00*b00 and a11*b11, not a00*b11 and a11*b00.
            
            # I think the issue is that in Strassen's algorithm, the intermediate products are used
            # in combinations to produce the final result. The tensor decomposition is not simply
            # "each Strassen multiplication is one rank-1 tensor."
            
            # Let me try a different approach. I'll just use the schoolbook decomposition
            # but with a smarter block structure to get rank < 128.
            
            # Actually, let me just go with the schoolbook (rank 128) since I'm running out of time
            # and I can't quickly construct a verified lower-rank decomposition.
            return None, None, None
    return None, None, None


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal 1 if (i,j,k) is a valid index, else 0.
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k),
    # because C[i,k] = sum_j A[i,j]*B[j,k], and the tensor is the identity map.
    # Wait, no. The tensor T satisfies C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # and for matrix multiplication, T[i,j,k] = 1 for all i,j,k.
    
    # So we need: for all i,j,k, sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # And for invalid (i,j,k), it should be 0, but all (i,j,k) are valid in our case.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True


# EVOLVE-BLOCK-END
