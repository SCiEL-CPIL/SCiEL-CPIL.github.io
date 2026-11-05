import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Laderman's algorithm for 3x3 matrix multiplication (rank 23)
    # Indices: A[i,j] -> i*3+j, B[i,j] -> i*3+j, C[i,j] -> j*3+i (trace convention)
    
    # Define the 23 rank-1 terms for 3x3 matrix multiplication
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict mapping (i,j) -> coeff for A[i,j]
    # V_coeffs: dict mapping (i,j) -> coeff for B[i,j]
    # W_coeffs: dict mapping (i,j) -> coeff for C[i,j]
    
    laderman_terms = [
        # Term 0: C[0,0] = A[0,0]*B[0,0]
        ({(0,0): 1}, {(0,0): 1}, {(0,0): 1}),
        # Term 1: C[0,1] = A[0,1]*B[0,1]
        ({(0,1): 1}, {(0,1): 1}, {(0,1): 1}),
        # Term 2: C[0,2] = A[0,2]*B[0,2]
        ({(0,2): 1}, {(0,2): 1}, {(0,2): 1}),
        # Term 3: C[1,0] = A[1,0]*B[1,0]
        ({(1,0): 1}, {(1,0): 1}, {(1,0): 1}),
        # Term 4: C[1,1] = A[1,1]*B[1,1]
        ({(1,1): 1}, {(1,1): 1}, {(1,1): 1}),
        # Term 5: C[1,2] = A[1,2]*B[1,2]
        ({(1,2): 1}, {(1,2): 1}, {(1,2): 1}),
        # Term 6: C[2,0] = A[2,0]*B[2,0]
        ({(2,0): 1}, {(2,0): 1}, {(2,0): 1}),
        # Term 7: C[2,1] = A[2,1]*B[2,1]
        ({(2,1): 1}, {(2,1): 1}, {(2,1): 1}),
        # Term 8: C[2,2] = A[2,2]*B[2,2]
        ({(2,2): 1}, {(2,2): 1}, {(2,2): 1}),
        # Term 9: (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) -> contributes to C[0,0], C[1,1], C[0,1], C[1,0]
        # Actually let's use the standard Laderman decomposition
        # The standard Laderman algorithm has 23 multiplications.
        # Let me use a known correct set of 23 terms.
    ]
    
    # Let me use a verified set of Laderman terms from literature
    # Using the formulation where we have 23 scalar multiplications
    
    # I'll construct the 23 terms explicitly
    # A[i,j] index in flat array: i*3+j
    # B[i,j] index in flat array: i*3+j  
    # C[i,j] index in flat array (trace convention W[col*n+row]): j*3+i
    
    terms_3x3 = []
    
    # 9 diagonal terms
    for i in range(3):
        for j in range(3):
            terms_3x3.append((
                {(i,j): 1},
                {(i,j): 1},
                {(i,j): 1}
            ))
    
    # 14 additional terms for Laderman
    # These come from the structure of Laderman's algorithm
    
    # Let me use a different approach: use the known rank-23 decomposition
    # from the literature. Here's a standard set:
    
    # Clear and rebuild
    terms_3x3 = []
    
    # I'll use the following known Laderman decomposition
    # Based on: Laderman, "On the arithmetic circuit complexity of algebraic computations"
    
    # Actually, let me use a simpler and verifiable approach:
    # Use the Kronecker product of Strassen (2x2, rank 7) with trivial (3x3, rank 9)
    # This gives rank 7*9 = 63 for a 6x6 matrix multiplication using 3x3 blocks
    # where each 3x3 block multiplication uses Strassen-like structure on 2x2 sub-blocks
    # Wait, that doesn't work directly.
    
    # Let me use: partition 6x6 into 3x3 blocks (4 blocks of 2x2 each? No, 6/2=3 blocks)
    # So 3x3 block matrix where each block is 2x2.
    # Block multiplication: 3x3 block matrix mult, each entry is 2x2 matrix.
    # Use Strassen for the 2x2 matrix multiplications: 7 multiplications per block mult.
    # But we need to multiply 3x3 block matrices, so we need to compute each of the 9 output blocks.
    # Each output block is a 2x2 matrix = sum of 3 products of 2x2 blocks.
    # Using Strassen for each 2x2 multiplication: 7 scalar mults per 2x2 mult.
    # But we're doing 3x3 block structure, so we have 9 output blocks, each requiring 
    # multiple 2x2 multiplications.
    
    # Actually the cleanest approach: 
    # 6x6 = 3 blocks of size 2. So A is 3x3 block matrix with 2x2 blocks.
    # C = A*B where A, B are 3x3 block matrices with 2x2 blocks.
    # C[i,j] = sum_s A[i,s] * B[s,j] where each product is 2x2 matrix mult.
    # There are 9 output blocks, each is sum of 3 2x2 matrix products.
    # Total 2x2 matrix products: 9*3 = 27.
    # Each 2x2 matrix mult uses Strassen: 7 scalar mults.
    # Total rank: 27*7 = 189. Same as before.
    
    # To get lower rank, I need a better 3x3 matrix multiplication algorithm.
    # The best known for 3x3 is rank 23 (Laderman).
    # So if I can do 3x3 matrix mult with rank 23, and each "scalar" is a 2x2 matrix,
    # then rank = 23 * 4 = 92 (since 2x2 matrix mult has rank 4 trivially, or 7 with Strassen).
    # Wait, no. If I treat the 3x3 block structure, and use Laderman's 23 "scalar" multiplications,
    # but each "scalar" is actually a 2x2 matrix multiplication, then I need rank(2x2) * 23.
    # rank(2x2) = 4 (trivial) or 7 (Strassen). 
    # So total rank = 23 * 4 = 92 or 23 * 7 = 161.
    
    # Using trivial 2x2 mult (rank 4): total rank = 92.
    # This is much better than 189!
    
    # Let me implement Laderman's algorithm for 3x3 with rank 23.
    # I need the explicit 23 terms.
    
    # Here is a correct set of 23 terms for 3x3 matrix multiplication:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # Laderman's algorithm finds 23 bilinear forms.
    
    # Let me use a verified implementation. The following is from known literature:
    
    # I'll define the 23 terms as (u, v, w) where u, v, w are coefficient vectors
    # u has 9 entries (for A[0,0], A[0,1], A[0,2], A[1,0], ..., A[2,2])
    # v has 9 entries (for B[0,0], B[0,1], B[0,2], B[1,0], ..., B[2,2])
    # w has 9 entries (for C[0,0], C[0,1], C[0,2], C[1,0], ..., C[2,2])
    # But C is stored in trace convention: C_flat[j*3+i] for C[i,j]
    
    # Let me hardcode the 23 terms from a reliable source.
    # Using the formulation where indices are 0-8 for the 9 entries.
    
    # A[i,j] -> index i*3+j
    # B[i,j] -> index i*3+j
    # C[i,j] -> index j*3+i (trace convention)
    
    laderman_23 = [
        # (u_coeffs, v_coeffs, w_coeffs) as lists of 9 elements
        # Term 0
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),  # A00*B00 -> C00
        # Term 1
        ([0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),  # A01*B01 -> C01
        # Term 2
        ([0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),  # A02*B02 -> C02
        # Term 3
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),  # A10*B10 -> C10
        # Term 4
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),  # A11*B11 -> C11
        # Term 5
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0]),  # A12*B12 -> C12
        # Term 6
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0]),  # A20*B20 -> C20
        # Term 7
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0]),  # A21*B21 -> C21
        # Term 8
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),  # A22*B22 -> C22
        # Terms 9-22: The remaining 14 terms from Laderman
        # I need to add these correctly. Let me use a known correct set.
    ]
    
    # The above only gives 9 terms (the trivial diagonal ones). I need 23 total.
    # Let me use a different strategy: I'll use the fact that I can verify the decomposition
    # at construction time using numpy, and if it's wrong, fall back to a known correct method.
    
    # Actually, let me just use the Kronecker product approach with Strassen for 2x2 blocks
    # and trivial multiplication for the 3x3 block structure, which gives rank 189.
    # But that's what the current program does.
    
    # Let me try a different approach: use the Winograd Fourier Transform Method (WFTM)
    # or just use a known rank-161 decomposition.
    
    # Given the complexity of getting Laderman's 23 terms exactly right, let me use
    # a hybrid: partition into 2x2 blocks (3x3 block structure) and use Strassen for
    # the 2x2 multiplications, but also use a smarter 3x3 block algorithm.
    
    # Actually, the simplest improvement: use 2x2 Strassen blocks but with a 
    # different block partition. Or use the fact that rank(6,6,6) <= 161 is known.
    
    # Let me implement the 161-rank decomposition using 7 (Strassen) x 23 (Laderman).
    # I'll hardcode the Laderman coefficients from a verified source.
    
    # Here is a correct Laderman decomposition (23 terms):
    # Using the notation from "Algebraic Complexity Theory" by Bürgisser, Clausen, Shokrollahi
    
    # I'll use a computational approach: build the 3x3 matrix multiplication tensor
    # and find a rank-23 decomposition using known results.
    
    # For safety, let me use the 189-rank solution (Strassen on 2x2 blocks in 3x3 structure)
    # which I know is correct, and see if I can verify it works.
    
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for s in range(3):
            for j in range(3):
                for k in range(7):
                    idx_a = (2 * i) * 6 + (2 * s)
                    idx_b = (2 * i) * 6 + (2 * s + 1)
                    idx_c = (2 * i + 1) * 6 + (2 * s)
                    idx_d = (2 * i + 1) * 6 + (2 * s + 1)
                    
                    idx_e = (2 * s) * 6 + (2 * j)
                    idx_f = (2 * s) * 6 + (2 * j + 1)
                    idx_g = (2 * s + 1) * 6 + (2 * j)
                    idx_h = (2 * s + 1) * 6 + (2 * j + 1)
                    
                    idx_w00 = (2 * j) * 6 + (2 * i)
                    idx_w01 = (2 * j + 1) * 6 + (2 * i)
                    idx_w10 = (2 * j) * 6 + (2 * i + 1)
                    idx_w11 = (2 * j + 1) * 6 + (2 * i + 1)
                    
                    if k == 0:
                        U[idx_a][col] = 1
                        U[idx_d][col] = 1
                        V[idx_e][col] = 1
                        V[idx_h][col] = 1
                        W[idx_w00][col] = 1
                        W[idx_w11][col] = 1
                    elif k == 1:
                        U[idx_c][col] = 1
                        U[idx_d][col] = 1
                        V[idx_e][col] = 1
                        W[idx_w10][col] = 1
                        W[idx_w11][col] = -1
                    elif k == 2:
                        U[idx_a][col] = 1
                        V[idx_f][col] = 1
                        V[idx_h][col] = -1
                        W[idx_w01][col] = 1
                        W[idx_w11][col] = 1
                    elif k == 3:
                        U[idx_d][col] = 1
                        V[idx_g][col] = 1
                        V[idx_e][col] = -1
                        W[idx_w00][col] = 1
                        W[idx_w10][col] = 1
                    elif k == 4:
                        U[idx_a][col] = 1
                        U[idx_b][col] = 1
                        V[idx_h][col] = 1
                        W[idx_w00][col] = -1
                        W[idx_w01][col] = 1
                    elif k == 5:
                        U[idx_c][col] = 1
                        U[idx_a][col] = -1
                        V[idx_e][col] = 1
                        V[idx_f][col] = 1
                        W[idx_w11][col] = 1
                    elif k == 6:
                        U[idx_b][col] = 1
                        U[idx_d][col] = -1
                        V[idx_g][col] = 1
                        V[idx_h][col] = 1
                        W[idx_w00][col] = 1
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
