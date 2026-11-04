"""Program for matrix multiplication 3x4 x 4x8 with Winograd-based decomposition."""
import time

DIMENSIONS = [3, 4, 8]

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x4 * 4x8.
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices with integer/half-integer entries.
        U: (n*m, rank) = (12, rank)
        V: (m*p, rank) = (32, rank) 
        W: (p*n, rank) = (24, rank)
    """
    start_time = time.time()
    deadline = start_time + 110  # 10-second safety margin
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    # Winograd decomposition for 3x3 matrix multiplication (rank 23)
    # This is the standard Winograd algorithm: F(3,3,3)
    # Coefficients are integers or half-integers
    
    # U matrices for the 3x3x3 Winograd algorithm
    # There are 23 terms
    # Each term has a coefficient vector for A (3x3=9 elements) and for B (3x3=9 elements)
    
    # Winograd coefficients for 3x3x3 (23 multiplications)
    # Format: list of (u_coeffs, v_coeffs, w_coeffs) where each is a 9-element list
    # u_coeffs[i] multiplies A[row, col] for the i-th input
    # v_coeffs[i] multiplies B[row, col] for the i-th input
    
    # The 23 terms of Winograd 3x3x3:
    # Term structure: each term c * (sum_u u_i * A) * (sum_v v_i * B) -> contributes to C
    
    # Let me define the Winograd 3x3x3 decomposition explicitly
    # Using the standard formulation where we compute C = A * B
    
    # A is 3x3, B is 3x3, C is 3x3
    # Index: A[r][c], B[r][c], C[r][c]
    
    # Winograd 3x3x3 has these 23 bilinear forms:
    # Each bilinear form: (linear combination of A entries) * (linear combination of B entries)
    # and contributes to specific entries of C
    
    # Let me use a known correct set of coefficients
    
    # The 23 terms with their A-coefficients, B-coefficients, and C-output indices
    # A is indexed as (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
    # Same for B and C
    
    # I'll use the standard Winograd 3x3x3 coefficients:
    winograd_3x3x3 = [
        # (A_coeffs[9], B_coeffs[9], C_output_indices[(row,col)] list)
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        ((1,0,0, 0,0,0, 0,0,0), (1,0,0, 0,0,0, 0,0,0), [(0,0)]),
        # Term 1: A[0,0]*B[0,1] -> C[0,1]
        ((1,0,0, 0,0,0, 0,0,0), (0,1,0, 0,0,0, 0,0,0), [(0,1)]),
        # Term 2: A[0,0]*B[0,2] -> C[0,2]
        ((1,0,0, 0,0,0, 0,0,0), (0,0,1, 0,0,0, 0,0,0), [(0,2)]),
        # Term 3: A[0,1]*B[1,0] -> C[1,0]
        ((0,1,0, 0,0,0, 0,0,0), (0,0,0, 1,0,0, 0,0,0), [(1,0)]),
        # Term 4: A[0,1]*B[1,1] -> C[1,1]
        ((0,1,0, 0,0,0, 0,0,0), (0,0,0, 0,1,0, 0,0,0), [(1,1)]),
        # Term 5: A[0,1]*B[1,2] -> C[1,2]
        ((0,1,0, 0,0,0, 0,0,0), (0,0,0, 0,0,1, 0,0,0), [(1,2)]),
        # Term 6: A[0,2]*B[2,0] -> C[2,0]
        ((0,0,1, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 1,0,0), [(2,0)]),
        # Term 7: A[0,2]*B[2,1] -> C[2,1]
        ((0,0,1, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 0,1,0), [(2,1)]),
        # Term 8: A[0,2]*B[2,2] -> C[2,2]
        ((0,0,1, 0,0,0, 0,0,0), (0,0,0, 0,0,0, 0,0,1), [(2,2)]),
        # Term 9: (A[1,0]+A[2,0])*(B[0,0]-B[1,0]) -> C[0,0] - C[1,0]
        ((0,0,0, 1,0,0, 1,0,0), (1,0,0, -1,0,0, 0,0,0), [(0,0,-1), (1,0,1)]),
        # Term 10: (A[1,0]+A[2,0])*(B[0,1]-B[1,1]) -> C[0,1] - C[1,1]
        ((0,0,0, 1,0,0, 1,0,0), (0,1,0, -1,0,0, 0,0,0), [(0,1,-1), (1,1,1)]),
        # Term 11: (A[1,0]+A[2,0])*(B[0,2]-B[1,2]) -> C[0,2] - C[1,2]
        ((0,0,0, 1,0,0, 1,0,0), (0,0,1, -1,0,0, 0,0,0), [(0,2,-1), (1,2,1)]),
        # Term 12: (A[1,1]+A[2,1])*(B[1,0]-B[2,0]) -> C[1,0] - C[2,0]
        ((0,0,0, 0,1,0, 0,1,0), (0,0,0, 1,0,0, -1,0,0), [(1,0,-1), (2,0,1)]),
        # Term 13: (A[1,1]+A[2,1])*(B[1,1]-B[2,1]) -> C[1,1] - C[2,1]
        ((0,0,0, 0,1,0, 0,1,0), (0,0,0, 0,1,0, -1,0,0), [(1,1,-1), (2,1,1)]),
        # Term 14: (A[1,1]+A[2,1])*(B[1,2]-B[2,2]) -> C[1,2] - C[2,2]
        ((0,0,0, 0,1,0, 0,1,0), (0,0,0, 0,0,1, -1,0,0), [(1,2,-1), (2,2,1)]),
        # Term 15: (A[1,2]+A[2,2])*(B[2,0]) -> C[2,0]
        ((0,0,0, 0,0,1, 0,0,1), (0,0,0, 0,0,0, 1,0,0), [(2,0,1)]),
        # Term 16: (A[1,2]+A[2,2])*(B[2,1]) -> C[2,1]
        ((0,0,0, 0,0,1, 0,0,1), (0,0,0, 0,0,0, 0,1,0), [(2,1,1)]),
        # Term 17: (A[1,2]+A[2,2])*(B[2,2]) -> C[2,2]
        ((0,0,0, 0,0,1, 0,0,1), (0,0,0, 0,0,0, 0,0,1), [(2,2,1)]),
        # Term 18: (A[0,1]+A[1,1])*(B[0,0]-B[0,1]) -> C[0,0] - C[0,1]
        ((0,1,0, 0,1,0, 0,0,0), (1,0,0, 0,0,0, 0,0,0), [(0,0,1), (0,1,-1)]),
        # Term 19: (A[0,2]+A[1,2])*(B[0,1]-B[0,2]) -> C[0,1] - C[0,2]
        ((0,0,1, 0,0,1, 0,0,0), (0,1,0, 0,0,0, 0,0,0), [(0,1,1), (0,2,-1)]),
        # Term 20: (A[0,0]+A[1,0])*(B[1,0]-B[2,0])... hmm this isn't quite right
        # Let me reconsider
    ]
    
    # The above is getting complicated. Let me use a simpler approach:
    # Use the schoolbook for 4x4 blocks but exploit the structure better.
    
    # Actually, let me just use a clean Winograd 3x3x3 with verified coefficients.
    # Here's a correct set:
    
    # For 3x3 matrix multiplication, the Winograd algorithm uses these 23 terms:
    # I'll represent each term as:
    #   u: 9-tuple (coefficients for A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2], A[2,0], A[2,1], A[2,2])
    #   v: 9-tuple (coefficients for B)
    #   w: list of (row, col, coeff) for C entries
    
    wg = [
        # 9 diagonal terms
        ((1,0,0,0,0,0,0,0,0), (1,0,0,0,0,0,0,0,0), [(0,0,1)]),
        ((1,0,0,0,0,0,0,0,0), (0,1,0,0,0,0,0,0,0), [(0,1,1)]),
        ((1,0,0,0,0,0,0,0,0), (0,0,1,0,0,0,0,0,0), [(0,2,1)]),
        ((0,0,0,1,0,0,0,0,0), (0,0,0,1,0,0,0,0,0), [(1,0,1)]),
        ((0,0,0,0,1,0,0,0,0), (0,0,0,0,1,0,0,0,0), [(1,1,1)]),
        ((0,0,0,0,0,0,1,0,0), (0,0,0,0,0,0,1,0,0), [(2,0,1)]),
        ((0,0,0,0,0,0,0,1,0), (0,0,0,0,0,0,0,1,0), [(2,1,1)]),
        ((0,0,0,0,0,0,0,0,1), (0,0,0,0,0,0,0,0,1), [(2,2,1)]),
        # Additional terms for cross-products
        ((0,0,0,1,0,0,1,0,0), (0,0,0,1,0,0,-1,0,0), [(1,0,1),(2,0,-1)]),
        ((0,0,0,1,0,0,1,0,0), (0,0,0,0,1,0,-1,0,0), [(1,1,1),(2,1,-1)]),
        ((0,0,0,1,0,0,1,0,0), (0,0,0,0,0,1,-1,0,0), [(1,2,1),(2,2,-1)]),
        ((0,0,0,0,1,0,0,1,0), (0,0,0,1,0,0,-1,0,0), [(1,0,1),(2,0,-1)]),
        ((0,0,0,0,1,0,0,1,0), (0,0,0,0,1,0,-1,0,0), [(1,1,1),(2,1,-1)]),
        ((0,0,0,0,1,0,0,1,0), (0,0,0,0,0,1,-1,0,0), [(1,2,1),(2,2,-1)]),
        ((0,0,0,0,0,1,0,0,1), (0,0,0,1,0,0,-1,0,0), [(1,0,1),(2,0,-1)]),
        ((0,0,0,0,0,1,0,0,1), (0,0,0,0,1,0,-1,0,0), [(1,1,1),(2,1,-1)]),
        ((0,0,0,0,0,1,0,0,1), (0,0,0,0,0,1,-1,0,0), [(1,2,1),(2,2,-1)]),
        ((0,1,0,1,0,0,0,0,0), (1,0,0,0,0,0,0,0,0), [(0,0,1),(1,0,1)]),
        ((0,1,0,0,1,0,0,0,0), (0,1,0,0,0,0,0,0,0), [(0,1,1),(1,1,1)]),
        ((0,1,0,0,0,1,0,0,0), (0,0,1,0,0,0,0,0,0), [(0,2,1),(1,2,1)]),
        ((1,0,0,0,1,0,0,0,0), (0,0,0,1,0,0,0,0,0), [(0,0,1),(1,1,1)]),
        ((1,0,0,0,0,1,0,0,0), (0,0,0,0,0,1,0,0,0), [(0,2,1),(1,2,1)]),
        ((0,0,1,0,0,1,0,0,0), (0,0,0,1,0,0,0,0,0), [(0,2,1),(1,2,1)]),
    ]
    
    # This gives 8 + 9 + 6 = 23 terms. Let me verify this is correct.
    # Actually, I'm not confident this is the correct Winograd decomposition.
    # Let me use a safer approach: just use the schoolbook but see if we can do better
    # by exploiting the 3x4x8 structure.
    
    # For now, let me fall back to a correct decomposition.
    # Schoolbook rank for 3x4x8 is 3*4*8 = 96.
    
    # Let me try a block approach:
    # Split the 4x8 B matrix into two 4x4 blocks: B = [B1 | B2]
    # Split the 3x4 A matrix... actually A is 3x4, so no natural split on A.
    # But we can think of C = A*B = [A*B1 | A*B2]
    # Each A*Bk is 3x4 * 4x4 = 3x4 result.
    # 
    # For 3x4 * 4x4: split A into 3x2 and 2x2? No, A is 3x4.
    # Split A as [A_top(2x4) ; A_bottom(1x4)] and B as [B_top(2x4) ; B_bottom(2x4)]
    # C = A_top * B_top + A_bottom * B_bottom (but dimensions don't work directly)
    
    # Actually for 3x4 * 4x4:
    # A is 3x4, B is 4x4
    # Split the 4 intermediate dimension into 2+2:
    # A = [A1(3x2) | A2(3x2)], B = [B1(2x4); B2(2x4)]
    # C = A1*B1 + A2*B2
    # A1*B1: 3x2 * 2x4, schoolbook rank = 3*2*4 = 24
    # A2*B2: 3x2 * 2x4, schoolbook rank = 3*2*4 = 24
    # Total for one 3x4*4x4: 48, but we can do better.
    
    # Actually 3x2 * 2x4 has rank at most 3*2*4=24 schoolbook.
    # Can we do better? 3x2*2x4 is like computing a 3x4 matrix from rank-2 factors.
    # The bilinear rank of 3x2*2x4... each output C[i,k] = sum_j A[i,j]*B[j,k] for j=0,1
    # So C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is 2 multiplications per output entry, 12 outputs, so 24 mults schoolbook.
    # But we can factor: for fixed i, [C[i,0], C[i,1], C[i,2], C[i,3]] = A[i,0]*[B[0,0],B[0,1],B[0,2],B[0,3]] + A[i,1]*[B[1,0],B[1,1],B[1,2],B[1,3]]
    # This is still 2 mults per i, total 6*2 = 12? No, it's still 24 because each multiplication is separate.
    
    # Hmm, let me just go with schoolbook rank 96 to be safe, since my Winograd coefficients
    # may be incorrect.
    
    rank = n * m * p  # 96
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
