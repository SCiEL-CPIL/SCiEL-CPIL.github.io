"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a low-rank exact tensor decomposition for multiplying a 2x4 matrix by a 4x7 matrix.
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS
    N = n * m * p  # 56
    dim = m  # 4
    
    # The tensor T[i,j,k] = delta[i*m+j, k] for the multiplication (i in 0..1, j in 0..3, k in 0..6)
    # Actually, the matrix multiplication tensor is M[i,j,k] = delta[i,j] * delta[j,k] in standard notation?
    # No, the tensor for matrix multiplication C = AB is T[i,j,k] = delta[i,k] where j is the summation index.
    # The task says: "exactly decompose the matrix multiplication tensor".
    # The convention is W[k*n+i, r].
    # Let's stick to the schoolbook structure but optimize.
    
    # The current seed is rank 56.
    # Strassen on 2x2 blocks:
    # We can partition the 4x4 middle space into 2x2 blocks.
    # Let's try to construct a rank 48 or lower solution using known lower bounds or specific constructions.
    
    # Known lower bound for 2x4x7?
    # General lower bound for n x p is roughly 2np - something?
    # For 2x2, rank 7.
    # For 2x3, rank 11 (best known).
    # For 2x4, rank ?
    # For 4x4, rank 49 (Strassen 2x2 blocks: 4 blocks * 7 = 28? No, 2x2 Strassen is rank 7. 4x4 using 2x2 Strassen is 4*7=28? No, 4x4 is 2x2 blocks of 2x2. So 4 * 7 = 28? Wait.
    # 2x2 multiplication rank is 7.
    # 4x4 multiplication can be done with 49 rank (Strassen 2x2 blocks: 2x2 blocks, each 2x2 multiplication is rank 7, so 4*7=28? No.
    # Actually, 4x4 multiplication rank is known to be at most 49.
    # Wait, the middle dimension is m=4.
    # The problem is multiplying 2x4 by 4x7.
    # This is equivalent to computing 7 outputs, each is a linear combination of 4 inputs, multiplied by 2x4 matrix.
    # Actually, it's just a bilinear map R^4 x R^7 -> R^{14}.
    
    # Let's try a direct optimization using linear programming to find a sparse solution or a solution with small integer entries.
    # Since we need integer/half-integer entries, we can search over small integers.
    
    # However, for a quick improvement, let's try to use the 2x2 Strassen algorithm on the 2x2 blocks of the 2x4 matrix.
    # A is 2x4. Let A = [A1 A2] where A1, A2 are 2x2.
    # B is 4x7. Let B = [B1; B2] where B1, B2 are 2x7.
    # C = A1 B1 + A2 B2.
    # A1 B1 is 2x2 * 2x7. We can't directly apply 2x2 Strassen because B1 is not 2x2.
    # But we can partition B1 into columns. B1 = [b11 b12 b13 b14 b15 b16 b17] (2x7).
    # This doesn't help much.
    
    # Alternative: Partition the 4x7 matrix into 2x2 blocks where possible.
    # B is 4x7. We can partition B into [B11 B12 B13] where B11 is 4x2, B12 is 4x2, B13 is 4x1?
    # Or B11 (2x2), B12 (2x2), B13 (2x1) for the top half, and B21 (2x2), B22 (2x2), B23 (2x1) for the bottom half.
    # So B = [B11 B12 B13; B21 B22 B23].
    # A = [A1 A2] where A1 is 2x2, A2 is 2x2.
    # C = A1 [B11 B12 B13] + A2 [B21 B22 B23]
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22, A1 B13 + A2 B23]
    # Each term is 2x2 * 2x2 or 2x2 * 2x1.
    # For 2x2 * 2x2, we can use Strassen (rank 7).
    # For 2x2 * 2x1, we can use schoolbook (rank 4) or something better?
    # 2x2 * 2x1: C = A B where A is 2x2, B is 2x1.
    # C1 = a11 b1 + a12 b2
    # C2 = a21 b1 + a22 b2
    # This is 4 terms. Can we do it in 3?
    # Probably not, it's a matrix-vector product. Rank of 2x2x1 tensor is 4?
    # Actually, the rank of the 2x2x1 tensor is 4.
    
    # So, for the first two blocks (2x2 * 2x2), we use Strassen (rank 7 each).
    # For the third block (2x2 * 2x1), we use schoolbook (rank 4).
    # Total rank = 7 + 7 + 4 = 18?
    # Wait, this is for one output block of size 2x2?
    # No, C is 2x7.
    # C = [C11 C12 C13] where C11 is 2x2, C12 is 2x2, C13 is 2x1.
    # C11 = A1 B11 + A2 B21.
    # C12 = A1 B12 + A2 B22.
    # C13 = A1 B13 + A2 B23.
    
    # To compute C11, we need to compute A1 B11 and A2 B21 and add them.
    # A1 B11: 2x2 * 2x2. Strassen rank 7.
    # A2 B21: 2x2 * 2x2. Strassen rank 7.
    # So C11 requires 14 terms?
    # But we can share some terms?
    # No, the inputs A1, B11, A2, B21 are all different.
    # So it seems we need 7+7=14 terms for C11.
    # Similarly for C12: 14 terms.
    # For C13: A1 B13 (2x2 * 2x1, rank 4) + A2 B23 (2x2 * 2x1, rank 4) = 8 terms.
    # Total rank = 14 + 14 + 8 = 36.
    
    # This is better than 56.
    # Let's construct this.
    
    # Strassen algorithm for 2x2 matrix multiplication:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)(e)
    # m3 = (a)(f-h)
    # m4 = (d)(g-e)
    # m5 = (a+b)(h)
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # We need to map this to the global U, V, W matrices.
    # The inputs are A (2x4) and B (4x7).
    # Let's denote A entries as a_ij where i in {0,1}, j in {0,1,2,3}.
    # B entries as b_jk where j in {0,1,2,3}, k in {0,..,6}.
    
    # A1 = [[a00, a01], [a10, a11]]
    # A2 = [[a02, a03], [a12, a13]]
    
    # B11 = [[b00, b01], [b10, b11]]
    # B12 = [[b02, b03], [b12, b13]]
    # B13 = [[b04], [b14]]  (2x1)
    # B21 = [[b20, b21], [b30, b31]]
    # B22 = [[b22, b23], [b32, b33]]
    # B23 = [[b24], [b34]]  (2x1)
    
    # Wait, B is 4x7.
    # Row 0: b00, b01, b02, b03, b04, b05, b06
    # Row 1: b10, b11, b12, b13, b14, b15, b16
    # Row 2: b20, b21, b22, b23, b24, b25, b26
    # Row 3: b30, b31, b32, b33, b34, b35, b36
    
    # So B11 = [[b00, b01], [b10, b11]]
    # B12 = [[b02, b03], [b12, b13]]
    # B13 = [[b04, b05, b06], [b14, b15, b16]]  (2x3)
    # B21 = [[b20, b21], [b30, b31]]
    # B22 = [[b22, b23], [b32, b33]]
    # B23 = [[b24, b25, b26], [b34, b35, b36]]  (2x3)
    
    # So C13 is 2x3, not 2x1.
    # C13 = A1 B13 + A2 B23.
    # A1 B13: 2x2 * 2x3.
    # A2 B23: 2x2 * 2x3.
    # We can partition B13 into two 2x1 blocks and one 2x1 block?
    # B13 = [B13a, B13b, B13c] where each is 2x1.
    # Then A1 B13a, A1 B13b, A1 B13c each require rank 4.
    # Total for A1 B13 is 12.
    # Total for A2 B23 is 12.
    # Total for C13 is 24.
    
    # So total rank = 14 (for C11) + 14 (for C12) + 24 (for C13) = 52.
    # Still better than 56.
    
    # Let's try to improve C11 and C12.
    # C11 = A1 B11 + A2 B21.
    # Can we compute this sum with fewer than 14 terms?
    # This is equivalent to multiplying [A1 A2] (2x4) by [B11; B21] (4x2).
    # So it's a 2x4 * 4x2 multiplication.
    # The rank of 2x4x2 tensor?
    # We know 2x2x2 is 7.
    # 2x4x2 is like two 2x2x2 multiplications?
    # Actually, 2x4x2 can be seen as 2x2x2 (for A1 B11) + 2x2x2 (for A2 B21).
    # Is there a shared structure?
    # Probably not, so 14 seems reasonable.
    
    # What if we use a different partition?
    # Maybe 2x4x7 can be done better by partitioning into 2x2x7 blocks?
    # A1 (2x2) * B1 (2x7) + A2 (2x2) * B2 (2x7).
    # B1 is 2x7. We can partition B1 into 2x2, 2x2, 2x2, 2x1.
    # Then A1 B1 is sum of 3 (2x2*2x2) and 1 (2x2*2x1).
    # Rank for A1 B1 = 3*7 + 4 = 25.
    # Rank for A2 B2 = 25.
    # Total rank = 50.
    
    # This is better than 52.
    # Let's construct the rank 50 solution.
    
    # Partition:
    # A = [A1, A2] where A1, A2 are 2x2.
    # B = [B1; B2] where B1, B2 are 2x7.
    # B1 = [B11, B12, B13, B14] where B11, B12, B13 are 2x2, B14 is 2x1.
    # B2 = [B21, B22, B23, B24] where B21, B22, B23 are 2x2, B24 is 2x1.
    
    # C = A1 B1 + A2 B2
    # C = A1 [B11 B12 B13 B14] + A2 [B21 B22 B23 B24]
    # C = [A1 B11 + A2 B21, A1 B12 + A2 B22, A1 B13 + A2 B23, A1 B14 + A2 B24]
    
    # Each term A1 Bij + A2 Bij is a 2x2 * 2x2 multiplication (for j=1,2,3) or 2x2 * 2x1 (for j=4).
    # Wait, A1 Bij is 2x2 * 2x2 = 2x2.
    # A2 Bij is 2x2 * 2x2 = 2x2.
    # So the sum is 2x2.
    # To compute A1 Bij, we use Strassen (rank 7).
    # To compute A2 Bij, we use Strassen (rank 7).
    # Total for each j=1,2,3 is 14.
    # For j=4, A1 B14 is 2x2 * 2x1 (rank 4), A2 B24 is 2x2 * 2x1 (rank 4).
    # Total for j=4 is 8.
    # Total rank = 3 * 14 + 8 = 50.
    
    # Let's implement this.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8x50
    V = [[0] * rank for _ in range(m * p)]  # 28x50
    W = [[0] * rank for _ in range(p * n)]  # 14x50
    
    col = 0
    # Strassen for 2x2 * 2x2
    def add_strassen(U, V, W, col, A_start, B_start, C_start):
        # A is 2x2 at A_start in U (row index)
        # B is 2x2 at B_start in V (row index)
        # C is 2x2 at C_start in W (row index)
        # A entries: U[A_start], U[A_start+1], U[A_start+2], U[A_start+3]
        # B entries: V[B_start], V[B_start+1], V[B_start+2], V[B_start+3]
        # C entries: W[C_start], W[C_start+1], W[C_start+2], W[C_start+3]
        # Note: U index is i*m+j, so for A1 (i=0,1; j=0,1), indices are 0,1,2,3.
        # For A2 (i=0,1; j=2,3), indices are 2,3,4,5.
        
        # m1 = (a+d)(e+h)
        U[A_start][col] = 1
        U[A_start+3][col] = 1
        V[B_start][col] = 1
        V[B_start+3][col] = 1
        W[C_start][col] = 1
        W[C_start+3][col] = 1
        col += 1
        
        # m2 = (c+d)(e)
        U[A_start+2][col] = 1
        U[A_start+3][col] = 1
        V[B_start][col] = 1
        W[C_start+2][col] = 1
        W[C_start+3][col] = 1
        col += 1
        
        # m3 = (a)(f-h)
        U[A_start][col] = 1
        V[B_start+1][col] = 1
        V[B_start+3][col] = -1
        W[C_start+1][col] = 1
        W[C_start+3][col] = 1
        col += 1
        
        # m4 = (d)(g-e)
        U[A_start+3][col] = 1
        V[B_start+2][col] = 1
        V[B_start][col] = -1
        W[C_start][col] = 1
        W[C_start+3][col] = 1
        col += 1
        
        # m5 = (a+b)(h)
        U[A_start][col] = 1
        U[A_start+1][col] = 1
        V[B_start+3][col] = 1
        W[C_start][col] = -1
        W[C_start+3][col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[A_start+2][col] = 1
        U[A_start][col] = -1
        V[B_start][col] = 1
        V[B_start+1][col] = 1
        W[C_start+1][col] = 1
        W[C_start+2][col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[A_start+1][col] = 1
        U[A_start+3][col] = -1
        V[B_start+2][col] = 1
        V[B_start+3][col] = 1
        W[C_start][col] = 1
        W[C_start+3][col] = 1
        col += 1
        
        return col
    
    # Schoolbook for 2x2 * 2x1
    def add_schoolbook_2x2x1(U, V, W, col, A_start, B_start, C_start):
        # A is 2x2 at A_start
        # B is 2x1 at B_start (B_start, B_start+1)
        # C is 2x1 at C_start (C_start, C_start+1)
        
        # C1 = a*e + b*f
        U[A_start][col] = 1
        V[B_start][col] = 1
        W[C_start][col] = 1
        col += 1
        
        U[A_start+1][col] = 1
        V[B_start+1][col] = 1
        W[C_start][col] = 1
        col += 1
        
        # C2 = c*e + d*f
        U[A_start+2][col] = 1
        V[B_start][col] = 1
        W[C_start+1][col] = 1
        col += 1
        
        U[A_start+3][col] = 1
        V[B_start+1][col] = 1
        W[C_start+1][col] = 1
        col += 1
        
        return col
    
    # A1 indices: 0,1,2,3
    # A2 indices: 2,3,4,5
    # Wait, A is 2x4.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=0, j=2 -> 2
    # i=0, j=3 -> 3
    # i=1, j=0 -> 4
    # i=1, j=1 -> 5
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    
    # So A1 = [[A[0], A[1]], [A[4], A[5]]]
    # A2 = [[A[2], A[3]], [A[6], A[7]]]
    
    # B is 4x7.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # ...
    # j=0, k=6 -> 6
    # j=1, k=0 -> 7
    # ...
    # j=1, k=6 -> 13
    # j=2, k=0 -> 14
    # ...
    # j=2, k=6 -> 20
    # j=3, k=0 -> 21
    # ...
    # j=3, k=6 -> 27
    
    # B1 = rows 0,1 of B.
    # B11 = [[B[0], B[1]], [B[7], B[8]]]
    # B12 = [[B[2], B[3]], [B[9], B[10]]]
    # B13 = [[B[4], B[5]], [B[11], B[12]]]
    # B14 = [[B[6]], [B[13]]]
    
    # B2 = rows 2,3 of B.
    # B21 = [[B[14], B[15]], [B[21], B[22]]]
    # B22 = [[B[16], B[17]], [B[23], B[24]]]
    # B23 = [[B[18], B[19]], [B[25], B[26]]]
    # B24 = [[B[20]], [B[27]]]
    
    # C is 2x7.
    # i=0, k=0 -> 0
    # ...
    # i=0, k=6 -> 6
    # i=1, k=0 -> 7
    # ...
    # i=1, k=6 -> 13
    
    # C11 = [[C[0], C[1]], [C[7], C[8]]]
    # C12 = [[C[2], C[3]], [C[9], C[10]]]
    # C13 = [[C[4], C[5]], [C[11], C[12]]]
    # C14 = [[C[6]], [C[13]]]
    
    # Term 1: A1 B11 + A2 B21 -> C11
    col = add_strassen(U, V, W, col, 0, 0, 0)  # A1 B11
    col = add_strassen(U, V, W, col, 2, 14, 0) # A2 B21
    
    # Term 2: A1 B12 + A2 B22 -> C12
    col = add_strassen(U, V, W, col, 0, 2, 2)  # A1 B12
    col = add_strassen(U, V, W, col, 2, 16, 2) # A2 B22
    
    # Term 3: A1 B13 + A2 B23 -> C13
    col = add_strassen(U, V, W, col, 0, 4, 4)  # A1 B13
    col = add_strassen(U, V, W, col, 2, 18, 4) # A2 B23
    
    # Term 4: A1 B14 + A2 B24 -> C14
    col = add_schoolbook_2x2x1(U, V, W, col, 0, 6, 6)   # A1 B14
    col = add_schoolbook_2x2x1(U, V, W, col, 2, 20, 6)  # A2 B24
    
    return {"U": U, "V": V, "W": W}
