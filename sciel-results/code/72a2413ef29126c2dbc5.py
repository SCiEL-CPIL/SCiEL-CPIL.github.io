"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 5 matrix.
    Output factor follows the official trace convention W[k*n+i,r].
    Returns: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    n, m, p = 2, 3, 5
    # Schoolbook rank is n*m*p = 2*3*5 = 30.
    
    # We can potentially reduce the rank by exploiting structure.
    # For 2x3 * 3x5, we can split the 3x5 into two parts? No, that's not a square block.
    # However, we can split the middle dimension m=3 into 1+2.
    # Let A be 2x3, B be 3x5.
    # A = [A1 | A2] where A1 is 2x1 and A2 is 2x2.
    # B = [B1; B2] where B1 is 1x5 and B2 is 2x5.
    # C = A1*B1 + A2*B2.
    # A2 is 2x2, B2 is 2x5.
    # We can use Strassen for the 2x2 * 2x2 part of B2?
    # B2 is 2x5, not square.
    # Let's split B2 into B2a (2x2) and B2b (2x3).
    # C = A1*B1 + A2*B2a + A2*B2b.
    # A2 (2x2) * B2a (2x2) can be done in rank 7 (Strassen).
    # A2 (2x2) * B2b (2x3) can be done in rank 2*2*3 = 12? Or less?
    # A1 (2x1) * B1 (1x5) is rank 2*1*5 = 10.
    # Total rank = 10 (A1*B1) + 7 (A2*B2a) + 12 (A2*B2b) = 29.
    # This is a reduction from 30 to 29.
    
    # Can we do better?
    # What if we split m=3 into 2+1?
    # A = [A1 | A2] where A1 is 2x2, A2 is 2x1.
    # B = [B1; B2] where B1 is 2x5, B2 is 1x5.
    # C = A1*B1 + A2*B2.
    # A1 (2x2) * B1 (2x5). Split B1 into B1a (2x2) and B1b (2x3).
    # A1*B1a is 2x2*2x2 -> rank 7.
    # A1*B1b is 2x2*2x3 -> rank 12.
    # A2 (2x1) * B2 (1x5) -> rank 10.
    # Total rank = 7 + 12 + 10 = 29.
    
    # What if we split p=5?
    # B is 3x5. Split B into B1 (3x2) and B2 (3x3).
    # C = A*B1 + A*B2.
    # A (2x3) * B1 (3x2) -> rank 12.
    # A (2x3) * B2 (3x3) -> rank 18.
    # Total = 30. No gain.
    
    # What if we split n=2? No, n=2 is prime.
    
    # Let's check if 29 is the minimum.
    # The tensor is M_{2,3,5}.
    # Known lower bounds:
    # Rank >= n+m-1 = 2+3-1 = 4? No, that's for matrix multiplication of (n-1)x(n-1) and (m-1)x(m-1) or similar.
    # Actually, for M_{n,m,p}, rank >= max(n,m,p) is not true.
    # Rank >= n+m+p-2? No.
    # Bini's lower bound: rank >= n+m+p-2?
    # For 2x3x5, n+m+p-2 = 2+3+5-2 = 8.
    # But we are at 29.
    
    # Is there a rank 28 decomposition?
    # If we use Strassen for 2x2x2, we save 1 rank.
    # If we can find two disjoint 2x2x2 blocks, we save 2 ranks -> 28.
    # Can we partition 2x3x5 into two 2x2x2 blocks and the rest?
    # The dimensions are 2, 3, 5.
    # A 2x2x2 block requires 2 rows of A, 2 cols of A, 2 rows of B, 2 cols of B.
    # We have 2 rows of A. So we must use all rows of A.
    # We have 3 cols of A. We can pick 2.
    # We have 3 rows of B. We must pick the same 2 cols of A as rows of B.
    # We have 5 cols of B. We can pick 2.
    # So we can identify a 2x2x2 sub-tensor.
    # Let's pick cols 0,1 of A and rows 0,1 of B.
    # Let's pick cols 0,1 of B.
    # This corresponds to C[0:2, 0:2] += A[0:2, 0:2] * B[0:2, 0:2].
    # This is a 2x2x2 block. Rank 7.
    # The remaining parts:
    # A[0:2, 2] * B[2, 0:5] -> 2x1 * 1x5 -> rank 10.
    # A[0:2, 0:2] * B[0:2, 2:5] -> 2x2 * 2x3 -> rank 12.
    # A[0:2, 2] * B[0:2, 2:5] -> 2x1 * 2x3 -> rank 6.
    # Total rank = 7 + 10 + 12 + 6 = 35. This is worse.
    # Why? Because Strassen only works if the multiplication is exactly 2x2*2x2.
    # If we mix it with other terms, we can't just add the ranks.
    # We need to construct the full tensor decomposition.
    
    # Let's stick to the 29 rank construction first, as it is a clear improvement over 30.
    # Construction for rank 29:
    # Split A = [A1 (2x2) | A2 (2x1)]
    # Split B = [B1 (2x5); B2 (1x5)]
    # C = A1*B1 + A2*B2
    # Split B1 = [B1a (2x2) | B1b (2x3)]
    # C = A1*B1a + A1*B1b + A2*B2
    # Term 1: A1 (2x2) * B1a (2x2) -> Strassen (rank 7)
    # Term 2: A1 (2x2) * B1b (2x3) -> Schoolbook (rank 12)
    # Term 3: A2 (2x1) * B2 (1x5) -> Schoolbook (rank 10)
    
    # Let's build this.
    # Indices:
    # A: i (0..1), j (0..2)
    # B: j (0..2), k (0..4)
    # C: i (0..1), k (0..4)
    
    # Split j=0,1 into A1, j=2 into A2.
    # Split k=0,1 into B1a, k=2,3,4 into B1b for B1.
    # B1 is rows 0,1 of B. B2 is row 2 of B.
    
    # Term 1: A[0:2, 0:2] * B[0:2, 0:2]
    # This contributes to C[0:2, 0:2].
    # Strassen for 2x2*2x2.
    # P1 = (A00+A11)(B00+B11) -> C00, C11
    # P2 = (A10+A11)B00 -> C10
    # P3 = (A10+A11)B11 -> C01, C11
    # P4 = (A00-A10)(B00+B10) -> C00, C10
    # P5 = (A01-A11)(B10+B11) -> C00, C01
    # P6 = A00(B01-B11) -> C11
    # P7 = A11(B10-B00) -> C00, C11
    
    # Let's map these to global indices.
    # A indices: i, j. B indices: j, k.
    # A[0:2, 0:2] means i in {0,1}, j in {0,1}.
    # B[0:2, 0:2] means j in {0,1}, k in {0,1}.
    
    # P1: U: A00, A11. V: B00, B11. W: C00, C11.
    # P2: U: A10, A11. V: B00. W: C10.
    # P3: U: A10, A11. V: B11. W: C01, C11.
    # P4: U: A00, A10. V: B00, B10. W: C00, C10.
    # P5: U: A01, A11. V: B10, B11. W: C00, C01.
    # P6: U: A00. V: B01, B11. W: C11.
    # P7: U: A11. V: B10, B00. W: C00, C11.
    
    # Term 2: A[0:2, 0:2] * B[0:2, 2:5]
    # A indices: i in {0,1}, j in {0,1}.
    # B indices: j in {0,1}, k in {2,3,4}.
    # Schoolbook: 2*2*3 = 12 terms.
    # For each i, j, k in {0,1}x{0,1}x{2,3,4}:
    # U: A[i,j]. V: B[j,k]. W: C[i,k].
    
    # Term 3: A[0:2, 2] * B[2, 0:5]
    # A indices: i in {0,1}, j=2.
    # B indices: j=2, k in {0,1,2,3,4}.
    # Schoolbook: 2*1*5 = 10 terms.
    # For each i, k in {0,1}x{0,1,2,3,4}:
    # U: A[i,2]. V: B[2,k]. W: C[i,k].
    
    # Total rank = 7 + 12 + 10 = 29.
    
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        # u, v, w are dicts: {index: coeff}
        # We need to align them into columns.
        pass
        
    # Let's use a list of (u_dict, v_dict, w_dict) and then convert to matrices.
    terms = []
    
    # Helper to add a term
    def add_strassen_term(u_indices, v_indices, w_indices, coeffs):
        # This is getting complex. Let's just build the matrices directly.
        pass
        
    # Let's build U, V, W as lists of lists.
    # Rank R = 29.
    R = 29
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    
    col = 0
    
    # Term 1: Strassen
    # P1: A00+A11, B00+B11, C00+C11
    U[0*m+0][col] += 1 # A00
    U[1*m+1][col] += 1 # A11
    V[0*p+0][col] += 1 # B00
    V[1*p+1][col] += 1 # B11
    W[0*n+0][col] += 1 # C00
    W[1*n+1][col] += 1 # C11
    col += 1
    
    # P2: A10+A11, B00, C10
    U[1*m+0][col] += 1 # A10
    U[1*m+1][col] += 1 # A11
    V[0*p+0][col] += 1 # B00
    W[0*n+1][col] += 1 # C10 (k=0, i=1 -> W[0*2+1])
    col += 1
    
    # P3: A10+A11, B11, C01+C11
    U[1*m+0][col] += 1 # A10
    U[1*m+1][col] += 1 # A11
    V[1*p+1][col] += 1 # B11
    W[1*n+0][col] += 1 # C01 (k=1, i=0 -> W[1*2+0])
    W[1*n+1][col] += 1 # C11 (k=1, i=1 -> W[1*2+1])
    col += 1
    
    # P4: A00-A10, B00+B10, C00+C10
    U[0*m+0][col] += 1 # A00
    U[1*m+0][col] += -1 # A10
    V[0*p+0][col] += 1 # B00
    V[0*p+1][col] += 1 # B10 (j=0, k=1)
    W[0*n+0][col] += 1 # C00
    W[0*n+1][col] += 1 # C10
    col += 1
    
    # P5: A01-A11, B10+B11, C00+C01
    U[0*m+1][col] += 1 # A01
    U[1*m+1][col] += -1 # A11
    V[1*p+0][col] += 1 # B10 (j=1, k=0)
    V[1*p+1][col] += 1 # B11 (j=1, k=1)
    W[0*n+0][col] += 1 # C00
    W[1*n+0][col] += 1 # C01
    col += 1
    
    # P6: A00, B01-B11, C11
    U[0*m+0][col] += 1 # A00
    V[0*p+1][col] += 1 # B01
    V[1*p+1][col] += -1 # B11
    W[1*n+1][col] += 1 # C11
    col += 1
    
    # P7: A11, B10-B00, C00+C11
    U[1*m+1][col] += 1 # A11
    V[1*p+0][col] += 1 # B10
    V[0*p+0][col] += -1 # B00
    W[0*n+0][col] += 1 # C00
    W[1*n+1][col] += 1 # C11
    col += 1
    
    # Term 2: Schoolbook for A[0:2, 0:2] * B[0:2, 2:5]
    # i in 0,1; j in 0,1; k in 2,3,4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    # Term 3: Schoolbook for A[0:2, 2] * B[2, 0:5]
    # i in 0,1; j=2; k in 0,1,2,3,4
    for i in range(2):
        for k in range(5):
            U[i*m+2][col] = 1
            V[2*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
