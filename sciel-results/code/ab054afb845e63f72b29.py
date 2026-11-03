import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def verify(U, V, W):
        """Verify exact decomposition"""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                    if total != 1:
                        return False
        return True
    
    def strassen_2x2():
        """Strassen decomposition for 2x2 * 2x2 -> 2x2. Returns (U, V, W) for the 2x2 block."""
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        # Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        U = []
        V = []
        W = []
        
        # For each Strassen term, we need to map to the tensor decomposition
        # The 2x2 block has indices: A is 2x2 (rows 0-1, cols 0-1), B is 2x2 (rows 0-1, cols 0-1), C is 2x2 (rows 0-1, cols 0-1)
        # U has shape (2*2, rank) = (4, rank) for A entries
        # V has shape (2*2, rank) = (4, rank) for B entries  
        # W has shape (2*2, rank) = (4, rank) for C entries
        
        # M1 = (a+d)(e+h): terms (a*e, a*h, d*e, d*h)
        # In the full 5x5, a=A[0,0], d=A[1,1], e=B[0,0], h=B[1,1]
        # U indices: A[0,0]=0, A[1,1]=3 (row-major: 0,1,2,3 for 2x2)
        # V indices: B[0,0]=0, B[1,1]=3
        # W indices: C[0,0]=0, C[1,1]=3
        
        # Let me use a helper to add terms
        def add_term(U_idx, V_idx, W_idx, sign=1):
            U.append([0]*1)
            V.append([0]*1)
            W.append([0]*1)
            U[-1][0] = sign
            V[-1][0] = 1
            W[-1][0] = 1
            # Store indices
            U_idx_list.append(U_idx)
            V_idx_list.append(V_idx)
            W_idx_list.append(W_idx)
        
        # This approach is getting complicated. Let me use a different strategy.
        # I'll build the full 5x5 decomposition directly using Strassen on sub-blocks.
        return None
    
    # Let me take a different approach: directly construct the tensor decomposition
    # using Strassen splits for the 5x5 case.
    
    # Split 5 into 2+3.
    # A (5x5) blocks: A11(2x2), A12(2x3), A21(3x2), A22(3x3)
    # B (5x5) blocks: B11(2x2), B12(2x3), B21(3x2), B22(3x3)
    # C (5x5) blocks: C11(2x2), C12(2x3), C21(3x2), C22(3x3)
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For each block product, I'll use the best known decomposition.
    
    # Let me just use schoolbook for all blocks but optimize where possible.
    # Actually, let me implement Strassen properly for the square blocks.
    
    def make_strassen_2x2():
        """
        Returns list of (U_entries, V_entries, W_entries) for Strassen 2x2.
        Each entry is a dict mapping (row,col) -> coefficient.
        """
        terms = []
        
        # M1 = (a+d)(e+h)
        # a=A(0,0), d=A(1,1), e=B(0,0), h=B(1,1)
        # Contributes to C(0,0), C(1,1)
        # C11 += M1, C22 += M1
        # So: a*e -> C11, a*h -> C11, d*e -> C11, d*h -> C11
        #      a*e -> C22, a*h -> C22, d*e -> C22, d*h -> C22
        # But that's not right. M1 is a single product term in Strassen.
        # In tensor decomposition, each rank-1 term is U[i,j]*V[k,l]*W[m,n]
        # For M1 = (a+d)(e+h), we need to expand this into rank-1 terms.
        # (a+d)(e+h) = a*e + a*h + d*e + d*h
        # Each of these is a rank-1 term in the 2x2*2x2->2x2 tensor.
        
        # Actually, for the full 5x5 tensor, I need to be more careful.
        # Let me just build the full decomposition directly.
        return terms
    
    # Given the complexity, let me use a simpler approach:
    # Use Strassen for 3x3 blocks (which saves 2 terms per 3x3*3x3 product)
    # and schoolbook for the rest.
    
    # 3x3 Strassen:
    # A(3x3) * B(3x3) -> C(3x3)
    # Split each into 2x2, 2x1, 1x2, 1x1 blocks? No, standard Strassen splits into 2x2 sub-blocks for 2x2 matrices.
    # For 3x3, we can split into 2+1.
    # A11(2x2), A12(2x1), A21(1x2), A22(1x1)
    # B11(2x2), B12(2x1), B21(1x2), B22(1x1)
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # A11*B11: 2x2*2x2 -> use Strassen (rank 7)
    # A12*B21: 2x1 * 1x2 -> rank = 2*1*2 = 4 (schoolbook)
    # A11*B12: 2x2 * 2x1 -> rank = 2*2*1 = 4
    # A12*B22: 2x1 * 1x1 -> rank = 2*1*1 = 2
    # A21*B11: 1x2 * 2x2 -> rank = 1*2*2 = 4
    # A22*B21: 1x1 * 1x2 -> rank = 1*1*2 = 2
    # A21*B12: 1x2 * 2x1 -> rank = 1*2*1 = 2
    # A22*B22: 1x1 * 1x1 -> rank = 1
    
    # Total for 3x3: 7 + 4 + 4 + 2 + 4 + 2 + 2 + 1 = 26
    # Schoolbook 3x3: 27
    # So we save 1 term!
    
    # For the full 5x5 with 2+3 split:
    # C11 = A11(2x2)*B11(2x2) + A12(2x3)*B21(3x2)
    #   A11*B11: 2x2*2x2 -> Strassen rank 7
    #   A12*B21: 2x3*3x2 -> schoolbook rank 2*3*2 = 12
    #   Subtotal: 19
    # C12 = A11(2x2)*B12(2x3) + A12(2x3)*B22(3x3)
    #   A11*B12: 2x2*2x3 -> schoolbook rank 2*2*3 = 12
    #   A12*B22: 2x3*3x3 -> schoolbook rank 2*3*3 = 18
    #   Subtotal: 30
    # C21 = A21(3x2)*B11(2x2) + A22(3x3)*B21(3x2)
    #   A21*B11: 3x2*2x2 -> schoolbook rank 3*2*2 = 12
    #   A22*B21: 3x3*3x2 -> schoolbook rank 3*3*2 = 18
    #   Subtotal: 30
    # C22 = A21(3x2)*B12(2x3) + A22(3x3)*B22(3x3)
    #   A21*B12: 3x2*2x3 -> schoolbook rank 3*2*3 = 18
    #   A22*B22: 3x3*3x3 -> use Strassen-like (rank 26)
    #   Subtotal: 44
    
    # Total: 19 + 30 + 30 + 44 = 123
    # That's the same as schoolbook! The Strassen savings are offset by the rectangular blocks.
    
    # Let me try a different split: 1+4
    # A11(1x1), A12(1x4), A21(4x1), A22(4x4)
    # B11(1x1), B12(1x4), B21(4x1), B22(4x4)
    # C11 = A11*B11: 1x1*1x1 -> rank 1
    # C12 = A11*B12 + A12*B22: 1x1*1x4 (rank 4) + 1x4*4x4 (rank 16) = 20
    # C21 = A21*B11 + A22*B21: 4x1*1x1 (rank 4) + 4x4*4x1 (rank 16) = 20
    # C22 = A21*B12 + A22*B22: 4x1*1x4 (rank 4) + 4x4*4x4 (rank 64) = 68
    # Total: 1 + 20 + 20 + 68 = 109
    # That's worse than 123.
    
    # Let me try 2+3 but use Strassen on the 3x3 block:
    # Actually, I realize my calculation above for 3x3 Strassen gave 26 vs 27 schoolbook.
    # Let me re-examine the 5x5 decomposition more carefully.
    
    # The issue is that the cross terms (like A12*B21) are rectangular and don't benefit from Strassen.
    
    # Let me just implement the schoolbook decomposition with the 2+3 split and see if I can find
    # any optimizations. For now, let me return a clean schoolbook decomposition.
    
    def schoolbook():
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
    
    return schoolbook()
