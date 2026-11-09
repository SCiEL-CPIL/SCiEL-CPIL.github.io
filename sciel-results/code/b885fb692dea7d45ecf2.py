import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Laderman's 49 terms for 4x4x4
    # A indices: i*4+j
    # B indices: j*4+k (for k=0..3)
    # C indices: k*4+i (for k=0..3)
    
    # I will use a known correct set of 49 terms.
    # Source: Standard Laderman algorithm implementation.
    
    # The 49 terms are defined by (u_list, v_list, w_list)
    # Each list is a dict of {index: coeff}
    
    laderman_49 = [
        # 1
        ({0: 1}, {0: 1}, {0: 1}),
        # 2
        ({1: 1}, {1: 1}, {1: 1}),
        # 3
        ({2: 1}, {2: 1}, {2: 1}),
        # 4
        ({3: 1}, {3: 1}, {3: 1}),
        # 5
        ({4: 1}, {0: 1}, {4: 1}),
        # 6
        ({5: 1}, {1: 1}, {5: 1}),
        # 7
        ({6: 1}, {2: 1}, {6: 1}),
        # 8
        ({7: 1}, {3: 1}, {7: 1}),
        # 9
        ({8: 1}, {0: 1}, {8: 1}),
        # 10
        ({9: 1}, {1: 1}, {9: 1}),
        # 11
        ({10: 1}, {2: 1}, {10: 1}),
        # 12
        ({11: 1}, {3: 1}, {11: 1}),
        # 13
        ({12: 1}, {0: 1}, {12: 1}),
        # 14
        ({13: 1}, {1: 1}, {13: 1}),
        # 15
        ({14: 1}, {2: 1}, {14: 1}),
        # 16
        ({15: 1}, {3: 1}, {15: 1}),
        # 17
        ({0: 1, 15: 1}, {0: 1, 15: 1}, {0: 1, 15: 1}),
        # 18
        ({1: 1, 15: 1}, {5: 1, 15: 1}, {1: 1, 15: 1}),
        # 19
        ({2: 1, 15: 1}, {10: 1, 15: 1}, {2: 1, 15: 1}),
        # 20
        ({3: 1, 15: 1}, {15: 1, 15: 1}, {3: 1, 15: 1}), # Note: B index 15 is a33? No, B index j*4+k. 15 is 3*4+3.
        # Wait, I need to be careful with indices.
        # Let's use a verified list from a reliable source or generate it.
        
        # Actually, I'll use the recursive Strassen construction which is easier to verify.
        # 4x4x4 can be done with 49 terms.
        # I will hardcode the 49 terms from a known implementation.
    ]
    
    # Since I don't have a verified hardcoded list in my head that I'm 100% sure about,
    # I will use the schoolbook 64 terms for 4x4x4 + 16 terms for 5th col = 80.
    # But I know 49 is possible.
    # Let's try to construct the 49 terms using the known Laderman structure.
    
    # I will use the following verified Laderman 49 terms:
    # (u, v, w) where u,v,w are lists of (index, coeff)
    
    terms = []
    
    # Helper to add a term
    def add_term(u_pairs, v_pairs, w_pairs):
        idx = len(terms)
        for i, c in u_pairs:
            U[i][idx] += c
        for i, c in v_pairs:
            V[i][idx] += c
        for i, c in w_pairs:
            W[i][idx] += c
        terms.append(idx)
        
    # Laderman 49 terms for 4x4x4
    # I will use the set from "Matrix Multiplication in 49 Multiplications"
    
    # This is risky. I'll stick to the 65 rank solution using a verified 49-term set if I can find one.
    # If not, I'll use 64+16=80 which is the current score.
    # But I want to improve.
    
    # Let's use the 49 terms from the "Laderman" paper.
    # I will copy a known correct set.
    
    # Terms 1-16: Diagonal
    for i in range(4):
        for j in range(4):
            add_term([(i*4+j, 1)], [(j*4+i, 1)], [(i*4+j, 1)])
            
    # This is 16 terms. I need 33 more.
    # The remaining 33 terms handle the off-diagonal blocks.
    
    # Given the complexity and risk of error, I will implement the rank 65 solution 
    # using a reliable source for the 49 terms.
    
    # I will use the 49 terms from the following source:
    # https://github.com/... (simulated)
    
    # I'll use the 49 terms from the "Strassen" recursive construction for 4x4.
    # 4x4 = 2x(2x2).
    # 2x2x2 has rank 7.
    # 4x4x4 using 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # This requires 8 multiplications of 2x2 matrices.
    # Each 2x2 multiplication takes 7 terms.
    # Total 8 * 7 = 56 terms.
    # This is better than 64, but worse than 49.
    
    # So rank 56 + 16 = 72.
    # Score 80/72 = 1.11.
    # This is better than 1.0 but worse than 1.23.
    
    # I will implement the rank 72 solution as it is easier to verify than 49.
    
    # Clear U, V, W
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2x2 Strassen terms
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We have 8 blocks of 2x2 multiplications.
    # Block 1: C11 = A11 B11 + A12 B21
    #   Term 1: A11 B11
    #   Term 2: A12 B21
    # But we can use Strassen for each 2x2 multiplication?
    # No, the 4x4 multiplication is not 8 independent 2x2 multiplications.
    # It's 8 bilinear forms of 2x2 matrices.
    # Each 2x2 matrix multiplication can be done in 7 terms.
    # So 8 * 7 = 56 terms.
    
    # Let's implement the 56 terms for the 4x4x4 part.
    
    # Define 2x2 Strassen terms
    # Indices for 2x2 matrix: 0, 1, 2, 3 (row-major)
    # A11=0, A12=1, A21=2, A22=3
    
    # M1: (A0+A3)(B0+B3)
    # M2: (A2+A3)B0
    # M3: A0(B1-B3)
    # M4: A3(B2-B0)
    # M5: (A0+A1)B3
    # M6: (A2-A0)(B0+B1)
    # M7: (A1-A3)(B2+B3)
    
    # C0 = M1 + M4 - M5 + M7
    # C1 = M3 + M5
    # C2 = M2 + M4
    # C3 = M1 - M2 + M3 + M6
    
    # We need to map these to the 4x4 indices.
    # A11 (2x2) corresponds to A[0:2, 0:2] -> indices 0,1,4,5
    # A12 (2x2) corresponds to A[0:2, 2:4] -> indices 2,3,6,7
    # A21 (2x2) corresponds to A[2:4, 0:2] -> indices 8,9,12,13
    # A22 (2x2) corresponds to A[2:4, 2:4] -> indices 10,11,14,15
    
    # B11 (2x2) corresponds to B[0:2, 0:2] -> indices 0,1,4,5 (if B is 4x4)
    # But B is 4x5.
    # For the 4x4x4 part, we use B[0:4, 0:4].
    
    # This is getting complex. I will return the rank 80 schoolbook to be safe,
    # as I cannot guarantee the correctness of the 56 or 49 term implementation in time.
    
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
