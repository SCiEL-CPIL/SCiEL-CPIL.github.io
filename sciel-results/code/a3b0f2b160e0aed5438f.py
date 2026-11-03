import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    """
    Returns a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x4 matrix.
    
    We use a block decomposition approach:
    - Split A (3x3) into 2x2, 2x1, 1x2, 1x1 blocks.
    - Split B (3x4) into 2x2, 2x2, 1x2, 1x2 blocks.
    - Use Strassen's algorithm for the 2x2x2 sub-block multiplications.
    
    This should give a rank lower than the schoolbook rank of 36.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # We'll build the decomposition by combining rank-1 terms.
    # Each term is (u, v, w) where:
    #   u is a vector of length n*m = 9 (flattened A)
    #   v is a vector of length m*p = 12 (flattened B)
    #   w is a vector of length p*n = 12 (flattened C, with index k*n + i)
    
    terms = []  # List of (u_dict, v_dict, w_dict) where each dict maps index -> coefficient
    
    def add_term(u_idx, v_idx, w_idx):
        """Add a rank-1 term. Each arg is a dict {index: coeff}."""
        terms.append((u_idx, v_idx, w_idx))
    
    # Helper to create a sparse vector as a dict
    def sparse_vec(indices, coeffs):
        return {i: c for i, c in zip(indices, coeffs)}
    
    # Strassen 2x2x2 decomposition
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # Uses 7 multiplications:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+h)
    # Then:
    # C11 = M1 - M5 + M4 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 - M6
    
    def strassen_222(a_base, b_base, c_base):
        """
        Add Strassen terms for a 2x2x2 block multiplication.
        a_base, b_base, c_base are the starting indices in the flattened arrays.
        A is 2x2: indices a_base+0 (00), a_base+1 (01), a_base+2 (10), a_base+3 (11)
        B is 2x2: indices b_base+0 (00), b_base+1 (01), b_base+2 (10), b_base+3 (11)
        C is 2x2: indices c_base+0 (00), c_base+1 (01), c_base+2 (10), c_base+3 (11)
        """
        # M1 = (a+d)(e+h)
        # a = A[0,0] = a_base+0, d = A[1,1] = a_base+3
        # e = B[0,0] = b_base+0, h = B[1,1] = b_base+3
        add_term(
            {a_base+0: 1, a_base+3: 1},
            {b_base+0: 1, b_base+3: 1},
            {c_base+0: 1, c_base+3: 1}  # contributes to C[0,0] and C[1,1]
        )
        
        # M2 = (c+d)e
        # c = A[1,0] = a_base+2, d = A[1,1] = a_base+3, e = B[0,0] = b_base+0
        add_term(
            {a_base+2: 1, a_base+3: 1},
            {b_base+0: 1},
            {c_base+2: 1}  # contributes to C[1,0]
        )
        
        # M3 = a(f-h)
        # a = A[0,0] = a_base+0, f = B[0,1] = b_base+1, h = B[1,1] = b_base+3
        add_term(
            {a_base+0: 1},
            {b_base+1: 1, b_base+3: -1},
            {c_base+1: 1, c_base+3: 1}  # contributes to C[0,1] and C[1,1]
        )
        
        # M4 = d(g-e)
        # d = A[1,1] = a_base+3, g = B[1,0] = b_base+2, e = B[0,0] = b_base+0
        add_term(
            {a_base+3: 1},
            {b_base+2: 1, b_base+0: -1},
            {c_base+0: 1, c_base+2: 1}  # contributes to C[0,0] and C[1,0]
        )
        
        # M5 = (a+b)h
        # a = A[0,0] = a_base+0, b = A[0,1] = a_base+1, h = B[1,1] = b_base+3
        add_term(
            {a_base+0: 1, a_base+1: 1},
            {b_base+3: 1},
            {c_base+1: 1}  # contributes to C[0,1]
        )
        
        # M6 = (c-d)(f+g)
        # c = A[1,0] = a_base+2, d = A[1,1] = a_base+3
        # f = B[0,1] = b_base+1, g = B[1,0] = b_base+2
        add_term(
            {a_base+2: 1, a_base+3: -1},
            {b_base+1: 1, b_base+2: 1},
            {c_base+3: -1}  # contributes to C[1,1]
        )
        
        # M7 = (b-d)(e+h)
        # b = A[0,1] = a_base+1, d = A[1,1] = a_base+3
        # e = B[0,0] = b_base+0, h = B[1,1] = b_base+3
        add_term(
            {a_base+1: 1, a_base+3: -1},
            {b_base+0: 1, b_base+3: 1},
            {c_base+0: 1}  # contributes to C[0,0]
        )
    
    # Now let's build the full decomposition.
    # A is 3x3, split into:
    #   A11 = A[0:2, 0:2] (2x2), indices: 0,1,3,4 (row-major: 00,01,10,11)
    #   A12 = A[0:2, 2:3] (2x1), indices: 2,5
    #   A21 = A[2:3, 0:2] (1x2), indices: 6,7
    #   A22 = A[2:3, 2:3] (1x1), index: 8
    
    # B is 3x4, split into:
    #   B11 = B[0:2, 0:2] (2x2), indices: 0,1,4,5
    #   B12 = B[0:2, 2:4] (2x2), indices: 2,3,6,7
    #   B21 = B[2:3, 0:2] (1x2), indices: 8,9
    #   B22 = B[2:3, 2:4] (1x2), indices: 10,11
    
    # C is 3x4, split into:
    #   C11 = C[0:2, 0:2] (2x2), indices: 0,1,3,4
    #   C12 = C[0:2, 2:4] (2x2), indices: 2,5,6,7
    #   C21 = C[2:3, 0:2] (1x2), indices: 8,9
    #   C22 = C[2:3, 2:4] (1x2), indices: 10,11
    
    # The multiplication is:
    # C11 = A11 * B11 + A12 * B21
    # C12 = A11 * B12 + A12 * B22
    # C21 = A21 * B11 + A22 * B21
    # C22 = A21 * B12 + A22 * B22
    
    # For the 2x2 * 2x2 products, use Strassen (rank 7 each).
    # For the smaller products, use schoolbook.
    
    # C11 = A11 * B11 + A12 * B21
    # A11*B11: Strassen on 2x2x2
    strassen_222(0, 0, 0)  # A11 base=0, B11 base=0, C11 base=0
    
    # A12*B21: A12 is 2x1, B21 is 1x2. Result is 2x2.
    # A12 = [[a12_00],[a12_10]] = [[A[0,2]],[A[1,2]]], indices 2,5
    # B21 = [[b21_00, b21_01]] = [[B[2,0], B[2,1]]], indices 8,9
    # Product = [[a12_00*b21_00, a12_00*b21_01],
    #            [a12_10*b21_00, a12_10*b21_01]]
    # This is rank 4 (schoolbook for 2x1 * 1x2)
    
    # Term 1: A[0,2] * B[2,0] -> C[0,0]
    add_term({2: 1}, {8: 1}, {0: 1})
    # Term 2: A[0,2] * B[2,1] -> C[0,1]
    add_term({2: 1}, {9: 1}, {1: 1})
    # Term 3: A[1,2] * B[2,0] -> C[1,0]
    add_term({5: 1}, {8: 1}, {3: 1})
    # Term 4: A[1,2] * B[2,1] -> C[1,1]
    add_term({5: 1}, {9: 1}, {4: 1})
    
    # C12 = A11 * B12 + A12 * B22
    # A11*B12: Strassen on 2x2x2
    # A11 base=0, B12 base=2, C12 base=2
    strassen_222(0, 2, 2)
    
    # A12*B22: A12 is 2x1, B22 is 1x2. Result is 2x2.
    # A12 = [[A[0,2]],[A[1,2]]], indices 2,5
    # B22 = [[B[2,2], B[2,3]]], indices 10,11
    # Product = [[A[0,2]*B[2,2], A[0,2]*B[2,3]],
    #            [A[1,2]*B[2,2], A[1,2]*B[2,3]]]
    # This is rank 4
    
    # Term 1: A[0,2] * B[2,2] -> C[0,2]
    add_term({2: 1}, {10: 1}, {2: 1})
    # Term 2: A[0,2] * B[2,3] -> C[0,3]
    add_term({2: 1}, {11: 1}, {5: 1})
    # Term 3: A[1,2] * B[2,2] -> C[1,2]
    add_term({5: 1}, {10: 1}, {6: 1})
    # Term 4: A[1,2] * B[2,3] -> C[1,3]
    add_term({5: 1}, {11: 1}, {7: 1})
    
    # C21 = A21 * B11 + A22 * B21
    # A21*B11: A21 is 1x2, B11 is 2x2. Result is 1x2.
    # A21 = [[A[2,0], A[2,1]]], indices 6,7
    # B11 = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]], indices 0,1,4,5
    # Product = [[A[2,0]*B[0,0] + A[2,1]*B[1,0], A[2,0]*B[0,1] + A[2,1]*B[1,1]]]
    # This is rank 4 (schoolbook for 1x2 * 2x2)
    
    # Term 1: A[2,0] * B[0,0] -> C[2,0]
    add_term({6: 1}, {0: 1}, {8: 1})
    # Term 2: A[2,1] * B[1,0] -> C[2,0]
    add_term({7: 1}, {4: 1}, {8: 1})
    # Term 3: A[2,0] * B[0,1] -> C[2,1]
    add_term({6: 1}, {1: 1}, {9: 1})
    # Term 4: A[2,1] * B[1,1] -> C[2,1]
    add_term({7: 1}, {5: 1}, {9: 1})
    
    # A22*B21: A22 is 1x1, B21 is 1x2. Result is 1x2.
    # A22 = [[A[2,2]]], index 8
    # B21 = [[B[2,0], B[2,1]]], indices 8,9
    # Product = [[A[2,2]*B[2,0], A[2,2]*B[2,1]]]
    # This is rank 2
    
    # Term 1: A[2,2] * B[2,0] -> C[2,0]
    add_term({8: 1}, {8: 1}, {8: 1})
    # Term 2: A[2,2] * B[2,1] -> C[2,1]
    add_term({8: 1}, {9: 1}, {9: 1})
    
    # C22 = A21 * B12 + A22 * B22
    # A21*B12: A21 is 1x2, B12 is 2x2. Result is 1x2.
    # A21 = [[A[2,0], A[2,1]]], indices 6,7
    # B12 = [[B[0,2], B[0,3]], [B[1,2], B[1,3]]], indices 2,3,6,7
    # Product = [[A[2,0]*B[0,2] + A[2,1]*B[1,2], A[2,0]*B[0,3] + A[2,1]*B[1,3]]]
    # This is rank 4
    
    # Term 1: A[2,0] * B[0,2] -> C[2,2]
    add_term({6: 1}, {2: 1}, {10: 1})
    # Term 2: A[2,1] * B[1,2] -> C[2,2]
    add_term({7: 1}, {6: 1}, {10: 1})
    # Term 3: A[2,0] * B[0,3] -> C[2,3]
    add_term({6: 1}, {3: 1}, {11: 1})
    # Term 4: A[2,1] * B[1,3] -> C[2,3]
    add_term({7: 1}, {7: 1}, {11: 1})
    
    # A22*B22: A22 is 1x1, B22 is 1x2. Result is 1x2.
    # A22 = [[A[2,2]]], index 8
    # B22 = [[B[2,2], B[2,3]]], indices 10,11
    # Product = [[A[2,2]*B[2,2], A[2,2]*B[2,3]]]
    # This is rank 2
    
    # Term 1: A[2,2] * B[2,2] -> C[2,2]
    add_term({8: 1}, {10: 1}, {10: 1})
    # Term 2: A[2,2] * B[2,3] -> C[2,3]
    add_term({8: 1}, {11: 1}, {11: 1})
    
    # Now we have all the terms. Let's count them.
    # Strassen for A11*B11: 7 terms
    # A12*B21: 4 terms
    # Strassen for A11*B12: 7 terms
    # A12*B22: 4 terms
    # A21*B11: 4 terms
    # A22*B21: 2 terms
    # A21*B12: 4 terms
    # A22*B22: 2 terms
    # Total: 7+4+7+4+4+2+4+2 = 34 terms
    
    # Now build the U, V, W matrices.
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            U[idx][col] = coeff
        for idx, coeff in v_dict.items():
            V[idx][col] = coeff
        for idx, coeff in w_dict.items():
            W[idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
