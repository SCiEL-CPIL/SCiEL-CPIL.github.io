import time
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4
    # Rank 16 for 4x4x1
    total_rank = 65
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_idx, v_idx, w_idx, value
    def add_term(u_idx, v_idx, w_idx, val):
        pass # Will implement logic inline

    # Strassen 2x2x2 decomposition
    # Returns lists of (u_coeff, v_coeff, w_coeff) for the 7 products
    # Actually, we need the full U, V, W for the 2x2x2 tensor
    # U is 4x7, V is 4x7, W is 4x7
    # Indices for 2x2: u: i*2+j, v: j*2+k, w: k*2+i
    
    # The 7 Strassen products P1..P7
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # We need to decompose each Pk (which is a 2x2x2 tensor) into 7 rank-1 terms.
    # So we need a function that given the linear combinations of A and B blocks,
    # outputs the coefficients for the rank-1 terms of the 2x2x2 multiplication.
    
    # Let's hardcode the 49 terms.
    # This is tedious but reliable.
    
    # Alternative: Use the recursive structure.
    # Let Strassen2x2(A, B) return the 7 products.
    # Each product is a 2x2 matrix.
    # We can represent the 49 terms by iterating over the 7 top-level products,
    # and for each, iterating over the 7 bottom-level products.
    
    # Top level:
    # A blocks: A00, A01, A10, A11 (2x2 each)
    # B blocks: B00, B01, B10, B11 (2x2 each)
    
    # P1: (A00+A11) * (B00+B11)
    # P2: (A10+A11) * B00
    # P3: A00 * (B01-B11)
    # P4: A11 * (B10-B00)
    # P5: (A00+A01) * B11
    # P6: (A10-A00) * (B00+B01)
    # P7: (A01-A11) * (B10+B11)
    
    # For each Pk, we have a 2x2 matrix product X * Y.
    # X and Y are linear combinations of the 2x2 blocks.
    # We need to decompose X * Y into 7 rank-1 terms of the 2x2x2 tensor.
    # The 2x2x2 tensor T has indices i,j,k (0..1).
    # T_{ijk} = sum_l X_{il} Y_{lk}.
    # The rank-7 decomposition of 2x2x2 is:
    # Q1 = (X00+X11)(Y00+Y11)
    # Q2 = (X10+X11)Y00
    # Q3 = X00(Y01-Y11)
    # Q4 = X11(Y10-Y00)
    # Q5 = (X00+X01)Y11
    # Q6 = (X10-X00)(Y00+Y01)
    # Q7 = (X01-X11)(Y10+Y11)
    # C00 = Q1 + Q4 - Q5 + Q7
    # C01 = Q3 + Q5
    # C10 = Q2 + Q4
    # C11 = Q1 - Q2 + Q3 + Q6
    
    # So, for each top-level Pk, we substitute X and Y with their block expressions.
    # Then for each of the 7 Q's, we substitute X and Y again.
    # This results in a sum of terms like (linear comb of A blocks) * (linear comb of B blocks).
    # Each such term is a 2x2x2 tensor.
    # We need to expand this into rank-1 terms of the 4x4x4 tensor.
    # A rank-1 term of the 2x2x2 tensor is u * v * w where u,v,w are 2x2 matrices? No.
    # The rank-1 term of the 2x2x2 tensor is $x \otimes y \otimes z$ where x,y,z are vectors?
    # No, the tensor is $T_{ijk}$.
    # A rank-1 term is $u_i v_j w_k$.
    # In the Strassen decomposition, each Q is a product of two linear forms.
    # Q = (sum a_{il} X_{il}) (sum b_{lk} Y_{lk})? No.
    # Q is a scalar (the product of two 2x2 matrices? No, Q is one of the 7 products).
    # Wait, Strassen computes 7 scalar products? No.
    # Strassen computes 7 products of 2x2 matrices.
    # Each product is a 2x2 matrix.
    # So Q1 is a 2x2 matrix.
    # To decompose Q1 (a 2x2 matrix resulting from product of two 2x2 matrices) into rank-1 terms of the 2x2x2 tensor...
    # The 2x2x2 tensor itself has rank 7.
    # The 7 terms in the Strassen decomposition of the 2x2x2 tensor are:
    # Term 1: u1 * v1 * w1
    # ...
    # Term 7: u7 * v7 * w7
    # where u, v, w are vectors of length 4? No.
    # The tensor T has size 2x2x2.
    # Flattened U is 4x7, V is 4x7, W is 4x7.
    
    # I will generate the 49 terms by explicit expansion.
    
    # This is getting complex. I will use a known library or pre-computed values if possible.
    # Since I cannot use external libraries for the decomposition itself (only for computation),
    # I will write a small recursive generator.
    
    def strassen_2x2_decomp():
        # Returns U, V, W for 2x2x2
        # U, V, W are 4x7
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # Indices: u: i*2+j, v: j*2+k, w: k*2+i
        
        # P1 = (A00+A11)(B00+B11)
        # A00 is u[0], A11 is u[3]
        # B00 is v[0], B11 is v[3]
        # The product is a 2x2 matrix.
        # We need to add this product to the tensor.
        # The product M = X*Y.
        # M_{ik} = sum_j X_{ij} Y_{jk}.
        # This is the standard tensor.
        # But Strassen replaces this with 7 terms.
        # The 7 terms are the rank-1 decomposition of the 2x2x2 tensor.
        # Let's hardcode the 7 terms of the 2x2x2 Strassen decomposition.
        
        # Term 1: (A00+A11) * (B00+B11)
        # This term contributes to C00, C01, C10, C11.
        # C00 += T1
        # C01 += T1
        # C10 += T1
        # C11 += T1
        # Wait, the coefficients are different.
        
        # Let's use the explicit rank-1 vectors for the 2x2x2 Strassen decomp.
        # Ref: https://en.wikipedia.org/wiki/Strassen_algorithm
        
        # The 7 products are:
        # 1: (A11+A22)(B11+B22)
        # 2: (A21+A22)B11
        # 3: A11(B12-B22)
        # 4: A22(B21-B11)
        # 5: (A11+A12)B22
        # 6: (A21-A11)(B11+B12)
        # 7: (A12-A22)(B21+B22)
        
        # And the reconstruction:
        # C11 = 1 + 4 - 5 + 7
        # C12 = 3 + 5
        # C21 = 2 + 4
        # C22 = 1 - 2 + 3 + 6
        
        # Each product is a 2x2 matrix.
        # To convert this to the rank-1 decomposition of the 2x2x2 tensor,
        # we need to expand each 2x2 product into 4 rank-1 terms?
        # No, the rank of 2x2x2 is 7.
        # The 7 products are the rank-1 terms?
        # No, each product is a matrix, not a scalar.
        # The rank-1 terms of the tensor T_{ijk} are $u_i v_j w_k$.
        # The Strassen algorithm computes 7 scalar products of linear forms?
        # No, it computes 7 products of 2x2 matrices.
        # This is a different level.
        
        # The rank-7 decomposition of the 2x2x2 tensor is:
        # T = sum_{k=1}^7 u_k \otimes v_k \otimes w_k
        # The vectors u_k, v_k, w_k are of length 4 (flattened 2x2).
        
        # I will find the explicit u, v, w for the 2x2x2 Strassen decomp.
        
        # Term 1:
        # u1: [1, 0, 0, 1] (A11+A22)
        # v1: [1, 0, 0, 1] (B11+B22)
        # w1: [1, 1, 1, 1] (Contributes to all C)
        # Wait, w1 must be such that C11, C12, C21, C22 all get +1.
        # C11 is w[0], C12 is w[1], C21 is w[2], C22 is w[3].
        # So w1 = [1, 1, 1, 1].
        
        # Term 2:
        # u2: [0, 0, 1, 1] (A21+A22)
        # v2: [1, 0, 0, 0] (B11)
        # w2: [0, 0, 1, -1] (C21 += 1, C22 += -1)
        
        # Term 3:
        # u3: [1, 0, 0, 0] (A11)
        # v3: [0, 1, 0, -1] (B12-B22)
        # w3: [0, 1, 0, 1] (C12 += 1, C22 += 1)
        
        # Term 4:
        # u4: [0, 0, 0, 1] (A22)
        # v4: [0, 0, 1, -1] (B21-B11)
        # w4: [1, 0, 1, 0] (C11 += 1, C21 += 1)
        
        # Term 5:
        # u5: [1, 1, 0, 0] (A11+A12)
        # v5: [0, 0, 0, 1] (B22)
        # w5: [-1, 1, 0, 0] (C11 += -1, C12 += 1)
        
        # Term 6:
        # u6: [0, 0, 1, -1] (A21-A11)
        # v6: [1, 1, 0, 0] (B11+B12)
        # w6: [0, 0, 0, 1] (C22 += 1)
        
        # Term 7:
        # u7: [0, 1, 0, -1] (A12-A22)
        # v7: [0, 0, 1, 1] (B21+B22)
        # w7: [1, 0, 0, 0] (C11 += 1)
        
        # Let's verify C11 = 1 + 4 - 5 + 7
        # w1[0]=1, w4[0]=1, w5[0]=-1, w7[0]=1. Sum=2?
        # My w vectors might be wrong.
        
        # Correct Strassen reconstruction:
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # So:
        # w1 = [1, 0, 0, 1] (P1 in C11, C22)
        # w2 = [0, 0, 1, -1] (P2 in C21, C22)
        # w3 = [0, 1, 0, 1] (P3 in C12, C22)
        # w4 = [1, 0, 1, 0] (P4 in C11, C21)
        # w5 = [-1, 1, 0, 0] (P5 in C11, C12)
        # w6 = [0, 0, 0, 1] (P6 in C22)
        # w7 = [1, 0, 0, 0] (P7 in C11)
        
        # Check C11: 1 + 1 - 1 + 1 = 2? No.
        # C11 = P1 + P4 - P5 + P7.
        # w1[0]=1, w4[0]=1, w5[0]=-1, w7[0]=1. Sum=2.
        # But C11 should be 1.
        # Ah, P1 is (A11+A22)(B11+B22).
        # The term P1 contributes to C11.
        # The coefficient is 1.
        # So w1[0] should be 1.
        # Why did I get 2?
        # Maybe my w vectors are not the coefficients for the final C.
        # The rank-1 term is $u \otimes v \otimes w$.
        # The contribution to C is $w$.
        # So the sum of $w_k$ for terms contributing to C11 must be 1.
        
        # Let's re-verify the Strassen formulas.
        # P1 = (A11+A22)(B11+B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P7 = (A12-A22)(B21+B22)
        
        # C11 = P1 + P4 - P5 + P7
        # Let's check with A=I, B=I.
        # P1 = 2*2 = 4
        # P4 = 1*(1-1) = 0
        # P5 = 2*1 = 2
        # P7 = (1-1)*2 = 0
        # C11 = 4 + 0 - 2 + 0 = 2.
        # But I*I = I, so C11 should be 1.
        # So the formula is wrong or I am misinterpreting.
        
        # Standard Strassen:
        # P1 = (A11+A22)(B11+B22)
        # P2 = (A21+A22)B11
        # P3 = A11(B12-B22)
        # P4 = A22(B21-B11)
        # P5 = (A11+A12)B22
        # P6 = (A21-A11)(B11+B12)
        # P7 = (A12-A22)(B21+B22)
        
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # Let's re-calculate for A=I, B=I.
        # A11=1, A12=0, A21=0, A22=1.
        # B11=1, B12=0, B21=0, B22=1.
        # P1 = (1+1)(1+1) = 4
        # P2 = (0+1)1 = 1
        # P3 = 1(0-1) = -1
        # P4 = 1(0-1) = -1
        # P5 = (1+0)1 = 1
        # P6 = (0-1)(1+0) = -1
        # P7 = (0-1)(0+1) = -1
        
        # C11 = 4 + (-1) - 1 + (-1) = 1. Correct.
        # C12 = -1 + 1 = 0. Correct.
        # C21 = 1 + (-1) = 0. Correct.
        # C22 = 4 - 1 + (-1) + (-1) = 1. Correct.
        
        # So the w vectors are:
        # w1 = [1, 0, 0, 1]
        # w2 = [0, 0, 1, -1]
        # w3 = [0, 1, 0, 1]
        # w4 = [1, 0, 1, 0]
        # w5 = [-1, 1, 0, 0]
        # w6 = [0, 0, 0, 1]
        # w7 = [1, 0, 0, 0]
        
        # And the u, v vectors are:
        # u1 = [1, 0, 0, 1]
        # v1 = [1, 0, 0, 1]
        
        # u2 = [0, 0, 1, 1]
        # v2 = [1, 0, 0, 0]
        
        # u3 = [1, 0, 0, 0]
        # v3 = [0, 1, 0, -1]
        
        # u4 = [0, 0, 0, 1]
        # v4 = [0, 0, 1, -1]
        
        # u5 = [1, 1, 0, 0]
        # v5 = [0, 0, 0, 1]
        
        # u6 = [0, 0, 1, -1]
        # v6 = [1, 1, 0, 0]
        
        # u7 = [0, 1, 0, -1]
        # v7 = [0, 0, 1, 1]
        
        return U, V, W

    # I will use these 7 vectors to build the 49 terms.
    
    # For the 4x4x4 case:
    # The 49 terms are formed by taking each of the 7 top-level products,
    # and for each, taking the 7 bottom-level products.
    
    # This is effectively a tensor product of the two decompositions.
    
    # I will implement the 49 terms by iterating.
    
    # For now, I will return the schoolbook 80 to ensure correctness, 
    # and then I will refine to 65.
    
    # Actually, I can just return 65 by combining the 49 and 16.
    
    # Let's generate the 49 terms.
    
    # This code is getting long. I will output the 65 rank solution.
    
    pass

# I will write the full code in the final block.
