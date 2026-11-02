import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    
    # Total rank estimate:
    # Block 1 (3x3 A, 3x3 B): 27
    # Block 2 (3x3 A, 3x3 B): 27
    # Block 3 (3x3 A, 3x1 B): 3
    # Block 4 (3x2 A, 2x3 B): 12
    # Block 5 (3x2 A, 2x3 B): 12
    # Block 6 (3x2 A, 2x1 B): 2
    # Total: 83
    
    rank = 83
    
    U = [[0] * rank for _ in range(n * m)] # 15x83
    V = [[0] * rank for _ in range(m * p)] # 35x83
    W = [[0] * rank for _ in range(p * n)] # 21x83
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Helper to add schoolbook decomposition for sub-block
    # u_base: offset in U for top-left of A sub-block
    # v_base: offset in V for top-left of B sub-block
    # w_base: offset in W for top-left of C sub-block
    # ni, nj, nk: dimensions of A, B, C sub-blocks
    # Note: W index is k*n + i. 
    # If C sub-block starts at row ci, col ck in global C.
    # Global W index for local (i, k) is (ck+k)*n + (ci+i).
    
    def add_schoolbook(ni, nj, nk, u_base_i, u_base_j, v_base_i, v_base_j, w_base_i, w_base_k):
        # u_base_i, u_base_j: local (0,0) corresponds to global (u_base_i, u_base_j)
        # Global U index: (u_base_i + i) * m + (u_base_j + j)
        # Global V index: (v_base_i + j) * p + (v_base_j + k)
        # Global W index: (w_base_k + k) * n + (w_base_i + i)
        
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    u_idx = (u_base_i + i) * m + (u_base_j + j)
                    v_idx = (v_base_i + j) * p + (v_base_j + k)
                    w_idx = (w_base_k + k) * n + (w_base_i + i)
                    add_term(u_idx, v_idx, w_idx)

    # Helper to add Strassen decomposition for 2x2x2
    def add_strassen(ni, nj, nk, u_base_i, u_base_j, v_base_i, v_base_j, w_base_i, w_base_k):
        # Assumes ni=2, nj=2, nk=2
        # Strassen 7 terms
        # Local indices 0,1
        # Global U: (u_base_i + i) * m + (u_base_j + j)
        
        # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # Term 2: M2 = (A10+A11)B00 -> C10 += M2
        # Term 3: M3 = A00(B10-B11) -> C01 += M3
        # Term 4: M4 = A11(B01-B00) -> C01 += -M4? No, C01 += M4?
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M6
        
        # We add terms to global W.
        # Each Strassen term is a product of 2x2 matrices.
        # We need to decompose the product of linear combinations into rank-1 terms?
        # No, Strassen terms are NOT rank-1 in the original tensor space.
        # They are bilinear forms.
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This requires 4 rank-1 terms if we expand.
        # But Strassen reduces the number of MULTIPLICATIONS.
        # In tensor decomposition, each "multiplication" is a rank-1 term?
        # No. The rank of the tensor is the number of rank-1 tensors in the sum.
        # Strassen's algorithm computes 7 products of 2x2 matrices.
        # Each 2x2 matrix product is a tensor of rank 4 (schoolbook) or 7 (Strassen)?
        # Wait. The tensor for 2x2x2 has rank 7.
        # So we can use the rank-7 decomposition for each 2x2 block product?
        # No, that would be 7 * 7 = 49 terms.
        # The point of Strassen is that the TOTAL rank for the 2x2x2 tensor is 7.
        # So we must hardcode the 7 rank-1 vectors for the 2x2x2 tensor.
        
        # The 7 rank-1 vectors for 2x2x2 (indices 00, 01, 10, 11):
        # 1. (1,0; 0,1) x (1,0; 0,1) -> (1,0; 0,1)? No.
        
        # Let's use the known rank-7 decomposition for 2x2x2.
        # u1 = [1,0,0,0], v1 = [1,0,0,0], w1 = [1,0,0,0] (A00B00->C00)
        # u2 = [0,0,1,0], v2 = [0,0,1,0], w2 = [0,0,0,1] (A10B10->C11? No)
        
        # This is getting complicated to hardcode manually without a reference.
        # Given the risk of error, I will fallback to Schoolbook for the 2x2x2 blocks as well.
        # If I use Schoolbook for 2x2x2 (Rank 4):
        # Block 4 (3x2x3):
        # A11(2x2) B11(2x2) -> 4 terms
        # A11(2x2) B12(2x1) -> 2 terms
        # A21(1x2) B11(2x2) -> 2 terms
        # A21(1x2) B12(2x1) -> 1 term
        # Total 9 terms.
        
        # So Rank 9 for 3x2x3.
        # Then Part 2: 9 + 9 + 2 = 20.
        # Part 1: 27 + 27 + 3 = 57.
        # Total: 77.
        
        # 77 is better than 83.
        # I will implement Schoolbook for all sub-blocks.
        # It is simple and guaranteed correct.
        
        pass # Placeholder, will just call add_schoolbook

    # Block 1: A(0:3, 0:3) B(0:3, 0:3) -> C(0:3, 0:3)
    add_schoolbook(3, 3, 3, 0, 0, 0, 0, 0, 0)
    
    # Block 2: A(0:3, 0:3) B(0:3, 3:6) -> C(0:3, 3:6)
    add_schoolbook(3, 3, 3, 0, 0, 0, 3, 0, 3)
    
    # Block 3: A(0:3, 0:3) B(0:3, 6:7) -> C(0:3, 6:7)
    add_schoolbook(3, 3, 1, 0, 0, 0, 6, 0, 6)
    
    # Block 4: A(0:3, 3:5) B(3:5, 0:3) -> C(0:3, 0:3)
    # A is 3x2, B is 2x3
    add_schoolbook(3, 2, 3, 0, 3, 3, 0, 0, 0)
    
    # Block 5: A(0:3, 3:5) B(3:5, 3:6) -> C(0:3, 3:6)
    add_schoolbook(3, 2, 3, 0, 3, 3, 3, 0, 3)
    
    # Block 6: A(0:3, 3:5) B(3:5, 6:7) -> C(0:3, 6:7)
    add_schoolbook(3, 2, 1, 0, 3, 3, 6, 0, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
