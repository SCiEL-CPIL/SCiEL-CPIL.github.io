import numpy as np
import random
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank calculation
    # Split m=4 into two 2s.
    # Each 2x2x7 split into three 2x2x2 (Strassen) and one 2x2x1 (Standard).
    # Rank(2x2x2) = 7. Rank(2x2x1) = 4.
    # Rank per 2x2x7 = 3*7 + 4 = 25.
    # Total Rank = 2 * 25 = 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a product
    # a_coeffs: dict { (i, j): coeff } for A
    # b_coeffs: dict { (j, k): coeff } for B
    # c_coeffs: dict { (i, k): coeff } for C
    def add_product(a_coeffs, b_coeffs, c_coeffs):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        
        for (i, j), coeff in a_coeffs.items():
            if coeff != 0:
                U[i * m + j][col] = coeff
                
        for (j, k), coeff in b_coeffs.items():
            if coeff != 0:
                V[j * p + k][col] = coeff
                
        for (i, k), coeff in c_coeffs.items():
            if coeff != 0:
                # W index is k * n + i
                W[k * n + i][col] = coeff
                
        col += 1

    # Strassen coefficients for 2x2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    # M2 = (A10+A11)B00 -> C10+=1
    # M3 = A00(B01-B11) -> C01+=1, C11+=1
    # M4 = A11(B10-B00) -> C00+=1, C10+=1
    # M5 = (A00+A01)B11 -> C00+=-1, C01+=1
    # M6 = (A10-A00)(B00+B01) -> C11+=1
    # M7 = (A01-A11)(B10+B11) -> C00+=1
    
    # Note: The C coefficients in Strassen are relative to the 2x2 block.
    # We need to map them to global indices.
    
    def add_strassen_block(base_i, base_j, base_k):
        """
        Adds 7 products for a 2x2x2 block.
        base_i: starting row index for A (0 or 1? No, A is 2x4, so i is always 0,1)
        Actually, A is 2x4. i is 0,1. j is 0,1,2,3.
        B is 4x7. j is 0,1,2,3. k is 0..6.
        C is 2x7. i is 0,1. k is 0..6.
        
        For a block at j-offset bj and k-offset bk:
        A indices: i in {0,1}, j in {bj, bj+1}
        B indices: j in {bj, bj+1}, k in {bk, bk+1}
        C indices: i in {0,1}, k in {bk, bk+1}
        """
        
        # M1
        add_product(
            {(0, base_j): 1, (1, base_j+1): 1},
            {(base_j, base_k): 1, (base_j+1, base_k+1): 1},
            {(0, base_k): 1, (1, base_k+1): 1}
        )
        
        # M2
        add_product(
            {(1, base_j): 1, (1, base_j+1): 1},
            {(base_j, base_k): 1},
            {(1, base_k): 1}
        )
        
        # M3
        add_product(
            {(0, base_j): 1},
            {(base_j, base_k+1): 1, (base_j+1, base_k+1): -1},
            {(0, base_k+1): 1, (1, base_k+1): 1}
        )
        
        # M4
        add_product(
            {(1, base_j+1): 1},
            {(base_j+1, base_k): 1, (base_j, base_k): -1},
            {(0, base_k): 1, (1, base_k): 1}
        )
        
        # M5
        add_product(
            {(0, base_j): 1, (0, base_j+1): 1},
            {(base_j+1, base_k+1): 1},
            {(0, base_k): -1, (0, base_k+1): 1}
        )
        
        # M6
        add_product(
            {(1, base_j): 1, (0, base_j): -1},
            {(base_j, base_k): 1, (base_j, base_k+1): 1},
            {(1, base_k+1): 1}
        )
        
        # M7
        add_product(
            {(0, base_j+1): 1, (1, base_j+1): -1},
            {(base_j+1, base_k): 1, (base_j+1, base_k+1): 1},
            {(0, base_k): 1}
        )

    def add_standard_block_2x2x1(base_j, base_k):
        """
        Adds 4 products for a 2x2x1 block.
        A: i in {0,1}, j in {base_j, base_j+1}
        B: j in {base_j, base_j+1}, k in {base_k}
        C: i in {0,1}, k in {base_k}
        """
        # C[0, base_k] = A[0, base_j]*B[base_j, base_k] + A[0, base_j+1]*B[base_j+1, base_k]
        add_product(
            {(0, base_j): 1},
            {(base_j, base_k): 1},
            {(0, base_k): 1}
        )
        add_product(
            {(0, base_j+1): 1},
            {(base_j+1, base_k): 1},
            {(0, base_k): 1}
        )
        
        # C[1, base_k] = A[1, base_j]*B[base_j, base_k] + A[1, base_j+1]*B[base_j+1, base_k]
        add_product(
            {(1, base_j): 1},
            {(base_j, base_k): 1},
            {(1, base_k): 1}
        )
        add_product(
            {(1, base_j+1): 1},
            {(base_j+1, base_k): 1},
            {(1, base_k): 1}
        )

    # Iterate over the two j-blocks
    for b in range(2):
        base_j = b * 2
        
        # k-blocks 0, 1, 2 are 2x2x2 (Strassen)
        # k-block 3 is 2x2x1 (Standard)
        
        # k=0,1
        add_strassen_block(base_j, 0)
        
        # k=2,3
        add_strassen_block(base_j, 2)
        
        # k=4,5
        add_strassen_block(base_j, 4)
        
        # k=6
        add_standard_block_2x2x1(base_j, 6)

    if col != rank:
        # If col < rank, we have extra zero columns?
        # We should trim or ensure exact rank.
        # But rank is fixed by the algorithm.
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
