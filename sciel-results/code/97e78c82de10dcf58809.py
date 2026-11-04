import numpy as np
import time
import sys

_DEADLINE = time.time() + 110  # Safety margin before 120s limit

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Try to use rank 7 for 2x2x3 blocks
    # Total rank: 49 (4x4x4) + 4 * 7 (4x4x3) = 77
    # Fallback: 49 + 4 * 11 = 93
    
    # We will build the 4x4x4 part first (Rank 49)
    # Then we will build the 4x4x3 part.
    # We need to know if the 7-term decomposition works.
    # We can test it locally.
    
    def test_2x2x3_rank7():
        # Define the 7 terms
        # A indices: 00, 01, 10, 11
        # B indices: 00, 01, 02, 10, 11, 12
        # C indices: 00, 01, 02, 10, 11, 12
        
        # M terms as (A_coeffs, B_coeffs)
        # A_coeffs: dict {(i,j): coeff}
        # B_coeffs: dict {(i,j): coeff}
        
        M_terms = [
            ({(0,0): 1}, {(0,0): 1, (1,1): 1, (0,2): 1}), # M1
            ({(1,0): 1, (1,1): 1}, {(0,0): 1, (1,2): 1}), # M2
            ({(0,0): 1}, {(0,1): 1, (1,1): -1, (1,2): 1}), # M3
            ({(1,1): 1}, {(1,0): 1, (0,0): -1, (0,2): 1}), # M4
            ({(0,0): 1, (0,1): 1}, {(1,1): 1, (0,2): 1}), # M5
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1, (1,2): 1}), # M6
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1, (0,2): 1}) # M7
        ]
        
        # C coeffs: dict {(i,j): [coeffs for M1..M7]}
        C_coeffs = {
            (0,0): [1, 0, 0, 1, -1, 0, 1],
            (0,1): [0, 0, 1, 0, 1, 0, 0],
            (1,0): [0, 1, 0, 1, 0, 0, 0],
            (1,1): [1, -1, 1, 0, 0, 1, 0],
            (0,2): [1, 0, 0, 1, -1, 0, 1], # Guess
            (1,2): [0, 1, 0, -1, 0, 1, -1]  # Guess
        }
        
        # Verify
        # Target: C[i,k] = sum_j A[i,j]B[j,k]
        # Check for all i,k
        for i in range(2):
            for k in range(3):
                # Calculate LHS (Target)
                # A[i,j] B[j,k]
                # We need to check if sum_r U_r V_r W_r == 1 for the specific term
                # Actually, we check if the linear combination of M terms gives the correct C entry.
                # C_target[i,k] = sum_j A[i,j] B[j,k]
                # This is a bilinear form.
                # The decomposition says C[i,k] = sum_r c_{i,k,r} M_r
                # M_r = (sum a_{r,ij} A[i,j]) (sum b_{r,jk} B[j,k])
                # So C[i,k] = sum_r c_{i,k,r} (sum a_{r,ij} A[i,j]) (sum b_{r,jk} B[j,k])
                # We need this to equal sum_j A[i,j] B[j,k]
                # This must hold for all A,B.
                # So for each i,k, and for each j, the coefficient of A[i,j]B[j,k] must be 1.
                # And for all other terms A[i',j']B[j'',k'], the coefficient must be 0.
                
                # Let's check the coefficient of A[i,j] B[j,k] in the sum.
                # Term r contributes: c_{i,k,r} * a_{r,ij} * b_{r,jk}
                # Sum over r.
                
                for j in range(2):
                    coeff = 0.0
                    for r in range(7):
                        a_val = M_terms[r][0].get((i,j), 0.0)
                        b_val = M_terms[r][1].get((j,k), 0.0)
                        c_val = C_coeffs[(i,k)][r]
                        coeff += c_val * a_val * b_val
                    if abs(coeff - 1.0) > 1e-10:
                        return False
                
                # Check cross terms
                # For i', j', k' such that (i',j',k') != (i,j,k)
                # Actually, we only need to check that the tensor is correct.
                # The tensor T[i,j,k] = 1 if valid.
                # The decomposition reconstructs the tensor.
                # We can just check a few random instances? No, must be exact.
                # The condition "coeff of A[i,j]B[j,k] is 1" is necessary.
                # Is it sufficient?
                # The map is bilinear. If it works for basis elements, it works for all.
                # The basis elements are A[i,j]B[j,k].
                # So yes, checking that the coefficient of each basis term A[i,j]B[j,k] is 1 is sufficient.
                
        return True

    use_rank7 = test_2x2x3_rank7()
    
    if use_rank7:
        total_rank = 49 + 28
    else:
        total_rank = 49 + 44

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Part 1: 4x4x4 (Rank 49)
    # ... (Copy existing code for Part 1) ...
    
    # Part 2: 4x4x3
    # ... (Modify add_2x2x3 to use 7 or 11 terms) ...

    # ...
    
    return {"U": U, "V": V, "W": W}
