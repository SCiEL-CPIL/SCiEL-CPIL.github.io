# EVOLVE-BLOCK-START
import time

def construct():
    """Find rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = 4, 4, 6
    
    # Total rank: 7 outer Strassen steps * 11 inner terms (7 Strassen + 4 Schoolbook)
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Outer Strassen Steps for 4x4 * 4x6
    # A is 4x4, B is 4x6, C is 4x6
    # Partition A into 2x2 blocks of size 2x2: A00, A01, A10, A11
    # Partition B into 2x2 blocks of size 2x3: B00, B01, B10, B11
    #   B00: rows 0,1; cols 0,1,2
    #   B01: rows 0,1; cols 3,4,5
    #   B10: rows 2,3; cols 0,1,2
    #   B11: rows 2,3; cols 3,4,5
    # Partition C into 2x2 blocks of size 2x3: C00, C01, C10, C11
    
    outer_steps = [
        # M1 = (A00 + A11) * (B00 + B11) -> C00 + C11
        (([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)])),
        # M2 = (A10 + A11) * B00 -> C10 - C11
        (([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)])),
        # M3 = A00 * (B01 - B11) -> C01 + C11
        (([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)])),
        # M4 = A11 * (B10 - B00) -> C00 + C10
        (([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)])),
        # M5 = (A00 + A01) * B11 -> C01 - C00
        (([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)])),
        # M6 = (A10 - A00) * (B00 + B01) -> C11
        (([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)])),
        # M7 = (A01 - A11) * (B10 + B11) -> C00
        (([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]))
    ]
    
    # Helper to map local block coordinates to global indices
    def get_global_A(block_id, r, s):
        I = (block_id // 2) * 2 + r
        J = (block_id % 2) * 2 + s
        return I, J

    def get_global_B(block_id, s, k):
        J = (block_id // 2) * 2 + s
        K = (block_id % 2) * 3 + k
        return J, K

    def get_global_C(block_id, r, k):
        I = (block_id // 2) * 2 + r
        K = (block_id % 2) * 3 + k
        return I, K

    # Combined inner decomposition for 2x2 * 2x3 (11 terms)
    # First 7: Strassen for 2x2 * 2x2 (cols 0,1)
    # Last 4: Schoolbook for 2x2 * 2x1 (col 2)
    inner_11 = [
        # S1: (A00+A11)(B00+B11) -> C00+C11
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])),
        # S2: (A10+A11)B00 -> C10-C11
        (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])),
        # S3: A00(B01-B11) -> C01+C11
        (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])),
        # S4: A11(B10-B00) -> C00+C10
        (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])),
        # S5: (A00+A01)B11 -> C01-C00
        (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,1), (0,0,-1)])),
        # S6: (A10-A00)(B00+B01) -> C11
        (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])),
        # S7: (A01-A11)(B10+B11) -> C00
        (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])),
        # T1: A00 * B02 -> C02
        (([(0,0,1)], [(0,2,1)], [(0,2,1)])),
        # T2: A01 * B12 -> C02
        (([(0,1,1)], [(1,2,1)], [(0,2,1)])),
        # T3: A10 * B02 -> C12
        (([(1,0,1)], [(0,2,1)], [(1,2,1)])),
        # T4: A11 * B12 -> C12
        (([(1,1,1)], [(1,2,1)], [(1,2,1)]))
    ]
    
    # Construct the decomposition
    for i, (A_outer, B_outer, C_outer) in enumerate(outer_steps):
        base_rank = i * 11
        
        for j, (A_inner, B_inner, C_inner) in enumerate(inner_11):
            c_global_rank = base_rank + j
            
            # Update U
            for (r, s, coeff) in A_inner:
                for (block_id, b_coeff) in A_outer:
                    I, J = get_global_A(block_id, r, s)
                    U[I * m + J][c_global_rank] += coeff * b_coeff
                    
            # Update V
            for (s, k, coeff) in B_inner:
                for (block_id, b_coeff) in B_outer:
                    J, K = get_global_B(block_id, s, k)
                    V[J * p + K][c_global_rank] += coeff * b_coeff
                    
            # Update W: W index is K * n + I
            for (r, k, coeff) in C_inner:
                for (block_id, b_coeff) in C_outer:
                    I, K = get_global_C(block_id, r, k)
                    W[K * n + I][c_global_rank] += coeff * b_coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
