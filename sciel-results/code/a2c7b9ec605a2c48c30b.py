import sys
import time
import numpy as np

def construct():
    """
    Rank-50 tensor decomposition for 2x4 by 4x7 matrix multiplication.
    
    Decomposition strategy:
    - Split B (4x7) into column blocks: [0,1], [2,3], [4,5], [6]
    - For each 2-column block of B: use two Strassen (2x2x2) blocks
      (one for top 2 rows of B, one for bottom 2 rows)
    - For the 1-column block: use two schoolbook (2x2x1) blocks
    - Total rank: 3 * 14 + 8 = 50
    
    W index convention: W[k*n+i, r] where C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = 2, 4, 7
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]  # (8, 50)
    V = [[0] * rank for _ in range(m * p)]  # (28, 50)
    W = [[0] * rank for _ in range(p * n)]  # (14, 50)
    
    col = 0
    
    def add_strassen_block(A_indices, B_indices, C_indices):
        """
        Adds a rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
        
        A_indices: list of 4 tuples (row, col) for A's 2x2 block, indexed 0-3
        B_indices: list of 4 tuples (row, col) for B's 2x2 block, indexed 0-3
        C_indices: list of 4 tuples (row, col) for C's 2x2 block, indexed 0-3
        
        Local index mapping: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        Strassen formulas:
        M1 = (a00+a11)(b00+b11)
        M2 = (a10+a11)b00
        M3 = a00(b01-b11)
        M4 = a11(b10-b00)
        M5 = (a00+a01)b11
        M6 = (a10-a00)(b00+b01)
        M7 = (a01-a11)(b10+b11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        nonlocal col
        
        def process_mk(A_coeffs, B_coeffs, C_coeffs):
            nonlocal col
            for local_a, coeff in A_coeffs.items():
                global_a = A_indices[local_a]
                U[global_a[0] * m + global_a[1]][col] += coeff
            for local_b, coeff in B_coeffs.items():
                global_b = B_indices[local_b]
                V[global_b[0] * p + global_b[1]][col] += coeff
            for local_c, coeff in C_coeffs.items():
                global_c = C_indices[local_c]
                # W index: W[k*n + i] where C[i,k] is the output element
                W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1
        
        # M1: (a00+a11)(b00+b11)
        process_mk({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
        # M2: (a10+a11)b00
        process_mk({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
        # M3: a00(b01-b11)
        process_mk({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
        # M4: a11(b10-b00)
        process_mk({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
        # M5: (a00+a01)b11
        process_mk({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
        # M6: (a10-a00)(b00+b01)
        process_mk({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
        # M7: (a01-a11)(b10+b11)
        process_mk({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    
    def add_schoolbook_221(A_indices, B_indices, C_indices):
        """
        Adds a rank-4 schoolbook decomposition for 2x2 * 2x1 -> 2x1.
        
        A_indices: list of 4 tuples (row, col) for A's 2x2 block
        B_indices: list of 2 tuples (row, col) for B's 2x1 block
        C_indices: list of 2 tuples (row, col) for C's 2x1 block
        
        C00 = a00*b00 + a01*b10
        C10 = a10*b00 + a11*b10
        """
        nonlocal col
        
        def process_mult(A_idx, B_idx, C_idx, coeff=1):
            nonlocal col
            global_a = A_indices[A_idx]
            global_b = B_indices[B_idx]
            global_c = C_indices[C_idx]
            
            U[global_a[0] * m + global_a[1]][col] += coeff
            V[global_b[0] * p + global_b[1]][col] += coeff
            W[global_c[1] * n + global_c[0]][col] += coeff
            col += 1
        
        # C00 contributions
        process_mult(0, 0, 0, 1)  # a00 * b00 -> c00
        process_mult(1, 1, 0, 1)  # a01 * b10 -> c00
        # C10 contributions
        process_mult(2, 0, 1, 1)  # a10 * b00 -> c10
        process_mult(3, 1, 1, 1)  # a11 * b10 -> c10
    
    # Define A blocks (2x4 matrix)
    # A_L: columns 0,1
    A_L = [(0, 0), (0, 1), (1, 0), (1, 1)]
    # A_R: columns 2,3
    A_R = [(0, 2), (0, 3), (1, 2), (1, 3)]
    
    # Block 1: B columns 0,1
    B_1_L = [(0, 0), (0, 1), (1, 0), (1, 1)]  # rows 0,1, cols 0,1
    B_1_R = [(2, 0), (2, 1), (3, 0), (3, 1)]  # rows 2,3, cols 0,1
    C_1 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    add_strassen_block(A_L, B_1_L, C_1)
    add_strassen_block(A_R, B_1_R, C_1)
    
    # Block 2: B columns 2,3
    B_2_L = [(0, 2), (0, 3), (1, 2), (1, 3)]  # rows 0,1, cols 2,3
    B_2_R = [(2, 2), (2, 3), (3, 2), (3, 3)]  # rows 2,3, cols 2,3
    C_2 = [(0, 2), (0, 3), (1, 2), (1, 3)]
    
    add_strassen_block(A_L, B_2_L, C_2)
    add_strassen_block(A_R, B_2_R, C_2)
    
    # Block 3: B columns 4,5
    B_3_L = [(0, 4), (0, 5), (1, 4), (1, 5)]  # rows 0,1, cols 4,5
    B_3_R = [(2, 4), (2, 5), (3, 4), (3, 5)]  # rows 2,3, cols 4,5
    C_3 = [(0, 4), (0, 5), (1, 4), (1, 5)]
    
    add_strassen_block(A_L, B_3_L, C_3)
    add_strassen_block(A_R, B_3_R, C_3)
    
    # Block 4: B column 6
    B_4_L = [(0, 6), (1, 6)]  # rows 0,1, col 6
    B_4_R = [(2, 6), (3, 6)]  # rows 2,3, col 6
    C_4 = [(0, 6), (1, 6)]
    
    add_schoolbook_221(A_L, B_4_L, C_4)
    add_schoolbook_221(A_R, B_4_R, C_4)
    
    # Verify col == rank
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition():
    """Verify the decomposition produces correct matrix multiplication."""
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    
    n, m, p = 2, 4, 7
    rank = U.shape[1]
    
    # Build the tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and check it equals the matrix multiplication tensor
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    C_computed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            total = 0.0
            for r in range(rank):
                for j in range(m):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
            C_computed[i, k] = total
    
    max_err = np.max(np.abs(C_computed - C_expected))
    assert max_err < 1e-10, f"Verification failed with max error {max_err}"
    return True


if __name__ == "__main__":
    _verify_decomposition()
    print("Verification passed!")
