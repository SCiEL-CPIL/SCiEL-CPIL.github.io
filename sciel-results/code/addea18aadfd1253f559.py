"""
Rewritten program for 2x4x5 matrix multiplication tensor decomposition.
Uses Winograd/Fourier-based approach: decompose into 2x2x2 blocks (Strassen)
and 2x2x1 blocks (schoolbook), but with careful index mapping.
"""

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    
    # Total rank: 2 blocks of (2 Strassen 2x2x2 + 1 schoolbook 2x2x1)
    # Each 2x2x2 Strassen: rank 7
    # Each 2x2x1 schoolbook: rank 4
    # Per block: 7 + 7 + 4 = 18
    # Two blocks: 36
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    col = 0
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        """Add one rank-1 term defined by sparse dictionaries."""
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1
    
    def add_strassen_2x2x2(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        """
        Add Strassen decomposition for 2x2 * 2x2 block.
        
        Parameters:
        - a_base_i, a_base_j: global (row, col) start of A block
        - b_base_i, b_base_j: global (row, col) start of B block
        - c_base_i, c_base_j: global (row, col) start of C block
        
        Indexing:
        - A index: i * m + j
        - B index: j * p + k
        - W index (C output): k * n + i
        """
        def map_a(local_i, local_j):
            return (a_base_i + local_i) * m + (a_base_j + local_j)
        
        def map_b(local_i, local_j):
            return (b_base_i + local_i) * p + (b_base_j + local_j)
        
        def map_c(local_i, local_j):
            # C[local_i, local_j] -> global row = c_base_i + local_i, col = c_base_j + local_j
            # W index: k * n + i, where i is row of C, k is col of C
            i_global = c_base_i + local_i
            k_global = c_base_j + local_j
            return k_global * n + i_global
        
        # Strassen 7 multiplications for 2x2 * 2x2
        # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
        add_rank1_term(
            {map_a(0, 0): 1, map_a(1, 1): 1},
            {map_b(0, 0): 1, map_b(1, 1): 1},
            {map_c(0, 0): 1, map_c(1, 1): 1}
        )
        
        # M2 = (A10 + A11)B00 -> C10 += M2, C11 -= M2
        add_rank1_term(
            {map_a(1, 0): 1, map_a(1, 1): 1},
            {map_b(0, 0): 1},
            {map_c(1, 0): 1, map_c(1, 1): -1}
        )
        
        # M3 = A00(B01 - B11) -> C01 += M3, C11 += M3
        add_rank1_term(
            {map_a(0, 0): 1},
            {map_b(0, 1): 1, map_b(1, 1): -1},
            {map_c(0, 1): 1, map_c(1, 1): 1}
        )
        
        # M4 = A11(B10 - B00) -> C00 += M4, C10 += M4
        add_rank1_term(
            {map_a(1, 1): 1},
            {map_b(1, 0): 1, map_b(0, 0): -1},
            {map_c(0, 0): 1, map_c(1, 0): 1}
        )
        
        # M5 = (A00 + A01)B11 -> C00 -= M5, C01 += M5
        add_rank1_term(
            {map_a(0, 0): 1, map_a(0, 1): 1},
            {map_b(1, 1): 1},
            {map_c(0, 0): -1, map_c(0, 1): 1}
        )
        
        # M6 = (A10 - A00)(B00 + B01) -> C11 += M6
        add_rank1_term(
            {map_a(1, 0): 1, map_a(0, 0): -1},
            {map_b(0, 0): 1, map_b(0, 1): 1},
            {map_c(1, 1): 1}
        )
        
        # M7 = (A01 - A11)(B10 + B11) -> C00 += M7
        add_rank1_term(
            {map_a(0, 1): 1, map_a(1, 1): -1},
            {map_b(1, 0): 1, map_b(1, 1): 1},
            {map_c(0, 0): 1}
        )
    
    def add_schoolbook_2x2x1(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        """
        Add schoolbook decomposition for 2x2 * 2x1 block.
        C[i, k] = sum_{j=0}^{1} A[i, j] * B[j, k]
        where k is fixed (1 column).
        
        Parameters:
        - a_base_i, a_base_j: global (row, col) start of A block (2x2)
        - b_base_i, b_base_j: global (row, col) start of B block (2x1)
        - c_base_i, c_base_j: global (row, col) start of C block (2x1)
        """
        def map_a(local_i, local_j):
            return (a_base_i + local_i) * m + (a_base_j + local_j)
        
        def map_b(local_i, local_j):
            return (b_base_i + local_i) * p + (b_base_j + local_j)
        
        def map_c(local_i, local_j):
            i_global = c_base_i + local_i
            k_global = c_base_j + local_j
            return k_global * n + i_global
        
        # 4 terms: for each i in {0,1}, j in {0,1}
        for i in range(2):
            for j in range(2):
                add_rank1_term(
                    {map_a(i, j): 1},
                    {map_b(j, 0): 1},
                    {map_c(i, 0): 1}
                )
    
    # Block 1: A cols 0,1; B rows 0,1
    # Split B cols into {0,1}, {2,3}, {4}
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)  # A[0:2,0:2] * B[0:2,0:2]
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2)  # A[0:2,0:2] * B[0:2,2:4]
    add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4)  # A[0:2,0:2] * B[0:2,4:5]
    
    # Block 2: A cols 2,3; B rows 2,3
    # Split B cols into {0,1}, {2,3}, {4}
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0)  # A[0:2,2:4] * B[2:4,0:2]
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2)  # A[0:2,2:4] * B[2:4,2:4]
    add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4)  # A[0:2,2:4] * B[2:4,4:5]
    
    return {"U": U, "V": V, "W": W}
