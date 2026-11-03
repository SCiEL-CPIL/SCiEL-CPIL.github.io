"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[a*2+b, r], V[b*2+c, r], W[c*2+a, r]
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Product 1: M1 = (A11+A22)*(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Product 2: M2 = (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # Product 3: M3 = A11*(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Product 4: M4 = A22*(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Product 5: M5 = (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Product 6: M6 = (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Product 7: M7 = (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _laderman_3x3():
    """Return U, V, W for 3x3 matrix multiplication using Laderman's algorithm.
    U has shape (9, 15), V has shape (9, 15), W has shape (9, 15).
    Indexing: U[a*3+b, r], V[b*3+c, r], W[c*3+a, r]
    """
    U = [[0]*15 for _ in range(9)]
    V = [[0]*15 for _ in range(9)]
    W = [[0]*15 for _ in range(9)]
    
    # Product 0: A[0,0] * B[0,0] -> C[0,0]
    U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
    
    # Product 1: (A[0,0]+A[0,1]) * (B[0,1]-B[1,1]) -> C[0,1]
    U[0][1] = 1; U[1][1] = 1
    V[1][1] = 1; V[4][1] = -1
    W[1][1] = 1
    
    # Product 2: (A[0,0]+A[0,2]) * (B[0,2]-B[2,2]) -> C[0,2]
    U[0][2] = 1; U[2][2] = 1
    V[2][2] = 1; V[8][2] = -1
    W[2][2] = 1
    
    # Product 3: (A[0,1]+A[0,2]) * (B[1,2]-B[2,1]) -> C[0,2] (additional)
    U[1][3] = 1; U[2][3] = 1
    V[5][3] = 1; V[3][3] = -1
    W[2][3] = 1
    
    # Product 4: A[1,0] * B[0,0] -> C[1,0]
    U[3][4] = 1; V[0][4] = 1; W[3][4] = 1
    
    # Product 5: (A[1,0]+A[1,1]) * (B[0,1]-B[1,1]) -> C[1,1]
    U[3][5] = 1; U[4][5] = 1
    V[1][5] = 1; V[4][5] = -1
    W[4][5] = 1
    
    # Product 6: (A[1,0]+A[1,2]) * (B[0,2]-B[2,2]) -> C[1,2]
    U[3][6] = 1; U[5][6] = 1
    V[2][6] = 1; V[8][6] = -1
    W[5][6] = 1
    
    # Product 7: (A[1,1]+A[1,2]) * (B[1,2]-B[2,1]) -> C[1,2] (additional)
    U[4][7] = 1; U[5][7] = 1
    V[5][7] = 1; V[3][7] = -1
    W[5][7] = 1
    
    # Product 8: A[2,0] * B[0,0] -> C[2,0]
    U[6][8] = 1; V[0][8] = 1; W[6][8] = 1
    
    # Product 9: (A[2,0]+A[2,1]) * (B[0,1]-B[1,1]) -> C[2,1]
    U[6][9] = 1; U[7][9] = 1
    V[1][9] = 1; V[4][9] = -1
    W[7][9] = 1
    
    # Product 10: (A[2,0]+A[2,2]) * (B[0,2]-B[2,2]) -> C[2,2]
    U[6][10] = 1; U[8][10] = 1
    V[2][10] = 1; V[8][10] = -1
    W[8][10] = 1
    
    # Product 11: (A[2,1]+A[2,2]) * (B[1,2]-B[2,1]) -> C[2,2] (additional)
    U[7][11] = 1; U[8][11] = 1
    V[5][11] = 1; V[3][11] = -1
    W[8][11] = 1
    
    # Product 12: (A[0,0]+A[1,0]+A[2,0]) * (B[1,0]+B[2,0]) -> C[0,0]+C[1,0]+C[2,0]
    U[0][12] = 1; U[3][12] = 1; U[6][12] = 1
    V[3][12] = 1; V[6][12] = 1
    W[0][12] = 1; W[3][12] = 1; W[6][12] = 1
    
    # Product 13: (A[0,1]+A[1,1]+A[2,1]) * (B[1,0]+B[2,0]) -> C[0,1]+C[1,1]+C[2,1]
    U[1][13] = 1; U[4][13] = 1; U[7][13] = 1
    V[3][13] = 1; V[6][13] = 1
    W[1][13] = 1; W[4][13] = 1; W[7][13] = 1
    
    # Product 14: (A[0,2]+A[1,2]+A[2,2]) * (B[1,0]+B[2,0]) -> C[0,2]+C[1,2]+C[2,2]
    U[2][14] = 1; U[5][14] = 1; U[8][14] = 1
    V[3][14] = 1; V[6][14] = 1
    W[2][14] = 1; W[5][14] = 1; W[8][14] = 1
    
    return U, V, W

def _schoolbook_rect(n, m, p):
    """Return schoolbook decomposition for n×m times m×p multiplication.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _build_5x5_decomposition():
    """
    Build a 5x5 matrix multiplication decomposition using block structure.
    Partition indices into S={0,1,2} and T={3,4}.
    The tensor decomposes into 8 independent sub-tensors based on which
    partition a, b, c belong to.
    
    Total rank = sum of ranks of all 8 sub-tensors.
    """
    n = 5
    S = [0, 1, 2]  # size 3
    T = [3, 4]     # size 2
    
    # Sub-tensor specifications: (group_a, group_b, group_c) -> (n_a, n_b, n_c)
    # where n_a = len(group_a), n_b = len(group_b), n_c = len(group_c)
    # U dim = n_a * n_b, V dim = n_b * n_c, W dim = n_c * n_a
    
    blocks = [
        (S, S, S),  # (3,3,3) -> Laderman rank 15
        (S, S, T),  # (3,2,2) -> schoolbook rank 12
        (S, T, S),  # (2,3,3) -> schoolbook rank 18
        (S, T, T),  # (2,2,2) -> Strassen rank 7
        (T, S, S),  # (2,3,2) -> schoolbook rank 12
        (T, S, T),  # (2,2,2) -> Strassen rank 7
        (T, T, S),  # (2,2,3) -> schoolbook rank 12
        (T, T, T),  # (2,2,2) -> Strassen rank 7
    ]
    
    # Get decompositions for each block type
    def get_decomp(na, nb, nc):
        if na == 3 and nb == 3 and nc == 3:
            return _laderman_3x3()
        elif na == 2 and nb == 2 and nc == 2:
            return _strassen_2x2()
        else:
            return _schoolbook_rect(na, nb, nc)
    
    # Compute total rank
    total_rank = 0
    block_decomps = []
    for ga, gb, gc in blocks:
        na, nb, nc = len(ga), len(gb), len(gc)
        U_b, V_b, W_b = get_decomp(na, nb, nc)
        rank_b = len(U_b[0])
        total_rank += rank_b
        block_decomps.append((ga, gb, gc, na, nb, nc, U_b, V_b, W_b, rank_b))
    
    # Build full U, V, W matrices
    U = [[0]*total_rank for _ in range(n*n)]  # 25 rows
    V = [[0]*total_rank for _ in range(n*n)]  # 25 rows
    W = [[0]*total_rank for _ in range(n*n)]  # 25 rows
    
    col_offset = 0
    for ga, gb, gc, na, nb, nc, U_b, V_b, W_b, rank_b in block_decomps:
        # Map local indices to global indices
        # U row: a*5+b where a in ga, b in gb
        # V row: b*5+c where b in gb, c in gc
        # W row: c*5+a where c in gc, a in ga
        
        for i_local in range(na * nb):
            a_local = i_local // nb
            b_local = i_local % nb
            a_global = ga[a_local]
            b_global = gb[b_local]
            U_global_row = a_global * n + b_global
            
            for r in range(rank_b):
                U[U_global_row][col_offset + r] = U_b[i_local][r]
        
        for i_local in range(nb * nc):
            b_local = i_local // nc
            c_local = i_local % nc
            b_global = gb[b_local]
            c_global = gc[c_local]
            V_global_row = b_global * n + c_global
            
            for r in range(rank_b):
                V[V_global_row][col_offset + r] = V_b[i_local][r]
        
        for i_local in range(nc * na):
            c_local = i_local // na
            a_local = i_local % na
            c_global = gc[c_local]
            a_global = ga[a_local]
            W_global_row = c_global * n + a_global
            
            for r in range(rank_b):
                W[W_global_row][col_offset + r] = W_b[i_local][r]
        
        col_offset += rank_b
    
    return U, V, W

def _build_4x4_strassen():
    """
    Build a 4x4 matrix multiplication decomposition using Strassen's algorithm.
    Uses 7 recursive 2x2 Strassen multiplications.
    Returns U, V, W each of shape (16, 49) for rank 49.
    """
    n = 4
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Strassen 2x2 on 4x4:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # 
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2 product computed via Strassen (7 scalar products)
    # Total: 7 * 7 = 49 scalar products
    
    # Define the 7 Strassen combinations for 2x2 blocks
    # For each combination, we have coefficients for A-blocks and B-blocks
    
    # Strassen 2x2 formula:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # For 4x4, A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], etc.
    
    # Let's define helper to add a Strassen 2x2 decomposition at given offset
    def add_strassen_2x2(offset, a_blocks, b_blocks, c_blocks, sign_a, sign_b, sign_c):
        """
        Add a Strassen 2x2 decomposition.
        a_blocks: list of (row_start, row_end) for A subblocks
        b_blocks: list of (col_start, col_end) for B subblocks  
        c_blocks: list of (row_start, row_end) for C subblocks
        """
        # This is complex; let me build it directly
        
        pass
    
    # Direct construction:
    # For each of the 7 Strassen products, decompose into 7 scalar products
    
    # Strassen product 1: (A11+A22)*(B11+B22)
    # A11 = rows 0-1, cols 0-1 of A
    # A22 = rows 2-3, cols 2-3 of A
    # B11 = rows 0-1, cols 0-1 of B
    # B22 = rows 2-3, cols 2-3 of B
    
    # (A11+A22) has entries:
    # (0,0): A[0,0]+A[2,2]
    # (0,1): A[0,1]+A[2,3]
    # (1,0): A[1,0]+A[3,2]
    # (1,1): A[1,1]+A[3,3]
    
    # (B11+B22) has entries:
    # (0,0): B[0,0]+B[2,2]
    # (0,1): B[0,1]+B[2,3]
    # (1,0): B[1,0]+B[3,2]
    # (1,1): B[1,1]+B[3,3]
    
    # Product via Strassen 2x2 on these 2x2 matrices
    
    # Let me just use the recursive structure directly
    
    # Actually, let me build this more carefully using the tensor structure
    
    # The 4x4 multiplication tensor has rank at most 7^2 = 49 via Strassen
    
    # Let me construct it by composing two levels of Strassen
    
    # Level 1: 4x4 = 2x2 of 2x2 blocks
    # Level 2: each 2x2 block multiplication uses Strassen (rank 7)
    
    # Total rank = 7 * 7 = 49
    
    # Define the 7 Strassen combinations for the outer 2x2:
    outer_combos = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: dict mapping (block_row, block_col) -> coefficient
        # Each block is 2x2, so we need to track individual elements
        
        # M1 = (A11+A22)*(B11+B22)
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        
        # M2 = (A21+A22)*B11
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1}},
        
        # M3 = A11*(B12-B22)
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        
        # M4 = A22*(B21-B11)
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        
        # M5 = (A11+A12)*B22
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        
        # M6 = (A21-A11)*(B11+B12)
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        
        # M7 = (A12-A22)*(B21+B22)
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,0): 1}},
    ]
    
    # Inner Strassen 2x2 combinations (for computing each 2x2 product):
    inner_combos = [
        # M1 = (A11+A22)*(B11+B22)
        {'A': {(0,0): 1, (1,1): 1}, 'B': {(0,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,1): 1}},
        
        # M2 = (A21+A22)*B11
        {'A': {(1,0): 1, (1,1): 1}, 'B': {(0,0): 1}, 'C': {(1,0): 1}},
        
        # M3 = A11*(B12-B22)
        {'A': {(0,0): 1}, 'B': {(0,1): 1, (1,1): -1}, 'C': {(0,1): 1, (1,1): 1}},
        
        # M4 = A22*(B21-B11)
        {'A': {(1,1): 1}, 'B': {(1,0): 1, (0,0): -1}, 'C': {(0,0): 1, (1,0): 1}},
        
        # M5 = (A11+A12)*B22
        {'A': {(0,0): 1, (0,1): 1}, 'B': {(1,1): 1}, 'C': {(0,0): -1, (0,1): 1}},
        
        # M6 = (A21-A11)*(B11+B12)
        {'A': {(1,0): 1, (0,0): -1}, 'B': {(0,0): 1, (0,1): 1}, 'C': {(1,1): 1}},
        
        # M7 = (A12-A22)*(B21+B22)
        {'A': {(0,1): 1, (1,1): -1}, 'B': {(1,0): 1, (1,1): 1}, 'C': {(0,0): 1, (1,0): 1}},
    ]
    
    # Now build the full decomposition
    col = 0
    for outer_idx, outer in enumerate(outer_combos):
        for inner_idx, inner in enumerate(inner_combos):
            # For this combination, we need to find all (a,b,c) triples
            # where the coefficient is non-zero
            
            # The outer combination defines which 2x2 blocks of A and B are used
            # The inner combination defines which elements within those 2x2 blocks
            
            # A block (br, bc) corresponds to A elements:
            # (br*2 + i, bc*2 + j) for i,j in {0,1}
            
            # B block (br, bc) corresponds to B elements:
            # (br*2 + i, bc*2 + j) for i,j in {0,1}
            
            # C block (br, bc) corresponds to C elements:
            # (br*2 + i, bc*2 + j) for i,j in {0,1}
            
            # For the outer combo, A blocks are specified in outer['A']
            # For the inner combo, within each 2x2 block, elements are specified
            
            # This is getting complex. Let me take a different approach.
            
            pass
    
    # Simpler approach: directly enumerate all scalar products
    # For Strassen 4x4, we have 49 scalar products
    # Each corresponds to a specific combination of A, B, C elements
    
    # Let me build this by hand using the recursive structure
    
    # Actually, the cleanest way is to use the tensor product structure
    
    # The 4x4 multiplication tensor = tensor product of two 2x2 multiplication tensors
    # under the Strassen decomposition
    
    # Let me just construct it directly by enumerating
    
    # For each of the 7 outer Strassen products, and for each of the 7 inner
    # Strassen products, we get one scalar product
    
    # The scalar product involves:
    # - A elements: determined by outer A blocks and inner A elements
    # - B elements: determined by outer B blocks and inner B elements  
    # - C elements: determined by outer C blocks and inner C elements
    
    # Let me define helper functions
    
    def get_block_elements(block_row, block_col, combo, matrix_type):
        """Get the elements of a 2x2 block according to the Strassen combination."""
        if matrix_type == 'A':
            coeffs = combo['A']
        elif matrix_type == 'B':
            coeffs = combo['B']
        else:
            coeffs = combo['C']
        
        # The block is at position (block_row, block_col) in the 2x2 block structure
        # But the Strassen combination specifies which blocks are used
        # and their coefficients
        
        # Actually, the Strassen combination specifies linear combinations of blocks
        # We need to figure out which individual elements contribute
        
        # For a 2x2 block at position (br, bc):
        # Element (i,j) in the block corresponds to global position (br*2+i, bc*2+j)
        
        # The Strassen combination for this matrix type gives coefficients for blocks
        # We need to combine with the inner combination
        
        # This is still complex. Let me just hardcode the 49 products.
        
        pass
    
    # Given the complexity, let me use a different strategy:
    # Build the 4x4 decomposition by composing two 2x2 Strassen decompositions
    
    # The key insight: if we have U1, V1, W1 for 2x2 Strassen (rank 7),
    # then for 4x4, we can use U1 ⊗ U1, V1 ⊗ V1, W1 ⊗ W1 (Kronecker product)
    # but with appropriate index mappings
    
    # Actually, the tensor product of two Strassen decompositions gives
    # a decomposition of rank 7*7 = 49 for 4x4 multiplication
    
    # Let me implement this using Kronecker products
    
    U2, V2, W2 = _strassen_2x2()
    U2 = np.array(U2, dtype=np.int64)
    V2 = np.array(V2, dtype=np.int64)
    W2 = np.array(W2, dtype=np.int64)
    
    # The 4x4 multiplication can be seen as:
    # A (4x4) = [[A11, A12], [A21, A22]] where each Ai is 2x2
    # B (4x4) = [[B11, B12], [B21, B22]] where each Bi is 2x2
    # C (4x4) = A * B
    
    # Using Strassen on the 2x2 block structure:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Where M1 = (A11+A22)*(B11+B22), etc.
    
    # Each Mi is a 2x2 product, computed via Strassen (rank 7)
    
    # So the total decomposition has rank 7*7 = 49
    
    # To build this, I need to:
    # 1. For each outer Strassen product Mi, determine which A and B blocks are used
    # 2. For each such product, apply the inner Strassen decomposition
    # 3. Combine with the C block contributions
    
    # This is essentially a tensor product construction
    
    # Let me implement it step by step
    
    # Outer Strassen products:
    # M1: A blocks {11, 22}, B blocks {11, 22}, C blocks {11, 22}
    # M2: A blocks {21, 22}, B blocks {11}, C blocks {21}
    # M3: A blocks {11}, B blocks {12, 22}, C blocks {12, 22}
    # M4: A blocks {22}, B blocks {21, 11}, C blocks {11, 21}
    # M5: A blocks {11, 12}, B blocks {22}, C blocks {11, 12}
    # M6: A blocks {21, 11}, B blocks {11, 12}, C blocks {22}
    # M7: A blocks {12, 22}, B blocks {21, 22}, C blocks {11, 21}
    
    # For each Mi, the A and B blocks are linear combinations
    # Then each Mi is a 2x2 product computed via Strassen
    
    # Let me define the outer combinations more precisely
    
    outer_defs = [
        # (A_block_coeffs, B_block_coeffs, C_block_coeffs)
        # Block (i,j) means A[i*2:(i+1)*2, j*2:(j+1)*2]
        # Coefficients are for the linear combination
        
        # M1 = (A11+A22)*(B11+B22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        
        # M2 = (A21+A22)*B11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        
        # M3 = A11*(B12-B22)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        
        # M4 = A22*(B21-B11)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        
        # M5 = (A11+A12)*B22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        
        # M6 = (A21-A11)*(B11+B12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        
        # M7 = (A12-A22)*(B21+B22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,0): 1}),
    ]
    
    # For each outer product, we compute a 2x2 product using inner Strassen
    # The inner Strassen uses the same 7 combinations
    
    # For a given outer product with A blocks {a1, a2, ...} and B blocks {b1, b2, ...}
    # The effective A matrix is a 2x2 matrix where each entry is a linear combination
    # of 2x2 blocks. Similarly for B.
    
    # Then we apply inner Strassen to this 2x2 "block matrix"
    
    # This is getting very complex to implement correctly. Let me use a different
    # approach: directly construct the 49 scalar products by enumeration.
    
    # Alternative: use the fact that Strassen 4x4 has rank 49, and construct
    # it by composing the 2x2 Strassen decomposition with itself.
    
    # The Kronecker product approach:
    # If T2 is the 2x2 multiplication tensor with Strassen decomposition
    # U2, V2, W2 (rank 7), then the 4x4 multiplication tensor T4 can be
    # decomposed using the tensor product T2 ⊗ T2
    
    # But T4 is not exactly T2 ⊗ T2; it's a "block" version
    
    # Let me just implement the direct enumeration approach
    
    # For each of the 7 outer Strassen products:
    #   For each of the 7 inner Strassen products:
    #     Determine the scalar product
    
    # The scalar product is:
    # sum over a_elements, b_elements, c_elements of
    #   coeff_a * coeff_b * coeff_c * A[a] * B[b] * C[c]
    
    # Where the coefficients come from the outer and inner Strassen combinations
    
    # Let me define this more carefully
    
    # For outer product k and inner product l:
    # The A part: outer A blocks combined with inner A elements
    # The B part: outer B blocks combined with inner B elements
    # The C part: outer C blocks combined with inner C elements
    
    # For a specific (a, b, c) triple in the 4x4 matrices:
    # a = (a_block_row, a_block_col, a_inner_row, a_inner_col)
    # where a_block_row, a_block_col in {0,1} and a_inner_row, a_inner_col in {0,1}
    # Global position: (a_block_row*2 + a_inner_row, a_block_col*2 + a_inner_col)
    
    # The coefficient for A[a] in outer product k, inner product l is:
    #   outer_A_coeff[k][a_block] * inner_A_coeff[l][a_inner]
    # where a_block = (a_block_row, a_block_col) and a_inner = (a_inner_row, a_inner_col)
    
    # Similarly for B and C
    
    # So the full coefficient for the scalar product involving
    # A[a], B[b], C[c] is:
    #   outer_A_coeff[k][a_block] * inner_A_coeff[l][a_inner] *
    #   outer_B_coeff[k][b_block] * inner_B_coeff[l][b_inner] *
    #   outer_C_coeff[k][c_block] * inner_C_coeff[l][c_inner]
    
    # Let me implement this
    
    # First, define the inner Strassen coefficients
    # inner_A_coeffs[l] is a dict mapping (row, col) -> coeff for the l-th inner product
    # where (row, col) is in {0,1}x{0,1}
    
    inner_A = [
        {(0,0): 1, (1,1): 1},  # M1
        {(1,0): 1, (1,1): 1},  # M2
        {(0,0): 1},             # M3
        {(1,1): 1},             # M4
        {(0,0): 1, (0,1): 1},  # M5
        {(1,0): 1, (0,0): -1}, # M6
        {(0,1): 1, (1,1): -1}, # M7
    ]
    
    inner_B = [
        {(0,0): 1, (1,1): 1},  # M1
        {(0,0): 1},             # M2
        {(0,1): 1, (1,1): -1}, # M3
        {(1,0): 1, (0,0): -1}, # M4
        {(1,1): 1},             # M5
        {(0,0): 1, (0,1): 1},  # M6
        {(1,0): 1, (1,1): 1},  # M7
    ]
    
    inner_C = [
        {(0,0): 1, (1,1): 1},  # M1
        {(1,0): 1},             # M2
        {(0,1): 1, (1,1): 1},  # M3
        {(0,0): 1, (1,0): 1},  # M4
        {(0,0): -1, (0,1): 1}, # M5
        {(1,1): 1},             # M6
        {(0,0): 1, (1,0): 1},  # M7
    ]
    
    # Outer coefficients (same structure as inner, but for 4x4 blocks)
    outer_A = [
        {(0,0): 1, (1,1): 1},  # M1
        {(1,0): 1, (1,1): 1},  # M2
        {(0,0): 1},             # M3
        {(1,1): 1},             # M4
        {(0,0): 1, (0,1): 1},  # M5
        {(1,0): 1, (0,0): -1}, # M6
        {(0,1): 1, (1,1): -1}, # M7
    ]
    
    outer_B = [
        {(0,0): 1, (1,1): 1},  # M1
        {(0,0): 1},             # M2
        {(0,1): 1, (1,1): -1}, # M3
        {(1,0): 1, (0,0): -1}, # M4
        {(1,1): 1},             # M5
        {(0,0): 1, (0,1): 1},  # M6
        {(1,0): 1, (1,1): 1},  # M7
    ]
    
    outer_C = [
        {(0,0): 1, (1,1): 1},  # M1
        {(1,0): 1},             # M2
        {(0,1): 1, (1,1): 1},  # M3
        {(0,0): 1, (1,0): 1},  # M4
        {(0,0): -1, (0,1): 1}, # M5
        {(1,1): 1},             # M6
        {(0,0): 1, (1,0): 1},  # M7
    ]
    
    # Now build the 49 scalar products
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    col = 0
    for k in range(7):  # outer product index
        for l in range(7):  # inner product index
            # For this (k, l), determine which (a, b, c) triples have non-zero coefficient
            
            # A elements:
            # a_block in outer_A[k], a_inner in inner_A[l]
            # Global a = (a_block_row*2 + a_inner_row, a_block_col*2 + a_inner_col)
            
            # B elements:
            # b_block in outer_B[k], b_inner in inner_B[l]
            # Global b = (b_block_row*2 + b_inner_row, b_block_col*2 + b_inner_col)
            
            # C elements:
            # c_block in outer_C[k], c_inner in inner_C[l]
            # Global c = (c_block_row*2 + c_inner_row, c_block_col*2 + c_inner_col)
            
            # For each combination of a, b, c, add the scalar product
            
            for a_block, a_coeff_outer in outer_A[k].items():
                for a_inner, a_coeff_inner in inner_A[l].items():
                    a_global_row = a_block[0]*2 + a_inner[0]
                    a_global_col = a_block[1]*2 + a_inner[1]
                    a_idx = a_global_row * 4 + a_global_col
                    a_coeff = a_coeff_outer * a_coeff_inner
                    
                    for b_block, b_coeff_outer in outer_B[k].items():
                        for b_inner, b_coeff_inner in inner_B[l].items():
                            b_global_row = b_block[0]*2 + b_inner[0]
                            b_global_col = b_block[1]*2 + b_inner[1]
                            b_idx = b_global_row * 4 + b_global_col
                            b_coeff = b_coeff_outer * b_coeff_inner
                            
                            for c_block, c_coeff_outer in outer_C[k].items():
                                for c_inner, c_coeff_inner in inner_C[l].items():
                                    c_global_row = c_block[0]*2 + c_inner[0]
                                    c_global_col = c_block[1]*2 + c_inner[1]
                                    c_idx = c_global_row * 4 + c_global_col
                                    c_coeff = c_coeff_outer * c_coeff_inner
                                    
                                    # Add to U, V, W
                                    if a_coeff != 0 and b_coeff != 0 and c_coeff != 0:
                                        U[a_idx][col] += a_coeff
                                        V[b_idx][col] += b_coeff
                                        W[c_idx][col] += c_coeff
            
            col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Verify tensor decomposition exactly (for integer/half-integer entries)."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    T_target_8 = target * 8
    if np.max(np.abs(T_computed - T_target_8)) > 1e-6:
        return False
    return True

def _build_5x5_with_4x4():
    """
    Build a 5x5 matrix multiplication decomposition using 4x4 Strassen blocks.
    Partition indices into S={0,1,2,3} and T={4}.
    
    Blocks:
    (S,S,S): 4x4x4 -> Strassen rank 49
    (S,S,T): 4x4x1 -> schoolbook rank 16
    (S,T,S): 4x1x4 -> schoolbook rank 16
    (S,T,T): 4x1x1 -> schoolbook rank 4
    (T,S,S): 1x4x4 -> schoolbook rank 16
    (T,S,T): 1x4x1 -> schoolbook rank 4
    (T,T,S): 1x1x4 -> schoolbook rank 4
    (T,T,T): 1x1x1 -> rank 1
    
    Total rank = 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1 = 110
    """
    n = 5
    S = [0, 1, 2, 3]  # size 4
    T = [4]           # size 1
    
    blocks = [
        (S, S, S),  # (4,4,4) -> Strassen rank 49
        (S, S, T),  # (4,4,1) -> schoolbook rank 16
        (S, T, S),  # (4,1,4) -> schoolbook rank 16
        (S, T, T),  # (4,1,1) -> schoolbook rank 4
        (T, S, S),  # (1,4,4) -> schoolbook rank 16
        (T, S, T),  # (1,4,1) -> schoolbook rank 4
        (T, T, S),  # (1,1,4) -> schoolbook rank 4
        (T, T, T),  # (1,1,1) -> rank 1
    ]
    
    def get_decomp(na, nb, nc):
        if na == 4 and nb == 4 and nc == 4:
            return _build_4x4_strassen()
        else:
            return _schoolbook_rect(na, nb, nc)
    
    total_rank = 0
    block_decomps = []
    for ga, gb, gc in blocks:
        na, nb, nc = len(ga), len(gb), len(gc)
        U_b, V_b, W_b = get_decomp(na, nb, nc)
        rank_b = len(U_b[0])
        total_rank += rank_b
        block_decomps.append((ga, gb, gc, na, nb, nc, U_b, V_b, W_b, rank_b))
    
    U = [[0]*total_rank for _ in range(n*n)]
    V = [[0]*total_rank for _ in range(n*n)]
    W = [[0]*total_rank for _ in range(n*n)]
    
    col_offset = 0
    for ga, gb, gc, na, nb, nc, U_b, V_b, W_b, rank_b in block_decomps:
        for i_local in range(na * nb):
            a_local = i_local // nb
            b_local = i_local % nb
            a_global = ga[a_local]
            b_global = gb[b_local]
            U_global_row = a_global * n + b_global
            
            for r in range(rank_b):
                U[U_global_row][col_offset + r] = U_b[i_local][r]
        
        for i_local in range(nb * nc):
            b_local = i_local // nc
            c_local = i_local % nc
            b_global = gb[b_local]
            c_global = gc[c_local]
            V_global_row = b_global * n + c_global
            
            for r in range(rank_b):
                V[V_global_row][col_offset + r] = V_b[i_local][r]
        
        for i_local in range(nc * na):
            c_local = i_local // na
            a_local = i_local % na
            c_global = gc[c_local]
            a_global = ga[a_local]
            W_global_row = c_global * n + a_global
            
            for r in range(rank_b):
                W[W_global_row][col_offset + r] = W_b[i_local][r]
        
        col_offset += rank_b
    
    return U, V, W

def construct():
    deadline = time.time() + 135.0  # 15s safety margin
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try the 3+2 block decomposition
    if time.time() < deadline - 5:
        U_bd, V_bd, W_bd = _build_5x5_decomposition()
        rank_bd = len(U_bd[0])
        
        if _verify(U_bd, V_bd, W_bd, n, m, p, rank_bd):
            print(f"3+2 Block decomposition verified, rank {rank_bd}")
            if rank_bd < best_rank:
                best_rank = rank_bd
                best_result = {"U": U_bd, "V": V_bd, "W": W_bd}
        else:
            print(f"3+2 Block decomposition FAILED verification, rank {rank_bd}")
    
    # Try the 4+1 block decomposition
    if time.time() < deadline - 5:
        U_41, V_41, W_41 = _build_5x5_with_4x4()
        rank_41 = len(U_41[0])
        
        if _verify(U_41, V_41, W_41, n, m, p, rank_41):
            print(f"4+1 Block decomposition verified, rank {rank_41}")
            if rank_41 < best_rank:
                best_rank = rank_41
                best_result = {"U": U_41, "V": V_41, "W": W_41}
        else:
            print(f"4+1 Block decomposition FAILED verification, rank {rank_41}")
    
    return best_result
# EVOLVE-BLOCK-END
