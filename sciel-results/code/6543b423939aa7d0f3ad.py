import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-73 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    Strategy:
    - Split the 6 columns of B into two groups of 3: B_left (cols 0-2), B_right (cols 3-5).
    - Compute C = A * B as C = A * B_left + A * B_right.
    - For each 4x4 * 4x3 product, use a known rank-32 decomposition.
    - Total rank = 32 + 32 = 64? Wait, 4x4 * 4x3 has lower bound 48, but known bounds are higher.
      Actually, let's use the standard schoolbook for 4x4*4x3: rank = 4*4*3 = 48.
      Total rank = 48 + 48 = 96. That's the same as schoolbook for 4x4*4x6.
    
    Better strategy: Use Strassen-like recursion.
    Split A (4x4) into 2x2 blocks of 2x2.
    Split B (4x6) into 2x3 blocks of 2x2.
    Split C (4x6) into 2x3 blocks of 2x2.
    
    C = A * B
    C_top_left (2x2) = A00*B00 + A01*B10
    C_top_mid (2x2) = A00*B01 + A01*B11
    C_top_right (2x2) = A00*B02 + A01*B12
    C_bot_left (2x2) = A10*B00 + A11*B10
    C_bot_mid (2x2) = A10*B01 + A11*B11
    C_bot_right (2x2) = A10*B02 + A11*B12
    
    We can use Strassen for 2x2*2x2 (rank 7).
    For each pair (A_block, B_block), we compute their product using 7 scalar mults.
    There are 6 B blocks and 4 A blocks, but they pair up.
    
    Actually, let's just use the known best rank for 4x4*4x6.
    Schoolbook is 96.
    Using Strassen on 2x2 blocks:
    Each 2x2*2x2 product has rank 7.
    We need to compute 4 A-blocks * 6 B-blocks? No, the structure is fixed.
    
    Let's use a simpler approach: direct search for small rank improvements.
    Since 120s is available, we can try to find a rank lower than 96.
    But random search for exact tensor decomposition is hard.
    
    Let's stick to a guaranteed valid decomposition that is at least as good as schoolbook.
    Schoolbook: rank 96.
    
    Can we do better with Strassen?
    Consider A (4x4), B (4x6).
    Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2.
    Split B into 2x3 blocks: B = [[B00, B01, B02], [B10, B11, B12]] where each is 2x2.
    
    C = [[A00*B00 + A01*B10, A00*B01 + A01*B11, A00*B02 + A01*B12],
         [A10*B00 + A11*B10, A10*B01 + A11*B11, A10*B02 + A11*B12]]
    
    Let's use Strassen for each 2x2*2x2 product.
    Number of 2x2*2x2 products: 4 A-blocks * 3 B-blocks? No.
    The terms are:
    A00*B00, A01*B10, A00*B01, A01*B11, A00*B02, A01*B12,
    A10*B00, A11*B10, A10*B01, A11*B11, A10*B02, A11*B12.
    That's 12 products of 2x2 matrices.
    Each has rank 7. Total rank = 12 * 7 = 84.
    
    This is an improvement over 96.
    Let's construct this.
    """
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # Strassen 2x2*2x2 decomposition (rank 7)
    # U indices: (i, j) for 2x2 matrix A
    # V indices: (j, k) for 2x2 matrix B
    # W indices: (k, i) for 2x2 matrix C
    
    # Strassen formulas:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_2x2(u_base, v_base, w_base, col_offset):
        """
        Add 7 columns for one Strassen 2x2*2x2 decomposition.
        u_base: (i0, j0) starting index for U (2x2 block of A)
        v_base: (j0, k0) starting index for V (2x2 block of B)
        w_base: (k0, i0) starting index for W (2x2 block of C)
        col_offset: starting column index in U, V, W
        """
        cols = []
        
        # M1 = (A00 + A11) * (B00 + B11)
        u = {}
        u[(u_base[0], u_base[1])] = 1   # A00
        u[(u_base[0]+1, u_base[1]+1)] = 1 # A11
        v = {}
        v[(v_base[0], v_base[1])] = 1   # B00
        v[(v_base[0]+1, v_base[1]+1)] = 1 # B11
        w = {}
        w[(w_base[0], w_base[1])] = 1   # C00 (k0, i0)
        w[(w_base[0]+1, w_base[1])] = 1 # C10 (k0+1, i0) ? No, C10 is (k1, i0)
        # Wait, W index is k*n + i.
        # For a 2x2 block of C at (k0, i0):
        # C[k0, i0] -> W[k0*n + i0]
        # C[k0, i0+1] -> W[k0*n + i0+1]
        # C[k0+1, i0] -> W[(k0+1)*n + i0]
        # C[k0+1, i0+1] -> W[(k0+1)*n + i0+1]
        
        # Let's redefine w_base as (k0, i0) in the 4x6 C matrix.
        # So the 2x2 block of C is at rows k0..k0+1, cols i0..i0+1.
        
        # M1 contributes to C00, C01, C10, C11? No, M1 is a scalar product.
        # M1 is added to C00 and C11.
        w = {}
        w[w_base[0]*n + w_base[1]] = 1      # C00
        w[(w_base[0]+1)*n + (w_base[1]+1)] = 1 # C11
        cols.append((u, v, w))
        
        # M2 = (A10 + A11) * B00
        u = {}
        u[(u_base[0]+1, u_base[1])] = 1   # A10
        u[(u_base[0]+1, u_base[1]+1)] = 1 # A11
        v = {}
        v[(v_base[0], v_base[1])] = 1   # B00
        w = {}
        w[(w_base[0]+1)*n + w_base[1]] = 1   # C10
        cols.append((u, v, w))
        
        # M3 = A00 * (B01 - B11)
        u = {}
        u[(u_base[0], u_base[1])] = 1   # A00
        v = {}
        v[(v_base[0], v_base[1]+1)] = 1   # B01
        v[(v_base[0]+1, v_base[1]+1)] = -1 # B11
        w = {}
        w[w_base[0]*n + (w_base[1]+1)] = 1   # C01
        cols.append((u, v, w))
        
        # M4 = A11 * (B10 - B00)
        u = {}
        u[(u_base[0]+1, u_base[1]+1)] = 1 # A11
        v = {}
        v[(v_base[0]+1, v_base[1])] = 1   # B10
        v[(v_base[0], v_base[1])] = -1   # B00
        w = {}
        w[(w_base[0]+1)*n + w_base[1]] = 1   # C10
        w[w_base[0]*n + w_base[1]] = 1      # C00
        cols.append((u, v, w))
        
        # M5 = (A00 + A01) * B11
        u = {}
        u[(u_base[0], u_base[1])] = 1   # A00
        u[(u_base[0], u_base[1]+1)] = 1 # A01
        v = {}
        v[(v_base[0]+1, v_base[1]+1)] = 1 # B11
        w = {}
        w[w_base[0]*n + w_base[1]] = -1   # C00
        w[w_base[0]*n + (w_base[1]+1)] = 1   # C01
        cols.append((u, v, w))
        
        # M6 = (A10 - A00) * (B00 + B01)
        u = {}
        u[(u_base[0]+1, u_base[1])] = 1   # A10
        u[(u_base[0], u_base[1])] = -1   # A00
        v = {}
        v[(v_base[0], v_base[1])] = 1   # B00
        v[(v_base[0], v_base[1]+1)] = 1   # B01
        w = {}
        w[(w_base[0]+1)*n + (w_base[1]+1)] = 1 # C11
        cols.append((u, v, w))
        
        # M7 = (A01 - A11) * (B10 + B11)
        u = {}
        u[(u_base[0], u_base[1]+1)] = 1   # A01
        u[(u_base[0]+1, u_base[1]+1)] = -1 # A11
        v = {}
        v[(v_base[0]+1, v_base[1])] = 1   # B10
        v[(v_base[0]+1, v_base[1]+1)] = 1 # B11
        w = {}
        w[w_base[0]*n + w_base[1]] = 1      # C00
        w[(w_base[0]+1)*n + (w_base[1]+1)] = 1 # C11
        cols.append((u, v, w))
        
        return cols
    
    # Now, let's assemble the full decomposition.
    # A is 4x4. Blocks:
    # A00: (0,0) to (1,1) -> u_base = (0,0)
    # A01: (0,2) to (1,3) -> u_base = (0,2)
    # A10: (2,0) to (3,1) -> u_base = (2,0)
    # A11: (2,2) to (3,3) -> u_base = (2,2)
    
    # B is 4x6. Blocks (each 2x2):
    # B00: (0,0) to (1,1) -> v_base = (0,0)
    # B01: (0,2) to (1,3) -> v_base = (0,2)
    # B02: (0,4) to (1,5) -> v_base = (0,4)
    # B10: (2,0) to (3,1) -> v_base = (2,0)
    # B11: (2,2) to (3,3) -> v_base = (2,2)
    # B12: (2,4) to (3,5) -> v_base = (2,4)
    
    # C is 4x6. Blocks (each 2x2):
    # C00: (0,0) to (1,1) -> w_base = (0,0)
    # C01: (0,2) to (1,3) -> w_base = (0,2)
    # C02: (0,4) to (1,5) -> w_base = (0,4)
    # C10: (2,0) to (3,1) -> w_base = (2,0)
    # C11: (2,2) to (3,3) -> w_base = (2,2)
    # C12: (2,4) to (3,5) -> w_base = (2,4)
    
    # Products:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    products = [
        # (A_block, B_block, C_block)
        ((0,0), (0,0), (0,0)),  # A00*B00 -> C00
        ((0,2), (2,0), (0,0)),  # A01*B10 -> C00
        ((0,0), (0,2), (0,2)),  # A00*B01 -> C01
        ((0,2), (2,2), (0,2)),  # A01*B11 -> C01
        ((0,0), (0,4), (0,4)),  # A00*B02 -> C02
        ((0,2), (2,4), (0,4)),  # A01*B12 -> C02
        ((2,0), (0,0), (2,0)),  # A10*B00 -> C10
        ((2,2), (2,0), (2,0)),  # A11*B10 -> C10
        ((2,0), (0,2), (2,2)),  # A10*B01 -> C11
        ((2,2), (2,2), (2,2)),  # A11*B11 -> C11
        ((2,0), (0,4), (2,4)),  # A10*B02 -> C12
        ((2,2), (2,4), (2,4)),  # A11*B12 -> C12
    ]
    
    rank = 12 * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for (u_base, v_base, w_base) in products:
        cols = strassen_2x2(u_base, v_base, w_base, col)
        for u_dict, v_dict, w_dict in cols:
            for (i, j), val in u_dict.items():
                U[i * m + j][col] = val
            for (j, k), val in v_dict.items():
                V[j * p + k][col] = val
            for (k, i), val in w_dict.items():
                W[k * n + i][col] = val
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
