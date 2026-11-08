import time
import random

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """
    Return the 7 rank-1 terms for 2x2*2x2 matrix multiplication.
    Each term is (u_dict, v_dict, w_dict) where each dict maps flat index to coefficient.
    U index: i*2+j for element (i,j) of X (row-major)
    V index: j*2+k for element (j,k) of Y (row-major)
    W index: k*2+i for element (i,k) of Z (trace convention: col*dim+row)
    
    Strassen's 7 products:
    M1 = (x00+x11)(y00+y11)
    M2 = (x10+x11)y00
    M3 = x00(y01-y11)
    M4 = x11(y10-y00)
    M5 = (x00+x01)y11
    M6 = (x10+x11)(y00+y01)
    M7 = (x00-x10)(y11+y10)
    
    Result:
    z00 = M1 + M4 - M5 + M7
    z01 = M3 + M5
    z10 = M2 + M4
    z11 = M1 - M2 + M3 + M6
    """
    terms = []
    
    # M1: (x00+x11)(y00+y11)
    # U: x00=1, x11=1 -> indices 0, 3
    # V: y00=1, y11=1 -> indices 0, 3
    # W: z00=1, z11=1 -> indices 0*2+0=0, 1*2+1=3
    terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
    
    # M2: (x10+x11)y00
    # U: x10=1, x11=1 -> indices 2, 3
    # V: y00=1 -> index 0
    # W: z10=1 -> index 1*2+0=2
    terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1}))
    
    # M3: x00(y01-y11)
    # U: x00=1 -> index 0
    # V: y01=1, y11=-1 -> indices 1, 3
    # W: z01=1, z11=1 -> indices 0*2+1=1, 1*2+1=3
    terms.append(({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}))
    
    # M4: x11(y10-y00)
    # U: x11=1 -> index 3
    # V: y10=1, y00=-1 -> indices 2, 0
    # W: z00=1, z10=1 -> indices 0*2+0=0, 1*2+0=2
    terms.append(({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}))
    
    # M5: (x00+x01)y11
    # U: x00=1, x01=1 -> indices 0, 1
    # V: y11=1 -> index 3
    # W: z00=-1, z01=1 -> indices 0*2+0=0, 0*2+1=1
    terms.append(({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}))
    
    # M6: (x10+x11)(y00+y01)
    # U: x10=1, x11=1 -> indices 2, 3
    # V: y00=1, y01=1 -> indices 0, 1
    # W: z11=1 -> index 1*2+1=3
    terms.append(({2: 1, 3: 1}, {0: 1, 1: 1}, {3: 1}))
    
    # M7: (x00-x10)(y11+y10)
    # U: x00=1, x10=-1 -> indices 0, 2
    # V: y11=1, y10=1 -> indices 3, 2
    # W: z00=1 -> index 0*2+0=0
    terms.append(({0: 1, 2: -1}, {3: 1, 2: 1}, {0: 1}))
    
    return terms


def _strassen_4x4():
    """
    Return a rank-49 decomposition for 4x4 matrix multiplication using
    tensor product of two Strassen 2x2 decompositions.
    
    The 4x4 matrices are viewed as 2x2 block matrices of 2x2 blocks.
    We use Strassen's algorithm for the 2x2 block structure (7 block products),
    and each block product is itself a 2x2*2x2 matrix product decomposed
    using Strassen's algorithm (7 scalar products).
    
    Total rank = 7 * 7 = 49.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 block-level linear combinations for A and B.
    # Blocks: 0=A11, 1=A12, 2=A21, 3=A22 (and similarly for B)
    # Format: list of 7 dicts mapping block_idx to coeff
    A_combos = [
        {0: 1, 3: 1},       # A11 + A22
        {2: 1, 3: 1},       # A21 + A22
        {0: 1},             # A11
        {3: 1},             # A22
        {0: 1, 1: 1},       # A11 + A12
        {2: 1, 0: -1},      # A21 - A11
        {1: 1, 3: -1},      # A12 - A22
    ]
    
    B_combos = [
        {0: 1, 3: 1},       # B11 + B22
        {0: 1},             # B11
        {1: 1, 3: -1},      # B12 - B22
        {2: 1, 0: -1},      # B21 - B11
        {3: 1},             # B22
        {0: 1, 1: 1},       # B11 + B12
        {2: 1, 3: 1},       # B21 + B22
    ]
    
    # How each block product P[bp_idx] contributes to output blocks C[br][bc]
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    block_contrib = [
        # P1 (idx 0)
        [[1, 0], [0, 1]],
        # P2 (idx 1)
        [[0, 0], [1, -1]],
        # P3 (idx 2)
        [[0, 1], [0, 1]],
        # P4 (idx 3)
        [[1, 0], [1, 0]],
        # P5 (idx 4)
        [[-1, 1], [0, 0]],
        # P6 (idx 5)
        [[0, 0], [0, 1]],
        # P7 (idx 6)
        [[1, 0], [0, 0]],
    ]
    
    # Inner Strassen terms for 2x2*2x2
    inner_terms = _strassen_2x2_terms()
    
    col = 0
    for bp_idx in range(7):
        A_combo = A_combos[bp_idx]
        B_combo = B_combos[bp_idx]
        
        for sp_idx in range(7):
            u_vals = [0] * 16
            v_vals = [0] * 16
            w_vals = [0] * 16
            
            # Get the inner Strassen term
            u_dict, v_dict, w_dict = inner_terms[sp_idx]
            
            # Expand to 4x4 element level
            # For U: for each block (br, bc), the coefficient is A_combo[block_idx] * u_dict[block_idx]
            # The block (br, bc) occupies rows 2*br..2*br+1, cols 2*bc..2*bc+1 in the 4x4 matrix
            for block_idx, coeff_u in u_dict.items():
                br = block_idx // 2
                bc = block_idx % 2
                outer_coeff = A_combo.get(block_idx, 0)
                total_coeff = outer_coeff * coeff_u
                if total_coeff != 0:
                    for r in range(2):
                        for c in range(2):
                            row = 2 * br + r
                            col_idx = 2 * bc + c
                            flat_idx = row * 4 + col_idx
                            u_vals[flat_idx] += total_coeff
            
            # For V: similar but with B_combo
            for block_idx, coeff_v in v_dict.items():
                br = block_idx // 2
                bc = block_idx % 2
                outer_coeff = B_combo.get(block_idx, 0)
                total_coeff = outer_coeff * coeff_v
                if total_coeff != 0:
                    for r in range(2):
                        for c in range(2):
                            row = 2 * br + r
                            col_idx = 2 * bc + c
                            flat_idx = row * 4 + col_idx
                            v_vals[flat_idx] += total_coeff
            
            # For W: the inner W gives contributions to the 2x2 block result.
            # The block result R has entries R[br][bc] which are 2x2 blocks.
            # w_dict maps block_idx (br*2+bc) to coefficient for that block's contribution.
            # But wait - the inner Strassen W tells us how the scalar product contributes
            # to the 2x2 block result. Each entry of the 2x2 block result is a 2x2 matrix.
            # So w_dict[block_idx] is the coefficient for the entire 2x2 block at position block_idx.
            # Then we multiply by block_contrib[bp_idx][br][bc] for the outer contribution.
            
            for block_idx, coeff_w in w_dict.items():
                br = block_idx // 2
                bc = block_idx % 2
                outer_coeff = block_contrib[bp_idx][br][bc]
                total_coeff = outer_coeff * coeff_w
                if total_coeff != 0:
                    for r in range(2):
                        for c in range(2):
                            row = 2 * br + r
                            col_idx = 2 * bc + c
                            # W index: k*n + i where k is column, i is row
                            flat_idx = col_idx * 4 + row
                            w_vals[flat_idx] += total_coeff
            
            # Assign to U, V, W
            for i in range(16):
                U[i][col] = u_vals[i]
                V[i][col] = v_vals[i]
                W[i][col] = w_vals[i]
            
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1
                actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
                if abs(actual - expected) > tol:
                    return False
    return True


def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition
    for 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Try the rank-49 Strassen-based decomposition
    U, V, W = _strassen_4x4()
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 64 if verification fails
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
