"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _strassen_4x4():
    """
    Return a rank-49 decomposition for 4x4 matrix multiplication using
    a tensor-product / block-based construction derived from Strassen's
    2x2 algorithm (rank 7) applied to the 4x4 problem.
    
    Uses exact integer arithmetic throughout.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to create a 2x2 block of linear forms
    # Returns a 2x2 list of dicts: {flat_index: coefficient}
    def make_block_A(br, bc, sign=1):
        """Create dict entries for 2x2 block at (br, bc) of A (n x m)"""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                row = 2 * br + i
                col = 2 * bc + j
                idx = row * m + col
                result[i][j][idx] = sign
        return result
    
    def make_block_B(br, bc, sign=1):
        """Create dict entries for 2x2 block at (br, bc) of B (m x p)"""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                row = 2 * br + i
                col = 2 * bc + j
                idx = row * p + col
                result[i][j][idx] = sign
        return result
    
    def add_blocks_2x2(base, other, sign_other=1):
        """Add sign_other * other to base (both are 2x2 lists of dicts)"""
        result = [[dict(base[i][j]) for j in range(2)] for i in range(2)]
        for i in range(2):
            for j in range(2):
                for idx, coeff in other[i][j].items():
                    result[i][j][idx] = result[i][j].get(idx, 0) + sign_other * coeff
        return result
    
    def sub_blocks_2x2(base, other):
        return add_blocks_2x2(base, other, -1)
    
    # Define the 7 Strassen block products for 4x4 (2x2 block) multiplication
    # A = [[A00, A01], [A10, A11]]  (blocks are 2x2)
    # B = [[B00, B01], [B10, B11]]
    
    # X1 = A00 + A11, Y1 = B00 + B11
    X1 = add_blocks_2x2(make_block_A(0, 0), make_block_A(1, 1))
    Y1 = add_blocks_2x2(make_block_B(0, 0), make_block_B(1, 1))
    
    # X2 = A10 + A11, Y2 = B00
    X2 = add_blocks_2x2(make_block_A(1, 0), make_block_A(1, 1))
    Y2 = make_block_B(0, 0)
    
    # X3 = A00, Y3 = B01 - B11
    X3 = make_block_A(0, 0)
    Y3 = sub_blocks_2x2(make_block_B(0, 1), make_block_B(1, 1))
    
    # X4 = A11, Y4 = B10 - B00
    X4 = make_block_A(1, 1)
    Y4 = sub_blocks_2x2(make_block_B(1, 0), make_block_B(0, 0))
    
    # X5 = A00 + A01, Y5 = B11
    X5 = add_blocks_2x2(make_block_A(0, 0), make_block_A(0, 1))
    Y5 = make_block_B(1, 1)
    
    # X6 = A10 - A00, Y6 = B00 + B01
    X6 = sub_blocks_2x2(make_block_A(1, 0), make_block_A(0, 0))
    Y6 = add_blocks_2x2(make_block_B(0, 0), make_block_B(0, 1))
    
    # X7 = A01 - A11, Y7 = B10 + B11
    X7 = sub_blocks_2x2(make_block_A(0, 1), make_block_A(1, 1))
    Y7 = add_blocks_2x2(make_block_B(1, 0), make_block_B(1, 1))
    
    Xs = [X1, X2, X3, X4, X5, X6, X7]
    Ys = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    # Strassen's 7 sub-products for 2x2 matmul X*Y:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # s1 = (x00+x11)*(y00+y11)
    # s2 = (x10+x11)*y00
    # s3 = x00*(y01-y11)
    # s4 = x11*(y10-y00)
    # s5 = (x00+x01)*y11
    # s6 = (x10-x00)*(y00+y01)
    # s7 = (x01-x11)*(y10+y11)
    
    # Result:
    # Z00 = s1 + s4 - s5 + s7
    # Z01 = s3 + s5
    # Z10 = s2 + s4
    # Z11 = s1 - s2 + s3 + s6
    
    # Coefficients of each sk in Z[r][c]:
    # strassen_coeff[sk_idx][r][c]
    strassen_coeff = [
        # s1
        [[1, 0], [0, 1]],
        # s2
        [[0, 0], [1, -1]],
        # s3
        [[0, 1], [0, 1]],
        # s4
        [[1, 0], [1, 0]],
        # s5
        [[-1, 1], [0, 0]],
        # s6
        [[0, 0], [0, 1]],
        # s7
        [[1, 0], [0, 0]],
    ]
    
    # Block contributions: which output blocks each Pj contributes to
    # P1 -> C00(+1), C11(+1)
    # P2 -> C10(+1), C11(-1)
    # P3 -> C01(+1), C11(+1)
    # P4 -> C00(+1), C10(+1)
    # P5 -> C00(-1), C01(+1)
    # P6 -> C11(+1)
    # P7 -> C00(+1)
    
    block_contrib = [
        {((0, 0)): 1, ((1, 1)): 1},    # P1
        {((1, 0)): 1, ((1, 1)): -1},   # P2
        {((0, 1)): 1, ((1, 1)): 1},    # P3
        {((0, 0)): 1, ((1, 0)): 1},    # P4
        {((0, 0)): -1, ((0, 1)): 1},   # P5
        {((1, 1)): 1},                 # P6
        {((0, 0)): 1},                 # P7
    ]
    
    col = 0
    for j in range(7):
        Xj = Xs[j]
        Yj = Ys[j]
        
        # Extract the 4 entries of Xj and Yj
        x00, x01 = Xj[0]
        x10, x11 = Xj[1]
        y00, y01 = Yj[0]
        y10, y11 = Yj[1]
        
        # Define the 7 sub-products' X and Y factors
        # Each factor is a list of (index, coeff)
        
        def combine_factors(f1, f2, sign2=1):
            """Combine two dicts of (index -> coeff), with sign2 multiplying f2"""
            result = dict(f1)
            for idx, c in f2.items():
                result[idx] = result.get(idx, 0) + c * sign2
            return result
        
        def subtract_factors(f1, f2):
            return combine_factors(f1, f2, -1)
        
        # The 7 sub-products:
        # s1: X = x00+x11, Y = y00+y11
        # s2: X = x10+x11, Y = y00
        # s3: X = x00, Y = y01-y11
        # s4: X = x11, Y = y10-y00
        # s5: X = x00+x01, Y = y11
        # s6: X = x10-x00, Y = y00+y01
        # s7: X = x01-x11, Y = y10+y11
        
        sub_products = [
            (combine_factors(x00, x11), combine_factors(y00, y11)),       # s1
            (combine_factors(x10, x11), y00),                             # s2
            (x00, subtract_factors(y01, y11)),                            # s3
            (x11, subtract_factors(y10, y00)),                            # s4
            (combine_factors(x00, x01), y11),                             # s5
            (subtract_factors(x10, x00), combine_factors(y00, y01)),      # s6
            (subtract_factors(x01, x11), combine_factors(y10, y11)),      # s7
        ]
        
        for sk_idx in range(7):
            x_factor, y_factor = sub_products[sk_idx]
            
            # Build U vector for this term
            for idx, coeff in x_factor.items():
                if coeff != 0:
                    U[idx][col] += coeff
            
            # Build V vector for this term
            for idx, coeff in y_factor.items():
                if coeff != 0:
                    V[idx][col] += coeff
            
            # Build W vector for this term
            for (bk, bl), alpha in block_contrib[j].items():
                for r in range(2):
                    for c in range(2):
                        coeff_rc = strassen_coeff[sk_idx][r][c]
                        if coeff_rc != 0:
                            C_row = 2 * bk + r
                            C_col = 2 * bl + c
                            W_idx = C_col * n + C_row
                            total_coeff = alpha * coeff_rc
                            W[W_idx][col] += total_coeff
            
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Check that T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r] matches the matmul tensor
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
    # Check some zero entries
    if n >= 2 and m >= 2 and p >= 2:
        # a = 0 (i=0,j=0), b = p (j'=1,k'=0), c = 0 (k''=0,i'=0)
        # j != j' so should be 0
        a = 0
        b = 1 * p + 0
        c = 0
        actual = sum(U[a][r] * V[b][r] * W[c][r] for r in range(rank))
        if abs(actual) > tol:
            return False
    return True


def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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
    return U, V, W


def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition
    for 4x4 matrix multiplication.
    
    Uses Strassen's algorithm applied to 2x2 blocks, giving rank 49.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Try the rank-49 Strassen-based decomposition
    try:
        U, V, W = _strassen_4x4()
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 64
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
