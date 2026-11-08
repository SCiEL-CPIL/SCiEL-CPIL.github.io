"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

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


def _strassen_4x4():
    """
    Return a rank-49 decomposition for 4x4 matrix multiplication using
    a tensor-product / block-based construction derived from Strassen's
    2x2 algorithm (rank 7) applied to the 4x4 problem.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 formulas for 2x2 matmul:
    # Given X = [[x0, x1], [x2, x3]], Y = [[y0, y1], [y2, y3]]
    # s1 = (x0+x3)*(y0+y3)
    # s2 = (x2+x3)*y0
    # s3 = x0*(y1-y3)
    # s4 = x3*(y2-y0)
    # s5 = (x0+x1)*y3
    # s6 = (x2-x0)*(y0+y1)
    # s7 = (x1-x3)*(y2+y3)
    # Result:
    # [[s1+s4-s5+s7, s3+s5],
    #  [s2+s4, s1-s2+s3+s6]]
    
    # The 7 block products for 4x4 (2x2 block) multiplication:
    # P1 = (A11+A22)*(B11+B22)
    # P2 = (A21+A22)*B11
    # P3 = A11*(B12-B22)
    # P4 = A22*(B21-B11)
    # P5 = (A11+A12)*B22
    # P6 = (A21-A11)*(B11+B12)
    # P7 = (A12-A22)*(B21+B22)
    
    # Output blocks:
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    def make_block_combo(blocks_with_coeffs):
        """
        blocks_with_coeffs: list of (bi, bj, coeff)
        Returns a 2x2 array where each entry is a list of (flat_index, coeff)
        """
        result = [[[] for _ in range(2)] for _ in range(2)]
        for bi, bj, coeff in blocks_with_coeffs:
            for r in range(2):
                for c in range(2):
                    idx = (2*bi + r) * m + (2*bj + c)
                    result[r][c].append((idx, coeff))
        return result
    
    def make_block_combo_B(blocks_with_coeffs):
        """Same but for B"""
        result = [[[] for _ in range(2)] for _ in range(2)]
        for bi, bj, coeff in blocks_with_coeffs:
            for r in range(2):
                for c in range(2):
                    idx = (2*bi + r) * p + (2*bj + c)
                    result[r][c].append((idx, coeff))
        return result
    
    # Define X1..X7
    X1 = make_block_combo([(0,0,1), (1,1,1)])  # A11 + A22
    X2 = make_block_combo([(1,0,1), (1,1,1)])  # A21 + A22
    X3 = make_block_combo([(0,0,1)])           # A11
    X4 = make_block_combo([(1,1,1)])           # A22
    X5 = make_block_combo([(0,0,1), (0,1,1)])  # A11 + A12
    X6 = make_block_combo([(1,0,1), (0,0,-1)]) # A21 - A11
    X7 = make_block_combo([(0,1,1), (1,1,-1)]) # A12 - A22
    
    # Define Y1..Y7
    Y1 = make_block_combo_B([(0,0,1), (1,1,1)])  # B11 + B22
    Y2 = make_block_combo_B([(0,0,1)])           # B11
    Y3 = make_block_combo_B([(0,1,1), (1,1,-1)]) # B12 - B22
    Y4 = make_block_combo_B([(1,0,1), (0,0,-1)]) # B21 - B11
    Y5 = make_block_combo_B([(1,1,1)])           # B22
    Y6 = make_block_combo_B([(0,0,1), (0,1,1)])  # B11 + B12
    Y7 = make_block_combo_B([(1,0,1), (1,1,1)])  # B21 + B22
    
    Xs = [X1, X2, X3, X4, X5, X6, X7]
    Ys = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    # Strassen coefficients: strassen_table[sk][r][c] = coefficient
    # R[0,0]: s1(+1), s4(+1), s5(-1), s7(+1)
    # R[0,1]: s3(+1), s5(+1)
    # R[1,0]: s2(+1), s4(+1)
    # R[1,1]: s1(+1), s2(-1), s3(+1), s6(+1)
    strassen_table = [
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
    
    # Block contributions: block_contrib[j][(k,l)] = coefficient
    # P1 -> C11(+1), C22(+1)
    # P2 -> C21(+1), C22(-1)
    # P3 -> C12(+1), C22(+1)
    # P4 -> C11(+1), C21(+1)
    # P5 -> C11(-1), C12(+1)
    # P6 -> C22(+1)
    # P7 -> C11(+1)
    block_contrib = [
        {(0,0): 1, (1,1): 1},   # P1
        {(1,0): 1, (1,1): -1},  # P2
        {(0,1): 1, (1,1): 1},   # P3
        {(0,0): 1, (1,0): 1},   # P4
        {(0,0): -1, (0,1): 1},  # P5
        {(1,1): 1},             # P6
        {(0,0): 1},             # P7
    ]
    
    # Now construct the decomposition
    col = 0
    for j in range(7):  # 7 block products
        Xj = Xs[j]
        Yj = Ys[j]
        # Xj is 2x2, each entry is list of (A_idx, coeff)
        # Yj is 2x2, each entry is list of (B_idx, coeff)
        
        x0, x1 = Xj[0]
        x2, x3 = Xj[1]
        y0, y1 = Yj[0]
        y2, y3 = Yj[1]
        
        def add_factors(f1, f2, sign2=1):
            """Add two factor lists, with sign2 multiplying f2"""
            result = list(f1)
            for idx, c in f2:
                result.append((idx, c * sign2))
            return result
        
        def sub_factors(f1, f2):
            return add_factors(f1, f2, -1)
        
        # s1: X = x0+x3, Y = y0+y3
        # s2: X = x2+x3, Y = y0
        # s3: X = x0, Y = y1-y3
        # s4: X = x3, Y = y2-y0
        # s5: X = x0+x1, Y = y3
        # s6: X = x2-x0, Y = y0+y1
        # s7: X = x1-x3, Y = y2+y3
        
        sub_products = [
            (add_factors(x0, x3), add_factors(y0, y3)),       # s1
            (add_factors(x2, x3), y0),                        # s2
            (x0, sub_factors(y1, y3)),                        # s3
            (x3, sub_factors(y2, y0)),                        # s4
            (add_factors(x0, x1), y3),                        # s5
            (sub_factors(x2, x0), add_factors(y0, y1)),       # s6
            (sub_factors(x1, x3), add_factors(y2, y3)),       # s7
        ]
        
        for sk_idx in range(7):
            x_factor, y_factor = sub_products[sk_idx]
            
            # Build U vector for this term
            u_dict = {}
            for idx, coeff in x_factor:
                u_dict[idx] = u_dict.get(idx, 0) + coeff
            for idx in range(n * m):
                U[idx][col] = u_dict.get(idx, 0)
            
            # Build V vector for this term
            v_dict = {}
            for idx, coeff in y_factor:
                v_dict[idx] = v_dict.get(idx, 0) + coeff
            for idx in range(m * p):
                V[idx][col] = v_dict.get(idx, 0)
            
            # Build W vector for this term
            w_dict = {}
            for (kl), alpha in block_contrib[j].items():
                k, l = kl
                for r in range(2):
                    for c in range(2):
                        coeff_rc = strassen_table[sk_idx][r][c]
                        if coeff_rc != 0:
                            C_row = 2 * k + r
                            C_col = 2 * l + c
                            W_idx = C_col * n + C_row
                            total_coeff = alpha * coeff_rc
                            w_dict[W_idx] = w_dict.get(W_idx, 0) + total_coeff
            for idx in range(p * n):
                W[idx][col] = w_dict.get(idx, 0)
            
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n=4, m=4, p=4):
    """
    Verify that the tensor decomposition is correct by checking
    T[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r]
    equals 1 iff j==j' and k==k' and i==i', else 0.
    """
    rank = len(U[0])
    # Check a random sample of entries
    import random
    random.seed(42)
    for _ in range(200):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        j2 = random.randint(0, m-1)
        k = random.randint(0, p-1)
        k2 = random.randint(0, p-1)
        i2 = random.randint(0, n-1)
        
        a_idx = i * m + j
        b_idx = j2 * p + k
        c_idx = k2 * n + i2
        
        s = 0
        for r in range(rank):
            s += U[a_idx][r] * V[b_idx][r] * W[c_idx][r]
        
        expected = 1.0 if (j == j2 and k == k2 and i == i2) else 0.0
        if abs(s - expected) > 1e-10:
            return False
    return True


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
        if time.time() < deadline:
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 64
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
