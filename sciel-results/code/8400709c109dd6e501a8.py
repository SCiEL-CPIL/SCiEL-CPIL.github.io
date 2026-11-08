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
    Strassen's 2x2 algorithm applied to 2x2 blocks, with each block product
    itself decomposed via Strassen's 7-term 2x2 matmul.
    
    The 4x4 matrix A is partitioned into four 2x2 blocks:
        A = [[A00, A01],
             [A10, A11]]
    Similarly for B. The 2x2 block-level multiplication uses Strassen's
    7 products, and each 2x2 block product is computed via Strassen's
    7-term decomposition, giving 7 * 7 = 49 rank terms total.
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper: get the flat index for A[row, col]
    def a_idx(row, col):
        return row * m + col
    
    # Helper: get the flat index for B[row, col]
    def b_idx(row, col):
        return row * p + col
    
    # Helper: get the flat index for W corresponding to C[row, col]
    # Convention: W[k*n + i] where k is output col, i is output row
    # So for C[row, col], W index = col * n + row
    def w_idx(row, col):
        return col * n + row
    
    # Strassen's 7-term decomposition for 2x2 matmul:
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # z1 = (x00+x11)*(y00+y11)
    # z2 = (x10+x11)*y00
    # z3 = x00*(y01-y11)
    # z4 = x11*(y10-y00)
    # z5 = (x00+x01)*y11
    # z6 = (x10-x00)*(y00+y01)
    # z7 = (x01-x11)*(y10+y11)
    # Result Z = X*Y:
    # Z[0,0] = z1 + z4 - z5 + z7
    # Z[0,1] = z3 + z5
    # Z[1,0] = z2 + z4
    # Z[1,1] = z1 - z2 + z3 + z6
    
    # For each of the 7 block products Pj = Xj * Yj, we need:
    # - Xj: a 2x2 matrix whose entries are linear combos of A entries
    # - Yj: a 2x2 matrix whose entries are linear combos of B entries
    # - The contribution of Pj to each output 2x2 block Ckl with a coefficient
    
    # Block product definitions (Strassen for 2x2 block matmul):
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # Output block contributions:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Define each Xj and Yj as 2x2 arrays of dicts {flat_idx: coeff}
    
    def block_A(br, bc, sign=1):
        """2x2 block of A at block position (br, bc), each entry is a dict."""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                result[r][c] = {a_idx(2*br + r, 2*bc + c): sign}
        return result
    
    def block_B(br, bc, sign=1):
        """2x2 block of B at block position (br, bc), each entry is a dict."""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                result[r][c] = {b_idx(2*br + r, 2*bc + c): sign}
        return result
    
    def add_2x2(M1, M2, sign2=1):
        """Add sign2*M2 to M1, element-wise (dicts)."""
        result = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                d = dict(M1[r][c])
                for k, v in M2[r][c].items():
                    d[k] = d.get(k, 0) + sign2 * v
                result[r][c] = d
        return result
    
    def sub_2x2(M1, M2):
        return add_2x2(M1, M2, -1)
    
    # Define X1..X7 (2x2 matrices of linear forms in A)
    A00 = block_A(0, 0)
    A01 = block_A(0, 1)
    A10 = block_A(1, 0)
    A11 = block_A(1, 1)
    
    X1 = add_2x2(A00, A11)          # A00 + A11
    X2 = add_2x2(A10, A11)          # A10 + A11
    X3 = A00                        # A00
    X4 = A11                        # A11
    X5 = add_2x2(A00, A01)          # A00 + A01
    X6 = sub_2x2(A10, A00)          # A10 - A00
    X7 = sub_2x2(A01, A11)          # A01 - A11
    
    # Define Y1..Y7 (2x2 matrices of linear forms in B)
    B00 = block_B(0, 0)
    B01 = block_B(0, 1)
    B10 = block_B(1, 0)
    B11 = block_B(1, 1)
    
    Y1 = add_2x2(B00, B11)          # B00 + B11
    Y2 = B00                        # B00
    Y3 = sub_2x2(B01, B11)          # B01 - B11
    Y4 = sub_2x2(B10, B00)          # B10 - B00
    Y5 = B11                        # B11
    Y6 = add_2x2(B00, B01)          # B00 + B01
    Y7 = add_2x2(B10, B11)          # B10 + B11
    
    Xs = [X1, X2, X3, X4, X5, X6, X7]
    Ys = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    # Block contributions: block_contrib[j] = {(block_r, block_c): coeff}
    block_contrib = [
        {(0,0): 1, (1,1): 1},       # P1
        {(1,0): 1, (1,1): -1},      # P2
        {(0,1): 1, (1,1): 1},       # P3
        {(0,0): 1, (1,0): 1},       # P4
        {(0,0): -1, (0,1): 1},      # P5
        {(1,1): 1},                 # P6
        {(0,0): 1},                 # P7
    ]
    
    # Strassen sub-product coefficients: strassen_table[sk][r][c]
    # sk is 0..6 (z1..z7), r,c are 0..1
    strassen_table = [
        [[1, 0], [0, 1]],    # z1: Z[0,0]=+1, Z[1,1]=+1
        [[0, 0], [1, -1]],   # z2: Z[1,0]=+1, Z[1,1]=-1
        [[0, 1], [0, 1]],    # z3: Z[0,1]=+1, Z[1,1]=+1
        [[1, 0], [1, 0]],    # z4: Z[0,0]=+1, Z[1,0]=+1
        [[-1, 1], [0, 0]],   # z5: Z[0,0]=-1, Z[0,1]=+1
        [[0, 0], [0, 1]],    # z6: Z[1,1]=+1
        [[1, 0], [0, 0]],    # z7: Z[0,0]=+1
    ]
    
    # For each block product j and each Strassen sub-product sk:
    # The X factor is a linear combo of entries of Xj
    # The Y factor is a linear combo of entries of Yj
    
    # Strassen sub-product definitions:
    # z1: X_factor = x00 + x11, Y_factor = y00 + y11
    # z2: X_factor = x10 + x11, Y_factor = y00
    # z3: X_factor = x00,       Y_factor = y01 - y11
    # z4: X_factor = x11,       Y_factor = y10 - y00
    # z5: X_factor = x00 + x01, Y_factor = y11
    # z6: X_factor = x10 - x00, Y_factor = y00 + y01
    # z7: X_factor = x01 - x11, Y_factor = y10 + y11
    
    def merge_dicts(d1, d2, sign2=1):
        """Merge d2 into d1 with sign2, return new dict."""
        result = dict(d1)
        for k, v in d2.items():
            result[k] = result.get(k, 0) + sign2 * v
        return result
    
    col = 0
    for j in range(7):
        Xj = Xs[j]
        Yj = Ys[j]
        x00, x01 = Xj[0]
        x10, x11 = Xj[1]
        y00, y01 = Yj[0]
        y10, y11 = Yj[1]
        
        # Define the 7 sub-products' X and Y factors as dicts
        sub_products = [
            (merge_dicts(x00, x11), merge_dicts(y00, y11)),           # z1
            (merge_dicts(x10, x11), dict(y00)),                       # z2
            (dict(x00), merge_dicts(y01, y11, -1)),                   # z3
            (dict(x11), merge_dicts(y10, y00, -1)),                   # z4
            (merge_dicts(x00, x01), dict(y11)),                       # z5
            (merge_dicts(x10, x00, -1), merge_dicts(y00, y01)),       # z6
            (merge_dicts(x01, x11, -1), merge_dicts(y10, y11)),       # z7
        ]
        
        for sk in range(7):
            x_factor, y_factor = sub_products[sk]
            
            # Build U vector
            for idx, coeff in x_factor.items():
                U[idx][col] = coeff
            
            # Build V vector
            for idx, coeff in y_factor.items():
                V[idx][col] = coeff
            
            # Build W vector
            for (br, bc), alpha in block_contrib[j].items():
                for r in range(2):
                    for c in range(2):
                        coeff_rc = strassen_table[sk][r][c]
                        if coeff_rc != 0:
                            C_row = 2 * br + r
                            C_col = 2 * bc + c
                            W_flat = w_idx(C_row, C_col)
                            total_coeff = alpha * coeff_rc
                            W[W_flat][col] += total_coeff
            
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n=4, m=4, p=4):
    """
    Verify that the decomposition is correct by checking that
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for all consistent (i, j, k) and 0 otherwise.
    """
    rank = len(U[0])
    # Check a few random entries for speed
    import random
    random.seed(42)
    
    # The tensor T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T should be 1 when indices are consistent, 0 otherwise
    # Actually, for the matmul tensor:
    # C[i, k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[(i,j), (j,k), (k,i)] = 1 if j=j' and k=k' and i=i'
    # i.e., T[a, b, c] = 1 iff the middle index of a matches middle of b,
    # last of b matches first of c, and last of c matches first of a.
    
    # Let's just check: for each (i, j, k), the contribution should be 1
    # and for inconsistent indices, 0.
    
    # Check all 4*4*4 = 64 consistent triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                if abs(val - 1) > 1e-9:
                    return False
    
    # Check some inconsistent triples
    # e.g., a = i*m+j, b = j'*p+k where j' != j
    for _ in range(200):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        i2 = random.randint(0, n-1)
        j2 = random.randint(0, m-1)
        k2 = random.randint(0, p-1)
        # Make them inconsistent
        if j2 == j and k2 == k and i2 == i:
            continue
        a = i * m + j
        b = j2 * p + k
        c = k2 * n + i2
        val = 0
        for r in range(rank):
            val += U[a][r] * V[b][r] * W[c][r]
        if abs(val) > 1e-9:
            return False
    
    return True


def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition
    for 4x4 matrix multiplication.
    
    Uses Strassen's algorithm applied to 2x2 blocks, giving rank 49.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    
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
