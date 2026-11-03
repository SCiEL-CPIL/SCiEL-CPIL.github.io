"""Matrix multiplication 4x4 by 4x5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2 * 2x2 as (U, V, W) factor lists.
    Each term is (u, v, w) where u, v, w are 4-vectors (flattened 2x2)."""
    # Strassen's 7 products for computing C = A*B where A,B are 2x2:
    # C = [[c11, c12], [c21, c22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Flatten 2x2: index 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # A indices: a00=0, a01=1, a10=2, a11=3
    # B indices: b00=0, b01=1, b10=2, b11=3
    
    terms = []
    # M1: (a00+a11)(b00+b11)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, -1]  # contributes +1 to c00, -1 to c11
    terms.append((u1, v1, w1))
    
    # M2: (a10+a11)b00
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]  # contributes +1 to c10
    terms.append((u2, v2, w2))
    
    # M3: a00(b01-b11)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1]  # contributes +1 to c01, +1 to c11
    terms.append((u3, v3, w3))
    
    # M4: a11(b10-b00)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0]  # contributes +1 to c00, +1 to c10
    terms.append((u4, v4, w4))
    
    # M5: (a00+a01)b11
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 0, 0, 1]  # contributes -1 to c00, +1 to c11
    terms.append((u5, v5, w5))
    
    # M6: (a10-a00)(b00+b01)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]  # contributes +1 to c11
    terms.append((u6, v6, w6))
    
    # M7: (a01-a11)(b10+b11)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0]  # contributes +1 to c00
    terms.append((u7, v7, w7))
    
    return terms

def _build_strassen_block_decomposition(n, m, p):
    """Build a Strassen-based decomposition for n x m * m x p.
    
    We decompose into 2x2 blocks where possible.
    For 4x4 * 4x5:
    - A is 4x4: can be viewed as 2x2 blocks of 2x2
    - B is 4x5: can be viewed as 2x2 blocks (with padding)
    - C is 4x5: output
    
    Strategy: Use Strassen for the 2x2 block structure.
    A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    B = [[B00, B01, B02], [B10, B11, B12]] where B00,B01 are 2x2, B02 is 2x1
         Actually, B is 4x5, so:
         B = [[B00, B01, B02], [B10, B11, B12]]
         where B00, B01, B10, B11 are 2x2, B02, B12 are 2x1
    
    C = A * B:
    C = [[C00, C01, C02], [C10, C11, C12]]
    where C00 = A00*B00 + A01*B10 (2x2)
           C01 = A00*B01 + A01*B11 (2x2)
           C02 = A00*B02 + A01*B12 (2x1)
           C10 = A10*B00 + A11*B10 (2x2)
           C11 = A10*B01 + A11*B11 (2x2)
           C12 = A10*B02 + A11*B12 (2x1)
    
    For each 2x2 * 2x2 product, use Strassen (7 terms).
    For 2x2 * 2x1 products, use a simpler algorithm.
    """
    n, m, p = 4, 4, 5
    
    # We'll build U, V, W as lists of terms
    # Each term has U_vec (length n*m=16), V_vec (length m*p=20), W_vec (length p*n=20)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Helper to create zero vectors
    def zero_U():
        return [Fraction(0)] * (n * m)
    def zero_V():
        return [Fraction(0)] * (m * p)
    def zero_W():
        return [Fraction(0)] * (p * n)
    
    # Strassen terms for 2x2 * 2x2
    strassen_terms = _strassen_2x2x2_terms()
    
    # For 2x2 * 2x1, we can use 3 terms:
    # C = A * b where A is 2x2, b is 2x1
    # C0 = a00*b0 + a01*b1
    # C1 = a10*b0 + a11*b1
    # Term 1: a00*b0 -> C0 += 1
    # Term 2: a01*b1 -> C0 += 1
    # Term 3: a10*b0 -> C1 += 1
    # Term 4: a11*b1 -> C1 += 1
    # Or use 3 terms with Strassen-like:
    # M1 = (a00+a11)*b0
    # M2 = (a01-a10)*b1
    # M3 = (a00-a10)*b0
    # C0 = M1 + M2 - M3
    # C1 = M1 - M2 + M3
    # Let's just use 4 terms for simplicity
    
    def add_term(u_vec, v_vec, w_vec):
        U_terms.append(u_vec)
        V_terms.append(v_vec)
        W_terms.append(w_vec)
    
    # Process each output block
    # Output C is 4x5, viewed as 2x2 blocks plus right column
    # C = [[C00, C01, C02], [C10, C11, C12]]
    # C00: 2x2, from A00*B00 + A01*B10
    # C01: 2x2, from A00*B01 + A01*B11
    # C02: 2x1, from A00*B02 + A01*B12
    # C10: 2x2, from A10*B00 + A11*B10
    # C11: 2x2, from A10*B01 + A11*B11
    # C12: 2x1, from A10*B02 + A11*B12
    
    # Index mapping:
    # A is 4x4, flattened: A[i*m+j] where i=row, j=col
    # A00: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A01: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A10: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A11: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # B is 4x5, flattened: B[j*p+k] where j=row, k=col
    # B00: rows 0-1, cols 0-1 -> indices 0,1,5,6
    # B01: rows 0-1, cols 2-3 -> indices 2,3,7,8
    # B02: rows 0-1, col 4 -> indices 4,9
    # B10: rows 2-3, cols 0-1 -> indices 10,11,15,16
    # B11: rows 2-3, cols 2-3 -> indices 12,13,17,18
    # B12: rows 2-3, col 4 -> indices 14,19
    
    # C is 4x5, flattened: C[k*n+i] where k=row, i=col (per trace convention W[k*n+i])
    # Wait, the task says W[k*n+i, r] for output. Let me re-read.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) and W[k*n+i, c] is the coefficient for C[i,k]
    # C[i,k] where i is row (0..n-1), k is col (0..p-1)
    # W index: k*n + i
    
    # C00: i in {0,1}, k in {0,1} -> W indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
    # C01: i in {0,1}, k in {2,3} -> W indices: 2*4+0=8, 2*4+1=9, 3*4+0=12, 3*4+1=13
    # C02: i in {0,1}, k in {4} -> W indices: 4*4+0=16, 4*4+1=17
    # C10: i in {2,3}, k in {0,1} -> W indices: 0*4+2=2, 0*4+3=3, 1*4+2=6, 1*4+3=7
    # C11: i in {2,3}, k in {2,3} -> W indices: 2*4+2=10, 2*4+3=11, 3*4+2=14, 3*4+3=15
    # C12: i in {2,3}, k in {4} -> W indices: 4*4+2=18, 4*4+3=19
    
    # Function to add Strassen terms for a 2x2 * 2x2 product
    def add_strassen_2x2(a_block, b_block, c_block):
        """Add Strassen terms for A_block * B_block contributing to C_block.
        a_block: list of 4 indices in U (A)
        b_block: list of 4 indices in V (B)
        c_block: list of 4 indices in W (C)
        """
        for (u_s, v_s, w_s) in strassen_terms:
            u_vec = zero_U()
            v_vec = zero_V()
            w_vec = zero_W()
            for idx, val in zip(a_block, u_s):
                u_vec[idx] += Fraction(val)
            for idx, val in zip(b_block, v_s):
                v_vec[idx] += Fraction(val)
            for idx, val in zip(c_block, w_s):
                w_vec[idx] += Fraction(val)
            add_term(u_vec, v_vec, w_vec)
    
    # Function to add terms for 2x2 * 2x1 product
    def add_2x2_2x1(a_block, b_vec, c_vec):
        """Add terms for A_block (2x2) * b_vec (2x1) contributing to c_vec (2x1).
        a_block: list of 4 indices in U
        b_vec: list of 2 indices in V
        c_vec: list of 2 indices in W
        """
        # 4 terms: each a_ij * b_j
        # C0 = a00*b0 + a01*b1
        # C1 = a10*b0 + a11*b1
        # a_block = [a00, a01, a10, a11]
        # b_vec = [b0, b1]
        # c_vec = [c0, c1]
        
        # Term 1: a00 * b0 -> c0
        u_vec = zero_U()
        v_vec = zero_V()
        w_vec = zero_W()
        u_vec[a_block[0]] = Fraction(1)
        v_vec[b_vec[0]] = Fraction(1)
        w_vec[c_vec[0]] = Fraction(1)
        add_term(u_vec, v_vec, w_vec)
        
        # Term 2: a01 * b1 -> c0
        u_vec = zero_U()
        v_vec = zero_V()
        w_vec = zero_W()
        u_vec[a_block[1]] = Fraction(1)
        v_vec[b_vec[1]] = Fraction(1)
        w_vec[c_vec[0]] = Fraction(1)
        add_term(u_vec, v_vec, w_vec)
        
        # Term 3: a10 * b0 -> c1
        u_vec = zero_U()
        v_vec = zero_V()
        w_vec = zero_W()
        u_vec[a_block[2]] = Fraction(1)
        v_vec[b_vec[0]] = Fraction(1)
        w_vec[c_vec[1]] = Fraction(1)
        add_term(u_vec, v_vec, w_vec)
        
        # Term 4: a11 * b1 -> c1
        u_vec = zero_U()
        v_vec = zero_V()
        w_vec = zero_W()
        u_vec[a_block[3]] = Fraction(1)
        v_vec[b_vec[1]] = Fraction(1)
        w_vec[c_vec[1]] = Fraction(1)
        add_term(u_vec, v_vec, w_vec)
    
    # A blocks (indices in U, flattened 4x4: row*4+col)
    A00 = [0, 1, 4, 5]    # a00, a01, a10, a11
    A01 = [2, 3, 6, 7]    # a02, a03, a12, a13
    A10 = [8, 9, 12, 13]  # a20, a21, a30, a31
    A11 = [10, 11, 14, 15] # a22, a23, a32, a33
    
    # B blocks (indices in V, flattened 4x5: row*5+col)
    B00 = [0, 1, 5, 6]    # b00, b01, b10, b11
    B01 = [2, 3, 7, 8]    # b02, b03, b12, b13
    B02 = [4, 9]          # b04, b14
    B10 = [10, 11, 15, 16] # b20, b21, b30, b31
    B11 = [12, 13, 17, 18] # b22, b23, b32, b33
    B12 = [14, 19]        # b24, b34
    
    # C blocks (indices in W, flattened: k*4+i where k=col, i=row)
    C00 = [0, 1, 4, 5]    # c00, c01, c10, c11
    C01 = [8, 9, 12, 13]  # c02, c03, c12, c13
    C02 = [16, 17]        # c04, c14
    C10 = [2, 3, 6, 7]    # c20, c21, c30, c31
    C11 = [10, 11, 14, 15] # c22, c23, c32, c33
    C12 = [18, 19]        # c24, c34
    
    # C00 = A00*B00 + A01*B10
    add_strassen_2x2(A00, B00, C00)
    add_strassen_2x2(A01, B10, C00)
    
    # C01 = A00*B01 + A01*B11
    add_strassen_2x2(A00, B01, C01)
    add_strassen_2x2(A01, B11, C01)
    
    # C02 = A00*B02 + A01*B12
    add_2x2_2x1(A00, B02, C02)
    add_2x2_2x1(A01, B12, C02)
    
    # C10 = A10*B00 + A11*B10
    add_strassen_2x2(A10, B00, C10)
    add_strassen_2x2(A11, B10, C10)
    
    # C11 = A10*B01 + A11*B11
    add_strassen_2x2(A10, B01, C11)
    add_strassen_2x2(A11, B11, C11)
    
    # C12 = A10*B02 + A11*B12
    add_2x2_2x1(A10, B02, C12)
    add_2x2_2x1(A11, B12, C12)
    
    # Count: 6 Strassen blocks * 7 terms + 4 (2x2*2x1) blocks * 4 terms = 42 + 16 = 58
    # This is already better than 80!
    
    return U_terms, V_terms, W_terms

def _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-10):
    """Verify that the decomposition is correct.
    Check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    rank = len(U_list)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for c in range(rank):
                    u_val = U_list[c][i * m + j]
                    v_val = V_list[c][j * p + k]
                    w_val = W_list[c][k * n + i]
                    total += u_val * v_val * w_val
                if abs(total - 1) > tol:
                    return False, (i, j, k), total
    return True, None, None

def _is_half_integer(val):
    """Check if a Fraction is an integer or half-integer."""
    return val.denominator in [1, 2]

def _verify_integer_half_integer(U_list, V_list, W_list):
    """Verify all entries are integers or half-integers."""
    for vec in U_list:
        for val in vec:
            if not _is_half_integer(val):
                return False
    for vec in V_list:
        for val in vec:
            if not _is_half_integer(val):
                return False
    for vec in W_list:
        for val in vec:
            if not _is_half_integer(val):
                return False
    return True

def _find_linear_dependencies(U_list, V_list, W_list):
    """Find linear dependencies among the terms.
    If term c can be expressed as a linear combination of other terms, we can eliminate it.
    
    We look for a vector d such that:
    sum_c d[c] * U[c] = 0
    sum_c d[c] * V[c] = 0
    sum_c d[c] * W[c] = 0
    
    But that's not quite right. The decomposition is:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    If we have a linear dependency in the sense that there exist coefficients d[c]
    such that for all i,j,k:
    sum_c d[c] * U[i,j,c] * V[j,k,c] * W[k,i,c] = 0
    
    Then we can use this to reduce rank. But finding such dependencies is hard.
    
    Alternative: check if any two terms are identical or negatives.
    """
    # Simple check: look for duplicate or opposite terms
    removed = set()
    n_terms = len(U_list)
    
    for c1 in range(n_terms):
        if c1 in removed:
            continue
        for c2 in range(c1 + 1, n_terms):
            if c2 in removed:
                continue
            # Check if term c2 is a scalar multiple of term c1
            # For simplicity, check if they're identical or negatives
            same = True
            neg = True
            for i in range(len(U_list[c1])):
                if U_list[c1][i] != U_list[c2][i]:
                    same = False
                if U_list[c1][i] != -U_list[c2][i]:
                    neg = False
            if same:
                # Can eliminate one
                removed.add(c2)
                break
            # Check V
            same = True
            neg = True
            for i in range(len(V_list[c1])):
                if V_list[c1][i] != V_list[c2][i]:
                    same = False
                if V_list[c1][i] != -V_list[c2][i]:
                    neg = False
            if same:
                removed.add(c2)
                break
            # Check W
            same = True
            neg = True
            for i in range(len(W_list[c1])):
                if W_list[c1][i] != W_list[c2][i]:
                    same = False
                if W_list[c1][i] != -W_list[c2][i]:
                    neg = False
            if same:
                removed.add(c2)
                break
    
    if removed:
        new_U = [U_list[c] for c in range(n_terms) if c not in removed]
        new_V = [V_list[c] for c in range(n_terms) if c not in removed]
        new_W = [W_list[c] for c in range(n_terms) if c not in removed]
        return new_U, new_V, new_W, len(removed)
    
    return U_list, V_list, W_list, 0

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition as fallback."""
    rank = n * m * p
    U = []
    V = []
    W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = [Fraction(0)] * (n * m)
                v_vec = [Fraction(0)] * (m * p)
                w_vec = [Fraction(0)] * (p * n)
                u_vec[i * m + j] = Fraction(1)
                v_vec[j * p + k] = Fraction(1)
                w_vec[k * n + i] = Fraction(1)
                U.append(u_vec)
                V.append(v_vec)
                W.append(w_vec)
    return U, V, W

def _to_float_list(vec):
    """Convert a list of Fractions to a list of floats."""
    return [float(v) for v in vec]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Build Strassen-based decomposition
    U_strassen, V_strassen, W_strassen = _build_strassen_block_decomposition(n, m, p)
    rank_strassen = len(U_strassen)
    
    # Verify
    valid, err, _ = _verify_decomposition(U_strassen, V_strassen, W_strassen, n, m, p)
    if not valid:
        # Fall back to schoolbook
        U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
        return {
            "U": _to_float_list(U_sb[0]) if False else [_to_float_list(u) for u in U_sb],
            "V": [_to_float_list(v) for v in V_sb],
            "W": [_to_float_list(w) for w in W_sb]
        }
    
    # Check integer/half-integer
    if not _verify_integer_half_integer(U_strassen, V_strassen, W_strassen):
        # Fall back to schoolbook
        U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
        return {
            "U": [_to_float_list(u) for u in U_sb],
            "V": [_to_float_list(v) for v in V_sb],
            "W": [_to_float_list(w) for w in W_sb]
        }
    
    # Try to find linear dependencies
    U_reduced, V_reduced, W_reduced, n_removed = _find_linear_dependencies(U_strassen, V_strassen, W_strassen)
    
    if n_removed > 0:
        # Verify reduced version
        valid2, _, _ = _verify_decomposition(U_reduced, V_reduced, W_reduced, n, m, p)
        if valid2 and _verify_integer_half_integer(U_reduced, V_reduced, W_reduced):
            U_strassen, V_strassen, W_strassen = U_reduced, V_reduced, W_reduced
            rank_strassen = len(U_strassen)
    
    # Build schoolbook for comparison
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    rank_schoolbook = len(U_sb)
    
    # Use the better one
    if rank_strassen <= rank_schoolbook:
        return {
            "U": [_to_float_list(u) for u in U_strassen],
            "V": [_to_float_list(v) for v in V_strassen],
            "W": [_to_float_list(w) for w in W_strassen]
        }
    else:
        return {
            "U": [_to_float_list(u) for u in U_sb],
            "V": [_to_float_list(v) for v in V_sb],
            "W": [_to_float_list(w) for w in W_sb]
        }
# EVOLVE-BLOCK-END
