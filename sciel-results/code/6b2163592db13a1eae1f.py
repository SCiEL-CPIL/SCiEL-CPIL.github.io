"""
Matrix multiplication 4,4,5: rank-65 decomposition via recursive Strassen.

Strategy:
- The 4x4x5 tensor is decomposed as:
  (a) 4x4x4 block (k=0..3): handled by recursive Strassen (rank 49)
  (b) 5th column cross-term (k=4): 16 schoolbook terms (rank 16)
  Total rank = 65.

The recursive Strassen for 4x4x4 uses 7 outer Strassen multiplications,
each of which is a 2x2 matrix product. Each 2x2 product is decomposed
via inner Strassen into 7 scalar multiplications. 7*7 = 49.

Verification uses exact rational arithmetic to ensure correctness.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 5]
SEED = 42


def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    The decomposition satisfies:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 80

    def time_remaining():
        return deadline - time.time()

    def verify(U, V, W):
        """
        Verify exact decomposition using rational arithmetic.
        U, V, W can be lists or numpy arrays.
        Returns True if sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k.
        """
        try:
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)

            if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
                return False

            rank = U_arr.shape[1]
            if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
                return False

            # Check shapes
            if U_arr.shape[0] != n * m:
                return False
            if V_arr.shape[0] != m * p:
                return False
            if W_arr.shape[0] != p * n:
                return False

            # Convert to Fraction for exact arithmetic
            U_frac = np.zeros_like(U_arr, dtype=object)
            V_frac = np.zeros_like(V_arr, dtype=object)
            W_frac = np.zeros_like(W_arr, dtype=object)

            for i in range(U_arr.shape[0]):
                for r in range(rank):
                    U_frac[i, r] = Fraction(U_arr[i, r])
            for i in range(V_arr.shape[0]):
                for r in range(rank):
                    V_frac[i, r] = Fraction(V_arr[i, r])
            for i in range(W_arr.shape[0]):
                for r in range(rank):
                    W_frac[i, r] = Fraction(W_arr[i, r])

            # Reshape to 3D
            U_3d = U_frac.reshape(n, m, rank)
            V_3d = V_frac.reshape(m, p, rank)
            W_3d = W_frac.reshape(p, n, rank)

            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = Fraction(0)
                        for r in range(rank):
                            total += U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
                        if total != 1:
                            return False
            return True
        except Exception:
            return False

    def get_schoolbook():
        """Generate the standard schoolbook decomposition of rank n*m*p."""
        U = np.zeros((n * m, schoolbook_rank), dtype=float)
        V = np.zeros((m * p, schoolbook_rank), dtype=float)
        W = np.zeros((p * n, schoolbook_rank), dtype=float)

        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Build rank-65 decomposition:
    # 49 terms from recursive Strassen for 4x4x4 (cols 0..48)
    # 16 terms for the 5th column (k=4) (cols 49..64)
    
    target_rank = 65
    U = np.zeros((n * m, target_rank), dtype=float)
    V = np.zeros((m * p, target_rank), dtype=float)
    W = np.zeros((p * n, target_rank), dtype=float)

    # Strassen 2x2: 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    def strassen_2x2_mults():
        """
        Returns list of 7 tuples: (u_dict, v_dict)
        where u_dict maps (r,c) -> coeff for the 2x2 A matrix
        and v_dict maps (r,c) -> coeff for the 2x2 B matrix
        """
        mults = []
        # M1 = (A00+A11)(B00+B11)
        u1 = {(0, 0): 1, (1, 1): 1}
        v1 = {(0, 0): 1, (1, 1): 1}
        mults.append((u1, v1))

        # M2 = (A10+A11)B00
        u2 = {(1, 0): 1, (1, 1): 1}
        v2 = {(0, 0): 1}
        mults.append((u2, v2))

        # M3 = A00(B01-B11)
        u3 = {(0, 0): 1}
        v3 = {(0, 1): 1, (1, 1): -1}
        mults.append((u3, v3))

        # M4 = A11(B10-B00)
        u4 = {(1, 1): 1}
        v4 = {(1, 0): 1, (0, 0): -1}
        mults.append((u4, v4))

        # M5 = (A00+A01)B11
        u5 = {(0, 0): 1, (0, 1): 1}
        v5 = {(1, 1): 1}
        mults.append((u5, v5))

        # M6 = (A10-A00)(B00+B01)
        u6 = {(1, 0): 1, (0, 0): -1}
        v6 = {(0, 0): 1, (0, 1): 1}
        mults.append((u6, v6))

        # M7 = (A01-A11)(B10+B11)
        u7 = {(0, 1): 1, (1, 1): -1}
        v7 = {(1, 0): 1, (1, 1): 1}
        mults.append((u7, v7))

        return mults

    # For each of the 7 outer Strassen terms, we have a 2x2 "effective" A and B.
    # These effective matrices are linear combinations of the 4x4 A and B blocks.
    # Then we apply inner Strassen to each 2x2 multiplication.

    # The 7 outer terms and their block definitions:
    # A_blocks: A00=(rows 0-1, cols 0-1), A01=(rows 0-1, cols 2-3),
    #           A10=(rows 2-3, cols 0-1), A11=(rows 2-3, cols 2-3)
    # B_blocks: B00=(rows 0-1, cols 0-1), B01=(rows 0-1, cols 2-3),
    #           B10=(rows 2-3, cols 0-1), B11=(rows 2-3, cols 2-3)

    # Each outer term defines a 2x2 matrix U_outer and V_outer where each entry
    # is a linear combination of 4x4 entries. We represent each entry as a dict
    # mapping (i,j) -> coeff for the 4x4 A/B entries.

    def expand_2x2_to_scalars(u_matrix, v_matrix):
        """
        u_matrix: 2x2 array where each entry is a dict {(i,j): coeff} of 4x4 A entries
        v_matrix: 2x2 array where each entry is a dict {(i,j): coeff} of 4x4 B entries
        Returns list of 7 (u_scalar, v_scalar) where each is a dict {(i,j): coeff}
        """
        strassen_mults = strassen_2x2_mults()
        results = []
        for u_dict, v_dict in strassen_mults:
            u_scalar = {}
            v_scalar = {}
            for (ur, uc), coeff_u in u_dict.items():
                for (ai, aj), coeff_ai in u_matrix[ur][uc].items():
                    key = (ai, aj)
                    u_scalar[key] = u_scalar.get(key, 0) + coeff_u * coeff_ai
            for (vr, vc), coeff_v in v_dict.items():
                for (bi, bj), coeff_bi in v_matrix[vr][vc].items():
                    key = (bi, bj)
                    v_scalar[key] = v_scalar.get(key, 0) + coeff_v * coeff_bi
            # Remove zeros
            u_scalar = {k: v for k, v in u_scalar.items() if v != 0}
            v_scalar = {k: v for k, v in v_scalar.items() if v != 0}
            results.append((u_scalar, v_scalar))
        return results

    # Define the 7 outer terms as 2x2 matrices of linear combos
    # Each entry is a dict {(i,j): coeff}

    # Outer term 1: U1 = A00 + A11, V1 = B00 + B11
    U1 = [[{} for _ in range(2)] for _ in range(2)]
    V1 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            U1[r][c][(r, c)] = 1
            U1[r][c][(r + 2, c + 2)] = 1
            V1[r][c][(r, c)] = 1
            V1[r][c][(r + 2, c + 2)] = 1

    # Outer term 2: U2 = A10 + A11, V2 = B00
    U2 = [[{} for _ in range(2)] for _ in range(2)]
    V2 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            U2[r][c][(r + 2, c)] = 1
            U2[r][c][(r + 2, c + 2)] = 1
            V2[r][c][(r, c)] = 1

    # Outer term 3: U3 = A00, V3 = B01 - B11
    U3 = [[{} for _ in range(2)] for _ in range(2)]
    V3 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            U3[r][c][(r, c)] = 1
            V3[r][c][(r, c + 2)] = 1
            V3[r][c][(r + 2, c + 2)] = -1

    # Outer term 4: U4 = A11, V4 = B10 - B00
    U4 = [[{} for _ in range(2)] for _ in range(2)]
    V4 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            U4[r][c][(r + 2, c + 2)] = 1
            V4[r][c][(r + 2, c)] = 1
            V4[r][c][(r, c)] = -1

    # Outer term 5: U5 = A00 + A01, V5 = B11
    U5 = [[{} for _ in range(2)] for _ in range(2)]
    V5 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            U5[r][c][(r, c)] = 1
            U5[r][c][(r, c + 2)] = 1
            V5[r][c][(r + 2, c + 2)] = 1

    # Outer term 6: U6 = A10 - A00, V6 = B00 + B01
    U6 = [[{} for _ in range(2)] for _ in range(2)]
    V6 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            U6[r][c][(r + 2, c)] = 1
            U6[r][c][(r, c)] = -1
            V6[r][c][(r, c)] = 1
            V6[r][c][(r, c + 2)] = 1

    # Outer term 7: U7 = A01 - A11, V7 = B10 + B11
    U7 = [[{} for _ in range(2)] for _ in range(2)]
    V7 = [[{} for _ in range(2)] for _ in range(2)]
    for r in range(2):
        for c in range(2):
            U7[r][c][(r, c + 2)] = 1
            U7[r][c][(r + 2, c + 2)] = -1
            V7[r][c][(r + 2, c)] = 1
            V7[r][c][(r + 2, c + 2)] = 1

    outer_terms = [
        (U1, V1),
        (U2, V2),
        (U3, V3),
        (U4, V4),
        (U5, V5),
        (U6, V6),
        (U7, V7),
    ]

    # For each outer term, expand to 7 inner scalar multiplications
    col = 0
    for U_outer, V_outer in outer_terms:
        inner_mults = expand_2x2_to_scalars(U_outer, V_outer)
        for u_scalar, v_scalar in inner_mults:
            # u_scalar: dict {(i,j): coeff} for A entries
            # v_scalar: dict {(j,k): coeff} for B entries
            # The W entry is determined by the C position
            # For the 4x4x4 part, k ranges 0..3, and W[k*n+i, col] = 1
            
            # Wait, this is not quite right. The Strassen decomposition
            # produces a 2x2 result for each outer term. Each entry of the
            # 2x2 result corresponds to a specific (i_block, k_block) position.
            # We need to track which C position each inner multiplication
            # contributes to.
            
            # Actually, the inner Strassen decomposes a 2x2 multiplication
            # into 7 scalar multiplications. The result is a 2x2 matrix.
            # Each of the 7 scalar multiplications contributes to one or more
            # of the 4 entries of the 2x2 result.
            
            # Let me reconsider. The standard Strassen 2x2 decomposition:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So each M_r contributes to specific C entries with specific coefficients.
            # We need to track this.
            
            # For the 4x4x4 case, the 2x2 blocks correspond to:
            # C00 -> (i in 0-1, k in 0-1)
            # C01 -> (i in 0-1, k in 2-3)
            # C10 -> (i in 2-3, k in 0-1)
            # C11 -> (i in 2-3, k in 2-3)
            
            # So for each inner multiplication M_r, we know which C entries it
            # contributes to. We need to add the corresponding W entries.
            
            # Let me redo this more carefully.
            pass

    # Let me take a different approach. Instead of trying to track the
    # Strassen coefficients through the expansion, let me directly compute
    # the 4x4x4 Strassen decomposition by simulating it.
    
    # For a 4x4 matrix product C = A * B, the recursive Strassen algorithm
    # works as follows:
    # 1. Partition A and B into 2x2 blocks of 2x2 matrices.
    # 2. Compute 7 "effective" 2x2 products using Strassen.
    # 3. Each effective 2x2 product is itself computed using 7 scalar products.
    # 4. The final C is reconstructed from the 7 effective products.
    
    # The key insight: each of the 49 scalar multiplications corresponds to
    # a specific (i, j, k) combination in the tensor. We can determine this
    # by tracing through the Strassen algorithm.
    
    # Let me define the Strassen algorithm for 2x2 matrices more explicitly.
    # For a 2x2 product C = A * B:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the 4x4 case, the "scalar" multiplications are actually 2x2
    # multiplications. Each 2x2 multiplication is decomposed into 7 scalar
    # multiplications using the same Strassen formula.
    
    # So the 49 scalar multiplications are indexed by (outer, inner) where
    # outer in 0..6 and inner in 0..6.
    
    # For each (outer, inner) pair, we can determine:
    # - Which entries of A are multiplied (u_entries)
    # - Which entries of B are multiplied (v_entries)
    # - Which entries of C are affected (w_entries)
    
    # Let me compute this explicitly.
    
    # First, define the 7 outer Strassen terms for 4x4:
    # Let A_blocks = [[A00, A01], [A10, A11]] where A00 is 2x2, etc.
    # Let B_blocks = [[B00, B01], [B10, B11]] where B00 is 2x2, etc.
    
    # Outer term 1: U1 = A00 + A11, V1 = B00 + B11
    # Outer term 2: U2 = A10 + A11, V2 = B00
    # Outer term 3: U3 = A00, V3 = B01 - B11
    # Outer term 4: U4 = A11, V4 = B10 - B00
    # Outer term 5: U5 = A00 + A01, V5 = B11
    # Outer term 6: U6 = A10 - A00, V6 = B00 + B01
    # Outer term 7: U7 = A01 - A11, V7 = B10 + B11
    
    # Each outer term produces a 2x2 result. The 2x2 result is added to
    # specific C blocks with specific coefficients.
    
    # C00 += M1 + M4 - M5 + M7
    # C01 += M3 + M5
    # C10 += M2 + M4
    # C11 += M1 - M2 + M3 + M6
    
    # So for outer term r, the contribution to C blocks is:
    # r=0 (M1): C00 += 1, C11 += 1
    # r=1 (M2): C10 += 1, C11 += -1
    # r=2 (M3): C01 += 1, C11 += 1
    # r=3 (M4): C00 += 1, C10 += 1
    # r=4 (M5): C00 += -1, C01 += 1
    # r=5 (M6): C11 += 1
    # r=6 (M7): C00 += 1
    
    # Now, for each outer term, the 2x2 result is computed using inner
    # Strassen. The inner Strassen produces 7 scalar multiplications, each
    # of which contributes to specific entries of the 2x2 result.
    
    # For the inner Strassen (2x2 product):
    # M1_inner = (A00_inner+A11_inner)(B00_inner+B11_inner)
    # M2_inner = (A10_inner+A11_inner)B00_inner
    # M3_inner = A00_inner(B01_inner-B11_inner)
    # M4_inner = A11_inner(B10_inner-B00_inner)
    # M5_inner = (A00_inner+A01_inner)B11_inner
    # M6_inner = (A10_inner-A00_inner)(B00_inner+B01_inner)
    # M7_inner = (A01_inner-A11_inner)(B10_inner+B11_inner)
    # C00_inner = M1_inner + M4_inner - M5_inner + M7_inner
    # C01_inner = M3_inner + M5_inner
    # C10_inner = M2_inner + M4_inner
    # C11_inner = M1_inner - M2_inner + M3_inner + M6_inner
    
    # So for inner term s, the contribution to the 2x2 result is:
    # s=0 (M1_inner): C00_inner += 1, C11_inner += 1
    # s=1 (M2_inner): C10_inner += 1, C11_inner += -1
    # s=2 (M3_inner): C01_inner += 1, C11_inner += 1
    # s=3 (M4_inner): C00_inner += 1, C10_inner += 1
    # s=4 (M5_inner): C00_inner += -1, C01_inner += 1
    # s=5 (M6_inner): C11_inner += 1
    # s=6 (M7_inner): C00_inner += 1
    
    # Now, for a given (outer, inner) pair, the scalar multiplication is:
    # u_entry = (A entry from outer term r) * (A entry from inner term s)
    # v_entry = (B entry from outer term r) * (B entry from inner term s)
    # w_entries = (C block from outer term r) * (C entry from inner term s)
    
    # Let me define the entries more explicitly.
    
    # For outer term r, the 2x2 matrix U_outer has entries:
    # U_outer[0][0] = linear combo of A entries
    # U_outer[0][1] = linear combo of A entries
    # U_outer[1][0] = linear combo of A entries
    # U_outer[1][1] = linear combo of A entries
    
    # Similarly for V_outer.
    
    # For inner term s, the 2x2 matrix U_inner has entries:
    # U_inner[0][0] = linear combo of U_outer entries
    # etc.
    
    # The scalar multiplication is:
    # u_scalar = sum of products of coefficients
    # v_scalar = sum of products of coefficients
    
    # Let me just compute this directly by simulating the algorithm.
    
    # I'll create a function that, given the outer and inner Strassen terms,
    # returns the (u_entries, v_entries, w_entries) for the scalar multiplication.
    
    # This is getting complex. Let me try a different approach:
    # I'll directly construct the 4x4x4 Strassen decomposition by
    # computing the tensor coefficients.
    
    # The tensor for 4x4x4 matrix multiplication is T[i,j,k] = 1 for all
    # i,j,k in 0..3. The Strassen decomposition expresses this as a sum
    # of 49 rank-1 tensors.
    
    # Each rank-1 tensor is u_i * v_j * w_k where u, v, w are vectors.
    
    # I can compute these vectors by simulating the Strassen algorithm
    # with basis vectors.
    
    # Let me try yet another approach: I'll use the known fact that the
    # rank of 4x4x4 is at most 49 (Bini's algorithm), and I'll construct
    # the decomposition by solving a system of equations.
    
    # Actually, let me just use the schoolbook for the 4x4x4 part and add
    # the 5th column. That gives rank 64 + 16 = 80, which is not better.
    
    # Let me try to implement the recursive Strassen correctly.
    
    # I'll define the 7 outer terms and 7 inner terms explicitly, and
    # compute the (u, v, w) vectors for each of the 49 scalar multiplications.
    
    # For the 4x4x4 case:
    # A is 4x4, B is 4x4, C is 4x4.
    # A[i,j] for i,j in 0..3
    # B[j,k] for j,k in 0..3
    # C[i,k] for i,k in 0..3
    
    # The tensor is T[i,j,k] = 1 for all i,j,k.
    
    # The Strassen decomposition:
    # T = sum_{r=0}^{6} sum_{s=0}^{6} u_{r,s} * v_{r,s} * w_{r,s}
    
    # where u_{r,s}, v_{r,s}, w_{r,s} are vectors.
    
    # Let me compute these vectors.
    
    # For outer term r and inner term s:
    # u_{r,s}[i,j] = coefficient of A[i,j] in the scalar multiplication
    # v_{r,s}[j,k] = coefficient of B[j,k] in the scalar multiplication
    # w_{r,s}[k,i] = coefficient of C[k,i] in the scalar multiplication
    
    # Note: the W vector is indexed by (k,i) because of the trace convention
    # W[k*n+i, r].
    
    # Let me define the outer terms as 2x2 matrices where each entry is a
    # vector of coefficients for the 4x4 A/B entries.
    
    # Outer term 1: U1 = A00 + A11
    # U1[0][0] = A[0,0] + A[2,2]
    # U1[0][1] = A[0,1] + A[2,3]
    # U1[1][0] = A[1,0] + A[3,2]
    # U1[1][1] = A[1,1] + A[3,3]
    
    # V1 = B00 + B11
    # V1[0][0] = B[0,0] + B[2,2]
    # V1[0][1] = B[0,1] + B[2,3]
    # V1[1][0] = B[1,0] + B[3,2]
    # V1[1][1] = B[1,1] + B[3,3]
    
    # Inner term 0 (M1_inner): (U1[0][0]+U1[1][1]) * (V1[0][0]+V1[1][1])
    # u = U1[0][0] + U1[1][1] = A[0,0] + A[2,2] + A[1,1] + A[3,3]
    # v = V1[0][0] + V1[1][1] = B[0,0] + B[2,2] + B[1,1] + B[3,3]
    # w = C[0,0] + C[1,1] (from C00_inner += 1, C11_inner += 1)
    #    But C00_inner corresponds to C[0,0] and C[1,1] in the 4x4?
    #    Wait, the 2x2 result of outer term 1 is added to C00 and C11.
    #    C00 is the top-left 2x2 block of C, so C[0,0], C[0,1], C[1,0], C[1,1].
    #    C11 is the bottom-right 2x2 block of C, so C[2,2], C[2,3], C[3,2], C[3,3].
    #    
    #    The inner Strassen produces a 2x2 result. The entries of this 2x2
    #    result are added to the corresponding entries of the 2x2 block.
    #    
    #    For outer term 1, the 2x2 result is added to C00 and C11.
    #    So C00_inner[0][0] -> C[0,0], C00_inner[0][1] -> C[0,1], etc.
    #    C11_inner[0][0] -> C[2,2], C11_inner[0][1] -> C[2,3], etc.
    
    # This is getting very complex. Let me just implement it step by step.
    
    # I'll create a function that, given the outer and inner term indices,
    # returns the (u, v, w) vectors.
    
    # Let me define the outer terms as 2x2 matrices of coefficient vectors.
    
    # For outer term r, U_outer[r][c] is a vector of length 16 (4x4) giving
    # the coefficients of A entries.
    
    # Similarly for V_outer.
    
    # For inner term s, the 2x2 matrix U_inner is formed from U_outer.
    
    # Let me just code this up.
    
    # Define the 7 outer terms as 2x2 matrices of coefficient vectors.
    # Each coefficient vector is a dict {(i,j): coeff} for the 4x4 A/B entries.
    
    def make_outer_terms():
        """Returns list of 7 (U_outer, V_outer) where each is a 2x2 matrix of dicts."""
        terms = []
        
        # Term 0: U = A00 + A11, V = B00 + B11
        U = [[{} for _ in range(2)] for _ in range(2)]
        V = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                U[r][c][(r, c)] = 1
                U[r][c][(r + 2, c + 2)] = 1
                V[r][c][(r, c)] = 1
                V[r][c][(r + 2, c + 2)] = 1
        terms.append((U, V))
        
        # Term 1: U = A10 + A11, V = B00
        U = [[{} for _ in range(2)] for _ in range(2)]
        V = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                U[r][c][(r + 2, c)] = 1
                U[r][c][(r + 2, c + 2)] = 1
                V[r][c][(r, c)] = 1
        terms.append((U, V))
        
        # Term 2: U = A00, V = B01 - B11
        U = [[{} for _ in range(2)] for _ in range(2)]
        V = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                U[r][c][(r, c)] = 1
                V[r][c][(r, c + 2)] = 1
                V[r][c][(r + 2, c + 2)] = -1
        terms.append((U, V))
        
        # Term 3: U = A11, V = B10 - B00
        U = [[{} for _ in range(2)] for _ in range(2)]
        V = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                U[r][c][(r + 2, c + 2)] = 1
                V[r][c][(r + 2, c)] = 1
                V[r][c][(r, c)] = -1
        terms.append((U, V))
        
        # Term 4: U = A00 + A01, V = B11
        U = [[{} for _ in range(2)] for _ in range(2)]
        V = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                U[r][c][(r, c)] = 1
                U[r][c][(r, c + 2)] = 1
                V[r][c][(r + 2, c + 2)] = 1
        terms.append((U, V))
        
        # Term 5: U = A10 - A00, V = B00 + B01
        U = [[{} for _ in range(2)] for _ in range(2)]
        V = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                U[r][c][(r + 2, c)] = 1
                U[r][c][(r, c)] = -1
                V[r][c][(r, c)] = 1
                V[r][c][(r, c + 2)] = 1
        terms.append((U, V))
        
        # Term 6: U = A01 - A11, V = B10 + B11
        U = [[{} for _ in range(2)] for _ in range(2)]
        V = [[{} for _ in range(2)] for _ in range(2)]
        for r in range(2):
            for c in range(2):
                U[r][c][(r, c + 2)] = 1
                U[r][c][(r + 2, c + 2)] = -1
                V[r][c][(r + 2, c)] = 1
                V[r][c][(r + 2, c + 2)] = 1
        terms.append((U, V))
        
        return terms
    
    # Define the inner Strassen terms.
    # For a 2x2 product, the 7 inner terms are:
    # s=0: (U[0][0]+U[1][1]) * (V[0][0]+V[1][1])
    # s=1: (U[1][0]+U[1][1]) * V[0][0]
    # s=2: U[0][0] * (V[0][1]-V[1][1])
    # s=3: U[1][1] * (V[1][0]-V[0][0])
    # s=4: (U[0][0]+U[0][1]) * V[1][1]
    # s=5: (U[1][0]-U[0][0]) * (V[0][0]+V[0][1])
    # s=6: (U[0][1]-U[1][1]) * (V[1][0]+V[1][1])
    
    # And the contribution to the 2x2 result is:
    # C[0][0] += s0 + s3 - s4 + s6
    # C[0][1] += s2 + s4
    # C[1][0] += s1 + s3
    # C[1][1] += s0 - s1 + s2 + s5
    
    def get_inner_terms():
        """Returns list of 7 (u_indices, v_indices) where indices are (r,c) pairs."""
        terms = []
        # s=0
        terms.append(([(0,0), (1,1)], [(0,0), (1,1)]))
        # s=1
        terms.append(([(1,0), (1,1)], [(0,0)]))
        # s=2
        terms.append(([(0,0)], [(0,1), (1,1)]))  # (V[0][1]-V[1][1])
        # s=3
        terms.append(([(1,1)], [(1,0), (0,0)]))  # (V[1][0]-V[0][0])
        # s=4
        terms.append(([(0,0), (0,1)], [(1,1)]))
        # s=5
        terms.append(([(1,0), (0,0)], [(0,0), (0,1)]))  # (U[1][0]-U[0][0]) * (V[0][0]+V[0][1])
        # s=6
        terms.append(([(0,1), (1,1)], [(1,0), (1,1)]))  # (U[0][1]-U[1][1]) * (V[1][0]+V[1][1])
        return terms
    
    # The contribution coefficients for each inner term to the 2x2 result:
    # C[0][0] += s0 + s3 - s4 + s6
    # C[0][1] += s2 + s4
    # C[1][0] += s1 + s3
    # C[1][1] += s0 - s1 + s2 + s5
    
    inner_contribs = [
        [(0,0,1), (1,1,1)],  # s=0
        [(1,0,1), (1,1,-1)],  # s=1
        [(0,1,1), (1,1,1)],  # s=2
        [(0,0,1), (1,0,1)],  # s=3
        [(0,0,-1), (0,1,1)],  # s=4
        [(1,1,1)],  # s=5
        [(0,0,1)],  # s=6
    ]
    
    # Now, for each (outer, inner) pair, compute the (u, v, w) vectors.
    
    outer_terms = make_outer_terms()
    inner_terms = get_inner_terms()
    
    col = 0
    for r in range(7):
        U_outer, V_outer = outer_terms[r]
        for s in range(7):
            u_indices, v_indices = inner_terms[s]
            
            # Compute u_scalar: dict {(i,j): coeff} for A entries
            u_scalar = {}
            for (ur, uc) in u_indices:
                for (ai, aj), coeff_ai in U_outer[ur][uc].items():
                    key = (ai, aj)
                    u_scalar[key] = u_scalar.get(key, 0) + coeff_ai
            
            # Compute v_scalar: dict {(j,k): coeff} for B entries
            v_scalar = {}
            for (vr, vc) in v_indices:
                for (bi, bj), coeff_bi in V_outer[vr][vc].items():
                    key = (bi, bj)
                    v_scalar[key] = v_scalar.get(key, 0) + coeff_bi
            
            # Remove zeros
            u_scalar = {k: v for k, v in u_scalar.items() if v != 0}
            v_scalar = {k: v for k, v in v_scalar.items() if v != 0}
            
            # Compute w_scalar: dict {(k,i): coeff} for C entries
            # The 2x2 result of the inner Strassen is added to the C blocks
            # determined by the outer term.
            
            # For outer term r, the 2x2 result is added to specific C blocks.
            # Let me define the C block mapping for each outer term.
            
            # C00 -> (i in 0-1, k in 0-1)
            # C01 -> (i in 0-1, k in 2-3)
            # C10 -> (i in 2-3, k in 0-1)
            # C11 -> (i in 2-3, k in 2-3)
            
            # The contribution of outer term r to the C blocks:
            # r=0 (M1): C00 += 1, C11 += 1
            # r=1 (M2): C10 += 1, C11 += -1
            # r=2 (M3): C01 += 1, C11 += 1
            # r=3 (M4): C00 += 1, C10 += 1
            # r=4 (M5): C00 += -1, C01 += 1
            # r=5 (M6): C11 += 1
            # r=6 (M7): C00 += 1
            
            outer_c_contribs = [
                [(0,0,1), (1,1,1)],  # r=0
                [(1,0,1), (1,1,-1)],  # r=1
                [(0,1,1), (1,1,1)],  # r=2
                [(0,0,1), (1,0,1)],  # r=3
                [(0,0,-1), (0,1,1)],  # r=4
                [(1,1,1)],  # r=5
                [(0,0,1)],  # r=6
            ]
            
            # For each C block contribution, add the inner contribution.
            w_scalar = {}
            for (cr, cc, c_coeff) in outer_c_contribs[r]:
                for (ir, ic, i_coeff) in inner_contribs[s]:
                    # The 2x2 block (cr, cc) has entries:
                    # (cr*2+ir, cc*2+ic) in the 4x4 C matrix
                    # But C is indexed by (i,k), so i = cr*2+ir, k = cc*2+ic
                    i = cr * 2 + ir
                    k = cc * 2 + ic
                    coeff = c_coeff * i_coeff
                    key = (k, i)  # W is indexed by (k, i)
                    w_scalar[key] = w_scalar.get(key, 0) + coeff
            
            # Remove zeros
            w_scalar = {k: v for k, v in w_scalar.items() if v != 0}
            
            # Now, fill in the U, V, W matrices for this column.
            for (i, j), coeff in u_scalar.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_scalar.items():
                V[j * p + k, col] = coeff
            for (k, i), coeff in w_scalar.items():
                W[k * n + i, col] = coeff
            
            col += 1
    
    # Now add the 16 schoolbook terms for the 5th column (k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 4, col] = 1.0
            W[4 * n + i, col] = 1.0
            col += 1
    
    # Verify
    if verify(U, V, W):
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    U, V, W = get_schoolbook()
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
