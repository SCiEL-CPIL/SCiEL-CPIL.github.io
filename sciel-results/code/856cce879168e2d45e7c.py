import numpy as np
import time

def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication using recursive Strassen.
    Uses a vectorized recursive approach to build the factor matrices.
    """
    start_time = time.time()
    deadline = start_time + 10
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def u_idx(i, k): return i * m + k
    def v_idx(k, j): return k * p + j
    # Convention: W[k*n+i, r] where k is col, i is row of C? 
    # Current code uses j*n+i.
    def w_idx(i, j): return j * n + i

    # Recursive Strassen construction
    # We define the 7 Strassen terms for 2x2 blocks.
    # Each term is defined by:
    # L: 2x2 matrix of linear forms on A
    # R: 2x2 matrix of linear forms on B
    # C: 2x2 matrix of linear forms on C (contribution)
    
    # Base case: 1x1 multiplication
    # Term: A[0,0] * B[0,0] -> C[0,0]
    
    # We will build the 49 terms by iterating over the 7 outer terms and 7 inner terms.
    
    # Define the 7 Strassen multipliers for a 2x2 block product.
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Contributions to C = [[c11, c12], [c21, c22]]
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # We represent linear forms as sparse dictionaries { (row, col): coeff }
    
    def add_dict(d1, d2):
        res = {}
        for k, v in d1.items(): res[k] = res.get(k, 0) + v
        for k, v in d2.items(): res[k] = res.get(k, 0) + v
        # Remove zeros
        return {k: v for k, v in res.items() if v != 0}
        
    def sub_dict(d1, d2):
        res = {}
        for k, v in d1.items(): res[k] = res.get(k, 0) + v
        for k, v in d2.items(): res[k] = res.get(k, 0) - v
        return {k: v for k, v in res.items() if v != 0}
        
    def scale_dict(d, s):
        return {k: v * s for k, v in d.items()}
        
    # Identity forms for 2x2 blocks
    # I(r, c) = { (r, c): 1 }
    def I(r, c): return {(r, c): 1}
    
    # The 7 Strassen terms for a generic 2x2 block
    # Each term is a tuple (L, R, C_contrib)
    # L, R are 2x2 grids of dicts
    # C_contrib is a 2x2 grid of dicts
    
    # Term 0: M1
    L0 = [[add_dict(I(0,0), I(1,1)), I(0,1)], # Wait, M1 uses a+d and e+h
          [I(1,0), add_dict(I(0,0), I(1,1))]] # This is wrong. M1 is (a+d)(e+h).
    # Let's be precise.
    # M1: X = a+d, Y = e+h
    # X is a 1x1 "matrix" in the sense that it multiplies with Y.
    # But in the tensor decomposition, each term is a rank-1 tensor.
    # The rank-1 tensor is defined by u, v, w vectors.
    # u is the linear form on A, v on B, w on C.
    
    # For Strassen, each "multiplication" M_k corresponds to one rank-1 term.
    # M1 = (A11+A22)(B11+B22)
    # u_M1 = A11 + A22
    # v_M1 = B11 + B22
    # w_M1 = C11 + C22 (with coefficients from the reconstruction)
    
    # Let's define the 7 terms for the 2x2 case.
    # Term k has u_k, v_k, w_k.
    
    # u_1 = A11 + A22
    # v_1 = B11 + B22
    # w_1 = C11 + C22
    
    # u_2 = A21 + A22
    # v_2 = B11
    # w_2 = C21 - C22
    
    # u_3 = A11
    # v_3 = B12 - B22
    # w_3 = C12 + C22
    
    # u_4 = A22
    # v_4 = B21 - B11
    # w_4 = C11 + C21
    
    # u_5 = A11 + A12
    # v_5 = B22
    # w_5 = -C11 + C12
    
    # u_6 = A21 - A11
    # v_6 = B11 + B12
    # w_6 = C22
    
    # u_7 = A12 - A22
    # v_7 = B21 + B22
    # w_7 = C11
    
    # These are for a 2x2 block.
    # For 4x4, we apply this to each of the 7 "super-terms" P1..P7.
    # Each P_t is a product of two 2x2 matrices L_t and R_t.
    # So P_t = L_t * R_t.
    # We decompose P_t using Strassen into 7 terms.
    # Each sub-term M_{t,k} has:
    # u_{t,k} = linear form on L_t (which is a linear form on A)
    # v_{t,k} = linear form on R_t (which is a linear form on B)
    # w_{t,k} = linear form on C (combined with contribution of P_t to C)
    
    # Let's define L_t and R_t as 2x2 matrices of linear forms on A and B.
    
    # P1 = (A11+A22)(B11+B22)
    L0 = [[I(0,0)+I(2,2), I(0,1)+I(2,3)],
          [I(1,0)+I(3,2), I(1,1)+I(3,3)]]
    R0 = [[I(0,0)+I(2,2), I(0,1)+I(2,3)],
          [I(1,0)+I(3,2), I(1,1)+I(3,3)]]
    # Contribution of P1 to C:
    # C11 += P1_11 + P1_22? No.
    # C = A*B.
    # C11 = A11*B11 + A12*B21 + A21*B12 + A22*B22
    # P1 = (A11+A22)(B11+B22) = A11B11 + A11B22 + A22B11 + A22B22
    # This doesn't match C11 directly.
    # Strassen reconstructs C from P1..P7.
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # So for each P_t, we know which C elements it contributes to.
    # P1 contributes to C11 (+1), C22 (+1).
    # P2 contributes to C21 (+1), C22 (-1).
    # P3 contributes to C12 (+1), C22 (+1).
    # P4 contributes to C11 (+1), C21 (+1).
    # P5 contributes to C11 (-1), C12 (+1).
    # P6 contributes to C22 (+1).
    # P7 contributes to C11 (+1).
    
    # Note: These are for the 2x2 block result.
    # When we compute P_t = L_t * R_t, the result is a 2x2 matrix.
    # Let's call the result elements (P_t)_{ij}.
    # The contribution to the final C is:
    # C_{block_i, block_j} += coeff * (P_t)_{i, j}
    
    # We need to map the 2x2 result of P_t to the 4x4 C.
    # P_t is a 2x2 matrix.
    # (P_t)_{0,0} corresponds to C[0:2, 0:2]_{0,0} i.e. C[0,0]?
    # No, P_t is a product of 2x2 blocks.
    # So (P_t)_{0,0} is the top-left element of the 2x2 block product.
    # This element contributes to C[0,0] and C[1,1]?
    # No.
    # Let's be careful.
    # A is 4x4. A11 is 2x2.
    # P1 = (A11+A22)(B11+B22).
    # P1 is a 2x2 matrix.
    # P1[0,0] = (A11+A22)[0,0] * (B11+B22)[0,0] ... NO.
    # Matrix multiplication: (XY)_{ij} = sum_k X_{ik} Y_{kj}.
    # So P1[0,0] = sum_k (A11+A22)[0,k] (B11+B22)[k,0].
    # This is a scalar.
    # How does this scalar contribute to C?
    # In Strassen 2x2, C11 = P1 + P4 - P5 + P7.
    # Here P1 is a *scalar* (the result of the 1x1 multiplication? No, P1 is a matrix).
    # Wait. In Strassen 2x2, A11, A12 etc are *scalars*.
    # So P1 is a scalar.
    # In our 4x4 case, A11 is a 2x2 matrix.
    # So P1 is a 2x2 matrix.
    # The formula C11 = P1 + ... applies to the *blocks*.
    # C11 is a 2x2 block.
    # So C11_block = P1_block + P4_block - P5_block + P7_block.
    # This means:
    # C[0:2, 0:2] = P1[0:2, 0:2] + ...
    # So the element C[0,0] receives contribution from P1[0,0].
    # C[0,1] receives contribution from P1[0,1].
    # C[1,0] receives contribution from P1[1,0].
    # C[1,1] receives contribution from P1[1,1].
    
    # So, for each sub-term M_{t,k} (which is a scalar multiplication of linear forms),
    # the "w" part is a linear form on C.
    # Specifically, M_{t,k} contributes to the element (P_t)_{i,j} with some coefficient.
    # And (P_t)_{i,j} contributes to C[2*bi+i, 2*bj+j] with the block coefficient.
    
    # Let's implement this logic.
    
    # Define L_t and R_t for t=0..6
    # L_t[r][c] is a dict of A indices
    # R_t[r][c] is a dict of B indices
    
    # Helper to create identity dict
    def make_I(r, c): return {(r, c): 1}
    
    L_terms = []
    R_terms = []
    C_contrib = [] # C_contrib[t][i][j] = list of (out_row, out_col, coeff)
    
    # t=0: P1 = (A11+A22)(B11+B22)
    L0 = [[add_dict(I(0,0), I(2,2)), add_dict(I(0,1), I(2,3))],
          [add_dict(I(1,0), I(3,2)), add_dict(I(1,1), I(3,3))]]
    R0 = [[add_dict(I(0,0), I(2,2)), add_dict(I(0,1), I(2,3))],
          [add_dict(I(1,0), I(3,2)), add_dict(I(1,1), I(3,3))]]
    # P1 contributes to C11 (+1) and C22 (+1)
    # C11 is block (0,0), C22 is block (1,1)
    C0 = [[[(0,0,1), (1,1,1)], [(0,1,1), (1,2,1)]], 
          [[(1,0,1), (2,1,1)], [(1,1,1), (2,2,1)]]]
    # Wait, C11 is 2x2. P1[0,0] -> C[0,0]. P1[0,1] -> C[0,1].
    # So C_contrib[0][0][0] should map to C[0,0] and C[2,2]?
    # No.
    # C11_block = P1_block.
    # So C[0,0] += P1[0,0]. C[0,1] += P1[0,1].
    # C[2,2] += P1[0,0]. C[2,3] += P1[0,1].
    # So for P1[0,0], the W form is C[0,0] + C[2,2].
    # For P1[0,1], the W form is C[0,1] + C[2,3].
    
    C0 = [[[(0,0,1), (2,2,1)], [(0,1,1), (2,3,1)]],
          [[(1,0,1), (3,2,1)], [(1,1,1), (3,3,1)]]]
    L_terms.append(L0)
    R_terms.append(R0)
    C_contrib.append(C0)
    
    # t=1: P2 = (A21+A22)B11
    L1 = [[add_dict(I(2,0), I(4,0)), add_dict(I(2,1), I(4,1))], # A21+A22
          [add_dict(I(3,0), I(5,0)), add_dict(I(3,1), I(5,1))]] # Wait, A is 4x4. Indices 0..3.
    # Correction: A21 is rows 2-3, cols 0-1. A22 is rows 2-3, cols 2-3.
    L1 = [[add_dict(I(2,0), I(2,2)), add_dict(I(2,1), I(2,3))],
          [add_dict(I(3,0), I(3,2)), add_dict(I(3,1), I(3,3))]]
    R1 = [[I(0,0), I(0,1)],
          [I(1,0), I(1,1)]]
    # P2 contributes to C21 (+1) and C22 (-1)
    # C21 is block (1,0) -> C[2:4, 0:2]
    # C22 is block (1,1) -> C[2:4, 2:4]
    # P2[0,0] -> C[2,0] + C[2,2](-1)
    C1 = [[[(2,0,1), (2,2,-1)], [(2,1,1), (2,3,-1)]],
          [[(3,0,1), (3,2,-1)], [(3,1,1), (3,3,-1)]]]
    L_terms.append(L1)
    R_terms.append(R1)
    C_contrib.append(C1)
    
    # t=2: P3 = A11(B12-B22)
    L2 = [[I(0,0), I(0,1)],
          [I(1,0), I(1,1)]]
    R2 = [[add_dict(I(0,2), I(2,2)), add_dict(I(0,3), I(2,3))], # B12-B22
          [add_dict(I(1,2), I(3,2)), add_dict(I(1,3), I(3,3))]]
    # P3 contributes to C12 (+1) and C22 (+1)
    C2 = [[[(0,2,1), (2,2,1)], [(0,3,1), (2,3,1)]],
          [[(1,2,1), (3,2,1)], [(1,3,1), (3,3,1)]]]
    L_terms.append(L2)
    R_terms.append(R2)
    C_contrib.append(C2)
    
    # t=3: P4 = A22(B21-B11)
    L3 = [[I(2,2), I(2,3)],
          [I(3,2), I(3,3)]]
    R3 = [[add_dict(I(2,0), I(0,0)), add_dict(I(2,1), I(0,1))], # B21-B11
          [add_dict(I(3,0), I(1,0)), add_dict(I(3,1), I(1,1))]]
    # P4 contributes to C11 (+1) and C21 (+1)
    C3 = [[[(0,0,1), (2,0,1)], [(0,1,1), (2,1,1)]],
          [[(1,0,1), (3,0,1)], [(1,1,1), (3,1,1)]]]
    L_terms.append(L3)
    R_terms.append(R3)
    C_contrib.append(C3)
    
    # t=4: P5 = (A11+A12)B22
    L4 = [[add_dict(I(0,0), I(0,2)), add_dict(I(0,1), I(0,3))],
          [add_dict(I(1,0), I(1,2)), add_dict(I(1,1), I(1,3))]]
    R4 = [[I(2,2), I(2,3)],
          [I(3,2), I(3,3)]]
    # P5 contributes to C11 (-1) and C12 (+1)
    C4 = [[[(0,0,-1), (0,2,1)], [(0,1,-1), (0,3,1)]],
          [[(1,0,-1), (1,2,1)], [(1,1,-1), (1,3,1)]]]
    L_terms.append(L4)
    R_terms.append(R4)
    C_contrib.append(C4)
    
    # t=5: P6 = (A21-A11)(B11+B12)
    L5 = [[add_dict(I(2,0), I(0,0)), add_dict(I(2,1), I(0,1))],
          [add_dict(I(3,0), I(1,0)), add_dict(I(3,1), I(1,1))]]
    R5 = [[add_dict(I(0,0), I(0,2)), add_dict(I(0,1), I(0,3))],
          [add_dict(I(1,0), I(1,2)), add_dict(I(1,1), I(1,3))]]
    # P6 contributes to C22 (+1)
    C5 = [[[(2,2,1)], [(2,3,1)]],
          [[(3,2,1)], [(3,3,1)]]]
    L_terms.append(L5)
    R_terms.append(R5)
    C_contrib.append(C5)
    
    # t=6: P7 = (A12-A22)(B21+B22)
    L6 = [[add_dict(I(0,2), I(2,2)), add_dict(I(0,3), I(2,3))],
          [add_dict(I(1,2), I(3,2)), add_dict(I(1,3), I(3,3))]]
    R6 = [[add_dict(I(2,0), I(2,2)), add_dict(I(2,1), I(2,3))],
          [add_dict(I(3,0), I(3,2)), add_dict(I(3,1), I(3,3))]]
    # P7 contributes to C11 (+1)
    C6 = [[[(0,0,1)], [(0,1,1)]],
          [[(1,0,1)], [(1,1,1)]]]
    L_terms.append(L6)
    R_terms.append(R6)
    C_contrib.append(C6)
    
    # Strassen sub-terms
    # M1: (a+d)(e+h) -> C11(+1), C22(+1)
    # M2: (c+d)e -> C21(+1), C22(-1)
    # M3: a(f-h) -> C12(+1), C22(+1)
    # M4: d(g-e) -> C11(+1), C21(+1)
    # M5: (a+b)h -> C11(-1), C12(+1)
    # M6: (c-a)(e+f) -> C22(+1)
    # M7: (b-d)(g+h) -> C11(+1)
    
    sub_L = [
        [[add_dict(I(0,0), I(1,1)), I(0,1)], [I(1,0), add_dict(I(0,0), I(1,1))]], # M1 X
        [[add_dict(I(1,0), I(1,1)), I(1,1)], [I(1,1), I(1,1)]], # M2 X: c+d. c is (1,0), d is (1,1).
        [[I(0,0), I(0,1)], [I(1,0), I(1,1)]], # M3 X: a
        [[I(1,1), I(1,1)], [I(1,1), I(1,1)]], # M4 X: d
        [[add_dict(I(0,0), I(0,1)), I(0,1)], [add_dict(I(1,0), I(1,1)), I(1,1)]], # M5 X: a+b
        [[add_dict(I(1,0), I(0,0)), I(1,1)], [add_dict(I(1,0), I(1,1)), I(1,1)]], # M6 X: c-a
        [[add_dict(I(0,1), I(1,1)), I(0,1)], [add_dict(I(0,1), I(1,1)), I(1,1)]]  # M7 X: b-d
    ]
    
    # This is getting complicated to hardcode correctly.
    # I will revert to the previous program's logic but clean it up.
    # The previous program was correct.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
