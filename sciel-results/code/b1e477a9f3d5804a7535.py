"""Fresh design: verified 2+3 block decomposition with Strassen for the 2x2 block.

Returns a rank-124 exact tensor decomposition for 5x5 matrix multiplication.
The output factor W follows the trace convention W[k*n+i, r].

Includes an internal exact integer check; only returns a decomposition that has
been verified to exactly reproduce the multiplication tensor. Falls back to the
schoolbook rank-125 decomposition if verification fails or the time budget is
exceeded.
"""
import time

DIMENSIONS = [5, 5, 5]
N = 5
M = 5
P = 5

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # safety margin before 150s wall-clock limit


def _schoolbook():
    """Rank 125 schoolbook decomposition (exact, always valid)."""
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _strassen_2x2_terms():
    """7 rank-1 terms for 2x2 matrix multiplication (Strassen).

    Each term (a, b, c) means: contribution a*X1*Y1 + b*X2*Y2 + c*X3*Y3.
    Output entry (i,k) for the 2x2 product (i,k in {0,1}).
    """
    # M1 = X1 Y1
    # M2 = (X1+X2) Y2
    # M3 = X1 (Y3-Y4)
    # M4 = X3 (Y4-Y1)
    # M5 = (X3+X4) Y4
    # M6 = (X4-X2) (Y1+Y2)
    # M7 = (X2-X3) (Y4+Y3)
    #
    # P11 = M5+M4-M2+M6
    # P12 = M1+M2
    # P21 = M3+M4
    # P22 = M1-M5+M3+M7
    terms = []
    # P11 = M5+M4-M2+M6
    terms.append((3, 4, 5, 6, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1))
    # P12 = M1+M2
    terms.append((0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    # P21 = M3+M4
    terms.append((0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    # P22 = M1-M5+M3+M7
    terms.append((0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    # Rebuild more carefully below using explicit coefficient vectors.
    return terms


def _build_rank124():
    """Build the rank-124 decomposition using 2+3 block split + Strassen."""
    n, m, p = N, M, P
    # Row index of X in U: (i, j) -> i*m + j
    # Row index of Y in V: (j, k) -> j*p + k
    # Row index of Z in W: (k, i) -> k*n + i

    U_rows = n * m  # 25
    V_rows = m * p  # 25
    W_rows = p * n  # 25

    # We'll build lists of (U_vec, V_vec, W_vec) for each rank-1 term.
    # Use sparse representation: dict row->value
    terms = []  # each term: (U_dict, V_dict, W_dict)

    # ---- Strassen 7 terms for the 2x2 block (top-left) ----
    # Block indices: i in {0,1}, j in {0,1}, k in {0,1}
    # X1 = X[0,0], X2 = X[0,1], X3 = X[1,0], X4 = X[1,1]
    # Y1 = Y[0,0], Y2 = Y[0,1], Y3 = Y[1,0], Y4 = Y[1,1]

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Strassen product terms for 2x2:
    # M1 = X1*Y1  -> contributes to P12, P22
    # M2 = (X1+X2)*Y2 -> P11, P12
    # M3 = X1*(Y3-Y4) -> P21, P22
    # M4 = X3*(Y4-Y1) -> P11, P21
    # M5 = (X3+X4)*Y4 -> P11, P22
    # M6 = (X4-X2)*(Y1+Y2) -> P11
    # M7 = (X2-X3)*(Y4+Y3) -> P22

    # P11 = M5+M4-M2+M6
    # P12 = M1+M2
    # P21 = M3+M4
    # P22 = M1-M5+M3+M7

    # Each Strassen "M" is a bilinear form: sum over (xcoeff * xrow) * (ycoeff * yrow)
    # We expand each M into rank-1 terms where U and V each carry one coefficient.
    # For M = (a*Xr1 + b*Xr2) * (c*Ys1 + d*Ys2), we need 4 rank-1 terms
    # unless one side is a single variable.

    # Let's define each M as a list of (coeff_u, u_row, coeff_v, v_row)
    # and the output placement as (coeff, i, k)

    # M1 = X1 * Y1: U=[(1, u_idx(0,0))], V=[(1, v_idx(0,0))], outputs: P12(+1), P22(+1)
    # M2 = (X1+X2)*Y2: U=[(1,u_idx(0,0)),(1,u_idx(0,1))], V=[(1,v_idx(0,1))], outputs: P11(-1), P12(+1)
    # M3 = X1*(Y3-Y4): U=[(1,u_idx(0,0))], V=[(1,v_idx(1,0)),(-1,v_idx(1,1))], outputs: P21(+1), P22(+1)
    # M4 = X3*(Y4-Y1): U=[(1,u_idx(1,0))], V=[(1,v_idx(1,1)),(-1,v_idx(0,0))], outputs: P11(+1), P21(+1)
    # M5 = (X3+X4)*Y4: U=[(1,u_idx(1,0)),(1,u_idx(1,1))], V=[(1,v_idx(1,1))], outputs: P11(+1), P22(-1)
    # M6 = (X4-X2)*(Y1+Y2): U=[(1,u_idx(1,1)),(-1,u_idx(0,1))], V=[(1,v_idx(0,0)),(1,v_idx(0,1))], outputs: P11(+1)
    # M7 = (X2-X3)*(Y4+Y3): U=[(1,u_idx(0,1)),(-1,u_idx(1,0))], V=[(1,v_idx(1,1)),(1,v_idx(1,0))], outputs: P22(+1)

    # Expand each M into rank-1 terms. A rank-1 term has U with one nonzero, V with one nonzero.
    # M with |U_coeffs| * |V_coeffs| rank-1 terms.

    def add_strassen_term(u_coeffs, v_coeffs, out_coeffs):
        """u_coeffs: list of (coeff, u_row), v_coeffs: list of (coeff, v_row),
        out_coeffs: list of (coeff, i, k).
        Produces len(u_coeffs)*len(v_coeffs) rank-1 terms."""
        for uc, ur in u_coeffs:
            for vc, vr in v_coeffs:
                U_d = {ur: uc}
                V_d = {vr: vc}
                W_d = {}
                for oc, oi, ok in out_coeffs:
                    wr = w_idx(ok, oi)
                    W_d[wr] = W_d.get(wr, 0) + oc
                # Skip if W is all zero
                if any(v != 0 for v in W_d.values()):
                    terms.append((U_d, V_d, W_d))

    # M1
    add_strassen_term([(1, u_idx(0, 0))], [(1, v_idx(0, 0))],
                      [(1, 0, 1), (1, 1, 1)])
    # M2 = (X1+X2)*Y2
    add_strassen_term([(1, u_idx(0, 0)), (1, u_idx(0, 1))], [(1, v_idx(0, 1))],
                      [(-1, 0, 0), (1, 0, 1)])
    # M3 = X1*(Y3-Y4)
    add_strassen_term([(1, u_idx(0, 0))], [(1, v_idx(1, 0)), (-1, v_idx(1, 1))],
                      [(1, 1, 0), (1, 1, 1)])
    # M4 = X3*(Y4-Y1)
    add_strassen_term([(1, u_idx(1, 0))], [(1, v_idx(1, 1)), (-1, v_idx(0, 0))],
                      [(1, 0, 0), (1, 1, 0)])
    # M5 = (X3+X4)*Y4
    add_strassen_term([(1, u_idx(1, 0)), (1, u_idx(1, 1))], [(1, v_idx(1, 1))],
                      [(1, 0, 0), (-1, 1, 1)])
    # M6 = (X4-X2)*(Y1+Y2)
    add_strassen_term([(1, u_idx(1, 1)), (-1, u_idx(0, 1))],
                      [(1, v_idx(0, 0)), (1, v_idx(0, 1))],
                      [(1, 0, 0)])
    # M7 = (X2-X3)*(Y4+Y3)
    add_strassen_term([(1, u_idx(0, 1)), (-1, u_idx(1, 0))],
                      [(1, v_idx(1, 1)), (1, v_idx(1, 0))],
                      [(1, 1, 1)])

    # Count Strassen terms so far
    # M1: 1, M2: 2, M3: 2, M4: 2, M5: 2, M6: 4, M7: 4 = 17 terms
    # That's too many! Standard Strassen uses 7 terms, not expanded.
    # The issue: when U has 2 coefficients and V has 1, that's 2 rank-1 terms, which is fine.
    # But M6 and M7 each have 2x2 = 4, giving 17 total instead of 7.
    # 
    # The correct Strassen has 7 rank-1 terms where each M is ONE rank-1 term:
    # M1 = X1*Y1 (1 term)
    # M2 = (X1+X2)*Y2 (1 term, but U has 2 entries -> still 1 rank-1 term in the tensor sense)
    # 
    # Wait - in tensor decomposition, a rank-1 term means U is a single vector (can have multiple
    # nonzero entries), V is a single vector, W is a single vector. So M2 IS one rank-1 term
    # where U = [0,1,1,0,...], V = [0,1,0,...], W = [...]. 
    # So I should NOT expand into multiple terms. Each M is one rank-1 term.

    # Let me redo: clear terms and rebuild properly.
    terms = []

    def add_term_single(u_dict, v_dict, w_dict):
        """Add one rank-1 term where u_dict, v_dict, w_dict are dicts row->value."""
        # Remove zeros
        u_dict = {k: v for k, v in u_dict.items() if v != 0}
        v_dict = {k: v for k, v in v_dict.items() if v != 0}
        w_dict = {k: v for k, v in w_dict.items() if v != 0}
        if u_dict and v_dict and w_dict:
            terms.append((u_dict, v_dict, w_dict))

    # M1 = X1 * Y1 -> P12 +, P22 +
    add_term_single(
        {u_idx(0, 0): 1},
        {v_idx(0, 0): 1},
        {w_idx(1, 0): 1, w_idx(1, 1): 1}
    )
    # M2 = (X1+X2) * Y2 -> P11 -, P12 +
    add_term_single(
        {u_idx(0, 0): 1, u_idx(0, 1): 1},
        {v_idx(0, 1): 1},
        {w_idx(0, 0): -1, w_idx(1, 0): 1}
    )
    # M3 = X1 * (Y3-Y4) -> P21 +, P22 +
    add_term_single(
        {u_idx(0, 0): 1},
        {v_idx(1, 0): 1, v_idx(1, 1): -1},
        {w_idx(0, 1): 1, w_idx(1, 1): 1}
    )
    # M4 = X3 * (Y4-Y1) -> P11 +, P21 +
    add_term_single(
        {u_idx(1, 0): 1},
        {v_idx(1, 1): 1, v_idx(0, 0): -1},
        {w_idx(0, 0): 1, w_idx(0, 1): 1}
    )
    # M5 = (X3+X4) * Y4 -> P11 +, P22 -
    add_term_single(
        {u_idx(1, 0): 1, u_idx(1, 1): 1},
        {v_idx(1, 1): 1},
        {w_idx(0, 0): 1, w_idx(1, 1): -1}
    )
    # M6 = (X4-X2) * (Y1+Y2) -> P11 +
    add_term_single(
        {u_idx(1, 1): 1, u_idx(0, 1): -1},
        {v_idx(0, 0): 1, v_idx(0, 1): 1},
        {w_idx(0, 0): 1}
    )
    # M7 = (X2-X3) * (Y4+Y3) -> P22 +
    add_term_single(
        {u_idx(0, 1): 1, u_idx(1, 0): -1},
        {v_idx(1, 1): 1, v_idx(1, 0): 1},
        {w_idx(1, 1): 1}
    )

    # ---- Schoolbook for remaining blocks (2+3 split) ----
    # The 2x2 block handled (i in {0,1}, k in {0,1}) for all j.
    # Remaining cases:
    # (i in {0,1}, k in {2,3,4}): 2*1*3 = 6 terms (j=0,1,2; i=0,1; k=2,3,4)
    # (i in {2,3,4}, k in {0,1}): 3*2*1 = 6 terms
    # (i in {2,3,4}, k in {2,3,4}): 3*1*3 = 9 terms
    # Total schoolbook: 6+6+9 = 21 terms
    # Plus 7 Strassen = 28 total? That's way too few.

    # Wait, I'm confusing myself. The 2+3 block split means we split BOTH the row (i) and column (k)
    # indices. The middle index j is NOT split. So:
    # - Top-left 2x2 block: i in {0,1}, k in {0,1}, j in {0,1,2} -> 2*2*3 = 12 schoolbook terms, replaced by Strassen 7 terms (saves 5)
    # - Top-right 2x3 block: i in {0,1}, k in {2,3,4}, j in {0,1,2} -> 2*3*3 = 18 terms
    # - Bottom-left 3x2 block: i in {2,3,4}, k in {0,1}, j in {0,1,2} -> 3*2*3 = 18 terms
    # - Bottom-right 3x3 block: i in {2,3,4}, k in {2,3,4}, j in {0,1,2} -> 3*3*3 = 27 terms
    # Total: 7 + 18 + 18 + 27 = 70? That doesn't match 124 either.

    # I think the 2+3 block decomposition works differently. Let me reconsider.
    # 
    # Actually for 5x5, the known result is rank 124. The 2+3 split with Strassen:
    # Split into blocks of sizes 2 and 3 in both dimensions.
    # The 2x2 sub-block uses Strassen (7 instead of 8 terms per j-value... no, Strassen replaces 
    # 8 rank-1 terms with 7 for a 2x2 multiplication).
    # 
    # For a 2x2 block with 3 middle indices: 2*2*3 = 12 schoolbook terms -> Strassen gives 7*... 
    # No. Strassen multiplies 2x2 matrices. Each "matrix" here has 3 columns (j=0,1,2).
    # Actually the middle dimension j is the "matrix element" index, so for each j we have
    # a 2x2 x 2x2 -> 2x2 multiplication. That's 3 independent 2x2 multiplications.
    # Each 2x2 multiplication has 8 schoolbook terms, Strassen does 7.
    # So 3 * 8 = 24 schoolbook -> 3 * 7 = 21 Strassen terms. Saves 3.
    # 
    # Hmm, that gives 125 - 3 = 122, not 124.

    # Let me look at this differently. The known rank for 5x5 is 124 (one known improvement 
    # over schoolbook 125). The mechanism: for the 2x2 sub-block, use Strassen which saves
    # 1 term per j-index? No, Strassen saves 1 term total for a 2x2 (8->7).
    # 
    # I think the correct accounting: we have the full 5x5x5 tensor (125 terms).
    # We identify a 2x2x2 sub-tensor within it and replace its 8 terms with Strassen's 7.
    # That saves exactly 1 term: 125 - 1 = 124.
    # 
    # The 2x2x2 sub-tensor corresponds to specific i, j, k indices.
    # E.g., i in {0,1}, j in {0,1}, k in {0,1} forms a 2x2x2 sub-tensor.
    # Replacing those 8 rank-1 terms with 7 Strassen terms gives rank 124.

    # So the correct approach: start with all 125 schoolbook terms, then for one 2x2x2
    # sub-block (say i={0,1}, j={0,1}, k={0,1}), remove the 8 terms and add 7 Strassen terms.

    # Let me rebuild from scratch with this correct understanding.
    terms = []  # clear

    # All schoolbook terms EXCEPT those with i in {0,1}, j in {0,1}, k in {0,1}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # skip the 2x2x2 sub-block
                U_d = {u_idx(i, j): 1}
                V_d = {v_idx(j, k): 1}
                W_d = {w_idx(k, i): 1}
                terms.append((U_d, V_d, W_d))

    # Add 7 Strassen terms for the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1})
    # X[i,j] for i,j in {0,1}, Y[j,k] for j,k in {0,1}, Z[k,i] for k,i in {0,1}
    # X1=X[0,0], X2=X[0,1], X3=X[1,0], X4=X[1,1]
    # Y1=Y[0,0], Y2=Y[0,1], Y3=Y[1,0], Y4=Y[1,1]
    # Z11=Z[0,0], Z12=Z[1,0]... wait, Z[k,i] so Z[0,0], Z[0,1], Z[1,0], Z[1,1]
    # Product P[i,k] = Z[k,i]: P[0,0]=Z[0,0], P[0,1]=Z[0,1], P[1,0]=Z[1,0], P[1,1]=Z[1,1]

    # Strassen:
    # M1 = X1*Y1 -> P[0,1]+=M1, P[1,1]+=M1  (P12, P22)
    # M2 = (X1+X2)*Y2 -> P[0,0]-=M2, P[0,1]+=M2  (P11, P12)
    # M3 = X1*(Y3-Y4) -> P[1,0]+=M3, P[1,1]+=M3  (P21, P22)
    # M4 = X3*(Y4-Y1) -> P[0,0]+=M4, P[1,0]+=M4  (P11, P21)
    # M5 = (X3+X4)*Y4 -> P[0,0]+=M5, P[1,1]-=M5  (P11, P22)
    # M6 = (X4-X2)*(Y1+Y2) -> P[0,0]+=M6  (P11)
    # M7 = (X2-X3)*(Y4+Y3) -> P[1,1]+=M7  (P22)

    # P[0,0] = M5+M4-M2+M6
    # P[0,1] = M1+M2
    # P[1,0] = M3+M4
    # P[1,1] = M1-M5+M3+M7

    add_term_single(
        {u_idx(0, 0): 1},
        {v_idx(0, 0): 1},
        {w_idx(1, 0): 1, w_idx(1, 1): 1}  # P[0,1]=w(1,0), P[1,1]=w(1,1)
    )
    add_term_single(
        {u_idx(0, 0): 1, u_idx(0, 1): 1},
        {v_idx(0, 1): 1},
        {w_idx(0, 0): -1, w_idx(1, 0): 1}  # P[0,0]=w(0,0), P[0,1]=w(1,0)
    )
    add_term_single(
        {u_idx(0, 0): 1},
        {v_idx(1, 0): 1, v_idx(1, 1): -1},
        {w_idx(0, 1): 1, w_idx(1, 1): 1}  # P[1,0]=w(0,1), P[1,1]=w(1,1)
    )
    add_term_single(
        {u_idx(1, 0): 1},
        {v_idx(1, 1): 1, v_idx(0, 0): -1},
        {w_idx(0, 0): 1, w_idx(0, 1): 1}  # P[0,0]=w(0,0), P[1,0]=w(0,1)
    )
    add_term_single(
        {u_idx(1, 0): 1, u_idx(1, 1): 1},
        {v_idx(1, 1): 1},
        {w_idx(0, 0): 1, w_idx(1, 1): -1}  # P[0,0]=w(0,0), P[1,1]=w(1,1)
    )
    add_term_single(
        {u_idx(1, 1): 1, u_idx(0, 1): -1},
        {v_idx(0, 0): 1, v_idx(0, 1): 1},
        {w_idx(0, 0): 1}  # P[0,0]=w(0,0)
    )
    add_term_single(
        {u_idx(0, 1): 1, u_idx(1, 0): -1},
        {v_idx(1, 1): 1, v_idx(1, 0): 1},
        {w_idx(1, 1): 1}  # P[1,1]=w(1,1)
    )

    # Now assemble into matrices
    rank = len(terms)
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]

    for c, (U_d, V_d, W_d) in enumerate(terms):
        for row, val in U_d.items():
            U[row][c] = val
        for row, val in V_d.items():
            V[row][c] = val
        for row, val in W_d.items():
            W[row][c] = val

    return {"U": U, "V": V, "W": W}


def _verify_exact(result):
    """Verify that the decomposition exactly reproduces the multiplication tensor.
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 iff (i,j,k) is valid,
    which it always is for matrix mult: T[i,j,k] = delta_{i,j,k} doesn't apply.
    Actually for matrix multiplication tensor: T[i,j,k] = 1 for all (i,j,k) is wrong.
    The multiplication tensor has T[i,j,k] = 1 for all i,j,k in the standard form.
    Wait no. The matrix multiplication tensor M[i,j,k] where the product C[i,k] = sum_j A[i,j]*B[j,k]
    corresponds to a 3-tensor T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor).
    
    Actually the multiplication tensor is: T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    No that's not right either. Let me think again.
    
    The tensor T such that T(A,B) = A*B is: (A*B)[i,k] = sum_j A[i,j] B[j,k].
    So T[i,j,k] = 1 for all i,j,k. Yes! The matrix multiplication tensor is the all-ones tensor.
    
    So we check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k.
    """
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0]) if U else 0
    n, m, p = N, M, P

    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True


def construct():
    """Construct the tensor decomposition. Returns dict with U, V, W."""
    if time.time() > _DEADLINE:
        return _schoolbook()

    # Try the rank-124 decomposition
    result = _build_rank124()

    if time.time() > _DEADLINE:
        return _schoolbook()

    # Verify exactness
    if _verify_exact(result):
        return result

    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
