import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 108.0

    n, m, p = 6, 6, 6
    nm, mp, pn = 36, 36, 36

    # Build the rank-216 schoolbook decomposition as a baseline
    rank_school = 216
    U_s = np.zeros((nm, rank_school))
    V_s = np.zeros((mp, rank_school))
    W_s = np.zeros((pn, rank_school))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U_s[i * m + j, col] = 1.0
                V_s[j * p + k, col] = 1.0
                W_s[k * n + i, col] = 1.0

    # BCLR rank-21 decomposition for 3x3 matrix multiplication
    # T_{i,j,k} = sum_r U_{i,j}^r * V_{j,k}^r * W_{k,i}^r
    # Using the known BCLR (Bini-Capovani-Lotti-Romani) decomposition
    # where the 21 terms are built from 7 Strassen-like terms each
    # contributing 3 terms from the 3x3 structure.
    
    r3 = 21
    U3 = np.zeros((9, r3))
    V3 = np.zeros((9, r3))
    W3 = np.zeros((9, r3))
    
    # BCLR decomposition: we use the standard 7-term Strassen structure
    # but each "block multiplication" of 3x3 matrices uses a rank-3
    # decomposition (the minimum for 3x3 over reals is 19, but we use
    # a structured approach). Actually, let's use the known BCLR rank-21.
    
    # The BCLR decomposition for 3x3: 
    # P1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    # P2 = (A21+A22)B11 -> C21(+), C22(-)
    # P3 = A11(B12-B22) -> C12(+), C22(+)
    # P4 = A22(B21-B11) -> C11(+), C21(+)
    # P5 = (A11+A12)B22 -> C12(+), C11(-)
    # P6 = (A21-A11)(B11+B12) -> C22(+)
    # P7 = (A12-A22)(B21+B22) -> C11(+)
    # But each of these involves 3x3 matrix products, which we decompose
    # into 3 scalar products each (using the diagonal structure).
    
    # Actually, the correct BCLR approach: each Strassen term P_t = X_t * Y_t
    # where X_t and Y_t are 3x3 matrices. We need to decompose each 3x3
    # matrix product into rank-3 terms. The key insight is that we can
    # use the fact that a 3x3 matrix product can be written with rank 3
    # if we allow the U, V, W factors to be structured.
    
    # Let me use the explicit BCLR rank-21 coefficients.
    # The 21 terms are indexed as (t, s) where t in 0..6 (Strassen term)
    # and s in 0..2 (inner rank-3 term).
    
    # For each Strassen term t, we have matrices A_t, B_t (3x3) such that
    # the contribution to C is sum_t (linear combo of A blocks) * (linear combo of B blocks)
    # mapped to C blocks. We decompose each 3x3 product into 3 rank-1 terms.
    
    # The rank-3 decomposition of a 3x3 matrix product M*N:
    # Use the fact that we can write it as sum of 3 outer products
    # by using the diagonal of one and appropriate combinations.
    # Actually, the standard approach: use the 3 diagonal elements.
    # (MN)_{ik} = sum_j M_{ij} N_{jk}
    # This is rank 9 in general. But for the BCLR, we use a different
    # structured decomposition.
    
    # Let me just directly code the known BCLR rank-21 for 3x3.
    # Reference: The 21 terms are as follows (using 1-indexed, converting to 0-indexed):
    
    # I'll use the explicit formulation where each of the 7 Strassen
    # "block" terms is expanded using 3 rank-1 terms based on the
    # structure of 3x3 matrix multiplication.
    
    # Term structure for 3x3: for a product X*Y of two 3x3 matrices,
    # we can write it as sum_{s=0}^{2} u_s * v_s * w_s where
    # u_s is a 3x3 matrix (for U), v_s is 3x3 (for V), w_s is 3x3 (for W)
    # such that sum_s u_s[i,j] * v_s[j,k] * w_s[k,i] = (X*Y)_{i,k}
    
    # A simple rank-3 decomposition: use the columns of X and rows of Y
    # but that gives rank 3 only if we're clever. Actually, let's use:
    # For s in 0,1,2:
    #   U_s[i,j] = X[i,s] (column s of X)
    #   V_s[j,k] = 1 if j==s else 0 (identity column)
    #   W_s[k,i] = Y[s,k] (row s of Y)
    # Check: sum_s X[i,s] * delta_{j,s} * Y[s,k] = X[i,j]*Y[j,k] ... 
    # Wait, that's not right. Let me reconsider.
    
    # (X*Y)_{i,k} = sum_j X[i,j] * Y[j,k]
    # If we set: for s in 0,1,2:
    #   U_s[i,j] = X[i,s] if j==s else 0  -> this is a matrix with only column s nonzero
    #   V_s[j,k] = 1 if j==s else 0  -> diagonal with 1 at (s,s)
    #   W_s[k,i] = Y[s,k]
    # Then: sum_s U_s[i,j] * V_s[j,k] * W_s[k,i]
    # = sum_s (X[i,s] if j==s) * (1 if j==s) * Y[s,k]
    # = sum_{s: s==j} X[i,s] * 1 * Y[s,k]
    # = X[i,j] * Y[j,k]
    # That only gives one term, not the full sum. This approach gives rank 9.
    
    # The BCLR trick is different. Let me look at this more carefully.
    # The rank of 3x3 matrix multiplication over R is 19 (proven).
    # Rank 21 is a known constructive upper bound.
    
    # Let me use the explicit BCLR coefficients. The decomposition is:
    # For 3x3 matrices A, B, C = A*B:
    # The 21 terms are given by specific linear combinations.
    
    # I'll use the formulation from the literature where the 21 terms
    # are constructed as follows. Let me denote the entries of A as a_{pq},
    # B as b_{qr}, C as c_{pr}.
    
    # BCLR rank-21 for 3x3:
    # The 7 Strassen terms each contribute 3 terms. For Strassen term t,
    # the 3 inner terms use a specific rank-3 decomposition.
    
    # Key insight: for the BCLR, each 3x3 block product is decomposed
    # using 3 terms where the U factor picks out specific rows of the
    # left matrix, V factor picks specific columns, and W factor maps back.
    
    # Let me use the following rank-3 decomposition for 3x3 products:
    # For product X*Y (both 3x3):
    # Term s (s=0,1,2):
    #   U_s[i,j] = X[i, (j+s) % 3]  (cyclic shift of columns)
    #   V_s[j,k] = 1 if j == k else 0  (identity)  
    #   W_s[k,i] = Y[(k-s) % 3, k]  ... this doesn't work simply.
    
    # Let me just hardcode the BCLR rank-21 coefficients directly.
    # After careful derivation, here's the structure:
    
    # For each of the 7 Strassen terms (t=0..6), we have:
    # A_t = sum of (coeff * A_block) for various 3x3 blocks
    # B_t = sum of (coeff * B_block) for various 3x3 blocks
    # The product A_t * B_t contributes to C blocks.
    # We decompose A_t * B_t into 3 rank-1 terms.
    
    # The rank-3 decomposition of a 3x3 matrix product X*Y:
    # Use: for s in {0,1,2}:
    #   u_s = X[:, s] (column s of X, as a 3-vector, extended to 3x3)
    #   v_s = e_s^T (row vector, extended)
    #   w_s = Y[s, :] (row s of Y, extended)
    # But this gives rank 9, not 3.
    
    # I think the correct approach for BCLR is different. Let me reconsider.
    # The BCLR decomposition achieves rank 21 for 3x3 by using the
    # Strassen-like 7-term structure where each term is a product of
    # two 3x3 matrices, and each such product is represented with
    # rank 3 by exploiting the specific linear combinations.
    
    # Actually, I realize the issue: in the BCLR decomposition, the
    # "rank" counts the total number of scalar multiplications. For
    # 3x3, the BCLR gives 21 scalar multiplications. This means we
    # need 21 terms where each term is a product of three scalars
    # (one from U, one from V, one from W) that together reconstruct
    # the full 3x3 matrix product.
    
    # So for the 3x3 case, we need:
    # C[i,k] = sum_{r=0}^{20} U3[i,r] * V3[r,k] * W3[k,r]  ... no
    # Actually: C[i,k] = sum_r U3[i*3+? , r] * ... 
    
    # Let me re-examine the tensor structure. For 3x3 matrix multiplication:
    # T_{i,j,k} = delta_{i,k} (the multiplication tensor)
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # Tensor decomposition: T_{i,j,k} = sum_r U_{i,j}^r * V_{j,k}^r * W_{k,i}^r
    # Wait, the trace convention: W[k*n+i, r] for n=p=3.
    # So: sum_r U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] = delta_{i,k} * 1 for all i,j,k
    # Actually no: the tensor T_{i,j,k} = 1 if the multiplication is valid,
    # which is always 1 for matrix multiplication. The tensor is:
    # T_{i,j,k} = 1 for all i,j,k (it's the all-ones tensor for 3x3 mult)
    # Wait no. The matrix multiplication tensor T has T_{i,j,k} = 1 for all
    # (i,j,k) in {0,1,2}^3. Because C_{i,k} = sum_j A_{i,j} B_{j,k},
    # the coefficient of A_{i,j} B_{j,k} in C_{i,k} is 1.
    
    # So we need: sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] = 1 for all i,j,k in {0,1,2}
    
    # The BCLR rank-21 achieves this with 21 terms.
    
    # Let me construct it using the 7 Strassen terms, each with 3 sub-terms.
    # For Strassen term t, we have linear combinations:
    # X_t = sum_{(a,b)} alpha_{t,(a,b)} * A_block_{a,b}
    # Y_t = sum_{(c,d)} beta_{t,(c,d)} * B_block_{c,d}
    # Then X_t * Y_t contributes to C blocks via the 3x3 product.
    
    # The 3x3 product X*Y = sum_j X_{i,j} Y_{j,k}
    # We decompose this as sum_{s=0}^{2} (X_{i, f(s)} * Y_{g(s), k}) 
    # for some functions f, g. This only works if we can factor the
    # sum over j into 3 terms, which requires the matrices to have
    # special structure.
    
    # I think the cleanest approach is to just verify: can we find
    # U3, V3, W3 of shape (9, 21) each such that the tensor identity holds?
    # Let me set up the system and solve it.
    
    # Actually, let me just use a known explicit BCLR decomposition.
    # Here's the standard BCLR rank-21 for 3x3:
    
    # The 21 terms are parameterized by (t, s) where t in {0,...,6}, s in {0,1,2}
    # For each (t,s), we define U3[.,.,(t,s)], V3[.,.,(t,s)], W3[.,.,(t,s)]
    
    # Strassen formulas for 2x2 blocks (applied to 3x3 "blocks" which are
    # actually 3x3 matrices, but we treat the 3x3 product as atomic for now):
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # For 3x3, each "block" is a 3x3 matrix, and the product of two 3x3
    # matrices needs rank 21/7 = 3 per Strassen term.
    
    # The rank-3 decomposition of 3x3 matrix product:
    # Key: for any 3x3 matrices X, Y, we can write XY = sum_{s=0}^{2} X_s * Y_s
    # where each X_s * Y_s is a rank-1 matrix product (outer product of vectors),
    # but that's not how tensor decomposition works here.
    
    # In the tensor decomposition framework:
    # (XY)_{i,k} = sum_j X_{i,j} Y_{j,k} = sum_s U_{i,j}^{(s)} V_{j,k}^{(s)} W_{k,i}^{(s)}
    # where the sum over s has 3 terms.
    
    # One valid rank-3 decomposition: 
    # For s in {0,1,2}:
    #   U_{i,j}^{(s)} = X_{i,s} * delta_{j,s}  (only nonzero when j=s, value X[i,s])
    #   V_{j,k}^{(s)} = delta_{j,s}  (identity at position s)
    #   W_{k,i}^{(s)} = Y_{s,k}
    # Check: sum_s U_{i,j}^{(s)} V_{j,k}^{(s)} W_{k,i}^{(s)}
    # = sum_s (X_{i,s} delta_{j,s}) * delta_{j,s} * Y_{s,k}
    # = sum_{s: j=s} X_{i,s} * 1 * Y_{s,k}
    # = X_{i,j} Y_{j,k}
    # This gives only ONE term contributing, not the sum over all j!
    # So this is wrong - it only captures the j=s term.
    
    # The correct rank-9 decomposition would use all 9 (i,j,k) triples.
    # A rank-3 decomposition is much more subtle.
    
    # I believe the BCLR rank-21 works by NOT decomposing each 3x3 product
    # independently, but by using the global structure of the 7 Strassen terms
    # to share terms across the 3x3 products.
    
    # Given the complexity, let me try a different approach: directly solve
    # for the BCLR coefficients using the known structure, or fall back to
    # the rank-189 solution if I can't verify rank-21 quickly.
    
    # Let me try to construct BCLR rank-21 using the following approach:
    # Use the 7 Strassen terms, and for each term, use 3 rank-1 terms
    # where the rank-1 decomposition of the 3x3 product uses the
    # "diagonal" structure.
    
    # For a 3x3 product X*Y, define for s in {0,1,2}:
    #   u_s[i,j] = X[i, (j+s) % 3]
    #   v_s[j,k] = 1 if j == k else 0
    #   w_s[k,i] = Y[(k-s) % 3, k]
    # This doesn't seem right either.
    
    # Let me just try the direct linear algebra approach: set up the
    # bilinear system and see if rank 21 is achievable with the
    # Strassen block structure.
    
    # For now, let me use a safe fallback: keep the rank-189 solution
    # but try to improve it with the greedy removal.
    
    # Actually, let me try a completely different approach: use the
    # known BCLR coefficients from the literature.
    
    # The BCLR rank-21 for 3x3 matrix multiplication:
    # I'll encode the 21 terms explicitly.
    
    # After research, the BCLR decomposition uses the following structure:
    # The 21 terms are (t, s) where t = 0..6, s = 0..2.
    # For each t, the Strassen linear combinations are fixed.
    # For each s, the rank-3 decomposition of the 3x3 product uses:
    #   The s-th "diagonal" of the product.
    
    # Let me define it as:
    # For term (t, s):
    #   The U factor for A-block (a,b) at position (i,j) within the 3x3:
    #     = alpha_{t,(a,b)} * A_{a*3+i, b*3+j} contribution
    #   Specifically, U3[i*3+j, (t,s)] = sum over blocks (a,b) of
    #     alpha_{t,(a,b)} * X_{i,j}^{(t)} where X^{(t)} is the t-th
    #     Strassen combination of 3x3 blocks.
    
    # This is getting too complex to derive from scratch. Let me use
    # the rank-189 solution as the primary and focus on optimizing
    # the greedy removal to find any further rank reduction.
    
    # Build rank-189 Strassen (2x2 blocks of 3x3) decomposition
    n3 = 3
    r3_school = 27
    U3_school = np.zeros((9, r3_school))
    V3_school = np.zeros((9, r3_school))
    W3_school = np.zeros((9, r3_school))
    for i3 in range(n3):
        for j3 in range(n3):
            for k3 in range(n3):
                col3 = i3 * 9 + j3 * 3 + k3
                U3_school[i3 * 3 + j3, col3] = 1.0
                V3_school[j3 * 3 + k3, col3] = 1.0
                W3_school[k3 * 3 + i3, col3] = 1.0

    total_rank = 7 * r3_school  # 189
    U6 = np.zeros((nm, total_rank))
    V6 = np.zeros((mp, total_rank))
    W6 = np.zeros((pn, total_rank))

    def add_term(start, a_blks, b_blks, c_blks):
        for r in range(r3_school):
            um = U3_school[:, r].reshape(3, 3)
            vm = V3_school[:, r].reshape(3, 3)
            wm = W3_school[:, r].reshape(3, 3)
            for bi, cf in a_blks:
                br, bc = bi // 2, bi % 2
                for ii in range(3):
                    for jj in range(3):
                        v = cf * um[ii, jj]
                        if v != 0:
                            U6[(br*3+ii)*6 + (bc*3+jj), start+r] += v
            for bi, cf in b_blks:
                br, bc = bi // 2, bi % 2
                for jj in range(3):
                    for kk in range(3):
                        v = cf * vm[jj, kk]
                        if v != 0:
                            V6[(br*3+jj)*6 + (bc*3+kk), start+r] += v
            for bi, cf in c_blks:
                br, bc = bi // 2, bi % 2
                for kk in range(3):
                    for ii in range(3):
                        v = cf * wm[kk, ii]
                        if v != 0:
                            W6[(bc*3+kk)*6 + (br*3+ii), start+r] += v

    # P1 = (A00+A11)(B00+B11) -> C00, C11
    add_term(0,
             [(0,1),(3,1)],
             [(0,1),(3,1)],
             [(0,1),(3,1)])
    # P2 = (A10+A11)B00 -> C10(+), C11(-)
    add_term(r3_school,
             [(2,1),(3,1)],
             [(0,1)],
             [(2,1),(3,-1)])
    # P3 = A00(B01-B11) -> C01(+), C11(+)
    add_term(2*r3_school,
             [(0,1)],
             [(1,1),(3,-1)],
             [(1,1),(3,1)])
    # P4 = A11(B10-B00) -> C00(+), C10(+)
    add_term(3*r3_school,
             [(3,1)],
             [(2,1),(0,-1)],
             [(0,1),(2,1)])
    # P5 = (A00+A01)B11 -> C01(+), C00(-)
    add_term(4*r3_school,
             [(0,1),(1,1)],
             [(3,1)],
             [(1,1),(0,-1)])
    # P6 = (A10-A00)(B00+B01) -> C11(+)
    add_term(5*r3_school,
             [(2,1),(0,-1)],
             [(0,1),(1,1)],
             [(3,1)])
    # P7 = (A01-A11)(B10+B11) -> C00(+)
    add_term(6*r3_school,
             [(1,1),(3,-1)],
             [(2,1),(3,1)],
             [(0,1)])

    # Verify decomposition
    target = np.ones((6, 6, 6))
    def check(U, V, W):
        Ug = U.reshape(6, 6, -1)
        Vg = V.reshape(6, 6, -1)
        Wg = W.reshape(6, 6, -1)
        rec = np.einsum('ijr,jkr,kir->ijk', Ug, Vg, Wg)
        return np.allclose(rec, target, atol=1e-9)

    if check(U6, V6, W6):
        best_U, best_V, best_W = U6, V6, W6
    else:
        best_U, best_V, best_W = U_s, V_s, W_s

    # Attempt to reduce rank by identifying linearly dependent rank terms.
    if check(U6, V6, W6) and time.time() < deadline - 10.0:
        try:
            rU = np.linalg.matrix_rank(U6, tol=1e-10)
            rV = np.linalg.matrix_rank(V6, tol=1e-10)
            rW = np.linalg.matrix_rank(W6, tol=1e-10)
            min_r = min(rU, rV, rW)

            if min_r < total_rank:
                keep = list(range(total_rank))
                improved = True
                while improved:
                    improved = False
                    for r in list(keep):
                        if len(keep) <= min_r:
                            break
                        new_keep = [k for k in keep if k != r]
                        if not new_keep:
                            continue
                        U_t = U6[:, new_keep]
                        V_t = V6[:, new_keep]
                        W_t = W6[:, new_keep]
                        if check(U_t, V_t, W_t):
                            keep = new_keep
                            best_U, best_V, best_W = U_t, V_t, W_t
                            improved = True
                            break
                    if time.time() > deadline - 5.0:
                        break
        except Exception:
            pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
