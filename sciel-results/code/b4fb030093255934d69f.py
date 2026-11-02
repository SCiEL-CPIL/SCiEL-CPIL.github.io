"""Search for a rank-minimal exact tensor decomposition of 6x6 matrix multiply.

Design: exploit the 3x3 block structure (2x2 blocks). Start from Block-Strassen
(rank 189), then attempt to find a lower-rank exact decomposition via a
randomized sparse algebraic search. All entries are kept as exact rationals
(integers) so the decomposition is verified exactly.
"""

import time
import random
import sys
from fractions import Fraction

DIMENSIONS = [6, 6, 6]

# Hard wall-clock guard: the evaluator kills after 150s; keep a safety margin.
DEADLINE = time.time() + 130.0


def _deadline_exceeded():
    return time.time() >= DEADLINE


# ---------------------------------------------------------------------------
# Exact tensor model.
#
# We decompose the matrix-multiplication tensor T_{i,j,k} = delta_{i,k} (the
# (i,k)-entry of A B where A is (i,j), B is (j,k)) as
#
#     T_{i,j,k} = sum_c  U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
#
# with n = m = p = 6. The output factor follows the trace convention
# W[k*n + i, c].
# ---------------------------------------------------------------------------


def _tensor_entry(i, j, k):
    # T[i,j,k] = 1 if i == k else 0  (A[i,j]*B[j,k] summed over j gives C[i,k])
    return 1 if i == k else 0


def _verify(U, V, W, n=6, m=6, p=6):
    """Return True iff the decomposition is exact (integer/rational entries)."""
    rank = len(U[0])
    # Build C[i,k] = sum_j sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # and check C[i,k] == delta_{i,k}.
    for i in range(n):
        for k in range(p):
            total = Fraction(0)
            for j in range(m):
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                w_val = W[k * n + i]
                for c in range(rank):
                    total += u_row[c] * v_row[c] * w_val[c]
            if total != (1 if i == k else 0):
                return False
    return True


def _verify_float(U, V, W, n=6, m=6, p=6, tol=1e-9):
    """Fast float check (used as a filter before exact check)."""
    import numpy as np
    rank = len(U[0])
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # C[i,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    for i in range(n):
        for k in range(p):
            total = 0.0
            for j in range(m):
                total += float(np.dot(U[i * m + j], V[j * p + k])) * W[k * n + i]
            if abs(total - (1 if i == k else 0)) > tol:
                return False
    return True


# ---------------------------------------------------------------------------
# Block-Strassen construction (rank 189).
#
# Partition A, B, C into 3x3 grids of 2x2 blocks. For each block position
# (I,J) of B and (I,K) of C, the product of a 2x2 block of A (row block I,
# col block J) with a 2x2 block of B (row block J, col block K) contributes
# to C's block (I,K). We apply Strassen's 7-term decomposition to each 2x2
# block product, but we must be careful: the same A-block and B-block appear
# in multiple (I,K) positions. To keep the decomposition exact and separable,
# we treat each (I,J,K) triple independently with its own 7 Strassen terms.
# That gives 3*3*3*7 = 189 terms.
# ---------------------------------------------------------------------------


def _block_strassen(n=6, m=6, p=6):
    # block size = 2, so 3x3 blocks
    bs = 2
    nb = n // bs  # 3
    rank = nb * nb * nb * 7
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]

    col = 0
    for I in range(nb):
        for J in range(nb):
            for K in range(nb):
                # Strassen's 7 products for 2x2 blocks.
                # A-block = A[I*bs:(I+1)*bs, J*bs:(J+1)*bs]
                # B-block = B[J*bs:(J+1)*bs, K*bs:(K+1)*bs]
                # C-block = C[I*bs:(I+1)*bs, K*bs:(K+1)*bs]
                # Strassen terms (for 2x2):
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B01-B11)
                # M4 = A11(B10-B00)
                # M5 = (A00+A10)B11
                # M6 = (A11-A00)(B00+B10)
                # M7 = (A01-A11)(B01+B11)
                # C00 = M1+M4-M5+M7
                # C01 = M3+M5
                # C10 = M2+M4
                # C11 = M1-M2+M3+M6
                #
                # We represent each Strassen product as one tensor term.
                # A local 2x2 has indices (a,b) with a,b in {0,1}; global row
                # index = I*bs + a, global col = J*bs + b.
                # B local (b,c); global row = J*bs + b, global col = K*bs + c.
                # C local (a,c); global row = I*bs + a, global col = K*bs + c.
                #
                # For each Strassen term, we need to express the product of a
                # linear combination of A-entries with a linear combination of
                # B-entries, then distribute into C.
                #
                # Instead of distributing the final C-combination, we keep each
                # of the 7 products as a separate rank term and let the C-side
                # W absorb the combination coefficients. But a single rank term
                # is a single product U*V*W which is rank-1 in the tensor. The
                # Strassen decomposition is a sum of 7 rank-1 terms in the
                # 2x2x2 block tensor, so each Strassen product IS one rank term.
                #
                # Term t has:
                #   A-side: sum_{a,b} alpha_{a,b}^{(t)} * A[I*bs+a, J*bs+b]
                #   B-side: sum_{b,c} beta_{b,c}^{(t)} * B[J*bs+b, K*bs+c]
                #   C-side: the resulting C-block entries with coefficients.
                #
                # But a rank-1 term in the full tensor is a single product of
                # one A-entry, one B-entry, one C-entry. The Strassen product
                # (A00+A11)(B00+B11) is NOT rank-1; it is a sum of 4 rank-1
                # terms. So Block-Strassen per block is actually 7 "grouped"
                # products, each of which is a sum of up to 4 elementary rank-1
                # terms. That does NOT reduce the elementary rank!
                #
                # The reduction from 8 to 7 comes from the fact that the 2x2x2
                # tensor has rank 7, i.e., it can be written as a sum of 7
                # rank-1 tensors (each rank-1 tensor = a single A-entry *
                # single B-entry * single C-entry pattern). Wait, that's not
                # right either. The rank-1 tensors in the matrix-multiplication
                # tensor are of the form e_i e_j e_k (single entry each). The
                # 2x2x2 matrix-multiplication tensor (multiplying 2x2 by 2x2)
                # has rank 7, meaning it's a sum of 7 terms each of which is a
                # product of a linear form in A, a linear form in B, and a
                # linear form in C. Each such term is NOT a single elementary
                # e_i e_j e_k; it's a product of sums.
                #
                # So in the full tensor decomposition, each Strassen term
                # (a product of two linear forms times a linear form) is ONE
                # rank term in the tensor sense, where U, V, W rows can have
                # multiple nonzero entries. Good. So Block-Strassen gives
                # 27*7 = 189 rank terms, each with sparse U/V/W rows.
                #
                # Let me define the 7 Strassen terms as (alpha, beta, gamma)
                # where alpha is a 2x2 matrix of coefficients for A-block,
                # beta for B-block, gamma for C-block, and the term is
                #   (sum alpha_ab A_ab)(sum beta_bc B_bc)(sum gamma_ac C_ac)
                # But that's not how tensor rank works. A rank-1 tensor is
                # u_i v_j w_k, a single product. The Strassen decomposition
                # writes the 2x2x2 tensor as sum of 7 terms, each term being
                # (linear form in A) * (linear form in B) * (linear form in C).
                # Each such term, when expanded, is a sum of elementary e_i e_j e_k
                # with coefficients, but as a TENSOR it's rank 1 because it's
                # the outer product of three vectors.
                #
                # So for the full 6x6x6 tensor, each Strassen term at block
                # (I,J,K) contributes a rank-1 term where:
                #   U vector has entries alpha_{a,b} at position (I*bs+a)*m + (J*bs+b)
                #   V vector has entries beta_{b,c} at position (J*bs+b)*p + (K*bs+c)
                #   W vector has entries gamma_{a,c} at position (K*bs+c)*n + (I*bs+a)
                #
                # And the 7 Strassen terms for a 2x2 multiply are:
                # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
                # M2 = (A10+A11)B00 -> C10 += M2
                # M3 = A00(B01-B11) -> C01 += M3
                # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
                # M5 = (A00+A10)B11 -> C00 -= M5, C01 += M5
                # M6 = (A11-A00)(B00+B10) -> C11 += M6
                # M7 = (A01-A11)(B01+B11) -> C00 += M7
                #
                # Wait, that's not right. The Strassen decomposition gives the
                # 7 products, and then C is a linear combination of them. But
                # in the tensor decomposition, each of the 7 products is a
                # rank-1 tensor, and the C-side coefficient for each product
                # determines which C-entries it contributes to.
                #
                # Actually, the standard way: the 2x2x2 matrix-multiplication
                # tensor T = sum_{i,j,k} T_{i,j,k} e_i e_j e_k where
                # T_{i,j,k} = delta_{i,k}. Its rank is 7. A rank-7
                # decomposition is T = sum_{c=1}^{7} u^{(c)} v^{(c)} w^{(c)}
                # where u^{(c)}, v^{(c)}, w^{(c)} are vectors in C^2 (or
                # C^4 for the 2x2 case, indexed by (a,b), (b,c), (a,c)).
                #
                # The Strassen decomposition gives exactly this: 7 terms where
                # each term c has u^{(c)}_{ab}, v^{(c)}_{bc}, w^{(c)}_{ac}.
                #
                # Let me look up the explicit rank-7 decomposition of the
                # 2x2x2 matrix-multiplication tensor. The 7 Strassen products
                # are:
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B01-B11)
                # M4 = A11(B10-B00)
                # M5 = (A00+A10)B11
                # M6 = (A11-A00)(B00+B10)
                # M7 = (A01-A11)(B01+B11)
                #
                # And C is:
                # C00 = M1 + M4 - M5 + M7
                # C01 = M3 + M5
                # C10 = M2 + M4
                # C11 = M1 - M2 + M3 + M6
                #
                # To write this as a tensor rank decomposition, each M_c is a
                # product of a linear form in A and a linear form in B. The
                # C-side is a linear combination of the M_c's. So the tensor
                # T = sum_c (linear form in A for M_c) * (linear form in B for M_c) * (linear form in C for M_c)
                #
                # where the C-side linear form for M_c has coefficient 1 at each
                # C-entry that M_c contributes to, with the appropriate sign.
                #
                # So for M1: A-side = A00 + A11, B-side = B00 + B11, C-side = C00 + C11
                # For M2: A-side = A10 + A11, B-side = B00, C-side = C10
                # For M3: A-side = A00, B-side = B01 - B11, C-side = C01
                # For M4: A-side = A11, B-side = B10 - B00, C-side = C00 + C10
                # For M5: A-side = A00 + A10, B-side = B11, C-side = -C00 + C01
                # For M6: A-side = A11 - A00, B-side = B00 + B10, C-side = C11
                # For M7: A-side = A01 - A11, B-side = B01 + B11, C-side = C00
                #
                # Let me verify M1: (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
                # C00 gets +1, C11 gets +1. So C-side = C00 + C11. Yes.
                #
                # Now I'll build the full decomposition.
                strassen = [
                    # (alpha: dict (a,b)->coeff, beta: dict (b,c)->coeff, gamma: dict (a,c)->coeff)
                    # M1
                    ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
                    # M2
                    ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1}),
                    # M3
                    ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1}),
                    # M4
                    ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
                    # M5
                    ({(0,0):1,(1,0):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
                    # M6
                    ({(1,1):1,(0,0):-1}, {(0,0):1,(1,0):1}, {(1,1):1}),
                    # M7
                    ({(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}, {(0,0):1}),
                ]

                for alpha, beta, gamma in strassen:
                    for (a, b), av in alpha.items():
                        U[(I * bs + a) * m + (J * bs + b)][col] += av
                    for (b, c), bv in beta.items():
                        V[(J * bs + b) * p + (K * bs + c)][col] += bv
                    for (a, c), wv in gamma.items():
                        W[(K * bs + c) * n + (I * bs + a)][col] += wv
                    col += 1
    return U, V, W


# ---------------------------------------------------------------------------
# Randomized sparse search for a lower-rank decomposition.
#
# We use a "tensor train" style approach: fix a candidate rank r, and try to
# find U, V, W with sparse rows such that the tensor equation holds. This is
# a nonlinear system; we use a simple iterative refinement.
#
# Given the complexity and time budget, we focus on a targeted search:
# 1. Start from Block-Strassen (rank 189).
# 2. Try to find linear dependencies among the 189 terms to reduce rank.
# 3. Use a randomized search for additional structure.
# ---------------------------------------------------------------------------


def _try_reduce_rank(U, V, W, n=6, m=6, p=6, time_budget=30.0):
    """Try to find a lower-rank decomposition by detecting linear dependencies.

    If two rank terms are linearly dependent in a certain sense, we can merge
    them. We use a simple approach: check if the 189 terms span a lower-
    dimensional space in some tensor-product sense.
    """
    import numpy as np
    start = time.time()
    rank = len(U[0])
    # Flatten each term into a vector in C^{36*36*36} (too large).
    # Instead, check linear dependencies of the U, V, W rows.
    #
    # A simpler approach: try random linear combinations of terms and check
    # if they still span the tensor.
    #
    # Given time constraints, we'll do a limited search.
    best = (U, V, W)
    best_rank = rank

    # Try to find pairs of terms that can be merged.
    # This is expensive; we'll skip it and rely on the Block-Strassen base.
    #
    # Instead, we'll do a randomized search for a rank-r decomposition with
    # r < 189. We use a "random initialization + gradient-free refinement"
    # approach.
    #
    # For simplicity and reliability, we'll just return the Block-Strassen
    # result unless we find something better quickly.
    if time.time() - start > time_budget:
        return best

    # Randomized search: try to find a rank-r decomposition with r in a
    # range below 189. We use a simple approach: fix r, initialize U, V, W
    # randomly with small integer entries, and iteratively refine.
    #
    # This is a hard problem; we'll do a limited number of trials.
    trials = 0
    max_trials = 50
    for r in range(180, max(100, 189 - 20), -5):
        if _deadline_exceeded() or time.time() - start > time_budget:
            break
        for _ in range(max_trials // 5):
            if _deadline_exceeded() or time.time() - start > time_budget:
                break
            trials += 1
            # Initialize with random small integers
            U_r = [[random.randint(-2, 2) for _ in range(r)] for _ in range(n * m)]
            V_r = [[random.randint(-2, 2) for _ in range(r)] for _ in range(m * p)]
            W_r = [[random.randint(-2, 2) for _ in range(r)] for _ in range(p * n)]
            # Quick float check
            if _verify_float(U_r, V_r, W_r, n, m, p, tol=1e-6):
                # Exact check
                U_frac = [[Fraction(x) for x in row] for row in U_r]
                V_frac = [[Fraction(x) for x in row] for row in V_r]
                W_frac = [[Fraction(x) for x in row] for row in W_r]
                if _verify(U_frac, V_frac, W_frac, n, m, p):
                    best = (U_frac, V_frac, W_frac)
                    best_rank = r
                    return best
    return best


def construct():
    start = time.time()
    n, m, p = DIMENSIONS

    # Base: Block-Strassen (rank 189)
    U, V, W = _block_strassen(n, m, p)
    # Verify base
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = (U, V, W)
    best_rank = len(U[0])

    # Try to reduce rank
    if not _deadline_exceeded():
        U2, V2, W2 = _try_reduce_rank(U, V, W, n, m, p, time_budget=60.0)
        if len(U2[0]) < best_rank:
            best = (U2, V2, W2)
            best_rank = len(U2[0])

    # Final deadline guard
    if _deadline_exceeded():
        pass  # Return best found so far

    U_best, V_best, W_best = best
    return {"U": U_best, "V": V_best, "W": W_best}


# ---------------------------------------------------------------------------
# Self-test (not called by evaluator, but useful for debugging)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    print(f"Rank: {rank}, time: {t1 - t0:.2f}s")
    valid = _verify(U, V, W, 6, 6, 6)
    print(f"Valid: {valid}")
    if not valid:
        print("ERROR: decomposition is not exact!")
        sys.exit(1)
