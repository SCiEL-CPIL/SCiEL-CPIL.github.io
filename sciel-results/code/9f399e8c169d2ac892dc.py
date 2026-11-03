"""
Matrix multiplication 6x6x6 tensor decomposition.
Uses a 3x3 block partition into 2x2 blocks with Strassen's rank-7 algorithm
for each block multiplication, giving rank 189.

The current seed program already achieves rank 189 via block-Strassen.
We keep this design family and try to find further reductions by:
1. Verifying the block-Strassen construction is correct.
2. Attempting to merge columns that are linearly dependent in both U and V.
3. Trying a 2x2 block partition into 3x3 blocks with Laderman's rank-17 algorithm
   for 4x4... wait, 6 = 2*3, so 2x2 blocks of 3x3. But Laderman is for 4x4.
   
Actually, let me reconsider. The directive says to KEEP the design and investigate
improvements within its family. The current design is block-Strassen with 3x3 grid
of 2x2 blocks, giving rank 189.

Let me think about whether there's a way to get below 189 within this family.

One idea: use a different block structure. 6 = 2 * 3, so we could have a 2x2 grid
of 3x3 blocks. For 3x3 matrix multiplication, the best known rank is 19 (or possibly
less, but 19 is standard). So 4 * 19 = 76? No wait, that's not right either.

Actually, for a 2x2 block matrix multiplication where each block is 3x3:
C = A * B where A, B are 2x2 block matrices with 3x3 blocks.
C[i,k] = sum_j A[i,j] * B[j,k] where each product is 3x3 * 3x3 -> 3x3.

So we need to multiply 3x3 matrices. If we use rank-r decomposition for 3x3,
then total rank = 4 * r (since there are 4 block positions: (0,0), (0,1), (1,0), (1,1)).

The best known rank for 3x3 matrix multiplication is 19 (from various sources).
So 4 * 19 = 76? That seems too low. Let me reconsider.

Wait, no. In a 2x2 block multiplication:
C00 = A00*B00 + A01*B10
C01 = A00*B01 + A01*B11
C10 = A10*B00 + A11*B10
C11 = A10*B01 + A11*B11

Each of these involves products of 3x3 matrices. If we use a rank-r decomposition
for 3x3 matrix multiplication, each 3x3*3x3 product takes r terms. But the sum
of two such products (e.g., A00*B00 + A01*B10) can't simply be decomposed as
2r terms because the decomposition is for a single product.

Actually, for the sum of two products, we need to be more careful. Let me think
differently.

For 2x2 block multiplication with 3x3 blocks, we can use the standard approach:
each block multiplication C[i,k] = sum_j A[i,j] * B[j,k] involves 2 products of
3x3 matrices. If each 3x3*3x3 product has rank r, then naively each C[i,k] block
needs 2r terms, and with 4 blocks, total = 8r. But there might be sharing.

Actually, the standard way to compose decompositions: if we have a rank-r decomposition
for multiplying 3x3 matrices, then for 2x2 block multiplication with 3x3 blocks,
the total rank is 4 * r (since there are 4 independent "positions" in the 2x2 block
structure, and each requires a full 3x3 multiplication).

Wait, I'm confusing myself. Let me be more careful.

The tensor for 6x6 matrix multiplication can be viewed as:
T[i,k,j] = delta where C[i,k] = sum_j A[i,j] B[j,k]

With block structure (2x2 blocks of 3x3):
Let A = [[A00, A01], [A10, A11]] where each Ai,j is 3x3.
Similarly for B and C.

C00 = A00*B00 + A01*B10
C01 = A00*B01 + A01*B11
C10 = A10*B00 + A11*B10
C11 = A10*B01 + A11*B11

Each term like A00*B00 is a 3x3 matrix product. If we have a rank-r decomposition
for 3x3 matrix multiplication, then A00*B00 = sum_{s=1}^r u_s(A00) * v_s(B00) * w_s
where u_s, v_s, w_s are the factor matrices.

But the sum A00*B00 + A01*B10 involves two different products. We can't directly
combine them unless we find a joint decomposition.

The standard composition rule: if M_{m x n x p} has rank r1 and M_{n x q x s} has
rank r2, then... actually, for block multiplication, the rank is multiplicative in
a specific way.

For 2x2 block multiplication with 3x3 blocks:
- The "outer" structure is 2x2x2 matrix multiplication (rank 7 via Strassen)
- The "inner" structure is 3x3x3 matrix multiplication (rank r3)
- Total rank = 7 * r3

So if r3 = 19 (best known for 3x3), total = 7 * 19 = 133.
If r3 = 27 (schoolbook for 3x3), total = 7 * 27 = 189.

So using a better decomposition for 3x3 could help! The current program uses
schoolbook for the 2x2 blocks (which gives rank 8 for 2x2, but Strassen gives 7).

Wait, I think I had it backwards. Let me re-examine.

Current program: 3x3 grid of 2x2 blocks, Strassen for 2x2.
- Outer: 3x3x3 matrix multiplication (schoolbook, rank 27)
- Inner: 2x2x2 matrix multiplication (Strassen, rank 7)
- Total: 27 * 7 = 189

Alternative: 2x2 grid of 3x3 blocks.
- Outer: 2x2x2 matrix multiplication (Strassen, rank 7)
- Inner: 3x3x3 matrix multiplication (rank r3)
- Total: 7 * r3

If we can find a good r3 for 3x3, this could be better.

The best known rank for 3x3 matrix multiplication over the reals is 19.
So 7 * 19 = 133, which is much better than 189!

Let me implement this. I need a rank-19 decomposition for 3x3 matrix multiplication.

Actually, finding a rank-19 decomposition for 3x3 is non-trivial. Let me check if
there's a standard one. The Laderman algorithm is for 4x4 with rank 17. For 3x3,
the best known is 19, but I'm not sure of the exact decomposition.

Let me try a different approach. Use the 2x2 block structure with Strassen for the
outer 2x2, and for the inner 3x3, use a known decomposition.

Actually, let me just implement the 2x2 block with 3x3 inner using Strassen for
the outer and a reasonable decomposition for the inner. If I can't find a rank-19
decomposition for 3x3, I'll use schoolbook (rank 27) for the inner, giving 7*27=189
(same as current). But if I can find something better, it'll help.

Let me try to use a known rank-19 decomposition for 3x3. One approach is to use
the fact that 3x3 multiplication can be done with 19 multiplications. Let me look
for a standard formulation.

Actually, for this problem, let me just implement the 2x2 block with 3x3 inner
using Strassen for the outer (rank 7) and schoolbook for the inner (rank 27),
giving 189. Then try to improve the inner decomposition.

Wait, that's the same as current. Let me think differently.

The current approach: 3x3 outer (schoolbook, 27) * 2x2 inner (Strassen, 7) = 189.
Alternative: 2x2 outer (Strassen, 7) * 3x3 inner (schoolbook, 27) = 189.

Both give 189. To do better, I need a better inner decomposition.

Let me try to implement a rank-19 decomposition for 3x3 matrix multiplication.
This is the key improvement.

After some research, the rank-19 decomposition for 3x3 is complex. Let me try
a different strategy: use the 2x2 outer with Strassen, and for the 3x3 inner,
try to find a decomposition with rank < 27.

Actually, let me just implement the 2x2 outer with Strassen and 3x3 inner with
schoolbook (rank 27), which gives 189, same as current. Then I'll add logic
to try to reduce the rank by finding linear dependencies.

But wait, the directive says to KEEP the design and investigate improvements.
The current design is block-Strassen. Let me stick with that and try to find
a better inner decomposition.

Let me implement the 2x2 block with 3x3 inner, using Strassen for the outer
and attempting to use a rank-19 decomposition for the inner 3x3.

For now, let me just implement the 2x2 outer with Strassen and 3x3 inner with
schoolbook, verify it's correct, and then try to reduce rank.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    """
    Construct a rank-reduced exact tensor decomposition for 6x6 matrix multiplication.
    Uses 2x2 block partition into 3x3 blocks with Strassen for the outer 2x2
    and schoolbook for the inner 3x3, giving rank 7*27 = 189.
    Then attempts rank reduction.
    """
    deadline = time.time() + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # Each entry is a list of (row, col, coeff) for the 2x2 block
    strassen_x = [
        [(0,0,1), (1,1,1)],      # M0: A00+A11
        [(1,0,1), (1,1,1)],      # M1: A10+A11
        [(0,0,1)],               # M2: A00
        [(1,1,1)],               # M3: A11
        [(0,0,1), (0,1,1)],      # M4: A00+A01
        [(1,0,1), (0,0,-1)],     # M5: A10-A00
        [(0,1,1), (1,1,-1)],     # M6: A01-A11
    ]
    strassen_y = [
        [(0,0,1), (1,1,1)],      # M0: B00+B11
        [(0,0,1)],               # M1: B00
        [(0,1,1), (1,1,-1)],     # M2: B01-B11
        [(1,0,1), (0,0,-1)],     # M3: B10-B00
        [(1,1,1)],               # M4: B11
        [(0,0,1), (0,1,1)],      # M5: B00+B01
        [(1,0,1), (1,1,1)],      # M6: B10+B11
    ]
    strassen_c = [
        [(0,0,1), (1,1,1)],      # C00 += M0, C11 += M0
        [(1,0,1), (1,1,-1)],     # C10 += M1, C11 -= M1
        [(0,1,1), (1,1,1)],      # C01 += M2, C11 += M2
        [(0,0,1), (1,0,1)],      # C00 += M3, C10 += M3
        [(0,0,-1), (0,1,1)],     # C00 -= M4, C01 += M4
        [(1,1,1)],               # C11 += M5
        [(0,0,1)],               # C00 += M6
    ]
    
    nb_outer = 2  # 2x2 outer block structure
    bs_inner = 3  # 3x3 inner blocks
    
    rank_total = nb_outer * nb_outer * nb_outer * 7 * bs_inner * bs_inner * bs_inner
    # = 8 * 7 * 27 = 1512? No, that's wrong.
    
    # Actually, the rank is: (number of outer terms) * (rank of inner decomposition)
    # Outer: 2x2x2 Strassen = 7 terms
    # Inner: 3x3x3 schoolbook = 27 terms
    # Total: 7 * 27 = 189
    
    rank_total = 7 * 27  # 189
    
    U = np.zeros((n*m, rank_total), dtype=np.float64)
    V = np.zeros((m*p, rank_total), dtype=np.float64)
    W = np.zeros((p*n, rank_total), dtype=np.float64)
    
    col = 0
    for s in range(7):
        for ii in range(bs_inner):
            for jj in range(bs_inner):
                for kk in range(bs_inner):
                    # For each Strassen term s and inner schoolbook term (ii,jj,kk)
                    # A block: outer row r, outer col c, inner (ii,jj)
                    # B block: outer row c, outer col d, inner (jj,kk)
                    # C block: outer row r, outer col d, inner (kk,ii)
                    
                    for (lr, lc, coeff_a) in strassen_x[s]:
                        # lr is outer row, lc is outer col for A
                        # Inner row: lr * bs_inner + ii
                        # Inner col: lc * bs_inner + jj
                        gi = lr * bs_inner + ii
                        gj = lc * bs_inner + jj
                        U[gi * m + gj, col] += coeff_a
                    
                    for (lr, lc, coeff_b) in strassen_y[s]:
                        # lr is outer row, lc is outer col for B
                        # Inner row: lr * bs_inner + jj
                        # Inner col: lc * bs_inner + kk
                        gj = lr * bs_inner + jj
                        gk = lc * bs_inner + kk
                        V[gj * p + gk, col] += coeff_b
                    
                    for (lr, lc, coeff_c) in strassen_c[s]:
                        # lr is outer row, lc is outer col for C
                        # Inner row: lr * bs_inner + kk
                        # Inner col: lc * bs_inner + ii
                        gi = lr * bs_inner + kk
                        gk = lc * bs_inner + ii
                        W[gk * n + gi, col] += coeff_a * coeff_b * coeff_c
                    
                    col += 1
    
    # Verify
    def verify_decomposition(U_mat, V_mat, W_mat, n=n, m=m, p=p, atol=1e-9):
        """Verify the decomposition."""
        rank = U_mat.shape[1]
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U_mat[:, r].reshape(n, m)
            v_col = V_mat[:, r].reshape(m, p)
            w_col = W_mat[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=atol)
    
    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p), dtype=np.float64)
        V = np.zeros((m*p, n*m*p), dtype=np.float64)
        W = np.zeros((p*n, n*m*p), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_total
    
    # Try to reduce rank by finding mergeable columns
    reduced_U = [U[:, c].copy() for c in range(rank_total)]
    reduced_V = [V[:, c].copy() for c in range(rank_total)]
    reduced_W = [W[:, c].copy() for c in range(rank_total)]
    
    used = [True] * rank_total
    merged = 0
    
    def find_ratio(vec_a, vec_b, tol=1e-9):
        """Find scalar r such that vec_b = r * vec_a, or None."""
        r = None
        for idx in range(len(vec_a)):
            if abs(vec_a[idx]) > 1e-12:
                if abs(vec_b[idx]) > 1e-12:
                    candidate = vec_b[idx] / vec_a[idx]
                    if r is None:
                        r = candidate
                    elif abs(candidate - r) > tol:
                        return None
                else:
                    if r is not None and abs(r) > tol:
                        return None
                    r = 0.0
        if r is not None:
            if not np.allclose(vec_b, r * vec_a, atol=tol):
                return None
        return r
    
    for i in range(rank_total):
        if not used[i]:
            continue
        for j in range(i+1, rank_total):
            if not used[j]:
                continue
            if time.time() > deadline - 15:
                break
            
            ratio_u = find_ratio(reduced_U[i], reduced_U[j])
            if ratio_u is None:
                continue
            
            ratio_v = find_ratio(reduced_V[i], reduced_V[j])
            if ratio_v is None:
                continue
            
            reduced_W[i] = reduced_W[i] + ratio_u * ratio_v * reduced_W[j]
            used[j] = False
            merged += 1
    
    if merged > 0:
        new_rank = sum(used)
        final_U = np.column_stack([reduced_U[i] for i in range(rank_total) if used[i]])
        final_V = np.column_stack([reduced_V[i] for i in range(rank_total) if used[i]])
        final_W = np.column_stack([reduced_W[i] for i in range(rank_total) if used[i]])
        
        if new_rank < best_rank and verify_decomposition(final_U, final_V, final_W):
            best_U, best_V, best_W = final_U, final_V, final_W
            best_rank = new_rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
