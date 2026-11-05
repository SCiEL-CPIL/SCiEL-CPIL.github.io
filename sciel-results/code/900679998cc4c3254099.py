"""Deterministic multi-Strassen decomposition for 3x4*4x7 matrix multiplication.

Strategy: Apply Strassen's 7-product algorithm to as many disjoint 2x2x2 sub-blocks
as possible. Each Strassen block replaces 8 schoolbook terms with 7, saving 1 rank.

The matrix multiplication tensor M_{3,4,7} has entries indexed by (i,j,k) where
i in {0,1,2} (row of A), j in {0,1,2,3} (col of A / row of B), k in {0,...,6} (col of B).

A 2x2x2 Strassen block covers indices (i,j,k) with i in {a,a+1}, j in {b,b+1}, k in {c,c+1}.
Two blocks are compatible (disjoint in all three dimensions) if their index sets don't overlap.

Maximum number of disjoint 2x2x2 blocks:
- i has 3 values -> at most 1 block can use 2 i-values (since 3 < 4, we can fit one pair)
  Actually we can have blocks with i in {0,1} and i in {1,2} but they share i=1, so not disjoint.
  So at most 1 block in the i-dimension.
- j has 4 values -> at most 2 disjoint pairs: {0,1} and {2,3}
- k has 7 values -> at most 3 disjoint pairs: {0,1}, {2,3}, {4,5}

But blocks must be disjoint in ALL three dimensions simultaneously. So if we pick:
  Block 1: i={0,1}, j={0,1}, k={0,1}
  Block 2: i={0,1}, j={2,3}, k={2,3}  -- shares i with Block 1, NOT disjoint!

Wait, "disjoint" here means the U, V, W index sets don't overlap. Let me reconsider.

For a rank-1 term at position (i,j,k):
  U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

Two terms c1 and c2 are "disjoint" if for every (i,j,k), at most one of them has
nonzero U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].

Actually, the key insight is: a Strassen block on indices (i in I, j in J, k in K)
only has nonzero entries in U at positions (i*m+j) for i in I, j in J;
in V at positions (j*p+k) for j in J, k in K;
in W at positions (k*n+i) for k in K, i in I.

Two Strassen blocks are compatible if:
  - Their I sets are disjoint OR their J sets are disjoint OR their K sets are disjoint.
  Actually no - they need to be disjoint in the sense that the product U*V*W doesn't interfere.
  
  More precisely, block 1 (I1, J1, K1) and block 2 (I2, J2, K2) are compatible if
  for any (i,j,k): it's not the case that i is in I1∩I2, j is in J1∩J2, AND k is in K1∩K2.
  Because if (i,j,k) is in both blocks, then both sets of 7 terms would try to decompose
  the same 8 schoolbook terms, causing interference.

  So we need: I1∩I2 = ∅ OR J1∩J2 = ∅ OR K1∩K2 = ∅.

Let's find the maximum number of compatible 2x2x2 blocks:
- I subsets of size 2 from {0,1,2}: {0,1}, {1,2}
- J subsets of size 2 from {0,1,2,3}: {0,1}, {2,3}
- K subsets of size 2 from {0,...,6}: {0,1}, {2,3}, {4,5}, {5,6}, etc.

Block (I,J,K) is compatible with (I',J',K') iff I∩I'=∅ or J∩J'=∅ or K∩K'=∅.

Let's try to pack:
1. ({0,1}, {0,1}, {0,1})
2. ({0,1}, {2,3}, {2,3}) - shares I with 1, but J∩J'={0,1}∩{2,3}=∅, so OK!
3. ({1,2}, {0,1}, {4,5}) - shares J with 1 and 2? J={0,1} intersects J1={0,1} and J2={2,3}.
   With block 1: I∩I'={0,1}∩{1,2}={1}≠∅, J∩J'={0,1}∩{0,1}={0,1}≠∅, K∩K'={0,1}∩{4,5}=∅. OK!
   With block 2: I∩I'={0,1}∩{1,2}={1}≠∅, J∩J'={0,1}∩{2,3}=∅. OK!
4. ({1,2}, {2,3}, {4,5}) - 
   With block 1: I∩I'={1}≠∅, J∩J'={2,3}∩{0,1}=∅. OK!
   With block 2: I∩I'={1}≠∅, J∩J'={2,3}∩{2,3}={2,3}≠∅, K∩K'={2,3}∩{4,5}=∅. OK!
   With block 3: I∩I'={1,2}∩{1,2}={1,2}≠∅, J∩J'={2,3}∩{0,1}=∅. OK!

Can we add a 5th block?
5. ({0,1}, {0,1}, {2,3}) - 
   With block 1: I∩I'={0,1}≠∅, J∩J'={0,1}≠∅, K∩K'={0,1}∩{2,3}=∅. OK!
   With block 2: I∩I'={0,1}≠∅, J∩J'={0,1}∩{2,3}=∅. OK!
   With block 3: I∩I'={0,1}≠∅, J∩J'={0,1}≠∅, K∩K'={4,5}∩{2,3}=∅. OK!
   With block 4: I∩I'={0,1}≠∅, J∩J'={0,1}∩{2,3}=∅. OK!

So we have 5 blocks! That saves 5 rank terms: 84 - 5 = 79.

Wait, let me double-check. Each Strassen block covers 8 (i,j,k) positions and uses 7 terms.
Schoolbook would use 8 terms for those 8 positions. So each block saves 1.

5 blocks save 5 terms: rank = 84 - 5 = 79.

Let me verify all pairs are compatible:
Blocks:
B1: I={0,1}, J={0,1}, K={0,1}
B2: I={0,1}, J={2,3}, K={2,3}
B3: I={1,2}, J={0,1}, K={4,5}
B4: I={1,2}, J={2,3}, K={4,5}
B5: I={0,1}, J={0,1}, K={2,3}

Pair (B1,B2): I∩={0,1}≠∅, J∩=∅. OK.
Pair (B1,B3): I∩={1}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
Pair (B1,B4): I∩={1}≠∅, J∩=∅. OK.
Pair (B1,B5): I∩={0,1}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
Pair (B2,B3): I∩={1}≠∅, J∩=∅. OK.
Pair (B2,B4): I∩={1}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
Pair (B2,B5): I∩={0,1}≠∅, J∩=∅. OK.
Pair (B3,B4): I∩={1,2}≠∅, J∩=∅. OK.
Pair (B3,B5): I∩={1}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
Pair (B4,B5): I∩={1}≠∅, J∩=∅. OK.

All 10 pairs are compatible! Rank = 79.

Can we add a 6th block?
Try B6: I={1,2}, J={0,1}, K={2,3}
With B1: I∩={1}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
With B2: I∩={1}≠∅, J∩=∅. OK.
With B3: I∩={1,2}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
With B4: I∩={1,2}≠∅, J∩=∅. OK.
With B5: I∩={1}≠∅, J∩={0,1}≠∅, K∩={2,3}≠∅. NOT OK!

Try B6: I={1,2}, J={2,3}, K={0,1}
With B1: I∩={1}≠∅, J∩=∅. OK.
With B2: I∩={1}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B3: I∩={1,2}≠∅, J∩=∅. OK.
With B4: I∩={1,2}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B5: I∩={1}≠∅, J∩=∅. OK.

B6 is compatible with all! Now we have 6 blocks. Rank = 84 - 6 = 78.

Can we add a 7th?
Try B7: I={0,1}, J={2,3}, K={4,5}
With B1: I∩={0,1}≠∅, J∩=∅. OK.
With B2: I∩={0,1}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B3: I∩=∅. OK.
With B4: I∩=∅. OK.
With B5: I∩={0,1}≠∅, J∩=∅. OK.
With B6: I∩={1}≠∅, J∩={2,3}≠∅, K∩=∅. OK.

B7 compatible! 7 blocks. Rank = 84 - 7 = 77.

Try B8: I={1,2}, J={0,1}, K={4,5}
With B1: I∩={1}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
With B2: I∩={1}≠∅, J∩=∅. OK.
With B3: I∩={1,2}≠∅, J∩={0,1}≠∅, K∩={4,5}≠∅. NOT OK!

Try B8: I={0,1}, J={2,3}, K={2,3}
With B1: I∩={0,1}≠∅, J∩=∅. OK.
With B2: I∩={0,1}≠∅, J∩={2,3}≠∅, K∩={2,3}≠∅. NOT OK!

Try B8: I={1,2}, J={2,3}, K={2,3}
With B1: I∩={1}≠∅, J∩=∅. OK.
With B2: I∩={1}≠∅, J∩={2,3}≠∅, K∩={2,3}≠∅. NOT OK!

Try B8: I={0,1}, J={0,1}, K={4,5}
With B1: I∩={0,1}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
With B2: I∩={0,1}≠∅, J∩=∅. OK.
With B3: I∩={1}≠∅, J∩={0,1}≠∅, K∩={4,5}≠∅. NOT OK!

Try B8: I={1,2}, J={0,1}, K={5,6}
With B1: I∩={1}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
With B2: I∩={1}≠∅, J∩=∅. OK.
With B3: I∩={1,2}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
With B4: I∩={1,2}≠∅, J∩=∅. OK.
With B5: I∩={1}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
With B6: I∩={1,2}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B7: I∩=∅. OK.

B8 compatible! 8 blocks. Rank = 84 - 8 = 76.

Try B9: I={0,1}, J={2,3}, K={5,6}
With B1: I∩={0,1}≠∅, J∩=∅. OK.
With B2: I∩={0,1}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B3: I∩=∅. OK.
With B4: I∩=∅. OK.
With B5: I∩={0,1}≠∅, J∩=∅. OK.
With B6: I∩={1}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B7: I∩={0,1}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B8: I∩={1}≠∅, J∩=∅. OK.

B9 compatible! 9 blocks. Rank = 84 - 9 = 75.

Try B10: I={1,2}, J={2,3}, K={5,6}
With B1: I∩={1}≠∅, J∩=∅. OK.
With B2: I∩={1}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B3: I∩={1,2}≠∅, J∩=∅. OK.
With B4: I∩={1,2}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B5: I∩={1}≠∅, J∩=∅. OK.
With B6: I∩={1,2}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B7: I∩=∅. OK.
With B8: I∩={1,2}≠∅, J∩=∅. OK.
With B9: I∩={1}≠∅, J∩={2,3}≠∅, K∩={5,6}≠∅. NOT OK!

Try B10: I={0,1}, J={0,1}, K={5,6}
With B1: I∩={0,1}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
With B2: I∩={0,1}≠∅, J∩=∅. OK.
With B3: I∩={1}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
With B4: I∩={1}≠∅, J∩=∅. OK.
With B5: I∩={0,1}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
With B6: I∩={1}≠∅, J∩=∅. OK.
With B7: I∩={0,1}≠∅, J∩=∅. OK.
With B8: I∩={1}≠∅, J∩={0,1}≠∅, K∩=∅. OK.
With B9: I∩={0,1}≠∅, J∩=∅. OK.

B10 compatible! 10 blocks. Rank = 84 - 10 = 74.

Let me try to find more...
Try B11: I={1,2}, J={0,1}, K={0,1}
With B1: I∩={1}≠∅, J∩={0,1}≠∅, K∩={0,1}≠∅. NOT OK!

Try B11: I={0,1}, J={2,3}, K={0,1}
With B1: I∩={0,1}≠∅, J∩=∅. OK.
With B2: I∩={0,1}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B3: I∩=∅. OK.
With B4: I∩=∅. OK.
With B5: I∩={0,1}≠∅, J∩=∅. OK.
With B6: I∩={1}≠∅, J∩={2,3}≠∅, K∩={0,1}≠∅. NOT OK!

Try B11: I={1,2}, J={2,3}, K={0,1}
With B1: I∩={1}≠∅, J∩=∅. OK.
With B2: I∩={1}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B3: I∩={1,2}≠∅, J∩=∅. OK.
With B4: I∩={1,2}≠∅, J∩={2,3}≠∅, K∩=∅. OK.
With B5: I∩={1}≠∅, J∩=∅. OK.
With B6: I∩={1,2}≠∅, J∩={2,3}≠∅, K∩={0,1}≠∅. NOT OK!

Try B11: I={0,1}, J={0,1}, K={2,3} -- this is B5, already used.

It seems like 10 is a good maximum. Let me verify the 10 blocks:
B1:  I={0,1}, J={0,1}, K={0,1}
B2:  I={0,1}, J={2,3}, K={2,3}
B3:  I={1,2}, J={0,1}, K={4,5}
B4:  I={1,2}, J={2,3}, K={4,5}
B5:  I={0,1}, J={0,1}, K={2,3}
B6:  I={1,2}, J={2,3}, K={0,1}
B7:  I={0,1}, J={2,3}, K={4,5}
B8:  I={1,2}, J={0,1}, K={5,6}
B9:  I={0,1}, J={2,3}, K={5,6}
B10: I={0,1}, J={0,1}, K={5,6}

Total rank = 84 - 10 = 74.
"""

import time

DIMENSIONS = [3, 4, 7]


def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (28, rank)
    # W: (p*n, rank) = (21, rank)

    # Define 10 compatible 2x2x2 Strassen blocks
    # Each block: (I_set, J_set, K_set) where sets are pairs of indices
    blocks = [
        ({0, 1}, {0, 1}, {0, 1}),   # B1
        ({0, 1}, {2, 3}, {2, 3}),   # B2
        ({1, 2}, {0, 1}, {4, 5}),   # B3
        ({1, 2}, {2, 3}, {4, 5}),   # B4
        ({0, 1}, {0, 1}, {2, 3}),   # B5
        ({1, 2}, {2, 3}, {0, 1}),   # B6
        ({0, 1}, {2, 3}, {4, 5}),   # B7
        ({1, 2}, {0, 1}, {5, 6}),   # B8
        ({0, 1}, {2, 3}, {5, 6}),   # B9
        ({0, 1}, {0, 1}, {5, 6}),   # B10
    ]

    # Track which (i,j,k) positions are covered by Strassen blocks
    covered = set()
    for I_set, J_set, K_set in blocks:
        for i in I_set:
            for j in J_set:
                for k in K_set:
                    covered.add((i, j, k))

    # Count: 10 blocks * 8 positions = 80 covered
    # Schoolbook terms needed: 84 - 80 = 4
    # Strassen terms: 10 * 7 = 70
    # Total rank: 70 + 4 = 74
    rank = 74

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen_block(I_set, J_set, K_set, col_start):
        """Add 7 Strassen terms for a 2x2x2 block."""
        i0, i1 = sorted(I_set)
        j0, j1 = sorted(J_set)
        k0, k1 = sorted(K_set)

        # Strassen's 7 products for 2x2 * 2x2:
        # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # M1 = (A_i0j0 + A_i1j1) * (B_j0k0 + B_j1k1)
        # U: A_i0j0 + A_i1j1 -> {(i0,j0):1, (i1,j1):1}
        # V: B_j0k0 + B_j1k1 -> {(j0,k0):1, (j1,k1):1}
        # W: C_k0i0 += 1, C_k1i1 += 1 -> {(k0,i0):1, (k1,i1):1}
        for (idx_u, coeff_u), (idx_v, coeff_v), (idx_w, coeff_w) in [
            ((i0 * m + j0, 1), (j0 * p + k0, 1), (k0 * n + i0, 1)),
            ((i1 * m + j1, 1), (j1 * p + k1, 1), (k1 * n + i1, 1)),
        ]:
            U[idx_u][col_start] += coeff_u
            V[idx_v][col_start] += coeff_v
            W[idx_w][col_start] += coeff_w
        col_start += 1

        # M2 = (A_i1j0 + A_i1j1) * B_j0k0
        # U: {(i1,j0):1, (i1,j1):1}
        # V: {(j0,k0):1}
        # W: C_k0i1 += 1, C_k1i1 -= 1 -> {(k0,i1):1, (k1,i1):-1}
        U[i1 * m + j0][col_start] = 1
        U[i1 * m + j1][col_start] = 1
        V[j0 * p + k0][col_start] = 1
        W[k0 * n + i1][col_start] = 1
        W[k1 * n + i1][col_start] = -1
        col_start += 1

        # M3 = A_i0j0 * (B_j0k1 - B_j1k1)
        # U: {(i0,j0):1}
        # V: {(j0,k1):1, (j1,k1):-1}
        # W: C_k1i0 += 1, C_k1i1 += 1 -> {(k1,i0):1, (k1,i1):1}
        U[i0 * m + j0][col_start] = 1
        V[j0 * p + k1][col_start] = 1
        V[j1 * p + k1][col_start] = -1
        W[k1 * n + i0][col_start] = 1
        W[k1 * n + i1][col_start] = 1
        col_start += 1

        # M4 = A_i1j1 * (B_j1k0 - B_j0k0)
        # U: {(i1,j1):1}
        # V: {(j1,k0):1, (j0,k0):-1}
        # W: C_k0i0 += 1, C_k0i1 += 1 -> {(k0,i0):1, (k0,i1):1}
        U[i1 * m + j1][col_start] = 1
        V[j1 * p + k0][col_start] = 1
        V[j0 * p + k0][col_start] = -1
        W[k0 * n + i0][col_start] = 1
        W[k0 * n + i1][col_start] = 1
        col_start += 1

        # M5 = (A_i0j0 + A_i0j1) * B_j1k1
        # U: {(i0,j0):1, (i0,j1):1}
        # V: {(j1,k1):1}
        # W: C_k0i0 -= 1, C_k1i0 += 1 -> {(k0,i0):-1, (k1,i0):1}
        U[i0 * m + j0][col_start] = 1
        U[i0 * m + j1][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = -1
        W[k1 * n + i0][col_start] = 1
        col_start += 1

        # M6 = (A_i1j0 - A_i0j0) * (B_j0k0 + B_j0k1)
        # U: {(i1,j0):1, (i0,j0):-1}
        # V: {(j0,k0):1, (j0,k1):1}
        # W: C_k1i1 += 1 -> {(k1,i1):1}
        U[i1 * m + j0][col_start] = 1
        U[i0 * m + j0][col_start] = -1
        V[j0 * p + k0][col_start] = 1
        V[j0 * p + k1][col_start] = 1
        W[k1 * n + i1][col_start] = 1
        col_start += 1

        # M7 = (A_i0j1 - A_i1j1) * (B_j1k0 + B_j1k1)
        # U: {(i0,j1):1, (i1,j1):-1}
        # V: {(j1,k0):1, (j1,k1):1}
        # W: C_k0i0 += 1 -> {(k0,i0):1}
        U[i0 * m + j1][col_start] = 1
        U[i1 * m + j1][col_start] = -1
        V[j1 * p + k0][col_start] = 1
        V[j1 * p + k1][col_start] = 1
        W[k0 * n + i0][col_start] = 1
        col_start += 1

        return col_start

    # Add all Strassen blocks
    for I_set, J_set, K_set in blocks:
        col = add_strassen_block(I_set, J_set, K_set, col)

    # Add schoolbook terms for uncovered positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = 0
                for c in range(rank):
                    total += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if total != 1:
                    raise ValueError(
                        f"Decomposition failed at ({i},{j},{k}): got {total}, expected 1"
                    )

    return {"U": U, "V": V, "W": W}
