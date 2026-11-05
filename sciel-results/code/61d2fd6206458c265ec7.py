"""Tensor decomposition for 3x5 * 5x7 matrix multiplication.

Strategy: Block decomposition using Strassen's rank-7 algorithm for 2x2x2.
Split the 3x5x7 problem into sub-problems:
- 3x2x7: use Strassen on 2x2x2 blocks (rank 7 per block, 7 blocks for k=0..6)
- 3x3x7: schoolbook (rank 63)
Total: 49 + 63 = 112.

Actually, let me think more carefully. The key insight is that we can use
Strassen recursively. For 3x5x7:
- Pad to 4x8x8? No, that's too big.
- Use the fact that 3 = 2+1, 5 = 2+2+1, 7 = 2+2+2+1

Better approach: Split A into A1 (2x5) and A2 (1x5), B into B1 (5x4) and B2 (5x3).
C = [A1; A2] * [B1 | B2] = [[A1*B1, A1*B2], [A2*B1, A2*B2]]

For A1*B1 (2x4): pad to 2x4x4, use Strassen? No, Strassen is for square.
Actually, 2x4x4 can be done as 2x2x2 (Strassen, rank 7) + 2x2x2 (Strassen, rank 7) = 14.
For A1*B2 (2x3): 2x2x2 (7) + 2x1x2 (2) = 9.
For A2*B1 (1x4): 1x2x2 (2) + 1x2x2 (2) = 4.
For A2*B2 (1x3): 1x2x2 (2) + 1x1x2 (1) = 3.
Total: 14 + 9 + 4 + 3 = 30. That seems too good.

Wait, I need to be more careful. The rank of a matrix multiplication tensor
[m x n] * [n x p] is the minimum number of rank-1 tensors needed.

For 2x2x2: rank = 7 (Strassen).
For 2x1x2: rank = 2 (just two terms).
For 1x2x2: rank = 2.
For 1x1x2: rank = 1.

So:
A1*B1 (2x4x4): Split B1 into B1a (5x2) and B1b (5x2).
  A1*B1a (2x2x2): rank 7
  A1*B1b (2x2x2): rank 7
  Total: 14

A1*B2 (2x3x3): Split B2 into B2a (5x2) and B2b (5x1).
  A1*B2a (2x2x2): rank 7
  A1*B2b (2x2x1): rank 2 (two terms: A1[0,0]*B2b[0,0] and A1[1,0]*B2b[0,0]... wait)
  Actually 2x2x1: C[i,0] = sum_j A[i,j]*B[j,0]. Two terms: (i=0,j=0), (i=1,j=0)? No.
  C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
  C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
  That's 4 schoolbook terms, but can we do better?
  Actually for 2x2x1, the rank is 2. We can write:
  C[0,0] = M1, C[1,0] = M2 where M1 = (A[0,0]+A[0,1])*(B[0,0]+B[1,0]) - ... no.
  Let me just use schoolbook for small cases: 2x2x1 = 4 terms.
  
  Total A1*B2: 7 + 4 = 11

A2*B1 (1x4x4): Split B1 into B1a (5x2) and B1b (5x2).
  A2*B1a (1x2x2): rank 2
  A2*B1b (1x2x2): rank 2
  Total: 4

A2*B2 (1x3x3): Split B2 into B2a (5x2) and B2b (5x1).
  A2*B2a (1x2x2): rank 2
  A2*B2b (1x2x1): rank 2 (schoolbook: 2 terms)
  Total: 4

Grand total: 14 + 11 + 4 + 4 = 33.

Hmm, that's way below 105. But I suspect I'm making an error. The issue is that
when I split, the sub-problems share the same A and B entries, so I can't just
add the ranks independently unless the decompositions are independent.

Actually, they ARE independent because each sub-product contributes to a different
block of C. So the total rank is the sum of the ranks of the sub-problems.

But wait, for A1*B1a (2x2x2), the A1 is 2x2 (rows 0-1, cols 0-1 of A) and B1a is
2x2 (rows 0-1, cols 0-1 of B). The Strassen decomposition uses specific linear
combinations. Similarly for A1*B1b, A1 is 2x2 (rows 0-1, cols 2-3 of A) and B1b
is 2x2 (rows 2-3, cols 0-1 of B). These are different sub-matrices, so the
decompositions are independent.

Let me implement this properly.

Actually, I realize the issue: when I say "A1*B1a (2x2x2)", I mean multiplying
a 2x2 matrix by a 2x2 matrix. But A1 is 2x5 and B1a is 5x2. So A1*B1a is a
2x5 * 5x2 = 2x2 product, which has rank 5*2*2 = 20 schoolbook, or can be done
better.

I was confusing myself. Let me restart.

The problem is 3x5x7: multiply a 3x5 matrix A by a 5x7 matrix B to get a 3x7 C.

Approach: Use Strassen's algorithm recursively.

3x5x7: 
- Split A into A_top (2x5) and A_bot (1x5)
- Split B into B_left (5x4) and B_right (5x3)
- C = [[A_top * B_left, A_top * B_right], [A_bot * B_left, A_bot * B_right]]

Each sub-product:
1. A_top * B_left: 2x5 * 5x4 = 2x4
2. A_top * B_right: 2x5 * 5x3 = 2x3
3. A_bot * B_left: 1x5 * 5x4 = 1x4
4. A_bot * B_right: 1x5 * 5x3 = 1x3

For each, I can further split:

1. 2x5 * 5x4: Split B_left into B_left1 (5x2) and B_left2 (5x2).
   - 2x5 * 5x2: Split A_top into A_top1 (2x3) and A_top2 (2x2)? No, the inner dim must match.
   Actually, the inner dim is 5. I need to split the inner dim.
   
   Let me split the inner dim (5) into 2+3:
   - A_top[:, :2] * B_left1[:2, :] (2x2 * 2x2 = 2x2, rank 7 via Strassen)
   - A_top[:, 2:] * B_left1[2:, :] (2x3 * 3x2 = 2x2, rank 6 schoolbook)
   - A_top[:, :2] * B_left2[:2, :] (2x2 * 2x2 = 2x2, rank 7)
   - A_top[:, 2:] * B_left2[2:, :] (2x3 * 3x2 = 2x2, rank 6)
   Total for 2x5*5x4: 7+6+7+6 = 26

This is getting complicated. Let me just implement a general recursive Strassen-like
decomposition.

Actually, the simplest correct approach that beats 105:

Use the known result that the rank of the 3x5x7 tensor is at most 49, achieved by
padding to 4x4x4 and using Strassen (rank 7^2 = 49 for 4x4x4).

But 3x5x7 can't be padded to 4x4x4 because the inner dim is 5 > 4.

Alternative: Pad to 4x8x4 (inner dim 8 = 4+4).
- A_pad: 4x8 (original 3x5 padded with zeros)
- B_pad: 8x4 (original 5x7 padded with zeros)  
- C_pad: 4x4

Split the inner dim 8 into 4+4:
- Part 1: A_pad[:, :4] * B_pad[:4, :] (4x4 * 4x4 = 4x4, rank 49 via Strassen)
- Part 2: A_pad[:, 4:] * B_pad[4:, :] (4x4 * 4x4 = 4x4, rank 49 via Strassen)
Total: 98.

But A_pad[:, 4:] only has column 4 non-zero (since original A is 3x5), and
B_pad[4:, :] only has row 4 non-zero (since original B is 5x7). So Part 2 is
actually rank 4*1*4 = 16 (one term per output entry, since only one inner index
contributes).

Total: 49 + 16 = 65.

For the remaining columns of C (k=4,5,6):
- C[:, 4:] = A * B[:, 4:] (3x5 * 5x3 = 3x3, rank 45 schoolbook)

Total: 65 + 45 = 110. Worse than 105.

OK, let me try a different split. Pad to 4x8x8:
- A_pad: 4x8, B_pad: 8x8, C_pad: 4x8
- Split inner 8 into 4+4:
  - Part 1: 4x4 * 4x8 (rank 4*4*8/2 * 7/8... no)
  
This is getting too complicated. Let me just implement the block decomposition
that was mentioned in the knowledge: split into 3x2x7 blocks using Strassen.

3x5x7: Split the middle dim 5 into 2+2+1:
- Block 1: 3x2x7 (A[:, :2] * B[:2, :])
- Block 2: 3x2x7 (A[:, 2:4] * B[2:4, :])
- Block 3: 3x1x7 (A[:, 4] * B[4, :])

For 3x2x7: Split the output dim 7 into 2+2+2+1:
- 3x2x2 (rank 6 schoolbook, or can we do better?)
- 3x2x2 (rank 6)
- 3x2x2 (rank 6)
- 3x2x1 (rank 3)
Total for 3x2x7: 6+6+6+3 = 21

For 3x1x7: rank 3*1*7 = 21 schoolbook.

Total: 21 + 21 + 21 = 63.

That's better than 105! Let me implement this.

Wait, but the decompositions for different blocks are independent, so I can just
concatenate them. Let me code this up.
"""
import time
import numpy as np

N, M, P = 3, 5, 7


def _strassen_2x2x2_terms():
    """Return 7 Strassen rank-1 terms for 2x2x2.
    
    Each term: (u, v, w) where u, v, w are dicts mapping (i,j), (j,k), (k,i) to coeff.
    """
    terms = []
    # M1 = (a00+a11)(b00+b11)
    terms.append((
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1},
        {(0,0):1, (1,1):1}
    ))
    # M2 = (a10+a11)b00
    terms.append((
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,0):1}
    ))
    # M3 = a00(b10-b00)
    terms.append((
        {(0,0):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1}
    ))
    # M4 = a11(b11-b01)
    terms.append((
        {(1,1):1},
        {(1,1):1, (0,1):-1},
        {(0,0):1}
    ))
    # M5 = (a00+a01)b11
    terms.append((
        {(0,0):1, (0,1):1},
        {(1,1):1},
        {(1,1):1}
    ))
    # M6 = (a10-a00)(b00+b01)
    terms.append((
        {(1,0):1, (0,0):-1},
        {(0,0):1, (0,1):1},
        {(1,1):1}
    ))
    # M7 = (a01-a11)(b10+b11)
    terms.append((
        {(0,1):1, (1,1):-1},
        {(1,0):1, (1,1):1},
        {(0,0):1}
    ))
    return terms


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = N, M, P  # 3, 5, 7
    
    # Strategy: Block decomposition
    # Split middle dim 5 into 2+2+1
    # Block 1: A[:, 0:2] * B[0:2, :] -> 3x2x7
    # Block 2: A[:, 2:4] * B[2:4, :] -> 3x2x7
    # Block 3: A[:, 4:5] * B[4:5, :] -> 3x1x7
    
    # For 3x2x7: split output dim 7 into 2+2+2+1
    #   3x2x2 (4 terms each, but can use Strassen-like for 2x2x2 part)
    #   Actually 3x2x2: C[i,k] = sum_j A[i,j]*B[j,k] for i=0..2, j=0..1, k=0..1
    #   That's 3*2*2 = 12 schoolbook terms.
    #   Can we do better? The 2x2x2 part (rows 0-1 of A, all of B) has rank 7.
    #   Row 2 of A: 2*2 = 4 terms.
    #   Total: 7 + 4 = 11 for 3x2x2.
    
    # For 3x2x1: C[i,0] = sum_j A[i,j]*B[j,0], 3*2 = 6 terms.
    
    # So 3x2x7: 11 + 11 + 11 + 6 = 39
    # 3x1x7: 3*1*7 = 21
    # Total: 39 + 39 + 21 = 99
    
    # Let me implement this.
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_dict, v_dict, w_dict):
        nonlocal rank
        U.append(u_dict)
        V.append(v_dict)
        W.append(w_dict)
        rank += 1
    
    # Helper: add schoolbook terms for a sub-block
    def add_schoolbook(i_off, j_off, k_off, n_sub, m_sub, p_sub):
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u = {(i_off + i, j_off + j): 1}
                    v = {(j_off + j, k_off + k): 1}
                    w = {(k_off + k, i_off + i): 1}
                    add_term(u, v, w)
    
    # Helper: add Strassen terms for 2x2x2 sub-block
    def add_strassen(i_off, j_off, k_off):
        base = _strassen_2x2x2_terms()
        for u_local, v_local, w_local in base:
            u = {}
            v = {}
            w = {}
            for (ii, jj), coeff in u_local.items():
                key = (i_off + ii, j_off + jj)
                u[key] = u.get(key, 0) + coeff
            for (jj, kk), coeff in v_local.items():
                key = (j_off + jj, k_off + kk)
                v[key] = v.get(key, 0) + coeff
            for (kk, ii), coeff in w_local.items():
                key = (k_off + kk, i_off + ii)
                w[key] = w.get(key, 0) + coeff
            add_term(u, v, w)
    
    # Block 1: A[:, 0:2] * B[0:2, :] -> 3x2x7
    # Split output into 2+2+2+1
    # Sub-block 1a: rows 0-1 of A, cols 0-1 of B, output cols 0-1: 2x2x2 (Strassen, rank 7)
    # Sub-block 1b: row 2 of A, cols 0-1 of B, output cols 0-1: 1x2x2 (schoolbook, rank 4)
    add_strassen(0, 0, 0)  # 2x2x2 for i=0-1, j=0-1, k=0-1
    add_schoolbook(2, 0, 0, 1, 2, 2)  # i=2, j=0-1, k=0-1
    
    # Sub-block 1c: rows 0-1, cols 0-1, output cols 2-3: 2x2x2
    # Sub-block 1d: row 2, cols 0-1, output cols 2-3: 1x2x2
    add_strassen(0, 0, 2)
    add_schoolbook(2, 0, 2, 1, 2, 2)
    
    # Sub-block 1e: rows 0-1, cols 0-1, output cols 4-5: 2x2x2
    # Sub-block 1f: row 2, cols 0-1, output cols 4-5: 1x2x2
    add_strassen(0, 0, 4)
    add_schoolbook(2, 0, 4, 1, 2, 2)
    
    # Sub-block 1g: rows 0-1, cols 0-1, output col 6: 2x2x1
    # Sub-block 1h: row 2, cols 0-1, output col 6: 1x2x1
    add_schoolbook(0, 0, 6, 2, 2, 1)
    add_schoolbook(2, 0, 6, 1, 2, 1)
    
    # Block 2: A[:, 2:4] * B[2:4, :] -> 3x2x7
    # Same structure, j offset by 2
    add_strassen(0, 2, 0)
    add_schoolbook(2, 2, 0, 1, 2, 2)
    
    add_strassen(0, 2, 2)
    add_schoolbook(2, 2, 2, 1, 2, 2)
    
    add_strassen(0, 2, 4)
    add_schoolbook(2, 2, 4, 1, 2, 2)
    
    add_schoolbook(0, 2, 6, 2, 2, 1)
    add_schoolbook(2, 2, 6, 1, 2, 1)
    
    # Block 3: A[:, 4:5] * B[4:5, :] -> 3x1x7 (schoolbook, rank 21)
    add_schoolbook(0, 4, 0, 3, 1, 7)
    
    # Convert to numpy arrays
    U_arr = np.zeros((n * m, rank))
    V_arr = np.zeros((m * p, rank))
    W_arr = np.zeros((p * n, rank))
    
    for col in range(rank):
        for (i, j), coeff in U[col].items():
            U_arr[i * m + j, col] = coeff
        for (j, k), coeff in V[col].items():
            V_arr[j * p + k, col] = coeff
        for (k, i), coeff in W[col].items():
            W_arr[k * n + i, col] = coeff
    
    return {"U": U_arr, "V": V_arr, "W": W_arr}
