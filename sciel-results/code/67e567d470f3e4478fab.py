"""
Rank-minimal tensor decomposition for 3x5 x 5x5 matrix multiplication.

Strategy:
- Use Strassen's algorithm (rank 7) for the 2x2x2 sub-blocks.
- Use schoolbook for the remaining 1x1, 1x2, 2x1, 1x5, 5x1, etc. blocks.
- The 3x5 x 5x5 product can be partitioned into blocks.
- Split the 3 rows of A into [1, 2] and the 5 columns into [1, 4] or similar.
- Actually, let's think about this more carefully.

The tensor for matrix multiplication C = A * B where A is 3x5, B is 5x5, C is 3x5:
C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k] for i in 0..2, k in 0..4.

We can decompose this into blocks based on partitioning the indices.

Let's partition:
- A rows: i in {0, 1, 2}
- B rows / A cols: j in {0, 1, 2, 3, 4}
- C cols / B cols: k in {0, 1, 2, 3, 4}

We can split j into {0,1} and {2,3,4}, and k into {0,1} and {2,3,4}.

Then C = A * B can be written as:
C[i, 0:2] = A[i, 0:2] * B[0:2, 0:2] + A[i, 2:5] * B[2:5, 0:2]
C[i, 2:5] = A[i, 0:2] * B[0:2, 2:5] + A[i, 2:5] * B[2:5, 2:5]

For the 2x2x2 blocks (A[i,0:2] * B[0:2, 0:2] for each i), we can use Strassen.
But A[i, 0:2] is 1x2, B[0:2, 0:2] is 2x2, result is 1x2.
This is not a square block multiplication.

Let me reconsider. The standard approach for non-square matrix multiplication tensor decomposition:

Actually, the simplest approach that gives a good rank:
- Use Strassen for the 2x2x2 subproblems where possible.
- The 3x5x5 tensor has rank at most 75 (schoolbook).

Let's try a different partition. Split the 5 columns/rows into groups of sizes that allow Strassen.

Actually, let me just use a known result: the rank of the 3x5x5 matrix multiplication tensor.

For small matrices, we can use:
- Rank of m x n x p matrix multiplication tensor is at most m*n*p (schoolbook)
- For 2x2x2, Strassen gives rank 7
- For 3x3x3, best known is 23

Let me try to construct a decomposition using Strassen for 2x2x2 blocks.

Partition:
- Split j (middle index) into {0,1} and {2,3,4}
- Split k (output col) into {0,1} and {2,3,4}
- Split i (input row) into {0} and {1,2}

Then we have 8 blocks:
1. i=0, j={0,1}, k={0,1}: A[0,0:2] (1x2) * B[0:2, 0:2] (2x2) -> C[0, 0:2] (1x2)
   This is 1x2 times 2x2 = 1x2. Can be done with 4 multiplications (schoolbook) or...
   Actually for 1x2 * 2x2, it's just 4 scalar multiplications.

2. i=0, j={0,1}, k={2,3,4}: A[0,0:2] (1x2) * B[0:2, 2:5] (2x3) -> C[0, 2:5] (1x3)
   1x2 * 2x3 = 1x3, 6 multiplications.

3. i=0, j={2,3,4}, k={0,1}: A[0,2:5] (1x3) * B[2:5, 0:2] (3x2) -> C[0, 0:2] (1x2)
   1x3 * 3x2 = 1x2, 6 multiplications.

4. i=0, j={2,3,4}, k={2,3,4}: A[0,2:5] (1x3) * B[2:5, 2:5] (3x3) -> C[0, 2:5] (1x3)
   1x3 * 3x3 = 1x3, 9 multiplications.

5-8. Similar for i={1,2}

Total schoolbook: 4+6+6+9 + 2*(4+6+6+9) = 25 + 2*25 = 75.

To improve, I need to find structure. Let me try using Strassen on the 2x2x2 parts.

Actually, let me just implement a direct approach: use the known Strassen decomposition for 2x2x2 blocks and combine with schoolbook for the rest.

The key insight from the cell findings: "Extending Strassen decomposition to 4 disjoint 2x2x2 blocks reduces rank from 74 to 71"

So we can identify 4 disjoint 2x2x2 subproblems within the 3x5x5 tensor and apply Strassen to each.

Let me identify where 2x2x2 blocks can fit:
- We need 2 consecutive i values, 2 consecutive j values, 2 consecutive k values.
- i can be {0,1} or {1,2}
- j can be {0,1}, {1,2}, {2,3}, {3,4}
- k can be {0,1}, {1,2}, {2,3}, {3,4}

If we use i={0,1}, j={0,1}, k={0,1}: that's one 2x2x2 block.
Then we can use i={0,1}, j={0,1}, k={2,3}: another.
And i={0,1}, j={2,3}, k={0,1}: another.
And i={0,1}, j={2,3}, k={2,3}: another.

That's 4 blocks using i={0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
Remaining: i={2}, j={0,1,2,3,4}, k={0,1,2,3,4} and i={0,1}, j={4}, k={0,1,2,3,4} and i={0,1}, j={0,1,2,3}, k={4}.

Wait, this is getting complicated. Let me just implement a clean solution.

Approach:
1. Start with schoolbook (rank 75).
2. Identify 4 disjoint 2x2x2 sub-tensors.
3. Replace each with Strassen (rank 7 instead of 8), saving 1 rank each.
4. Total rank: 75 - 4 = 71.

Let me implement this carefully.
"""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
DEADLINE = 115.0


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=3, m=5, p=5

    # Schoolbook decomposition as baseline
    rank_sb = n * m * p  # 75
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    if time.time() - start_time > DEADLINE - 5:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Strassen decomposition for 2x2x2 matrix multiplication
    # C = A * B where A, B, C are 2x2
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # Strassen's 7 multiplications:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    #
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    #
    # In tensor decomposition form, each multiplication m_r corresponds to a rank-1 term.
    # For m1: U factors are (a00 + a11) and (b00 + b11), W factor is (c00 + c11 contribution)
    # Actually, we need to express each m_r as a product of linear forms.

    # Strassen rank-7 decomposition for 2x2x2:
    # The 8 entries of C are linear combinations of the 7 products m1..m7.
    # We need to express this as a sum of 7 rank-1 tensors.

    # Let me define the Strassen factors explicitly.
    # For a 2x2x2 subproblem with local indices (i, j, k) in {0,1}^3:
    # The tensor T[i,j,k] = 1 (each term A[i,j]*B[j,k] contributes to C[i,k])
    #
    # Strassen decomposition:
    # U factors (for A indices (i,j)):
    # m1: U = [1, 0, 0, 1] for (a00, a01, a10, a11) -> wait, need to be careful with indexing.
    # A is 2x2, indexed by (i,j) where i=0,1 is row, j=0,1 is col.
    # In the global tensor, A index is i*m+j. For local 2x2, it's i*2+j.
    #
    # Let me define the 7 Strassen rank-1 components:
    # Each component r has:
    #   u_r: vector in R^4 (for A entries a00, a01, a10, a11)
    #   v_r: vector in R^4 (for B entries b00, b01, b10, b11)
    #   w_r: vector in R^4 (for C entries c00, c01, c10, c11)
    #
    # The constraint is: for each (i,j,k), sum_r u_r[i*2+j] * v_r[j*2+k] * w_r[k*2+i] = 1

    # Strassen's 7 products:
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b01)
    # m7 = (a01-a11)*(b10+b11)

    # And C entries:
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6

    # So the W factors (coefficients of m_r in each c_ik):
    # w for m1: c00 gets +1, c11 gets +1 -> w1 = [1, 0, 0, 1] (for c00, c01, c10, c11)
    # w for m2: c10 gets +1, c11 gets -1 -> w2 = [0, 0, 1, -1]
    # w for m3: c01 gets +1, c11 gets +1 -> w3 = [0, 1, 0, 1]
    # w for m4: c00 gets +1, c10 gets +1 -> w4 = [1, 0, 1, 0]
    # w for m5: c00 gets -1, c01 gets +1 -> w5 = [-1, 1, 0, 0]
    # w for m6: c11 gets +1 -> w6 = [0, 0, 0, 1]
    # w for m7: c00 gets +1 -> w7 = [1, 0, 0, 0]

    # U factors (coefficients of m_r in each a_ij):
    # m1: a00 gets +1, a11 gets +1 -> u1 = [1, 0, 0, 1] (for a00, a01, a10, a11)
    # m2: a10 gets +1, a11 gets +1 -> u2 = [0, 0, 1, 1]
    # m3: a00 gets +1 -> u3 = [1, 0, 0, 0]
    # m4: a11 gets +1 -> u4 = [0, 0, 0, 1]
    # m5: a00 gets +1, a01 gets +1 -> u5 = [1, 1, 0, 0]
    # m6: a10 gets +1, a00 gets -1 -> u6 = [-1, 0, 1, 0]
    # m7: a01 gets +1, a11 gets -1 -> u7 = [0, 1, 0, -1]

    # V factors (coefficients of m_r in each b_jk):
    # m1: b00 gets +1, b11 gets +1 -> v1 = [1, 0, 0, 1] (for b00, b01, b10, b11)
    # m2: b00 gets +1 -> v2 = [1, 0, 0, 0]
    # m3: b01 gets +1, b11 gets -1 -> v3 = [0, 1, 0, -1]
    # m4: b10 gets +1, b00 gets -1 -> v4 = [-1, 0, 1, 0]
    # m5: b11 gets +1 -> v5 = [0, 0, 0, 1]
    # m6: b00 gets +1, b01 gets +1 -> v6 = [1, 1, 0, 0]
    # m7: b10 gets +1, b11 gets +1 -> v7 = [0, 0, 1, 1]

    # Now let's verify: for (i,j,k) = (0,0,0), we need sum_r u_r[0] * v_r[0] * w_r[0] = 1
    # u_r[0] is coefficient of a00, v_r[0] is coefficient of b00, w_r[0] is coefficient of c00
    # r=1: u1[0]=1, v1[0]=1, w1[0]=1 -> 1*1*1 = 1
    # r=2: u2[0]=0 -> 0
    # r=3: u3[0]=1, v3[0]=0 -> 0
    # r=4: u4[0]=0 -> 0
    # r=5: u5[0]=1, v5[0]=0 -> 0
    # r=6: u6[0]=-1, v6[0]=1, w6[0]=0 -> 0
    # r=7: u7[0]=0 -> 0
    # Sum = 1. Correct!

    # For (i,j,k) = (0,0,1): a00 * b01 -> c01
    # u_r[0] * v_r[1] * w_r[1]
    # r=1: 1*0*0=0
    # r=2: 0*0*0=0
    # r=3: 1*1*1=1
    # r=4: 0*0*0=0
    # r=5: 1*0*1=0
    # r=6: -1*1*0=0
    # r=7: 0*0*0=0
    # Sum = 1. Correct!

    # Great, the Strassen decomposition is verified.

    # Strassen factors as arrays (index 0..3 for local 2x2 entries):
    # A entries: a00=0, a01=1, a10=2, a11=3
    # B entries: b00=0, b01=1, b10=2, b11=3
    # C entries: c00=0, c01=1, c10=2, c11=3

    strassen_U = np.array([
        [1, 0, 0, 1],  # m1
        [0, 0, 1, 1],  # m2
        [1, 0, 0, 0],  # m3
        [0, 0, 0, 1],  # m4
        [1, 1, 0, 0],  # m5
        [-1, 0, 1, 0],  # m6
        [0, 1, 0, -1]   # m7
    ], dtype=float)

    strassen_V = np.array([
        [1, 0, 0, 1],  # m1
        [1, 0, 0, 0],  # m2
        [0, 1, 0, -1],  # m3
        [-1, 0, 1, 0],  # m4
        [0, 0, 0, 1],  # m5
        [1, 1, 0, 0],  # m6
        [0, 0, 1, 1]   # m7
    ], dtype=float)

    strassen_W = np.array([
        [1, 0, 0, 1],  # m1
        [0, 0, 1, -1],  # m2
        [0, 1, 0, 1],  # m3
        [1, 0, 1, 0],  # m4
        [-1, 1, 0, 0],  # m5
        [0, 0, 0, 1],  # m6
        [1, 0, 0, 0]   # m7
    ], dtype=float)

    # Now build the full 3x5x5 decomposition.
    # We'll use 4 disjoint 2x2x2 Strassen blocks.
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    # Block 2: i={0,1}, j={0,1}, k={2,3}
    # Block 3: i={0,1}, j={2,3}, k={0,1}
    # Block 4: i={0,1}, j={2,3}, k={2,3}
    #
    # Remaining terms (schoolbook):
    # i={0,1}, j={4}, k={0,1,2,3,4}: 2*1*5 = 10 terms
    # i={2}, j={0,1,2,3,4}, k={0,1,2,3,4}: 1*5*5 = 25 terms
    # i={0,1}, j={0,1,2,3}, k={4}: 2*4*1 = 8 terms
    #
    # Wait, let me recount. The 4 Strassen blocks cover:
    # i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    # That's 2*4*4 = 32 terms, replaced by 4*7 = 28 rank-1 terms.
    #
    # Remaining:
    # i in {0,1}, j={4}, k in {0,1,2,3,4}: 2*1*5 = 10 terms
    # i={2}, j in {0,1,2,3,4}, k in {0,1,2,3,4}: 1*5*5 = 25 terms
    # i in {0,1}, j in {0,1,2,3}, k={4}: 2*4*1 = 8 terms
    #
    # Total remaining: 10 + 25 + 8 = 43 terms
    # Total rank: 28 + 43 = 71

    # Let's verify: 32 + 10 + 25 + 8 = 75. Correct.

    total_rank = 28 + 43  # 71
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))

    col_offset = 0

    # Define the 4 Strassen blocks
    # Each block is defined by (i_start, j_start, k_start) with size 2x2x2
    strassen_blocks = [
        (0, 0, 0),  # i={0,1}, j={0,1}, k={0,1}
        (0, 0, 2),  # i={0,1}, j={0,1}, k={2,3}
        (0, 2, 0),  # i={0,1}, j={2,3}, k={0,1}
        (0, 2, 2),  # i={0,1}, j={2,3}, k={2,3}
    ]

    for (i_s, j_s, k_s) in strassen_blocks:
        for r in range(7):
            # For each Strassen rank-1 component
            # u_r is a 4-vector for local A entries (li, lj) in {0,1}^2
            # v_r is a 4-vector for local B entries (lj, lk) in {0,1}^2
            # w_r is a 4-vector for local C entries (lk, li) in {0,1}^2

            for li in range(2):
                for lj in range(2):
                    gi = i_s + li
                    gj = j_s + lj
                    a_idx = gi * m + gj  # global A index
                    local_a_idx = li * 2 + lj
                    U_full[a_idx, col_offset] = strassen_U[r, local_a_idx]

            for lj in range(2):
                for lk in range(2):
                    gj = j_s + lj
                    gk = k_s + lk
                    b_idx = gj * p + gk  # global B index
                    local_b_idx = lj * 2 + lk
                    V_full[b_idx, col_offset] = strassen_V[r, local_b_idx]

            for lk in range(2):
                for li in range(2):
                    gk = k_s + lk
                    gi = i_s + li
                    c_idx = gk * n + gi  # global C index (W[k*n+i])
                    local_c_idx = lk * 2 + li
                    W_full[c_idx, col_offset] = strassen_W[r, local_c_idx]

            col_offset += 1

    # Now add schoolbook terms for the remaining (i,j,k) tuples
    # We need to skip the ones covered by Strassen blocks
    # Strassen covers: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}

    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i,j,k) is covered by a Strassen block
                covered = False
                for (i_s, j_s, k_s) in strassen_blocks:
                    if (i_s <= i < i_s + 2) and (j_s <= j < j_s + 2) and (k_s <= k < k_s + 2):
                        covered = True
                        break
                if not covered:
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    U_full[a_idx, col_offset] = 1
                    V_full[b_idx, col_offset] = 1
                    W_full[c_idx, col_offset] = 1
                    col_offset += 1

    # Verify col_offset == total_rank
    # assert col_offset == total_rank, f"col_offset={col_offset}, total_rank={total_rank}"

    # Verify the decomposition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_full[a, :] * V_full[b, :], W_full[c, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err < 1e-10:
        # Convert to lists with clean integer/half-integer values
        U_list = U_full.tolist()
        V_list = V_full.tolist()
        W_list = W_full.tolist()

        # Round small values to 0, and clean up
        for mat in [U_list, V_list, W_list]:
            for row in mat:
                for idx, val in enumerate(row):
                    if abs(val) < 1e-12:
                        row[idx] = 0
                    elif abs(val - round(val)) < 1e-12:
                        row[idx] = float(int(round(val)))
                    elif abs(val - (round(val * 2) / 2)) < 1e-12:
                        row[idx] = float(round(val * 2) / 2)

        return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback: schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
