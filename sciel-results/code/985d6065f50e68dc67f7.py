"""Rewritten program: Strassen-style block decomposition for 2x5*5x6 matrix multiplication."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 -> 2x2 multiplication using Strassen (rank 7).
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    
    Convention:
    - A is 2x2, indexed A[i,j] for i in {0,1}, j in {0,1}
    - B is 2x2, indexed B[j,k]
    - C[i,k] = sum_j A[i,j] * B[j,k]
    - U: rows indexed by (i,j) for A, so row i*2+j
    - V: rows indexed by (j,k) for B, so row j*2+k
    - W: rows indexed by (k,i) for C (trace convention: W[k*n+i, r] where n=2), so row k*2+i
    
    Strassen:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11) B00
    M3 = A00 (B01 - B11)
    M4 = A11 (B10 - B00)
    M5 = (A00 + A01) B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    # A part: A00 (row 0), A11 (row 3)
    # B part: B00 (row 0), B11 (row 3)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    # C part: C00 (row 0), C11 (row 3)
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11) B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1  # C10
    
    # M3 = A00 (B01 - B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1  # C01
    
    # M4 = A11 (B10 - B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1  # C00 and C10
    
    # M5 = (A00 + A01) B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1  # -C00 + C01
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    
    return U, V, W


def _combine_decompositions(decs, n, m, p):
    """
    Combine block decompositions into a full decomposition.
    decs: list of (U_block, V_block, W_block) for each (i,j,k) block combination
    Actually, we need to think about this differently.
    """
    pass


def construct():
    """
    Construct a rank-minimal (or near-minimal) tensor decomposition for 2x5 * 5x6 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strategy: Use Strassen on 2x2 blocks where possible.
    # We can decompose the 2x5 * 5x6 multiplication using block structure.
    # 
    # Key insight: If we can find a partition of the middle dimension (5) into blocks,
    # and use Strassen on compatible sub-blocks, we can reduce the rank.
    #
    # The schoolbook rank is 2*5*6 = 60.
    # 
    # One approach: partition m=5 into 2+3, p=6 into 2+4, n=2 into 1+1 or use directly.
    # 
    # Actually, let's think about this more carefully.
    # 
    # For a (n x m) * (m x p) multiplication, the tensor has rank at most n*m*p (schoolbook).
    # Strassen shows that 2x2*2x2 can be done in rank 7 instead of 8.
    # 
    # We can use a block decomposition:
    # - Partition A (2x5) into blocks, B (5x6) into blocks
    # - For each pair of compatible blocks, use an optimal decomposition
    #
    # Let's try: m=5 = 2+3, so A = [A1 (2x2), A2 (2x3)], B = [B1 (2x6); B2 (3x6)]
    # Then C = A1*B1 + A2*B2
    #
    # A1*B1 is 2x2 * 2x6. We can further partition B1 into 2x2 + 2x4.
    # A1*B1a is 2x2 * 2x2 -> use Strassen (rank 7)
    # A1*B1b is 2x2 * 2x4 -> partition B1b into 2x2 + 2x2, each uses Strassen (2*7=14)
    # So A1*B1 has rank 7 + 14 = 21
    #
    # A2*B2 is 2x3 * 3x6. Partition B2 into 3x2 + 3x4.
    # A2*B2a is 2x3 * 3x2. Partition A2 into 1x3 + 1x3, B2a into 3x1 + 3x1.
    #   Each 1x3 * 3x1 is rank 3 (dot product). Total: 2*3*2 = 12? No...
    #   Actually 1x3 * 3x1 is a scalar (rank 3 for the tensor).
    #   2x3 * 3x2: partition into 4 blocks of 1x3 * 3x1, each rank 3, total 12.
    #   Or partition A2 into 2x1 + 2x2, B2a into 1x2 + 2x2:
    #   2x1 * 1x2: rank 2 (outer product style)
    #   2x2 * 2x2: Strassen rank 7
    #   2x1 * 2x2: rank 4
    #   2x2 * 1x2: rank 4
    #   Total: 2+7+4+4 = 17
    #
    # This is getting complex. Let me use a more systematic approach.
    #
    # Alternative: Use the known result that the rank of matrix multiplication tensor
    # R(n,m,p) satisfies certain bounds. For small dimensions, we can try to find
    # good decompositions by combining Strassen blocks.
    #
    # Let me try a different strategy: direct construction using Strassen where applicable.
    
    # Let's build the decomposition by partitioning.
    # 
    # Partition:
    # n=2: keep as is (can't split 2 into smaller useful blocks for Strassen)
    # m=5: split into 2+3
    # p=6: split into 2+4
    #
    # A is 2x5 = [A_1 (2x2), A_2 (2x3)]
    # B is 5x6 = [B_1 (2x6); B_2 (3x6)]
    #
    # C = A_1 @ B_1 + A_2 @ B_2
    #
    # For A_1 @ B_1 (2x2 * 2x6):
    #   B_1 is 2x6, split into B_1a (2x2), B_1b (2x4)
    #   A_1 @ B_1a: 2x2 * 2x2 -> Strassen, rank 7
    #   A_1 @ B_1b: 2x2 * 2x4, split B_1b into B_1b1 (2x2), B_1b2 (2x2)
    #     A_1 @ B_1b1: Strassen, rank 7
    #     A_1 @ B_1b2: Strassen, rank 7
    #   Total for A_1 @ B_1: 7 + 7 + 7 = 21
    #
    # For A_2 @ B_2 (2x3 * 3x6):
    #   B_2 is 3x6, split into B_2a (3x2), B_2b (3x4)
    #   A_2 @ B_2a: 2x3 * 3x2
    #     Split A_2 into A_2a (2x1), A_2b (2x2)
    #     Split B_2a into B_2a1 (1x2), B_2a2 (2x2)
    #     A_2a @ B_2a1: 2x1 * 1x2 -> rank 2
    #     A_2a @ B_2a2: 2x1 * 2x2 -> rank 4
    #     A_2b @ B_2a1: 2x2 * 1x2 -> rank 4
    #     A_2b @ B_2a2: 2x2 * 2x2 -> Strassen, rank 7
    #     Total for A_2 @ B_2a: 2+4+4+7 = 17
    #   A_2 @ B_2b: 2x3 * 3x4
    #     Split B_2b into B_2b1 (3x2), B_2b2 (3x2)
    #     A_2 @ B_2b1: same as A_2 @ B_2a, rank 17
    #     A_2 @ B_2b2: same, rank 17
    #     Total for A_2 @ B_2b: 17+17 = 34
    #   Total for A_2 @ B_2: 17+34 = 51
    #
    # Total rank: 21 + 51 = 72. That's worse than schoolbook (60)!
    #
    # The problem is that small blocks don't benefit from Strassen.
    # 
    # Let me try a different partition.
    #
    # Try: m=5=1+4, p=6=2+4
    # A_1 (2x1) @ B_1 (1x6): rank 6 (outer product)
    # A_2 (2x4) @ B_2 (4x6): 
    #   Split B_2 into B_2a (4x2), B_2b (4x4)
    #   A_2 @ B_2a: 2x4 * 4x2, split A_2 into 2x2+2x2, B_2a into 2x2+2x2
    #     Four 2x2*2x2 blocks, each Strassen rank 7: total 28
    #   A_2 @ B_2b: 2x4 * 4x4, split A_2 into 2x2+2x2, B_2b into 4x2+4x2
    #     A_2a @ B_2b1: 2x2*4x2, split B_2b1 into 2x2+2x2: 2*7=14
    #     A_2a @ B_2b2: 14
    #     A_2b @ B_2b1: 14
    #     A_2b @ B_2b2: 14
    #     Total: 56
    #   Total A_2 @ B_2: 28+56=84
    # Total: 6+84=90. Worse.
    #
    # The issue is that Strassen only helps for 2x2*2x2. For other sizes, schoolbook is often better.
    #
    # Let me reconsider. The best approach might be to use Strassen only where we have exact 2x2*2x2 blocks,
    # and use schoolbook elsewhere.
    #
    # Schoolbook: 60
    # If we can find k disjoint 2x2*2x2 sub-problems, each saves 1 rank (8->7).
    # Max number of 2x2*2x2 blocks in 2x5*5x6:
    #   We need to pick 2 rows of A, 2 columns of A (which are rows of B), 2 columns of B.
    #   n=2: only one way to pick 2 rows.
    #   m=5: C(5,2)=10 ways to pick 2 middle indices.
    #   p=6: C(6,2)=15 ways to pick 2 output columns.
    #   But the blocks must be disjoint in the middle dimension.
    #   Max disjoint 2-element subsets of {0,1,2,3,4}: floor(5/2)=2.
    #   So we can have at most 2 Strassen blocks, saving 2 ranks.
    #   Rank: 60-2=58.
    #
    # But wait, we can also use other optimizations. Let me just go with the known best:
    # Use schoolbook but replace 2x2*2x2 sub-blocks with Strassen.
    
    # Let me implement a clean approach:
    # 1. Start with schoolbook (rank 60)
    # 2. Identify 2x2*2x2 sub-blocks and replace with Strassen
    #
    # For 2x5*5x6:
    #   Middle dimension indices: 0,1,2,3,4
    #   We can pair (0,1) and (2,3), leaving index 4 for schoolbook.
    #   For each pair of middle indices and each pair of output columns, we can use Strassen.
    #   But we need to be careful about how the blocks interact.
    #
    # Actually, the cleanest way: 
    #   Use Strassen for the sub-problem involving middle indices {0,1} and output columns {0,1}
    #   Use Strassen for the sub-problem involving middle indices {2,3} and output columns {2,3}
    #   Use schoolbook for everything else
    #
    # But this doesn't quite work because the blocks overlap in the output.
    #
    # Let me just implement the straightforward approach:
    # Build the full decomposition using a combination of Strassen and schoolbook.
    
    # I'll use a different strategy: directly construct the decomposition.
    # 
    # The tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # Rank decomposition: T = sum_r U[:,r] * V[:,r] * W[:,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    #
    # For schoolbook: each (i,j,k) triple gives one term.
    # For Strassen on a 2x2*2x2 block: 7 terms instead of 8.
    
    # Let me build it systematically.
    
    def schoolbook_term(i, j, k, n, p):
        """Return (u_idx, v_idx, w_idx) for schoolbook term A[i,j]*B[j,k] contributing to C[i,k]."""
        u_idx = i * m + j  # row in U (n*m rows)
        v_idx = j * p + k  # row in V (m*p rows)
        w_idx = k * n + i  # row in W (p*n rows, trace convention)
        return u_idx, v_idx, w_idx
    
    # Strategy: Use Strassen for two 2x2*2x2 blocks and schoolbook for the rest.
    # 
    # Block 1: middle indices {0,1}, output columns {0,1}
    # Block 2: middle indices {2,3}, output columns {2,3}
    # Rest: schoolbook
    #
    # But this is tricky because the Strassen blocks produce all 4 output entries C[i,k]
    # for i in {0,1}, k in {0,1} (block 1) and i in {0,1}, k in {2,3} (block 2).
    # The remaining schoolbook terms should only cover the non-Strassen parts.
    
    # Actually, let me think about this differently.
    # 
    # The full multiplication is:
    # C[i,k] = sum_{j=0}^{4} A[i,j] * B[j,k]
    #
    # We can write this as:
    # C[i,k] = sum_{j in S1} A[i,j]*B[j,k] + sum_{j in S2} A[i,j]*B[j,k] + A[i,4]*B[4,k]
    # where S1={0,1}, S2={2,3}.
    #
    # For fixed i and k, the sum over j in S1 is a 2-term dot product.
    # If we group by (i,k) pairs where i in {0,1} and k in {0,1}, we get a 2x2*2x2 sub-problem
    # with A_sub[i,j] = A[i,j] for j in S1, B_sub[j,k] = B[j,k] for k in {0,1}.
    # This sub-problem can be done in rank 7 (Strassen) instead of rank 8 (schoolbook).
    #
    # Similarly for S2 and k in {2,3}.
    #
    # The remaining terms involve:
    # - j in S1, k in {4,5}: 2 middle indices * 2 output cols * 2 rows = 8 terms
    # - j in S2, k in {0,1,4,5}: 2*4*2 = 16 terms
    # - j=4, k in {0,1,2,3,4,5}: 1*6*2 = 12 terms
    # Total remaining: 8+16+12 = 36 terms
    #
    # Total rank: 7 + 7 + 36 = 50
    #
    # Wait, that seems too optimistic. Let me recount.
    #
    # Total schoolbook terms: 2*5*6 = 60
    # Strassen block 1 replaces: i in {0,1}, j in {0,1}, k in {0,1} -> 2*2*2=8 terms with 7
    # Strassen block 2 replaces: i in {0,1}, j in {2,3}, k in {2,3} -> 2*2*2=8 terms with 7
    # Remaining: 60 - 8 - 8 = 44 terms
    # Total rank: 7 + 7 + 44 = 58
    #
    # So we save 2 ranks, going from 60 to 58.
    
    # Let me implement this.
    
    # First, let's identify which (i,j,k) triples are covered by Strassen blocks.
    strassen1_indices = set()
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                strassen1_indices.add((i, j, k))
    
    strassen2_indices = set()
    for i in [0, 1]:
        for j in [2, 3]:
            for k in [2, 3]:
                strassen2_indices.add((i, j, k))
    
    # Build the decomposition
    U = []
    V = []
    W = []
    
    # Add Strassen block 1
    U1, V1, W1 = _strassen_2x2()
    # U1 has rows for A[i,j] with i,j in {0,1} -> rows 0,1,2,3 of full U
    # V1 has rows for B[j,k] with j in {0,1}, k in {0,1} -> rows 0,1,2,3 of full V (since p=6, j*p+k)
    #   j=0: k=0,1 -> rows 0,1
    #   j=1: k=0,1 -> rows 6,7
    # Wait, V row index is j*p+k. For j in {0,1}, k in {0,1}:
    #   j=0,k=0: row 0
    #   j=0,k=1: row 1
    #   j=1,k=0: row 6
    #   j=1,k=1: row 7
    # So V1's rows correspond to V rows [0, 1, 6, 7]
    #
    # W1 has rows for C[i,k] with i in {0,1}, k in {0,1} -> W row index is k*n+i
    #   k=0,i=0: row 0
    #   k=0,i=1: row 1
    #   k=1,i=0: row 2
    #   k=1,i=1: row 3
    # So W1's rows correspond to W rows [0, 1, 2, 3]
    
    for r in range(7):
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        for u_row in range(4):
            U_col[u_row] = U1[u_row][r]  # u_row maps directly to U row (i*m+j for i,j in {0,1})
        # V row mapping: V1 row 0->V row 0, 1->V row 1, 2->V row 6, 3->V row 7
        v_row_map = [0, 1, 6, 7]
        for v_row in range(4):
            V_col[v_row_map[v_row]] = V1[v_row][r]
        # W row mapping: W1 row 0->W row 0, 1->W row 1, 2->W row 2, 3->W row 3
        for w_row in range(4):
            W_col[w_row] = W1[w_row][r]
        U.append(U_col)
        V.append(V_col)
        W.append(W_col)
    
    # Add Strassen block 2
    U2, V2, W2 = _strassen_2x2()
    # U2 rows: i in {0,1}, j in {2,3} -> U row index i*m+j
    #   i=0,j=2: row 2
    #   i=0,j=3: row 3
    #   i=1,j=2: row 7
    #   i=1,j=3: row 8
    # V2 rows: j in {2,3}, k in {2,3} -> V row index j*p+k
    #   j=2,k=2: row 14
    #   j=2,k=3: row 15
    #   j=3,k=2: row 20
    #   j=3,k=3: row 21
    # W2 rows: i in {0,1}, k in {2,3} -> W row index k*n+i
    #   k=2,i=0: row 4
    #   k=2,i=1: row 5
    #   k=3,i=0: row 6
    #   k=3,i=1: row 7
    
    u_row_map2 = [2, 3, 7, 8]  # U2 row 0->U row 2, 1->3, 2->7, 3->8
    v_row_map2 = [14, 15, 20, 21]  # V2 row 0->V row 14, etc.
    w_row_map2 = [4, 5, 6, 7]  # W2 row 0->W row 4, etc.
    
    for r in range(7):
        U_col = [0] * (n * m)
        V_col = [0] * (m * p)
        W_col = [0] * (p * n)
        for u_row in range(4):
            U_col[u_row_map2[u_row]] = U2[u_row][r]
        for v_row in range(4):
            V_col[v_row_map2[v_row]] = V2[v_row][r]
        for w_row in range(4):
            W_col[w_row_map2[w_row]] = W2[w_row][r]
        U.append(U_col)
        V.append(V_col)
        W.append(W_col)
    
    # Add schoolbook terms for remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen1_indices or (i, j, k) in strassen2_indices:
                    continue
                U_col = [0] * (n * m)
                V_col = [0] * (m * p)
                W_col = [0] * (p * n)
                u_idx, v_idx, w_idx = schoolbook_term(i, j, k, n, p)
                U_col[u_idx] = 1
                V_col[v_idx] = 1
                W_col[w_idx] = 1
                U.append(U_col)
                V.append(V_col)
                W.append(W_col)
    
    rank = len(U)
    
    # Verify the decomposition
    # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k
    # Actually, the tensor is: T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # But the decomposition is: T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # So T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    # And T[a,b,c] = 1 if the indices are consistent (same i,j,k), else 0.
    
    # Let's verify
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Check a few random entries
    valid = True
    for _ in range(1000):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        a = i * m + j
        b = j * p + k
        c = k * n + i
        # T[a,b,c] should be 1
        val = sum(U_arr[a, r] * V_arr[b, r] * W_arr[c, r] for r in range(rank))
        if abs(val - 1) > 1e-9:
            valid = False
            break
    
    # Also check that T[a,b,c] = 0 for inconsistent indices
    for _ in range(1000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        # Extract i,j from a, j,k from b, k,i from c
        i1 = a // m
        j1 = a % m
        j2 = b // p
        k1 = b % p
        k2 = c // n
        i2 = c % n
        if i1 == i2 and j1 == j2 and k1 == k2:
            continue  # consistent, skip
        val = sum(U_arr[a, r] * V_arr[b, r] * W_arr[c, r] for r in range(rank))
        if abs(val) > 1e-9:
            valid = False
            break
    
    if not valid:
        # Fallback to schoolbook
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        rank = n*m*p
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
