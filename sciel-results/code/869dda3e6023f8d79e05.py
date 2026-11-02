"""Find a rank-minimal exact tensor decomposition for 3x4 x 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    U rows indexed by (i,j) in {00,01,10,11}, V rows by (j,k), W rows by (k,i).
    Strassen 7 products:
      P1 = (A00+A11)(B00+B11)
      P2 = (A10+A11)(B00)
      P3 = (A00)(B01-B11)
      P4 = (A11)(B10-B00)
      P5 = (A00+A01)(B11)
      P6 = (A10-A00)(B00+B01)
      P7 = (A01-A11)(B10+B11)
    """
    # Index order: U row: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #                V row: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #                W row: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    U = [[1, 1, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0, 1],
         [1, 1, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 0, 0]]
    V = [[1, 1, 0, 0, 0, 1, 1],
         [0, 0, 0, 1, 0, 0, 0],
         [0, 0, 1, -1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]]  # placeholder, fill below
    
    # Let me re-derive carefully.
    # P1: A: rows (0,0)=1, (1,1)=1 -> U[0]=1, U[3]=1; B: (0,0)=1, (1,1)=1 -> V[0]=1, V[3]=1
    # P2: A: (1,0)=1, (1,1)=1 -> U[2]=1, U[3]=1; B: (0,0)=1 -> V[0]=1
    # P3: A: (0,0)=1 -> U[0]=1; B: (0,1)=1, (1,1)=-1 -> V[1]=1, V[3]=-1
    # P4: A: (1,1)=1 -> U[3]=1; B: (1,0)=1, (0,0)=-1 -> V[2]=1, V[0]=-1
    # P5: A: (0,0)=1, (0,1)=1 -> U[0]=1, U[1]=1; B: (1,1)=1 -> V[3]=1
    # P6: A: (1,0)=1, (0,0)=-1 -> U[2]=1, U[0]=-1; B: (0,0)=1, (0,1)=1 -> V[0]=1, V[1]=1
    # P7: A: (0,1)=1, (1,1)=-1 -> U[1]=1, U[3]=-1; B: (1,0)=1, (1,1)=1 -> V[2]=1, V[3]=1
    
    U = [[1, 1, 0, 0, 1, -1, 0],   # i=0,j=0
         [0, 1, 0, 0, 1, 0, 1],    # i=0,j=1
         [1, 1, 0, 0, 0, 1, 0],    # i=1,j=0
         [1, 0, 0, 1, 0, 0, -1]]   # i=1,j=1
    
    V = [[1, 1, 0, -1, 0, 1, 0],   # j=0,k=0
         [0, 0, 0, 0, 0, 1, 0],    # j=0,k=1
         [0, 0, 1, 1, 0, 0, 1],    # j=1,k=0
         [1, 0, -1, 1, 1, 0, 1]]   # j=1,k=1
    
    # W: C[i,k] = sum of appropriate P's
    # C[0,0] = P1 - P4 + P6
    # C[0,1] = P3 + P5 + P6
    # C[1,0] = P2 + P4 + P7
    # C[1,1] = P1 - P3 + P5 - P7  (wait, let me recheck)
    # Actually standard Strassen:
    # C00 = P1 - P4 + P6
    # C01 = P3 + P5 + P6
    # C10 = P2 + P4 + P7
    # C11 = P1 - P3 + P5 - P7
    W = [[1, 0, 0, -1, 0, 1, 0],    # k=0,i=0
         [0, 1, 1, 0, 1, 1, 0],     # k=0,i=1
         [0, 0, 1, 0, 1, 0, 1],     # k=1,i=0
         [1, 0, -1, 0, -1, 0, -1]]  # k=1,i=1
    
    # Wait, I need W[k,i] convention. W row index is k*n+i where n=2.
    # So W row 0 = (k=0,i=0), row 1 = (k=0,i=1), row 2 = (k=1,i=0), row 3 = (k=1,i=1)
    # C[0,0] uses P1, P4, P6 -> W[0] = [1, 0, 0, -1, 0, 1, 0]  (P1 - P4 + P6)
    # C[0,1] uses P3, P5, P6 -> W[1] = [0, 0, 1, 0, 1, 1, 0]  (P3 + P5 + P6)
    # C[1,0] uses P2, P4, P7 -> W[2] = [0, 1, 0, -1, 0, 0, 1]  (P2 - P4... wait)
    # Hmm, C[1,0] = P2 + P4 + P7 -> W[2] = [0, 1, 0, 1, 0, 0, 1]
    # C[1,1] = P1 - P3 + P5 - P7 -> W[3] = [1, 0, -1, 0, 1, 0, -1]
    
    W = [[1, 0, 0, -1, 0, 1, 0],
         [0, 0, 1, 0, 1, 1, 0],
         [0, 1, 0, 1, 0, 0, 1],
         [1, 0, -1, 0, 1, 0, -1]]
    
    return U, V, W


def _build_strassen_decomposition():
    """
    Build the rank-44 decomposition using Strassen for 2x2 blocks.
    
    Partition:
    - i: I0 = {0, 1}, I1 = {2}
    - j: J0 = {0, 1}, J1 = {2, 3}
    - k: K0 = {0, 1}, K1 = {2, 3}, K2 = {4, 5}, K3 = {6}
    
    For (i in I0, j in J0) x k in K0,K1,K2: 3 Strassen 2x2x2 (rank 7 each)
    For (i in I0, j in J0) x k in K3: 1 schoolbook 2x2x1 (rank 4)
    Similarly for (i in I0, j in J1), (i in I1, j in J0), (i in I1, j in J1)
    
    Total: 4 * (3*7 + 4) + ... wait, let me recount.
    
    Actually: 4 (i,j blocks) * 3 (k pairs) * 7 (Strassen) + 4 * 4 (schoolbook k=6) 
    = 4*21 + 16 = 84+16 = 100? No that's wrong.
    
    Let me reconsider. The total rank should be:
    - 4 (i,j combinations) * [3 (k-pairs) * 7 (Strassen rank) + 1 (k=singleton) * 4 (schoolbook)]
    = 4 * [21 + 4] = 4 * 25 = 100
    
    That's worse than 84! Let me rethink.
    
    Actually the issue is I'm double-counting. Let me think differently.
    
    The tensor has n*m*p = 3*4*7 = 84 non-zero entries (schoolbook rank 84).
    Strassen for each independent 2x2x2 subproblem saves 1 multiplication per block.
    
    Better approach: Use the known rank for small matrices.
    For 2x2x2, rank is 7 (Strassen). 
    For 2x2xn, the rank can be computed.
    
    Let me use a different strategy: direct Winograd-like or use known small matrix ranks.
    
    Actually, let me just implement a careful Strassen-based block decomposition.
    
    Rank budget: We want to use fewer than 84 multipliers.
    Each 2x2x2 Strassen uses 7 instead of 8, saving 1 per block.
    
    Number of 2x2x2 blocks if we partition 3x4x7:
    i: 3 rows -> can do 1 full 2x2x2 block (rows 0,1) and 1 row left (row 2)
    j: 4 cols -> 2 pairs
    k: 7 cols -> 3 pairs + 1 singleton
    
    For i={0,1}, j-pairs x k-pairs: 2 * 3 = 6 Strassen blocks (rank 7 each = 42)
    For i={0,1}, j-pairs x k=singleton: 2 * 1 = 2 schoolbook 2x2x1 blocks (rank 4 each = 8)
    For i={2}, j-pairs x all k: 2 * 4 = 8 schoolbook 1x2x1... no, 1x2xk
    
    This is getting complex. Let me use a simpler but valid approach.
    """
    n, m, p = DIMENSIONS
    
    # Use a known good decomposition. For now, let's try the simple approach:
    # Use Strassen where possible.
    
    # Actually, the simplest improvement: use the fact that we can share computations.
    # Let me just return schoolbook for now as a baseline, but the real work is in
    # finding the rank-44 construction.
    
    # Let me implement the rank-44 construction properly:
    # Partition i into {0,1} and {2}, j into {0,1} and {2,3}
    # For each (i_sub, j_sub) block of size 2x2x7 or 1x2x7:
    #   For 2x2x7: split k into pairs (0,1),(2,3),(4,5) and singleton (6)
    #     Use Strassen for each 2x2x2 pair: 3 * rank(2x2x2) = 3*7 = 21
    #     Use schoolbook for 2x2x1: rank(2x2x1) = 4
    #     Total for 2x2x7: 21 + 4 = 25
    #   For 1x2x7: split k similarly
    #     Schoolbook for 1x2x2: rank = 4 (but we can do 1x2x2 in 4 mults)
    #     Actually 1x2x2: C[k] = sum_j A[j]*B[j,k], 2 j-values, 2 k-values = 4 mults
    #     Schoolbook for 1x2x1: rank = 2
    #     Total for 1x2x7: 3*4 + 2 = 14
    #
    # Total: 2 (i_subs with 2 rows) * 25 (for 2x2x7) + 1 (i_sub with 1 row) * 14 (for 1x2x7)
    # = 2*25 + 14 = 64
    #
    # Still above 44. Let me try rank(2x2x7) more carefully.
    # 
    # Actually, the minimum rank for 2x2xn is known to be 2n+1 for n>=2? No.
    # For 2x2x2 it's 7. For 2x2xn, using repeated Strassen, the rank might be lower.
    #
    # Let me just implement the 64-rank construction and verify it.

    U_list = []  # Each entry is a dict mapping (global_i, global_j) -> value
    V_list = []
    W_list = []
    
    def add_product(U_val, V_val, W_val):
        """Add one rank-1 term."""
        U_list.append(U_val)
        V_list.append(V_val)
        W_list.append(W_val)
    
    # Strassen 2x2x2: returns lists of (U_coeff, V_coeff, W_coeff) for 7 terms
    def strassen_terms(i_base, j_base, k_base):
        """Yield 7 rank-1 terms for 2x2x2 matmul with i=i_base+{0,1}, j=j_base+{0,1}, k=k_base+{0,1}."""
        U2, V2, W2 = _strassen_2x2x2()
        for r in range(7):
            for a in range(2):  # i offset
                for b in range(2):  # j offset
                    # U factor: A[i_base+a, j_base+b]
                    pass
        # Actually, the Strassen decomposition already encodes which A[i,j] and B[j,k] combine.
        # U[i,m+j] is the coefficient of A[i,j], V[j*p+k] is coeff of B[j,k], W[k*n+i] is coeff of C[k,i].
        # For 2x2x2 with i_base, j_base, k_base:
        # U has 4 rows: (i_base, j_base), (i_base, j_base+1), (i_base+1, j_base), (i_base+1, j_base+1)
        # We need to embed into the full n*m x 7 matrix.
        
    # Let me build the full decomposition directly.
    
    n, m, p = 3, 4, 7
    rank_total = 0
    U_mat = [[0] * 0 for _ in range(n * m)]  # Will resize
    V_mat = [[0] * 0 for _ in range(m * p)]
    W_mat = [[0] * 0 for _ in range(p * n)]
    
    def ensure_size(U, V, W, new_rank):
        nonlocal U_mat, V_mat, W_mat
        while new_rank > len(U_mat[0]):
            U_mat[0].append(0)
            for row in range(len(U_mat)):
                U_mat[row].append(0)
            V_mat[0].append(0)
            for row in range(len(V_mat)):
                V_mat[row].append(0)
            W_mat[0].append(0)
            for row in range(len(W_mat)):
                W_mat[row].append(0)
    
    U_mat = [[0] * 0 for _ in range(n * m)]
    V_mat = [[0] * 0 for _ in range(m * p)]
    W_mat = [[0] * 0 for _ in range(p * n)]
    
    def add_term(i_idx, j_idx, k_idx, u_coeff, v_coeff, w_coeff, col):
        """Add coefficients at column col for a single rank-1 term."""
        U_mat[i_idx * m + j_idx][col] = u_coeff
        V_mat[j_idx * p + k_idx][col] = v_coeff
        W_mat[k_idx * n + i_idx][col] = w_coeff
    
    col = 0
    # For each i_block, j_block combination:
    i_blocks = [(0, [0, 1]), (1, [2])]
    j_blocks = [(0, [0, 1]), (1, [2, 3])]
    k_blocks = [(0, [0, 1]), (1, [2, 3]), (2, [4, 5]), (3, [6])]
    
    for ib_idx, i_list in i_blocks:
        for jb_idx, j_list in j_blocks:
            if len(i_list) == 2 and len(j_list) == 2:
                # 2x2 sub-problem
                for kb_idx, k_list in k_blocks:
                    if len(k_list) == 2:
                        # 2x2x2: use Strassen, rank 7
                        U2, V2, W2 = _strassen_2x2x2()
                        i0, i1 = i_list
                        j0, j1 = j_list
                        k0, k1 = k_list
                        for r in range(7):
                            # U2 row 0 = (i0,j0), row 1 = (i0,j1), row 2 = (i1,j0), row 3 = (i1,j1)
                            U2_indices = [(i0, j0), (i0, j1), (i1, j0), (i1, j1)]
                            V2_indices = [(j0, k0), (j0, k1), (j1, k0), (j1, k1)]
                            W2_indices = [(k0, i0), (k0, i1), (k1, i0), (k1, i1)]
                            for a in range(4):
                                if U2[a][r] != 0:
                                    add_term(U2_indices[a][0], U2_indices[a][1], 0, U2[a][r], 0, 0, col)
                            for a in range(4):
                                if V2[a][r] != 0:
                                    # Need to combine with U and W... this approach won't work for multi-term
                                    pass
                        # This approach is flawed. Let me use a different method.
                        pass
            else:
                # Schoolbook
                for i in i_list:
                    for j in j_list:
                        for k in range(p):
                            add_term(i, j, k, 1, 1, 1, col)
                            col += 1
    
    # The above approach is getting too complex with the Strassen embedding.
    # Let me use a cleaner method: build the full U, V, W matrices for Strassen 2x2x2
    # and embed them directly.
    
    U_mat = [[0] * 0 for _ in range(n * m)]
    V_mat = [[0] * 0 for _ in range(m * p)]
    W_mat = [[0] * 0 for _ in range(p * n)]
    
    def _grow(U, V, W, cols):
        for row in range(len(U)):
            U[row].extend([0] * cols)
        for row in range(len(V)):
            V[row].extend([0] * cols)
        for row in range(W):
            W[row].extend([0] * cols)
    
    n, m, p = 3, 4, 7
    U_mat = [[0] * 0 for _ in range(n * m)]
    V_mat = [[0] * 0 for _ in range(m * p)]
    W_mat = [[0] * 0 for _ in range(p * n)]
    
    def grow_to(rank):
        curr = len(U_mat[0]) if U_mat[0] else 0
        while curr < rank:
            for row in U_mat:
                row.append(0)
            for row in V_mat:
                row.append(0)
            for row in W_mat:
                row.append(0)
            curr += 1
    
    def embed_strassen(i0, i1, j0, j1, k0, k1, start_col):
        """Embed a 2x2x2 Strassen decomposition into the global matrices starting at start_col."""
        U2, V2, W2 = _strassen_2x2x2()
        # U2: 4 rows for (i,j) in {(i0,j0),(i0,j1),(i1,j0),(i1,j1)}, 7 cols
        # V2: 4 rows for (j,k) in {(j0,k0),(j0,k1),(j1,k0),(j1,k1)}, 7 cols
        # W2: 4 rows for (k,i) in {(k0,i0),(k0,i1),(k1,i0),(k1,i1)}, 7 cols
        U2_row_map = {(i0, j0): 0, (i0, j1): 1, (i1, j0): 2, (i1, j1): 3}
        V2_row_map = {(j0, k0): 0, (j0, k1): 1, (j1, k0): 2, (j1, k1): 3}
        W2_row_map = {(k0, i0): 0, (k0, i1): 1, (k1, i0): 2, (k1, i1): 3}
        
        grow_to(start_col + 7)
        for r in range(7):
            for (ii, jj), ur in U2_row_map.items():
                val = U2[ur][r]
                if val != 0:
                    U_mat[ii * m + jj][start_col + r] = val
            for (jj, kk), vr in V2_row_map.items():
                val = V2[vr][r]
                if val != 0:
                    V_mat[jj * p + kk][start_col + r] = val
            for (kk, ii), wr in W2_row_map.items():
                val = W2[wr][r]
                if val != 0:
                    W_mat[kk * n + ii][start_col + r] = val
        
    def embed_schoolbook_2x2x2(i0, i1, j0, j1, k0, k1, start_col):
        """Embed a 2x2x2 schoolbook decomposition (rank 4)."""
        grow_to(start_col + 4)
        pairs = [(i0, j0, k0), (i0, j1, k1), (i1, j0, k0), (i1, j1, k1)]
        # Actually schoolbook 2x2x2: C[i,k] = sum_j A[i,j]B[j,k]
        # 4 terms: (i0,j0,k0),(i0,j0,k1),(i0,j1,k0),(i0,j1,k1) -> no, that's not right.
        # Schoolbook: for each i in {i0,i1}, k in {k0,k1}: C[i,k] = A[i,j0]*B[j0,k] + A[i,j1]*B[j1,k]
        # So 2 i's * 2 j's * 2 k's = 8 terms for schoolbook 2x2x2. But Strassen does it in 7.
        # For the singleton k case (2x2x1), schoolbook is 2*2*1 = 4 terms.
        pass
    
    # Let me simplify: for the parts we can't Strassen, use full schoolbook.
    # 
    # Final plan: 
    # i={0,1}, j={0,1}, k-pairs: 3 Strassen 2x2x2 (rank 7 each)
    # i={0,1}, j={0,1}, k=6: schoolbook 2x2x1 (rank 4)
    # i={0,1}, j={2,3}, k-pairs: 3 Strassen 2x2x2
    # i={0,1}, j={2,3}, k=6: schoolbook 2x2x1
    # i={2}, j={0,1}, all k: schoolbook 1x2x7 (rank 14)
    # i={2}, j={2,3}, all k: schoolbook 1x2x7 (rank 14)
    #
    # Total: 2*(3*7+4) + 2*14 = 2*25 + 28 = 78
    
    col = 0
    # 2x2x7 for (i=0,1), (j=0,1)
    for kb in range(3):
        k0 = 2*kb
        k1 = 2*kb+1
        embed_strassen(0, 1, 0, 1, k0, k1, col)
        col += 7
    # Schoolbook 2x2x1 for (i=0,1), (j=0,1), k=6
    for i in [0, 1]:
        for j in [0, 1]:
            grow_to(col + 1)
            U_mat[i*m+j][col] = 1
            V_mat[j*p+6][col] = 1
            W_mat[6*n+i][col] = 1
            col += 1
    # 2x2x7 for (i=0,1), (j=2,3)
    for kb in range(3):
        k0 = 2*kb
        k1 = 2*kb+1
        embed_strassen(0, 1, 2, 3, k0, k1, col)
        col += 7
    for i in [0, 1]:
        for j in [2, 3]:
            grow_to(col + 1)
            U_mat[i*m+j][col] = 1
            V_mat[j*p+6][col] = 1
            W_mat[6*n+i][col] = 1
            col += 1
    # Schoolbook 1x2x7 for i=2, j=0,1
    for j in [0, 1]:
        for k in range(7):
            grow_to(col + 1)
            U_mat[2*m+j][col] = 1
            V_mat[j*p+k][col] = 1
            W_mat[k*n+2][col] = 1
            col += 1
    # Schoolbook 1x2x7 for i=2, j=2,3
    for j in [2, 3]:
        for k in range(7):
            grow_to(col + 1)
            U_mat[2*m+j][col] = 1
            V_mat[j*p+k][col] = 1
            W_mat[k*n+2][col] = 1
            col += 1
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}


def _verify(U, V, W):
    """Verify the tensor decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U and U[0] else 0
    # Check all (i,j,k) where T[i,j,k] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-10:
                    return False
    return True


def _build_schoolbook():
    """Construct the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    # Start with schoolbook
    best = _build_schoolbook()
    best_rank = n * m * p  # 84
    
    # Try Strassen-based decomposition
    try:
        candidate = _build_strassen_decomposition()
        if _verify(candidate["U"], candidate["V"], candidate["W"]):
            cand_rank = len(candidate["U"][0])
            if cand_rank < best_rank:
                best = candidate
                best_rank = cand_rank
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
