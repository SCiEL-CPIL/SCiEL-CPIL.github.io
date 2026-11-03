import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Strassen 2x2x2 matrix multiplication decomposition.
    Returns U (4x7), V (4x7), W (4x7) for multiplying 2x2 by 2x2.
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # U rows indexed by (i,j) for A[i,j], V rows by (j,k) for B[j,k], W rows by (k,i) for C[k,i]
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C indices: (0,0), (0,1), (1,0), (1,1) -> W[k*2+i]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    W[0, 0] = 1  # C[0,0]
    W[3, 0] = 1  # C[1,1]
    
    # M2 = (c+d)e
    U[2, 1] = 1  # c
    U[3, 1] = 1  # d
    V[0, 1] = 1  # e
    W[2, 1] = 1  # C[1,0]
    
    # M3 = a(f-h)
    U[0, 2] = 1  # a
    V[1, 2] = 1  # f
    V[3, 2] = -1  # h
    W[1, 2] = 1  # C[0,1]
    W[3, 2] = 1  # C[1,1]
    
    # M4 = d(g-e)
    U[3, 3] = 1  # d
    V[2, 3] = 1  # g
    V[0, 3] = -1  # e
    W[0, 3] = 1  # C[0,0]
    W[2, 3] = 1  # C[1,0]
    
    # M5 = (a+b)h
    U[0, 4] = 1  # a
    U[1, 4] = 1  # b
    V[3, 4] = 1  # h
    W[0, 4] = -1  # C[0,0]
    W[1, 4] = 1  # C[0,1]
    
    # M6 = (c-a)(e+f)
    U[2, 5] = 1  # c
    U[0, 5] = -1  # a
    V[0, 5] = 1  # e
    V[1, 5] = 1  # f
    W[3, 5] = 1  # C[1,1]
    
    # M7 = (b-d)(g+h)
    U[1, 6] = 1  # b
    U[3, 6] = -1  # d
    V[2, 6] = 1  # g
    V[3, 6] = 1  # h
    W[0, 6] = 1  # C[0,0]
    
    return U, V, W


def _schoolbook_nm_p(n, m, p):
    """Schoolbook decomposition for n x m times m x p."""
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    R = U.shape[1]
    U_3d = U.T.reshape(n, m, R)
    V_3d = V.T.reshape(m, p, R)
    W_3d = W.T.reshape(p, n, R)
    tensor = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    T = np.ones((n, m, p), dtype=np.float64)
    error = np.linalg.norm(tensor - T)
    return error < 1e-10, error


def _combine_decompositions(decomps, n, m, p):
    """
    Combine multiple independent sub-decompositions into one.
    Each decomp is (U_sub, V_sub, W_sub) covering a subset of indices.
    We need to map sub-indices to global indices.
    
    For now, handle the case where we split the middle dimension m into parts.
    """
    # This is a placeholder; the actual combination depends on how we partition.
    # For splitting m into m1 + m2:
    # U has rows for (i, j) where j goes 0..m-1
    # V has rows for (j, k) where j goes 0..m-1
    # W has rows for (k, i)
    
    # Actually, let's just concatenate if the sub-decompositions are independent
    # and cover disjoint sets of (i,j,k) triples.
    
    total_rank = sum(d[0].shape[1] for d in decomps)
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    for U_sub, V_sub, W_sub in decomps:
        R_sub = U_sub.shape[1]
        U_full[:, col_offset:col_offset + R_sub] = U_sub
        V_full[:, col_offset:col_offset + R_sub] = V_sub
        W_full[:, col_offset:col_offset + R_sub] = W_sub
        col_offset += R_sub
    
    return U_full, V_full, W_full


def _build_4x5_5x6_hybrid():
    """
    Build a hybrid decomposition for 4x5 times 5x6.
    
    Strategy: Split the 5 middle dimension into 2 + 3.
    - Use Strassen-style for the 4x2 part (split 4 into 2+2, use 2x2 Strassen blocks)
    - Use schoolbook for the 4x3 part
    
    Actually, let's try a different approach:
    Split the output 4x6 into four 2x3 blocks.
    For each 2x3 block, we need to compute A_2x5 * B_5x3.
    We can split the 5 into 2+3.
    
    Let me try: split m=5 into m1=2 and m2=3.
    A = [A1 (4x2) | A2 (4x3)]
    B = [B1 (2x6); B2 (3x6)]
    C = A1*B1 + A2*B2
    
    For A1*B1: 4x2 times 2x6. Split 4 into 2+2.
    Each 2x2 times 2x6 can use... hmm, this gets complex.
    
    Simpler approach: Use Strassen for 2x2x2 blocks where possible.
    
    Let me try splitting differently:
    - Split n=4 into 2+2
    - Split m=5 into 2+3  
    - Split p=6 into 3+3
    
    Block (i in {0,1}, j in {0,1}, k in {0,1,2}): 2x2x3 -> schoolbook rank 12
    Block (i in {0,1}, j in {2,3,4}, k in {0,1,2}): 2x3x3 -> schoolbook rank 18
    Block (i in {2,3}, j in {0,1}, k in {0,1,2}): 2x2x3 -> schoolbook rank 12
    Block (i in {2,3}, j in {2,3,4}, k in {0,1,2}): 2x3x3 -> schoolbook rank 18
    Total schoolbook: 60
    
    That's not better. Let me think about using Strassen more cleverly.
    
    For 2x2x2 blocks, Strassen gives rank 7 vs schoolbook 8.
    
    Let me try: partition into 2x2x2 blocks where possible.
    n=4=2+2, m=5=2+2+1, p=6=2+2+2
    
    This is getting complicated. Let me just use a known good hybrid.
    """
    
    n, m, p = 4, 5, 6
    
    # Approach: Use the fact that we can embed smaller Strassen decompositions.
    # Split m=5 into 2+3.
    # Part 1: A[:, :2] @ B[:2, :] -> 4x2 times 2x6
    # Part 2: A[:, 2:] @ B[2:, :] -> 4x3 times 3x6
    
    # For Part 1 (4x2 x 2x6):
    # Split n=4 into 2+2, p=6 into 3+3
    # Block (2x2 x 2x3) for k in 0..2: 2x2 times 2x3
    #   Use Strassen on the 2x2 part: rank 7 per 2x3 block? No, Strassen is for square.
    #   For 2x2 times 2x3: schoolbook is 12. 
    #   We can use 2x2 Strassen (rank 7) for the 2x2 part and multiply by 3rd column.
    #   Actually, 2x2 times 2x3 = (2x2 times 2x1) + (2x2 times 2x1) + (2x2 times 2x1)
    #   Each 2x2 times 2x1 has rank... let me think.
    #   2x2 times 2x1: C = A * b where A is 2x2, b is 2x1.
    #   C[0,0] = a*b[0] + c*b[1], C[1,0] = b*a[0] + d*b[1]
    #   This is bilinear in (a,b,c,d) and (b0,b1). Rank = 4 (schoolbook).
    #   Can we do better? Probably not for 2x2x1.
    #   So 2x2 times 2x3 has rank 4*3 = 12.
    
    # This approach isn't giving savings. Let me try a different strategy.
    
    # Strategy: Use the known result that for rectangular matrices,
    # we can sometimes do better than n*m*p by using Strassen on square subblocks.
    
    # Let me try: pad 4x5 x 5x6 to 8x8 x 8x8 and use Strassen recursively?
    # That would be 8^3 = 512 schoolbook, or with Strassen ~8^(log2(7)) ~ 298.
    # That's worse.
    
    # Better: Use the decomposition from the inspiration programs.
    # They achieved rank 101 by replacing 4x4x2 schoolbook (32) with 4 Strassen 2x2x2 (28).
    
    # Let me replicate that:
    # Split p=6 into 2+4.
    # Part A: 4x5 times 5x2 (rank 40 schoolbook)
    # Part B: 4x5 times 5x4 (rank 80 schoolbook)
    # Total: 120
    
    # For Part B (4x5 x 5x4): Split n=4 into 2+2, p=4 into 2+2.
    # Four 2x5 x 5x2 blocks. Each has rank 20.
    # For each 2x5 x 5x2: split m=5 into 2+3.
    #   2x2 x 2x2: Strassen rank 7
    #   2x3 x 3x2: schoolbook rank 12
    #   Total per block: 19
    # Four blocks: 76
    # Part A (4x5 x 5x2): Split n=4 into 2+2.
    #   Two 2x5 x 5x2 blocks.
    #   Each: split m=5 into 2+3.
    #     2x2 x 2x2: Strassen rank 7
    #     2x3 x 3x2: schoolbook rank 12
    #     Total per block: 19
    #   Two blocks: 38
    # Total: 76 + 38 = 114. Still not great.
    
    # Let me just build the known-good decomposition from inspirations.
    # They got 101. Let me try to get at least that.
    
    # Actually, let me try a cleaner approach:
    # Split m=5 into 2+3.
    # Term 1: A[:,:2] @ B[:2,:] -> 4x2 @ 2x6, schoolbook rank 48
    # Term 2: A[:,2:] @ B[2:,:] -> 4x3 @ 3x6, schoolbook rank 72
    # Total: 120. No improvement.
    
    # The key insight from inspirations: use Strassen on 2x2x2 subblocks.
    # Let me partition:
    # n=4: rows 0,1 and 2,3
    # m=5: cols 0,1 and 2,3,4  
    # p=6: cols 0,1,2,3,4,5
    
    # For the (n=0,1, m=0,1) subblock: 2x2 matrix A_top_left
    # It multiplies with B[0,1,:] which is 2x6.
    # Result is 2x6, contributing to C[0,1,:].
    # We can split p=6 into three 2-column groups.
    # For each 2-column group: 2x2 times 2x2 -> Strassen rank 7.
    # Three groups: 21.
    
    # For the (n=0,1, m=2,3,4) subblock: 2x3 matrix A_top_right
    # It multiplies with B[2,3,4,:] which is 3x6.
    # Result is 2x6.
    # Split p=6 into three 2-column groups.
    # For each 2-column group: 2x3 times 3x2 -> schoolbook rank 12.
    # Three groups: 36.
    
    # Similarly for n=2,3:
    # (n=2,3, m=0,1): 2x2 times 2x6 -> 3 * 7 = 21
    # (n=2,3, m=2,3,4): 2x3 times 3x6 -> 3 * 12 = 36
    
    # Total: 21 + 36 + 21 + 36 = 114.
    
    # Hmm, still not 101. Let me check the inspiration more carefully.
    # They said: "Replacing the 4x4x2 schoolbook block (rank 32) with 4 Strassen 2x2x2 blocks (rank 28)"
    # So they had a 4x4x2 block. That suggests splitting p=6 into 2+4.
    # The 4x4x2 part: n=4, m=4, p=2. Schoolbook rank 32.
    # They replaced with 4 Strassen 2x2x2 blocks, each rank 7, total 28.
    # Savings: 4.
    
    # So the decomposition is:
    # Split p=6 into p1=2 and p2=4.
    # Part 1: 4x5 times 5x2. Split m=5 into 4+1? Or n=4 into 2+2?
    #   If we use the 4x4x2 Strassen replacement, we need m=4.
    #   So split m=5 into m1=4 and m2=1.
    #   Part 1a: 4x4 times 4x2 -> use 4 Strassen 2x2x2 blocks, rank 28
    #   Part 1b: 4x1 times 1x2 -> schoolbook rank 8
    #   Part 1 total: 36
    # Part 2: 4x5 times 5x4. 
    #   Split n=4 into 2+2, p=4 into 2+2.
    #   Four 2x5 times 5x2 blocks.
    #   Each 2x5 times 5x2: split m=5 into 4+1.
    #     2x4 times 4x2: use 2 Strassen 2x2x2 blocks? 
    #       2x4: n=2, m=4. Split m=4 into 2+2.
    #       Two 2x2 times 2x2 blocks: 2 * 7 = 14.
    #     2x1 times 1x2: schoolbook rank 4.
    #     Total per 2x5x5x2 block: 18.
    #   Four blocks: 72.
    #   Part 2 total: 72.
    # Grand total: 36 + 72 = 108.
    
    # Still not 101. Let me just implement a reasonable hybrid and see.
    
    # For now, let me build a decomposition that uses Strassen where beneficial
    # and falls back to schoolbook otherwise. I'll aim for rank ~105-110.
    
    # Simplest improvement: identify all 2x2x2 subblocks and use Strassen.
    
    # Let me build the full decomposition by partitioning into 2x2x2 cells
    # where possible, and handling the remainder with schoolbook.
    
    # n=4: [0,1], [2,3]
    # m=5: [0,1], [2,3], [4]
    # p=6: [0,1], [2,3], [4,5]
    
    # Cells:
    # (n=0,1, m=0,1, p=0,1): 2x2x2 -> Strassen 7
    # (n=0,1, m=0,1, p=2,3): 2x2x2 -> Strassen 7
    # (n=0,1, m=0,1, p=4,5): 2x2x2 -> Strassen 7
    # (n=0,1, m=2,3, p=0,1): 2x2x2 -> Strassen 7
    # (n=0,1, m=2,3, p=2,3): 2x2x2 -> Strassen 7
    # (n=0,1, m=2,3, p=4,5): 2x2x2 -> Strassen 7
    # (n=0,1, m=4,    p=0,1): 2x1x2 -> schoolbook 4
    # (n=0,1, m=4,    p=2,3): 2x1x2 -> schoolbook 4
    # (n=0,1, m=4,    p=4,5): 2x1x2 -> schoolbook 4
    # (n=2,3, m=0,1, p=0,1): 2x2x2 -> Strassen 7
    # (n=2,3, m=0,1, p=2,3): 2x2x2 -> Strassen 7
    # (n=2,3, m=0,1, p=4,5): 2x2x2 -> Strassen 7
    # (n=2,3, m=2,3, p=0,1): 2x2x2 -> Strassen 7
    # (n=2,3, m=2,3, p=2,3): 2x2x2 -> Strassen 7
    # (n=2,3, m=2,3, p=4,5): 2x2x2 -> Strassen 7
    # (n=2,3, m=4,    p=0,1): 2x1x2 -> schoolbook 4
    # (n=2,3, m=4,    p=2,3): 2x1x2 -> schoolbook 4
    # (n=2,3, m=4,    p=4,5): 2x1x2 -> schoolbook 4
    
    # Total: 12 * 7 + 6 * 4 = 84 + 24 = 108.
    
    # That's rank 108, better than 120. Score = 120/108 = 1.111.
    # The inspirations got 101, so there's more to be done, but this is a start.
    
    # Let me implement this.
    
    n, m, p = 4, 5, 6
    U = np.zeros((n * m, 0))
    V = np.zeros((m * p, 0))
    W = np.zeros((p * n, 0))
    
    def add_decomp(U_sub, V_sub, W_sub, i_offset, j_offset, k_offset):
        """Add a sub-decomposition at given offsets."""
        nonlocal U, V, W
        R_sub = U_sub.shape[1]
        if R_sub == 0:
            return
        
        # Create new full matrices with additional columns
        new_U = np.zeros((n * m, U.shape[1] + R_sub))
        new_V = np.zeros((m * p, V.shape[1] + R_sub))
        new_W = np.zeros((p * n, W.shape[1] + R_sub))
        
        # Copy existing
        if U.shape[1] > 0:
            new_U[:, :U.shape[1]] = U
            new_V[:, :V.shape[1]] = V
            new_W[:, :W.shape[1]] = W
        
        # Add sub-decomp with offset
        for r in range(R_sub):
            # U_sub has rows for (i,j) in local coords
            for li in range(U_sub.shape[0] // (m - j_offset + 1) if False else 2):
                pass  # This is getting complicated. Let me use a different approach.
        
        U, V, W = new_U, new_V, new_W
    
    # This offset approach is messy. Let me build it differently.
    # I'll create the full matrices and fill in the appropriate entries.
    
    total_rank = 108  # 12*7 + 6*4
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Define the partition
    n_parts = [(0, 2), (2, 4)]
    m_parts = [(0, 2), (2, 4), (4, 5)]
    p_parts = [(0, 2), (2, 4), (4, 6)]
    
    for ni, (i1, i2) in enumerate(n_parts):
        for mi, (j1, j2) in enumerate(m_parts):
            for pi, (k1, k2) in enumerate(p_parts):
                ni_size = i2 - i1
                mi_size = j2 - j1
                pi_size = k2 - k1
                
                if ni_size == 2 and mi_size == 2 and pi_size == 2:
                    # Use Strassen 2x2x2
                    U_s, V_s, W_s = _strassen_2x2()
                    for r in range(7):
                        for li in range(2):
                            for lj in range(2):
                                U[(i1 + li) * m + (j1 + lj), col] = U_s[li * 2 + lj, r]
                        for lj in range(2):
                            for lk in range(2):
                                V[(j1 + lj) * p + (k1 + lk), col] = V_s[lj * 2 + lk, r]
                        for lk in range(2):
                            for li in range(2):
                                W[(k1 + lk) * n + (i1 + li), col] = W_s[lk * 2 + li, r]
                        col += 1
                elif ni_size == 2 and mi_size == 1 and pi_size == 2:
                    # 2x1x2 schoolbook
                    for li in range(2):
                        for lk in range(2):
                            U[(i1 + li) * m + j1, col] = 1
                            V[j1 * p + (k1 + lk), col] = 1
                            W[(k1 + lk) * n + (i1 + li), col] = 1
                            col += 1
                else:
                    # General schoolbook
                    for li in range(ni_size):
                        for lj in range(mi_size):
                            for lk in range(pi_size):
                                U[(i1 + li) * m + (j1 + lj), col] = 1
                                V[(j1 + lj) * p + (k1 + lk), col] = 1
                                W[(k1 + lk) * n + (i1 + li), col] = 1
                                col += 1
    
    # Trim if col < total_rank
    if col < total_rank:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
    
    # Verify
    valid, error = _verify_decomp(U, V, W, n, m, p)
    if not valid:
        # Fall back to schoolbook
        U, V, W = _schoolbook_nm_p(n, m, p)
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = 140  # 10s safety margin before 150s limit
    
    # Build the hybrid decomposition
    U, V, W = _build_4x5_5x6_hybrid()
    
    rank = U.shape[1]
    
    # Try to remove linearly dependent columns
    # Convert to exact fractions for better precision
    U_frac = np.array([[Fraction(int(x)) for x in row] for row in U])
    V_frac = np.array([[Fraction(int(x)) for x in row] for row in V])
    W_frac = np.array([[Fraction(int(x)) for x in row] for row in W])
    
    # Try removing columns one by one
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = rank
    
    for attempt in range(3):  # Multiple passes
        removed_any = False
        for c in range(current_rank - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing column c
            keep = [i for i in range(current_rank) if i != c]
            U_try = current_U[:, keep]
            V_try = current_V[:, keep]
            W_try = current_W[:, keep]
            
            valid, error = _verify_decomp(U_try, V_try, W_try, n, m, p)
            if valid:
                current_U, current_V, current_W = U_try, V_try, W_try
                current_rank -= 1
                removed_any = True
                break  # Restart from the end
        
        if not removed_any:
            break
    
    # Convert to float and verify
    U_final = current_U.astype(np.float64)
    V_final = current_V.astype(np.float64)
    W_final = current_W.astype(np.float64)
    
    valid, error = _verify_decomp(U_final, V_final, W_final, n, m, p)
    if not valid:
        # Fall back to the hybrid without column removal
        U_final, V_final, W_final = U, V, W
        valid, error = _verify_decomp(U_final, V_final, W_final, n, m, p)
    
    if not valid:
        # Last resort: schoolbook
        U_final, V_final, W_final = _schoolbook_nm_p(n, m, p)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

# EVOLVE-BLOCK-END
