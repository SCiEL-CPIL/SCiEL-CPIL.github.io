import numpy as np
import time
from fractions import Fraction
from itertools import product

def construct():
    start = time.time()
    n, m, p = 3, 4, 7
    
    def elapsed():
        return time.time() - start
    
    # Schoolbook: rank 84
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        for r in range(rank):
            i = r // (m * p)
            j = (r // p) % m
            k = r % p
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    def matmul_decompose(A_dims, B_dims):
        """Decompose A(A_dims[0] x A_dims[1]) * B(B_dims[1] x B_dims[2]).
        Returns (U_local, V_local, W_local, rank) where U_local has shape (A_dims[0]*A_dims[1], rank),
        V_local has shape (B_dims[1]*B_dims[2], rank), W_local has shape (B_dims[2]*A_dims[0], rank).
        """
        n_a, m_a = A_dims
        m_b, p_b = B_dims
        assert m_a == m_b, f"Inner dims mismatch: {m_a} vs {m_b}"
        
        # Check if Strassen applies: need both matrices splittable into 2x2 blocks of 2x2
        # i.e., n_a, m_a, p_b all even and >= 2
        if n_a >= 2 and m_a >= 2 and p_b >= 2 and n_a % 2 == 0 and m_a % 2 == 0 and p_b % 2 == 0:
            # Try Strassen
            n2 = n_a // 2
            m2 = m_a // 2
            p2 = p_b // 2
            s = _strassen_2x2(n2, m2, p2)
            if s is not None:
                return s
        
        # Schoolbook
        return schoolbook_nmp(n_a, m_a, p_b)
    
    def schoolbook_nmp(n_, m_, p_):
        rank = n_ * m_ * p_
        U = np.zeros((n_ * m_, rank), dtype=np.float64)
        V = np.zeros((m_ * p_, rank), dtype=np.float64)
        W = np.zeros((p_ * n_, rank), dtype=np.float64)
        for r in range(rank):
            i = r // (m_ * p_)
            j = (r // p_) % m_
            k = r % p_
            U[i * m_ + j, r] = 1.0
            V[j * p_ + k, r] = 1.0
            W[k * n_ + i, r] = 1.0
        return U, V, W, rank
    
    def _strassen_2x2(n2, m2, p2):
        """
        Full 2x2 Strassen for (2*n2 x 2*m2) * (2*m2 x 2*p2).
        Splits into 4 blocks:
          A = [[A11, A12], [A21, A22]]  each (n2 x m2)
          B = [[B11, B12], [B21, B22]]  each (m2 x p2)
        Result:
          C11 = P1 - P5 + P6 - P7
          C12 = P4 + P5
          C21 = P3 + P6
          C22 = P1 - P2 + P3 + P4
        where:
          P1 = (A11+A22)(B11+B22)
          P2 = (A21+A22) B11
          P3 = A11 (B12-B22)
          P4 = A22 (B21-B11)
          P5 = (A11+A12) B22
          P6 = (A21-A11)(B11+B12)
          P7 = (A12-A22)(B21+B22)
        """
        # For each P term, we need the decomposition of the relevant sub-multiplication
        # P1: (n2+m2) effectively... no, it's (A11+A22) which is (n2 x m2) times (B11+B22) which is (m2 x p2)
        # So each P term is a (n2 x m2) * (m2 x p2) multiplication
        
        # Decomposition of each (n2 x m2)*(m2 x p2):
        sub_decomp = []
        for name in range(7):
            U_s, V_s, W_s, r_s = schoolbook_nmp(n2, m2, p2)
            sub_decomp.append((U_s, V_s, W_s, r_s))
        
        # Total rank = 7 * (n2*m2*p2) ... but actually we can be smarter
        # For now, use schoolbook for each sub-multiplication
        # Actually, let's use schoolbook for the sub-multiplications since n2*m2*p2 is already small
        
        total_rank = sum(r_s for _, _, _, r_s in sub_decomp)
        U_all = np.zeros((2 * n2 * 2 * m2, total_rank))
        V_all = np.zeros((2 * m2 * 2 * p2, total_rank))
        W_all = np.zeros((2 * p2 * 2 * n2, total_rank))
        
        col = 0
        # P1 = (A11+A22)(B11+B22): contributes to C11(+1), C22(+1)
        # U factors: A11 gets +1, A22 gets +1
        # V factors: B11 gets +1, B22 gets +1
        # W: C11 gets +1, C22 gets +1
        for name, (U_s, V_s, W_s, r_s) in enumerate(sub_decomp):
            if name == 0:  # P1
                U_contribs = {0: 1.0, 2: 1.0}  # A11, A22
                V_contribs = {0: 1.0, 2: 1.0}  # B11, B22
                W_contribs = {0: 1.0, 3: 1.0}  # C11, C22
            elif name == 1:  # P2 = (A21+A22)B11
                U_contribs = {2: 1.0, 3: 1.0}
                V_contribs = {0: 1.0}
                W_contribs = {3: -1.0}  # C22: -P2
            elif name == 2:  # P3 = A11(B12-B22)
                U_contribs = {0: 1.0}
                V_contribs = {1: 1.0, 2: -1.0}
                W_contribs = {2: 1.0, 3: 1.0}  # C21: +P3, C22: +P3
            elif name == 3:  # P4 = A22(B21-B11)
                U_contribs = {3: 1.0}
                V_contribs = {2: 1.0, 0: -1.0}
                W_contribs = {1: 1.0, 3: 1.0}  # C12: +P4, C22: +P4
            elif name == 4:  # P5 = (A11+A12)B22
                U_contribs = {0: 1.0, 1: 1.0}
                V_contribs = {2: 1.0}
                W_contribs = {0: -1.0, 1: 1.0}  # C11: -P5, C12: +P5
            elif name == 5:  # P6 = (A21-A11)(B11+B12)
                U_contribs = {2: 1.0, 0: -1.0}
                V_contribs = {0: 1.0, 1: 1.0}
                W_contribs = {0: 1.0, 2: 1.0}  # C11: +P6, C21: +P6
            elif name == 6:  # P7 = (A12-A22)(B21+B22)
                U_contribs = {1: 1.0, 3: -1.0}
                V_contribs = {2: 1.0, 3: 1.0}
                W_contribs = {0: -1.0}  # C11: -P7
            
            for c in range(r_s):
                base_col = col + c
                # U: for each block contribution
                for block_idx, coeff in U_contribs.items():
                    row_off = block_idx * n2 * m2 + c
                    U_all[row_off, base_col] = coeff * U_s[c, 0] if r_s == 1 else 0
                    # Actually U_s has shape (n2*m2, r_s), so U_s[row, c]
                # Let me redo this more carefully
                for row in range(n2 * m2):
                    # This row corresponds to which block?
                    # Row r in the local U_s maps to A_{i,j} where i = r // m2, j = r % m2
                    # In the global A, block 0 = rows [0:n2], cols [0:m2]
                    # block 1 = rows [0:n2], cols [m2:2*m2]
                    # block 2 = rows [n2:2*n2], cols [0:m2]
                    # block 3 = rows [n2:2*n2], cols [m2:2*m2]
                    i_local = row // m2
                    j_local = row % m2
                    for block_idx, coeff in U_contribs.items():
                        gi = (block_idx // 2) * n2 + i_local
                        gj = (block_idx % 2) * m2 + j_local
                        U_all[gi * 2 * m2 + gj, base_col] = coeff * U_s[row, c]
                
                for row in range(m2 * p2):
                    j_local = row // p2
                    k_local = row % p2
                    for block_idx, coeff in V_contribs.items():
                        gj = (block_idx // 2) * m2 + j_local
                        gk = (block_idx % 2) * p2 + k_local
                        V_all[gj * 2 * p2 + gk, base_col] = coeff * V_s[row, c]
                
                for row in range(p2 * n2):
                    k_local = row // n2
                    i_local = row % n2
                    for block_idx, coeff in W_contribs.items():
                        gk = (block_idx // 2) * p2 + k_local
                        gi = (block_idx % 2) * n2 + i_local
                        W_all[gk * 2 * n2 + gi, base_col] = coeff * W_s[row, c]
            
            col += r_s
        
        return U_all, V_all, W_all, col
    
    # Strategy: Split 3x4 by 4 into 2+2
    # A is 3x4, B is 4x7
    # Split B into B_left (4x4) and B_right (4x3)
    # A * B_left: 3x4 * 4x4
    # A * B_right: 3x4 * 4x3
    # For 3x4*4x4: split A into A_top (2x4) + A_bottom (1x4)
    #   A_top * B_left: 2x4 * 4x4 -> can use Strassen (2x2 of 2x2)
    #   A_bottom * B_left: 1x4 * 4x4 -> schoolbook, rank 16
    # For 3x4*4x3: split A into A_top (2x4) + A_bottom (1x4)
    #   A_top * B_right: 2x4 * 4x3 -> Strassen? 2x4 is (2x2)(2x2), 4x3 is (2x2)(2x2)+(1x2)... no
    #   Actually 2x4 * 4x3: can split into (2x2)*(2x2) + (2x2)*(2x1) + ... 
    #   Let me just use schoolbook for the odd-sized ones.
    
    # Simpler: try various block splits and use Strassen where possible
    
    # Option 1: Split into (2x4)(4x4) + (1x4)(4x4) + (2x4)(4x3) + (1x4)(4x3)
    # (2x4)(4x4): Strassen -> 7 * (2*2*2) = 56... no, Strassen on 2x4*4x4:
    #   blocks are 2x2, so 7 * (2*2*2) = 56 rank terms
    # (1x4)(4x4): schoolbook = 1*4*4 = 16
    # (2x4)(4x3): not directly Strassen (3 is odd). Schoolbook = 2*4*3 = 24
    # (1x4)(4x3): schoolbook = 1*4*3 = 12
    # Total: 56 + 16 + 24 + 12 = 108. That's worse than 84!
    
    # Option 2: Just use Strassen on a 2x2x2 sub-block within the 3x4x7
    # Split: A = [A_2x4; A_1x4], B = [B_4x4 | B_4x3]
    # A_2x4 * B_4x4: use Strassen -> this is (2x4)*(4x4)
    #   Sub-blocks: A_top-left (2x2), A_top-right (2x2), etc.
    #   Strassen rank for 2x2*2x2 sub: 7 each, 4 sub-multiplications... 
    #   Actually (2x4)*(4x4) with Strassen:
    #     Split into 2x2 blocks: A has blocks [[A00,A01],[A10,A11]] each 2x2? No...
    #     A is 2x4, so split into [A_left(2x2), A_right(2x2)]
    #     B is 4x4, so split into [[B00,B01],[B10,B11]] each 2x2
    #     This is NOT a standard 2x2x2 Strassen. Standard Strassen needs both to be 2xN where N is even.
    #     2x4 * 4x4: split A into (2x2)+(2x2), B into 2x2 blocks.
    #     C_left = A_left * B00 + A_right * B10
    #     C_right = A_left * B01 + A_right * B11
    #     So 4 sub-multiplications of 2x2 * 2x2, each using Strassen (rank 7).
    #     Total: 4 * 7 = 28 rank terms for this block.
    #   vs schoolbook: 2*4*4 = 32. So saves 4.
    # 
    # A_1x4 * B_4x4: 1x4 * 4x4, schoolbook = 16 (can't really reduce)
    # A_2x4 * B_4x3: 2x4 * 4x3
    #   Split: C_left(2x2) = A_left(2x2)*B00(2x2) + A_right(2x2)*B10(2x2) -- wait B is 4x3
    #   B_4x3: split into B_top(2x3), B_bottom(2x3)
    #   C = A_left(2x2)*B_top(2x3) + A_right(2x2)*B_bottom(2x3)
    #   Each 2x2*2x3: schoolbook = 12, or can we do better? Not with standard methods.
    #   Total: 2*12 = 24. Same as schoolbook.
    # A_1x4 * B_4x3: 1x4 * 4x3, schoolbook = 12.
    # 
    # Total: 28 + 16 + 24 + 12 = 80. That's better than 84!
    
    # Let me implement this carefully.
    # A is 3x4, indexed A[i,j] for i in [0,3), j in [0,4)
    # B is 4x7, indexed B[j,k] for j in [0,4), k in [0,7)
    # C = A*B, C[i,k] for i in [0,3), k in [0,7)
    # 
    # Split A: A_top = A[0:2,:], A_bot = A[2:3,:]
    # Split B: B_L = B[:,0:4], B_R = B[:,4:7]
    # C_L = A_top @ B_L (2x4) and A_bot @ B_L (1x4) -> C[0:2,0:4] and C[2,0:4]
    # C_R = A_top @ B_R (2x4) and A_bot @ B_R (1x4) -> C[0:2,4:7] and C[2,4:7]
    # 
    # For C_top_L = A_top(2x4) @ B_L(4x4):
    #   Split A_top into AL(2x2) = A_top[:,0:2], AR(2x2) = A_top[:,2:4]
    #   Split B_L into [[B00,B01],[B10,B11]] each 2x2
    #   C_TL = AL @ B00 + AR @ B10  (2x2 result, cols 0-1)
    #   C_TR = AL @ B01 + AR @ B11  (2x2 result, cols 2-3)
    #   Each 2x2*2x2 uses Strassen with rank 7.
    #   Total rank for this block: 4 * 7 = 28
    # 
    # For C_bot_L = A_bot(1x4) @ B_L(4x4):
    #   Just dot product: sum_j A_bot[0,j] * B_L[j,k] for each k.
    #   Rank = 1*4*4 = 16 (schoolbook)
    # 
    # For C_top_R = A_top(2x4) @ B_R(4x3):
    #   Split A_top into AL(2x2), AR(2x2)
    #   Split B_R into BT(2x3), BB(2x3)
    #   C_top_R = AL @ BT + AR @ BB (each 2x2*2x3)
    #   Each 2x2*2x3: rank 12 (schoolbook)
    #   Total: 24
    # 
    # For C_bot_R = A_bot(1x4) @ B_R(4x3):
    #   Rank = 1*4*3 = 12 (schoolbook)
    # 
    # Grand total: 28 + 16 + 24 + 12 = 80
    # Improvement from 84 to 80!
    
    # Now let's implement the Strassen 2x2*2x2 decomposition
    def strassen_222():
        """Return U,V,W for 2x2 * 2x2 = 2x2 using Strassen (rank 7)."""
        U = np.zeros((4, 7), dtype=np.float64)  # (n*m=4, rank=7)
        V = np.zeros((4, 7), dtype=np.float64)  # (m*p=4, rank=7)
        W = np.zeros((4, 7), dtype=np.float64)  # (p*n=4, rank=7)
        # A = [[a,b],[c,d]], B = [[w,x],[y,z]], C = A*B
        # C = [[aw+cy, ax+cz], [bw+dy, bx+dz]]
        # Strassen:
        # P1 = (a+d)(w+z)
        # P2 = (c+d)w
        # P3 = a(x-z)
        # P4 = d(y-w)
        # P5 = (a+b)z
        # P6 = (c-a)(w+x)
        # P7 = (b-d)(y+z)
        # C11 = P1 - P5 + P6 - P7
        # C12 = P4 + P5
        # C21 = P3 + P6
        # C22 = P1 - P2 + P3 + P4
        
        # U[r, r] coefficient for A elements: A_flat = [a, b, c, d] (row-major: i*m+j, m=2)
        # V[r, r] coefficient for B elements: B_flat = [w, x, y, z] (j*p+k, p=2)
        # W[r, r] coefficient for C elements: C_flat = [C00, C01, C10, C11] (k*n+i, n=2)
        
        # r=0: P1
        U[:, 0] = [1, 0, 0, 1]  # a + d
        V[:, 0] = [1, 0, 0, 1]  # w + z
        W[:, 0] = [1, 0, 0, 1]  # C11 + C22
        
        # r=1: P2
        U[:, 1] = [0, 0, 1, 1]  # c + d
        V[:, 1] = [1, 0, 0, 0]  # w
        W[:, 1] = [0, 0, 0, -1]  # -C22
        
        # r=2: P3
        U[:, 2] = [1, 0, 0, 0]  # a
        V[:, 2] = [0, 1, 0, -1]  # x - z
        W[:, 2] = [0, 0, 1, 1]  # C21 + C22
        
        # r=3: P4
        U[:, 3] = [0, 0, 0, 1]  # d
        V[:, 3] = [0, 0, 1, -1]  # y - w
        W[:, 3] = [0, 1, 0, 1]  # C12 + C22
        
        # r=4: P5
        U[:, 4] = [1, 1, 0, 0]  # a + b
        V[:, 4] = [0, 0, 0, 1]  # z
        W[:, 4] = [-1, 1, 0, 0]  # -C11 + C12
        
        # r=5: P6
        U[:, 5] = [0, 0, 1, -1]  # c - a
        V[:, 5] = [1, 1, 0, 0]  # w + x
        W[:, 5] = [1, 0, 1, 0]  # C11 + C21
        
        # r=6: P7
        U[:, 6] = [0, 1, 0, -1]  # b - d
        V[:, 6] = [0, 0, 1, 1]  # y + z
        W[:, 6] = [-1, 0, 0, 0]  # -C11
        
        return U, V, W, 7
    
    def schoolbook_small(n_, m_, p_):
        return schoolbook_nmp(n_, m_, p_)
    
    # Build the full decomposition
    # Block 1: A_top(2x4) * B_L(4x4) using Strassen on 2x2 sub-blocks
    #   AL(2x2)*B00(2x2): Strassen, offset in A: rows 0-1, cols 0-1; B: rows 0-1, cols 0-1; C: rows 0-1, cols 0-1
    #   AL(2x2)*B10(2x2): Strassen, A: rows 0-1, cols 0-1; B: rows 2-3, cols 0-1; C: rows 0-1, cols 0-1
    #   AR(2x2)*B00(2x2): Strassen, A: rows 0-1, cols 2-3; B: rows 0-1, cols 0-1; C: rows 0-1, cols 0-1
    #   AR(2x2)*B11(2x2): Strassen, A: rows 0-1, cols 2-3; B: rows 2-3, cols 2-3; C: rows 0-1, cols 2-3
    #   AR(2x2)*B01(2x2): Strassen, A: rows 0-1, cols 2-3; B: rows 0-1, cols 2-3; C: rows 0-1, cols 2-3
    #   Wait, I need to be more careful.
    # 
    # C_top_L (2x4) = A_top(2x4) @ B_L(4x4)
    # C_TL[k, j] where k in [0,2) (row in C), j in [0,4) (col in C)
    # = sum_{l in [0,4)} A_top[k,l] * B_L[l,j]
    # 
    # Using block structure:
    # C_TL[:, 0:2] = AL @ B00 + AR @ B10
    # C_TL[:, 2:4] = AL @ B01 + AR @ B11
    # 
    # where AL = A_top[:, 0:2] (2x2), AR = A_top[:, 2:4] (2x2)
    #       B00 = B_L[0:2, 0:2] (2x2), B01 = B_L[0:2, 2:4] (2x2)
    #       B10 = B_L[2:4, 0:2] (2x2), B11 = B_L[2:4, 2:4] (2x2)
    # 
    # So 4 Strassen terms, each rank 7, total 28.
    
    # Similarly for C_top_R (2x3) = A_top(2x4) @ B_R(4x3)
    # C_top_R = AL @ BT + AR @ BB
    # where BT = B_R[0:2, :] (2x3), BB = B_R[2:4, :] (2x3)
    # Each 2x2*2x3: schoolbook rank 12. Total 24.
    
    # C_bot_L (1x4) = A_bot(1x4) @ B_L(4x4): rank 16
    # C_bot_R (1x3) = A_bot(1x4) @ B_R(4x3): rank 12
    
    total_rank = 28 + 16 + 24 + 12  # = 80
    U_full = np.zeros((12, total_rank), dtype=np.float64)  # 3*4=12
    V_full = np.zeros((28, total_rank), dtype=np.float64)  # 4*7=28
    W_full = np.zeros((21, total_rank), dtype=np.float64)  # 7*3=21
    
    col = 0
    # Block 1: 4 Strassen terms for A_top * B_L
    # Each term: (AL or AR) * (B00 or B10 or B01 or B11)
    strassen_U, strassen_V, strassen_W, _ = strassen_222()
    
    # Term 1: AL * B00 -> contributes to C[0:2, 0:2]
    for r in range(7):
        # AL: rows 0-1 of A, cols 0-1 -> A_flat indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5
        for local_row in range(4):
            i_l, j_l = local_row // 2, local_row % 2
            U_full[i_l * 4 + j_l, col + r] += strassen_U[local_row, r]
        # B00: rows 0-1 of B, cols 0-1 -> B_flat: 0*7+0=0, 0*7+1=1, 1*7+0=7, 1*7+1=8
        for local_row in range(4):
            j_l, k_l = local_row // 2, local_row % 2
            V_full[j_l * 7 + k_l, col + r] += strassen_V[local_row, r]
        # C[0:2, 0:2]: W_flat[k*3+i] for k in [0,2), i in [0,2)
        for local_row in range(4):
            k_l, i_l = local_row // 2, local_row % 2
            W_full[k_l * 3 + i_l, col + r] += strassen_W[local_row, r]
    col += 7
    
    # Term 2: AR * B10 -> contributes to C[0:2, 0:2]
    for r in range(7):
        for local_row in range(4):
            i_l, j_l = local_row // 2, local_row % 2
            U_full[i_l * 4 + (2 + j_l), col + r] += strassen_U[local_row, r]
        for local_row in range(4):
            j_l, k_l = local_row // 2, local_row % 2
            V_full[(2 + j_l) * 7 + k_l, col + r] += strassen_V[local_row, r]
        for local_row in range(4):
            k_l, i_l = local_row // 2, local_row % 2
            W_full[k_l * 3 + i_l, col + r] += strassen_W[local_row, r]
    col += 7
    
    # Term 3: AL * B01 -> contributes to C[0:2, 2:4]
    for r in range(7):
        for local_row in range(4):
            i_l, j_l = local_row // 2, local_row % 2
            U_full[i_l * 4 + j_l, col + r] += strassen_U[local_row, r]
        for local_row in range(4):
            j_l, k_l = local_row // 2, local_row % 2
            V_full[j_l * 7 + (2 + k_l), col + r] += strassen_V[local_row, r]
        for local_row in range(4):
            k_l, i_l = local_row // 2, local_row % 2
            W_full[(2 + k_l) * 3 + i_l, col + r] += strassen_W[local_row, r]
    col += 7
    
    # Term 4: AR * B11 -> contributes to C[0:2, 2:4]
    for r in range(7):
        for local_row in range(4):
            i_l, j_l = local_row // 2, local_row % 2
            U_full[i_l * 4 + (2 + j_l), col + r] += strassen_U[local_row, r]
        for local_row in range(4):
            j_l, k_l = local_row // 2, local_row % 2
            V_full[(2 + j_l) * 7 + (2 + k_l), col + r] += strassen_V[local_row, r]
        for local_row in range(4):
            k_l, i_l = local_row // 2, local_row % 2
            W_full[(2 + k_l) * 3 + i_l, col + r] += strassen_W[local_row, r]
    col += 7
    
    # Block 2: A_bot(1x4) * B_L(4x4), schoolbook rank 16
    # C[2, 0:4] = sum_{j=0}^{3} A[2,j] * B[j, 0:4]
    for j in range(4):
        for k in range(4):
            U_full[2 * 4 + j, col] = 1.0
            V_full[j * 7 + k, col] = 1.0
            W_full[k * 3 + 2, col] = 1.0
            col += 1
    # col is now 28 + 16 = 44
    
    # Block 3: A_top(2x4) * B_R(4x3), schoolbook rank 24
    # C[0:2, 4:7] = sum_{j=0}^{3} A[0:2, j] * B[j, 4:7]
    for i in range(2):
        for j in range(4):
            for k in range(3):
                U_full[i * 4 + j, col] = 1.0
                V_full[j * 7 + (4 + k), col] = 1.0
                W_full[(4 + k) * 3 + i, col] = 1.0
                col += 1
    # col is now 44 + 24 = 68
    
    # Block 4: A_bot(1x4) * B_R(4x3), schoolbook rank 12
    for j in range(4):
        for k in range(3):
            U_full[2 * 4 + j, col] = 1.0
            V_full[j * 7 + (4 + k), col] = 1.0
            W_full[(4 + k) * 3 + 2, col] = 1.0
            col += 1
    # col is now 68 + 12 = 80
    
    # Verify the decomposition
    def verify_decomp(U, V, W, n_, m_, p_):
        """Verify using random test cases."""
        np.random.seed(42)
        for trial in range(3):
            A = np.random.randn(n_, m_)
            B = np.random.randn(m_, p_)
            C_true = A @ B
            alpha = U.T @ A.flatten()
            beta = V.T @ B.flatten()
            C_comp = np.zeros((n_, p_))
            for i in range(n_):
                for k in range(p_):
                    C_comp[i, k] = np.dot(alpha * beta, W[k * n_ + i, :])
            if not np.allclose(C_comp, C_true, atol=1e-9):
                return False
        return True
    
    if verify_decomp(U_full, V_full, W_full, n, m, p):
        best_U, best_V, best_W, best_rank = U_full, V_full, W_full, col
    else:
        # Verification failed, return schoolbook
        pass
    
    if elapsed() > 140:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try even more aggressive decomposition
    # Can we also use Strassen-like tricks for the 1x4*4x4 or 2x2*2x3 blocks?
    # For 1x4*4x4: this is a dot product, essentially a vector-matrix product.
    # C_k = sum_j v_j * B[j,k]. Rank is m*p = 4*4 = 16 unless there's structure.
    # Not reducible with standard methods.
    
    # Try: can we split B_L differently to get more Strassen terms?
    # What if we split B into 7 columns as 4+3, but also try 7 = 2+2+2+1?
    # That gives more small blocks but each 2x2*2x2 is still Strassen.
    # Actually the current split 4+3 is fine.
    
    # Let's also try: split A into 1+1+1 rows instead of 2+1
    # Then we have 3 * (1x4) * B blocks, each (1x4)*(4x7) rank 28
    # Total: 3*28 = 84. Same as schoolbook. Worse.
    
    # Try: can we reduce the 2x2*2x3 schoolbook terms?
    # 2x2*2x3: C = A(2x2) * B(2x3). 
    # This is equivalent to two 2x2*2x1 and 2x2*2x2 multiplications.
    # The 2x2*2x2 part can use Strassen (rank 7), the 2x2*2x1 part... 
    # 2x2*2x1: C(2x1) = A(2x2)*B(2x1), C_i = A[i,0]*B[0] + A[i,1]*B[1], rank 4.
    # Total: 7 + 4 = 11 < 12! Save 1 per term, 2 terms = save 2.
    # New total: 80 - 2 = 78!
    
    # Let me implement 2x2*2x3 with rank 11.
    def matmul_2223():
        """2x2 * 2x3 decomposition with rank 11.
        A is 2x2, B is 2x3, C is 2x3.
        C = A @ B
        Split B into B_left(2x2) and B_right(2x1).
        C_left = A @ B_left (2x2*2x2, Strassen rank 7)
        C_right = A @ B_right (2x2*2x1, rank 4)
        Total rank: 11
        """
        # U has shape (2*2=4, 11), V has shape (2*3=6, 11), W has shape (3*2=6, 11)
        U = np.zeros((4, 11), dtype=np.float64)
        V = np.zeros((6, 11), dtype=np.float64)
        W = np.zeros((6, 11), dtype=np.float64)
        
        # Strassen part for 2x2*2x2 (B_left: cols 0,1)
        sU, sV, sW, _ = strassen_222()
        for r in range(7):
            U[:, r] += sU[:, r]
            # V: B_left is rows 0-1, cols 0-1 -> B_flat: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
            V[0, r] += sV[0, r]  # B[0,0]
            V[1, r] += sV[1, r]  # B[0,1]
            V[3, r] += sV[2, r]  # B[1,0]
            V[4, r] += sV[3, r]  # B[1,1]
            # W: C_left is 2x2, cols 0-1 of C (2x3)
            # W_flat[k*2+i] for k in [0,2), i in [0,2) -> indices 0,1,2,3
            W[0, r] += sW[0, r]
            W[1, r] += sW[1, r]
            W[2, r] += sW[2, r]
            W[3, r] += sW[3, r]
        
        # 2x2*2x1 part (B_right: col 2)
        # C_right = A @ B_col, where B_col = [b0, b1]^T
        # C[0,2] = a00*b0 + a01*b1
        # C[1,2] = a10*b0 + a11*b1
        # 4 terms:
        for t in range(4):
            r = 7 + t
            if t == 0:
                U[0, r] = 1.0; V[5, r] = 1.0; W[4, r] = 1.0  # a00*b0 -> C[0,2]
            elif t == 1:
                U[1, r] = 1.0; V[5, r] = 1.0; W[4, r] = 1.0  # a01*b0 -> C[0,2]
            elif t == 2:
                U[2, r] = 1.0; V[5, r] = 1.0; W[5, r] = 1.0  # a10*b1 -> C[1,2]
            elif t == 3:
                U[3, r] = 1.0; V[5, r] = 1.0; W[5, r] = 1.0  # a11*b1 -> C[1,2]
        
        return U, V, W, 11
    
    # Now rebuild with 2x2*2x3 using rank 11 instead of 12
    if elapsed() > 140:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    U_full2 = np.zeros((12, 78), dtype=np.float64)
    V_full2 = np.zeros((28, 78), dtype=np.float64)
    W_full2 = np.zeros((21, 78), dtype=np.float64)
    
    col2 = 0
    # Block 1: 4 Strassen terms (28 terms) - same as before
    for term in range(4):
        for r in range(7):
            if term == 0:  # AL*B00
                a_off_i, a_off_j = 0, 0
                b_off_i, b_off_j = 0, 0
                c_off_i, c_off_j = 0, 0
            elif term == 1:  # AR*B10
                a_off_i, a_off_j = 0, 2
                b_off_i, b_off_j = 2, 0
                c_off_i, c_off_j = 0, 0
            elif term == 2:  # AL*B01
                a_off_i, a_off_j = 0, 0
                b_off_i, b_off_j = 0, 2
                c_off_i, c_off_j = 0, 2
            elif term == 3:  # AR*B11
                a_off_i, a_off_j = 0, 2
                b_off_i, b_off_j = 2, 2
                c_off_i, c_off_j = 0, 2
            
            for local_row in range(4):
                i_l, j_l = local_row // 2, local_row % 2
                U_full2[(a_off_i + i_l) * 4 + (a_off_j + j_l), col2 + r] += strassen_U[local_row, r]
            for local_row in range(4):
                j_l, k_l = local_row // 2, local_row % 2
                V_full2[(b_off_i + j_l) * 7 + (b_off_j + k_l), col2 + r] += strassen_V[local_row, r]
            for local_row in range(4):
                k_l, i_l = local_row // 2, local_row % 2
                W_full2[(c_off_j + k_l) * 3 + (c_off_i + i_l), col2 + r] += strassen_W[local_row, r]
        col2 += 7
    
    # Block 2: A_bot * B_L (16 terms)
    for j in range(4):
        for k in range(4):
            U_full2[2 * 4 + j, col2] = 1.0
            V_full2[j * 7 + k, col2] = 1.0
            W_full2[k * 3 + 2, col2] = 1.0
            col2 += 1
    
    # Block 3: A_top * B_R using 2x2*2x3 with rank 11
    # AL(2x2) * BT(2x3): rank 11
    mm2223_U, mm2223_V, mm2223_W, r2223 = matmul_2223()
    for r in range(r2223):
        for local_row in range(4):
            i_l, j_l = local_row // 2, local_row % 2
            U_full2[i_l * 4 + j_l, col2 + r] += mm2223_U[local_row, r]
        for local_row in range(6):
            j_l, k_l = local_row // 3, local_row % 3
            V_full2[j_l * 7 + (4 + k_l), col2 + r] += mm2223_V[local_row, r]
        for local_row in range(6):
            k_l, i_l = local_row // 2, local_row % 2
            W_full2[(4 + k_l) * 3 + i_l, col2 + r] += mm2223_W[local_row, r]
    col2 += r2223
    
    # AR(2x2) * BB(2x3): rank 11
    for r in range(r2223):
        for local_row in range(4):
            i_l, j_l = local_row // 2, local_row % 2
            U_full2[i_l * 4 + (2 + j_l), col2 + r] += mm2223_U[local_row, r]
        for local_row in range(6):
            j_l, k_l = local_row // 3, local_row % 3
            V_full2[(2 + j_l) * 7 + (4 + k_l), col2 + r] += mm2223_V[local_row, r]
        for local_row in range(6):
            k_l, i_l = local_row // 2, local_row % 2
            W_full2[(4 + k_l) * 3 + i_l, col2 + r] += mm2223_W[local_row, r]
    col2 += r2223
    
    # Block 4: A_bot * B_R (12 terms)
    for j in range(4):
        for k in range(3):
            U_full2[2 * 4 + j, col2] = 1.0
            V_full2[j * 7 + (4 + k), col2] = 1.0
            W_full2[(4 + k) * 3 + 2, col2] = 1.0
            col2 += 1
    
    # col2 should be 28 + 16 + 11 + 11 + 12 = 78
    if col2 != 78:
        print(f"Warning: expected col2=78, got {col2}")
        # Trim or pad
        U_full2 = U_full2[:, :col2]
        V_full2 = V_full2[:, :col2]
        W_full2 = W_full2[:, :col2]
    
    if verify_decomp(U_full2, V_full2, W_full2, n, m, p):
        best_U, best_V, best_W, best_rank = U_full2, V_full2, W_full2, col2
    else:
        print("Verification failed for rank-78 decomposition")
    
    if elapsed() > 140:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to further optimize: can we reduce 1x4*4x4 or 1x4*4x3?
    # These are essentially vector-matrix products. 
    # 1x4*4x4: C_k = sum_j v_j B[jk]. This is a bilinear form with rank = 4*4 = 16 in general.
    # Unless B has special structure, we can't reduce this.
    
    # What about trying to merge some terms across blocks?
    # E.g., some of the A_bot terms might be expressible via the Strassen terms.
    # This would require finding linear dependencies in the combined system.
    
    # For now, return the best we have.
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0]) if result['U'] else 0}")
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0]) if result['U'] else 0})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0]) if result['V'] else 0})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0]) if result['W'] else 0})")
