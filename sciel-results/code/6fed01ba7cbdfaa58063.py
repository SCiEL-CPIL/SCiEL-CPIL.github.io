"""Matrix multiplication 4,4,6 tensor decomposition."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """Return a valid tensor decomposition for 4x4 * 4x6 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 4, 6

    # Build rank-7 Strassen for 2x2 * 2x2
    def build_2x2x2():
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))

        # Term 0: M1 = (A11+A22)(B11+B22)
        # A: (0,0)=1, (1,1)=1 -> U idx 0, 3
        # B: (0,0)=1, (1,1)=1 -> V idx 0, 3
        # C: (0,0)=1, (1,1)=1 -> W idx 0, 3
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # Term 1: M2 = (A21+A22)B11
        # A: (1,0)=1, (1,1)=1 -> U idx 2, 3
        # B: (0,0)=1 -> V idx 0
        # C: (1,0)=1, (1,1)=-1 -> W idx 2, 3
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1

        # Term 2: M3 = A11(B12-B22)
        # A: (0,0)=1 -> U idx 0
        # B: (0,1)=1, (1,1)=-1 -> V idx 1, 3
        # C: (0,1)=1, (1,1)=1 -> W idx 1, 3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1

        # Term 3: M4 = A22(B21-B11)
        # A: (1,1)=1 -> U idx 3
        # B: (1,0)=1, (0,0)=-1 -> V idx 2, 0
        # C: (0,0)=1, (1,0)=1 -> W idx 0, 2
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1

        # Term 4: M5 = (A11+A12)B22
        # A: (0,0)=1, (0,1)=1 -> U idx 0, 1
        # B: (1,1)=1 -> V idx 3
        # C: (0,0)=-1, (0,1)=1 -> W idx 0, 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1

        # Term 5: M6 = (A21-A11)(B11+B12)
        # A: (1,0)=1, (0,0)=-1 -> U idx 2, 0
        # B: (0,0)=1, (0,1)=1 -> V idx 0, 1
        # C: (1,1)=1 -> W idx 3
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        # Term 6: M7 = (A12-A22)(B21+B22)
        # A: (0,1)=1, (1,1)=-1 -> U idx 1, 3
        # B: (1,0)=1, (1,1)=1 -> V idx 2, 3
        # C: (0,0)=1 -> W idx 0
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1

        return U, V, W

    # Build rank-11 for 2x2 * 2x3
    # Decompose B (2x3) into B_left (2x1) and B_right (2x2)
    # A * B_left: rank 4
    # A * B_right: rank 7
    def build_2x2x3():
        rank = 11
        U = np.zeros((4, rank))
        V = np.zeros((6, rank))  # 2*3
        W = np.zeros((6, rank))  # 3*2

        # Part 1: A * B_left (rank 4)
        # B_left is column 0 of B (2x1)
        # C_left is column 0 of C (2x1)
        # Schoolbook:
        # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
        # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
        
        # Term 0: A[0,0] * B[0,0] -> C[0,0]
        # U idx 0, V idx 0 (j=0, k=0), W idx 0 (k=0, i=0)
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1

        # Term 1: A[0,1] * B[1,0] -> C[0,0]
        # U idx 1, V idx 3 (j=1, k=0), W idx 0 (k=0, i=0)
        U[1, 1] = 1
        V[3, 1] = 1
        W[0, 1] = 1

        # Term 2: A[1,0] * B[0,0] -> C[1,0]
        # U idx 2, V idx 0, W idx 1 (k=0, i=1)
        U[2, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1

        # Term 3: A[1,1] * B[1,0] -> C[1,0]
        # U idx 3, V idx 3, W idx 1
        U[3, 3] = 1
        V[3, 3] = 1
        W[1, 3] = 1

        # Part 2: A * B_right (rank 7)
        # B_right is columns 1,2 of B (2x2)
        # C_right is columns 1,2 of C (2x2)
        # Use Strassen for 2x2x2
        U_s, V_s, W_s = build_2x2x2()
        
        for c in range(7):
            for r in range(4):
                if U_s[r, c] != 0:
                    U[r, 4 + c] += U_s[r, c]
            
            # Map V_s indices (j*2+k_local) to global V indices (j*3 + (k_local+1))
            # V_s[0] (j=0, k=0) -> V[0*3+1] = V[1]
            # V_s[1] (j=0, k=1) -> V[0*3+2] = V[2]
            # V_s[2] (j=1, k=0) -> V[1*3+1] = V[4]
            # V_s[3] (j=1, k=1) -> V[1*3+2] = V[5]
            v_map = [1, 2, 4, 5]
            for r in range(4):
                if V_s[r, c] != 0:
                    V[v_map[r], 4 + c] += V_s[r, c]
            
            # Map W_s indices (k_local*2 + i) to global W indices ((k_local+1)*2 + i)
            # W_s[0] (k=0, i=0) -> W[1*2+0] = W[2]
            # W_s[1] (k=0, i=1) -> W[1*2+1] = W[3]
            # W_s[2] (k=1, i=0) -> W[2*2+0] = W[4]
            # W_s[3] (k=1, i=1) -> W[2*2+1] = W[5]
            w_map = [2, 3, 4, 5]
            for r in range(4):
                if W_s[r, c] != 0:
                    W[w_map[r], 4 + c] += W_s[r, c]

        return U, V, W

    # Build 4x4 * 4x6 using Strassen with 2x2x3 blocks
    def build_4x4x6():
        U_sub, V_sub, W_sub = build_2x2x3()
        rank_sub = 11
        total_rank = 7 * rank_sub
        
        U = np.zeros((16, total_rank))
        V = np.zeros((24, total_rank))
        W = np.zeros((24, total_rank))

        # Strassen terms
        # Blocks: 0:11, 1:12, 2:21, 3:22
        terms = [
            {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]}, # M1
            {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, -1)]},         # M2
            {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1), (3, 1)]},         # M3
            {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},         # M4
            {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(0, -1), (1, 1)]},         # M5
            {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(3, 1)]},         # M6
            {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(0, 1)]},         # M7
        ]

        # Offsets for U (4x4), V (4x6), W (6x4)
        # Block 0: (0:2, 0:2) for U, (0:2, 0:3) for V, (0:3, 0:2) for W
        # Block 1: (0:2, 2:4) for U, (0:2, 3:6) for V, (3:6, 0:2) for W
        # Block 2: (2:4, 0:2) for U, (2:4, 0:3) for V, (0:3, 2:4) for W
        # Block 3: (2:4, 2:4) for U, (2:4, 3:6) for V, (3:6, 2:4) for W
        
        u_offsets = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
        v_offsets = {0: (0, 0), 1: (0, 3), 2: (2, 0), 3: (2, 3)}
        w_offsets = {0: (0, 0), 1: (3, 0), 2: (0, 2), 3: (3, 2)}

        for t_idx, term in enumerate(terms):
            base_col = t_idx * rank_sub

            for c in range(rank_sub):
                # U
                for u_local in range(4):
                    u_val = U_sub[u_local, c]
                    if u_val == 0:
                        continue
                    i_loc = u_local // 2
                    j_loc = u_local % 2
                    for block, coeff in term['A']:
                        ro, co = u_offsets[block]
                        g_u = (i_loc + ro) * m + (j_loc + co)
                        U[g_u, base_col + c] += u_val * coeff

                # V
                for v_local in range(6):
                    v_val = V_sub[v_local, c]
                    if v_val == 0:
                        continue
                    j_loc = v_local // 3
                    k_loc = v_local % 3
                    for block, coeff in term['B']:
                        ro, co = v_offsets[block]
                        g_v = (j_loc + ro) * p + (k_loc + co)
                        V[g_v, base_col + c] += v_val * coeff

                # W
                for w_local in range(6):
                    w_val = W_sub[w_local, c]
                    if w_val == 0:
                        continue
                    k_loc = w_local // 2
                    i_loc = w_local % 2
                    for block, coeff in term['C']:
                        ro, co = w_offsets[block]
                        g_w = (k_loc + ro) * n + (i_loc + co)
                        W[g_w, base_col + c] += w_val * coeff

        return U, V, W

    U, V, W = build_4x4x6()
    
    # Convert to lists of integers (entries are integer)
    U_list = [[int(x) for x in row] for row in U]
    V_list = [[int(x) for x in row] for row in V]
    W_list = [[int(x) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
