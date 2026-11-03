"""Rank-reduction search: structured construction + local search for M_{4,4,7}."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    start_time = time.time()
    deadline = start_time + 140

    def strassen_2x2():
        """Returns U, V, W for 2x2 x 2x2 Strassen decomposition, each 4x7."""
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # p1 = (a+d)*(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # p2 = (c+d)*e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # p3 = a*(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # p4 = d*(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # p5 = (a+b)*h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # p6 = (c-a)*(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # p7 = (b-d)*(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def block_indices(br, bc, size=4):
        """Get flat indices for 2x2 block (br, bc) in a size x size matrix."""
        indices = []
        for r in range(2):
            for c in range(2):
                row = br * 2 + r
                col = bc * 2 + c
                indices.append(row * size + col)
        return indices

    def strassen_4x4x4():
        """Returns U, V, W for M_{4,4,4} using recursive Strassen, rank 49."""
        rank = 49
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        
        U2, V2, W2 = strassen_2x2()
        
        A11_idx = block_indices(0, 0)
        A12_idx = block_indices(0, 1)
        A21_idx = block_indices(1, 0)
        A22_idx = block_indices(1, 1)
        
        B11_idx = block_indices(0, 0)
        B12_idx = block_indices(0, 1)
        B21_idx = block_indices(1, 0)
        B22_idx = block_indices(1, 1)
        
        C11_idx = block_indices(0, 0)
        C12_idx = block_indices(0, 1)
        C21_idx = block_indices(1, 0)
        C22_idx = block_indices(1, 1)
        
        col = 0
        # P1 = (A11 + A22) * (B11 + B22)
        for r in range(7):
            for idx in A11_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A22_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B11_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B22_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C11_idx: W[idx, col] += W2[idx % 4, r]
            for idx in C22_idx: W[idx, col] -= W2[idx % 4, r]
            col += 1
        
        # P2 = (A21 + A22) * B11
        for r in range(7):
            for idx in A21_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A22_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B11_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C21_idx: W[idx, col] += W2[idx % 4, r]
            for idx in C22_idx: W[idx, col] -= W2[idx % 4, r]
            col += 1
        
        # P3 = A11 * (B12 - B22)
        for r in range(7):
            for idx in A11_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B12_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B22_idx: V[idx, col] -= V2[idx % 4, r]
            for idx in C12_idx: W[idx, col] += W2[idx % 4, r]
            for idx in C22_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P4 = A22 * (B21 - B11)
        for r in range(7):
            for idx in A22_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B21_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B11_idx: V[idx, col] -= V2[idx % 4, r]
            for idx in C11_idx: W[idx, col] += W2[idx % 4, r]
            for idx in C21_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P5 = (A11 + A12) * B22
        for r in range(7):
            for idx in A11_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A12_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B22_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C11_idx: W[idx, col] -= W2[idx % 4, r]
            for idx in C12_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P6 = (A21 - A11) * (B11 + B12)
        for r in range(7):
            for idx in A21_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A11_idx: U[idx, col] -= U2[idx % 4, r]
            for idx in B11_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B12_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C22_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P7 = (A12 - A22) * (B21 + B22)
        for r in range(7):
            for idx in A12_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A22_idx: U[idx, col] -= U2[idx % 4, r]
            for idx in B21_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B22_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C11_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        assert col == 49
        return U, V, W

    def schoolbook_4x4x1():
        """Returns U, V, W for M_{4,4,1} using schoolbook, rank 16."""
        rank = 16
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((4, rank), dtype=float)   # 4*1 = 4
        W = np.zeros((4, rank), dtype=float)   # 1*4 = 4
        
        col = 0
        for i in range(4):
            for j in range(4):
                U[i * 4 + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        
        assert col == 16
        return U, V, W

    def strassen_4x4x2():
        """
        Returns U, V, W for M_{4,4,2} using Strassen on 2x2x2 blocks.
        A is 4x4, B is 4x2, C is 4x2.
        Split B into B1 (4x1) and B2 (4x1), C into C1 (4x1) and C2 (4x1).
        Use Strassen for 2x2x1 multiplications.
        Rank: 2 * 7 = 14 (Strassen for 2x2x1) + 2 * 4 = 8 (schoolbook for 2x2x1)
        Actually, let's use a direct approach.
        """
        # For 4x4 x 4x2:
        # Split A into 2x2 blocks: A11, A12, A21, A22
        # B is 4x2: split into B_top (2x2) and B_bot (2x2)
        # C is 4x2: split into C_top (2x2) and C_bot (2x2)
        # C_top = A11 * B_top + A12 * B_bot
        # C_bot = A21 * B_top + A22 * B_bot
        # Each 2x2 x 2x2 multiplication uses Strassen (rank 7)
        # Total: 4 * 7 = 28
        
        rank = 28
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((8, rank), dtype=float)   # 4*2 = 8
        W = np.zeros((8, rank), dtype=float)   # 2*4 = 8
        
        U2, V2, W2 = strassen_2x2()
        
        # B indices: B is 4x2, row-major
        # B_top = rows 0-1, cols 0-1: indices [0, 1, 2, 3]
        # B_bot = rows 2-3, cols 0-1: indices [4, 5, 6, 7]
        B_top_idx = [0, 1, 2, 3]
        B_bot_idx = [4, 5, 6, 7]
        
        # C indices: C is 4x2, row-major
        # C_top = rows 0-1, cols 0-1: indices [0, 1, 2, 3]
        # C_bot = rows 2-3, cols 0-1: indices [4, 5, 6, 7]
        C_top_idx = [0, 1, 2, 3]
        C_bot_idx = [4, 5, 6, 7]
        
        # A blocks (4x4, row-major)
        A11_idx = block_indices(0, 0)  # [0, 1, 4, 5]
        A12_idx = block_indices(0, 1)  # [2, 3, 6, 7]
        A21_idx = block_indices(1, 0)  # [8, 9, 12, 13]
        A22_idx = block_indices(1, 1)  # [10, 11, 14, 15]
        
        col = 0
        # C_top = A11 * B_top + A12 * B_bot
        # A11 * B_top: Strassen 2x2 x 2x2
        for r in range(7):
            for idx in A11_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B_top_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C_top_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # A12 * B_bot: Strassen 2x2 x 2x2
        for r in range(7):
            for idx in A12_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B_bot_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C_top_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # C_bot = A21 * B_top + A22 * B_bot
        # A21 * B_top: Strassen 2x2 x 2x2
        for r in range(7):
            for idx in A21_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B_top_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C_bot_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # A22 * B_bot: Strassen 2x2 x 2x2
        for r in range(7):
            for idx in A22_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B_bot_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C_bot_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        assert col == 28
        return U, V, W

    def verify_decomposition(U, V, W, n, m, p):
        """Verify exact decomposition."""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        # Check all target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check non-target entries (sample)
        import random
        random.seed(42)
        for _ in range(500):
            u_idx = random.randint(0, n * m - 1)
            v_idx = random.randint(0, m * p - 1)
            w_idx = random.randint(0, p * n - 1)
            
            i = u_idx // m
            j = u_idx % m
            j2 = v_idx // p
            k = v_idx % p
            k2 = w_idx // n
            i2 = w_idx % n
            
            if i == i2 and j == j2 and k == k2:
                continue
            
            val = np.sum(U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :])
            if abs(val) > 1e-9:
                return False
        
        return True

    # Build the decomposition: 4x4x4 (Strassen, rank 49) + 4x4x2 (Strassen, rank 28) + 4x4x1 (schoolbook, rank 16)
    # Total rank: 49 + 28 + 16 = 93
    
    U1, V1, W1 = strassen_4x4x4()
    U2, V2, W2 = strassen_4x4x2()
    U3, V3, W3 = schoolbook_4x4x1()
    
    rank_total = 49 + 28 + 16
    U_full = np.zeros((16, rank_total), dtype=float)
    V_full = np.zeros((28, rank_total), dtype=float)
    W_full = np.zeros((28, rank_total), dtype=float)
    
    # Part 1: 4x4x4 (B cols 0-3, C cols 0-3)
    U_full[:, :49] = U1
    V_full[:16, :49] = V1
    W_full[:16, :49] = W1
    
    # Part 2: 4x4x2 (B cols 4-5, C cols 4-5)
    U_full[:, 49:77] = U2
    V_full[16:24, 49:77] = V2
    W_full[16:24, 49:77] = W2
    
    # Part 3: 4x4x1 (B col 6, C col 6)
    U_full[:, 77:93] = U3
    V_full[24:28, 77:93] = V3
    W_full[24:28, 77:93] = W3
    
    # Verify
    if not verify_decomposition(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
        # Fall back to pure schoolbook
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
    
    # Now try rank reduction via local search
    # Strategy: try to find linear dependencies among the rank-1 terms
    # If we can express one term as a linear combination of others, we can eliminate it
    
    best_rank = rank_total
    best_U = U_full.copy()
    best_V = V_full.copy()
    best_W = W_full.copy()
    
    # Try to eliminate terms by checking if removing a term still gives valid decomposition
    # This is a greedy approach: try removing each term, check validity
    
    def try_eliminate_term(U, V, W, term_idx, n, m, p):
        """Try to eliminate term_idx by checking if the remaining terms still decompose correctly."""
        if time.time() > deadline:
            return False, U, V, W
        
        rank = U.shape[1]
        if term_idx < 0 or term_idx >= rank:
            return False, U, V, W
        
        # Create new matrices without this term
        mask = [i for i in range(rank) if i != term_idx]
        U_new = U[:, mask]
        V_new = V[:, mask]
        W_new = W[:, mask]
        
        # Quick check: verify a subset of entries
        U_np = np.array(U_new, dtype=float)
        V_np = np.array(V_np, dtype=float) if False else np.array(V_new, dtype=float)
        W_np = np.array(W_new, dtype=float)
        
        # Check all target entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False, U, V, W
        
        # Check some non-target entries
        import random
        random.seed(42)
        for _ in range(200):
            u_idx = random.randint(0, n * m - 1)
            v_idx = random.randint(0, m * p - 1)
            w_idx = random.randint(0, p * n - 1)
            
            i = u_idx // m
            j = u_idx % m
            j2 = v_idx // p
            k = v_idx % p
            k2 = w_idx // n
            i2 = w_idx % n
            
            if i == i2 and j == j2 and k == k2:
                continue
            
            val = np.sum(U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :])
            if abs(val) > 1e-9:
                return False, U, V, W
        
        return True, U_new, V_new, W_new
    
    # Greedy elimination: try to remove terms one by one
    # Start from the end (schoolbook terms are most likely redundant if shared)
    current_U = U_full.copy()
    current_V = V_full.copy()
    current_W = W_full.copy()
    
    eliminated = set()
    for term_idx in range(rank_total - 1, -1, -1):
        if term_idx in eliminated:
            continue
        if time.time() > deadline:
            break
        
        success, new_U, new_V, new_W = try_eliminate_term(
            current_U, current_V, current_W, term_idx, n, m, p
        )
        
        if success:
            current_U = new_U
            current_V = new_V
            current_W = new_W
            eliminated.add(term_idx)
    
    # Check if we improved
    new_rank = current_U.shape[1]
    if new_rank < best_rank:
        best_rank = new_rank
        best_U = current_U
        best_V = current_V
        best_W = current_W
    
    # Final verification
    if not verify_decomposition(best_U.tolist(), best_V.tolist(), best_W.tolist(), n, m, p):
        # Fall back to the un-eliminated version
        if verify_decomposition(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
            best_U = U_full
            best_V = V_full
            best_W = W_full
            best_rank = rank_total
        else:
            # Pure schoolbook fallback
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
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
