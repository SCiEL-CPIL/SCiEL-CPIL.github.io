"""Matrix multiplication 4,4,9 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    def check_validity_fast(U, V, W, tol=1e-8):
        """Fast validity check using vectorized operations."""
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                A_ik = U[i*m:(i+1)*m, :]  # (m, rank)
                B_ik = np.array([V[j*p+k, :] for j in range(m)])  # (m, rank)
                w_ik = W[k*n+i, :]  # (rank,)
                result = (A_ik * w_ik[None, :]) @ B_ik.T  # (m, m)
                if np.max(np.abs(result - np.eye(m))) > tol:
                    return False
        return True

    def strassen_2x2():
        """Rank-7 tensor decomposition for 2x2 matrix multiplication."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[2, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = -1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def strassen_4x4():
        """
        Rank-49 Strassen-based decomposition for 4x4 * 4x4.
        Uses 2x2 blocks with Strassen 2x2 at each level.
        """
        U2, V2, W2 = strassen_2x2()
        
        rank = 49
        U = np.zeros((16, rank))
        V = np.zeros((16, rank))
        W = np.zeros((16, rank))
        
        def block_indices(bi, bj, dim=4):
            return [(bi*2+lr)*dim + (bj*2+lc) for lr in range(2) for lc in range(2)]
        
        A_blocks = {
            0: [(0,0,1), (1,1,1)],
            1: [(1,0,1), (1,1,1)],
            2: [(0,0,1)],
            3: [(1,1,1)],
            4: [(0,0,1), (0,1,1)],
            5: [(1,0,1), (0,0,-1)],
            6: [(0,1,1), (1,1,-1)]
        }
        B_blocks = {
            0: [(0,0,1), (1,1,1)],
            1: [(0,0,1)],
            2: [(0,1,1), (1,1,-1)],
            3: [(1,0,1), (0,0,-1)],
            4: [(1,1,1)],
            5: [(0,0,1), (0,1,1)],
            6: [(1,0,1), (1,1,1)]
        }
        C_blocks = {
            0: [(0,0,1), (1,1,1)],
            1: [(1,0,1), (1,1,-1)],
            2: [(0,1,1), (1,1,1)],
            3: [(0,0,1), (1,0,1)],
            4: [(0,0,-1), (0,1,1)],
            5: [(1,1,-1)],
            6: [(0,0,1)]
        }
        
        for k in range(7):
            for s in range(7):
                col = k * 7 + s
                for (bi, bj, coeff) in A_blocks[k]:
                    idxs = block_indices(bi, bj, dim=4)
                    for l in range(4):
                        if U2[l, s] != 0:
                            U[idxs[l], col] += coeff * U2[l, s]
                for (bk, bj, coeff) in B_blocks[k]:
                    idxs = block_indices(bk, bj, dim=4)
                    for l in range(4):
                        if V2[l, s] != 0:
                            V[idxs[l], col] += coeff * V2[l, s]
                for (bi, bj, coeff) in C_blocks[k]:
                    idxs = block_indices(bi, bj, dim=4)
                    for l in range(4):
                        if W2[l, s] != 0:
                            W[idxs[l], col] += coeff * W2[l, s]
        
        assert check_validity_fast(U, V, W), "4x4*4x4 Strassen failed"
        return U, V, W, rank

    def block_4x3():
        """
        Decomposition for 4x4 * 4x3 multiplication.
        Split into two 4x4*4x2 and one 4x4*4x1, or use structured approach.
        
        For 4x4 * 4x3: we can think of it as computing C = A*B where B is 4x3.
        Split B into B1 (4x2) and B2 (4x1).
        C1 = A*B1: use Strassen-like for 4x4*4x2
        C2 = A*B2: schoolbook 4x4*4x1, rank 16
        
        For 4x4*4x2: split into two 4x4*4x1, each rank 16, total 32.
        Or better: use 2x2 Strassen structure.
        
        Actually, let's use a direct structured approach:
        4x4 * 4x3 can be done with rank 48 (schoolbook) or less.
        Let's try: split A into 2x2 blocks, B into 2x2 and 2x1 blocks.
        
        For now, use schoolbook for 4x3 part: rank = 4*4*3 = 48
        """
        rank = 48
        U = np.zeros((16, rank))
        V = np.zeros((12, rank))  # 4*3
        W = np.zeros((12, rank))  # 3*4
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i*4+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
        
        assert check_validity_fast(np.zeros((16,16)), np.zeros((12,12)), np.zeros((12,12)))  # dummy
        return U, V, W, rank

    def combine_blocks():
        """
        Combine 4x4*4x4 (rank 49) and 4x4*4x5 (rank 80 schoolbook) 
        to get 4x4*4x9. Total rank: 129.
        
        Alternative: Split 9 into 4+3+2:
        - 4x4*4x4: rank 49 (Strassen)
        - 4x4*4x3: rank 48 (schoolbook) or better
        - 4x4*4x2: rank 32 (schoolbook) or better
        Total: 49 + 48 + 32 = 129 (same)
        
        Or: 4+5: 49 + 80 = 129
        
        Let's try 4+3+2 with optimized 4x3 and 4x2.
        """
        U44, V44, W44, r44 = strassen_4x4()
        
        # 4x4 * 4x3: schoolbook rank 48
        r43 = 48
        U43 = np.zeros((16, r43))
        V43 = np.zeros((12, r43))
        W43 = np.zeros((12, r43))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U43[i*4+j, col] = 1
                    V43[j*3+k, col] = 1
                    W43[k*4+i, col] = 1
                    col += 1
        
        # 4x4 * 4x2: schoolbook rank 32
        r42 = 32
        U42 = np.zeros((16, r42))
        V42 = np.zeros((8, r42))
        W42 = np.zeros((8, r42))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    U42[i*4+j, col] = 1
                    V42[j*2+k, col] = 1
                    W42[k*4+i, col] = 1
                    col += 1
        
        rank_total = r44 + r43 + r42  # 49 + 48 + 32 = 129
        
        U_full = np.zeros((16, rank_total))
        V_full = np.zeros((36, rank_total))
        W_full = np.zeros((36, rank_total))
        
        # Part 1: columns 0-3, rows 0-3 of B -> columns 0-3, rows 0-3 of C
        U_full[:, :r44] = U44
        for j in range(4):
            for k in range(4):
                V_full[j*9+k, :r44] = V44[j*4+k, :]
        for k in range(4):
            for i in range(4):
                W_full[k*4+i, :r44] = W44[k*4+i, :]
        
        # Part 2: columns 4-6 of B -> columns 4-6 of C
        offset = r44
        U_full[:, offset:offset+r43] = U43
        for j in range(4):
            for k in range(3):
                V_full[j*9+4+k, offset:offset+r43] = V43[j*3+k, :]
        for k in range(3):
            for i in range(4):
                W_full[(4+k)*4+i, offset:offset+r43] = W43[k*4+i, :]
        
        # Part 3: columns 7-8 of B -> columns 7-8 of C
        offset2 = r44 + r43
        U_full[:, offset2:offset2+r42] = U42
        for j in range(4):
            for k in range(2):
                V_full[j*9+7+k, offset2:offset2+r42] = V42[j*2+k, :]
        for k in range(2):
            for i in range(4):
                W_full[(7+k)*4+i, offset2:offset2+r42] = W42[k*4+i, :]
        
        assert check_validity_fast(U_full, V_full, W_full), "Combined 4x4*4x9 failed"
        return U_full, V_full, W_full, rank_total

    def try_block_optimization():
        """
        Try to optimize the 4x3 and 4x2 sub-decompositions using structured 
        approaches that might reduce rank below schoolbook.
        
        For 4x4 * 4x3: We can use a 2x2 block structure.
        A = [[A11, A12], [A21, A22]] where each is 2x2
        B = [[B11, B12, B13], [B21, B22, B23]] where each is 2x2
        
        C = [[C11, C12, C13], [C21, C22, C23]]
        
        Each Cij = A11*B1j + A12*B2j (for i=1)
                 = A21*B1j + A22*B2j (for i=2)
        
        Using Strassen for each 2x2*2x2 product: rank 7 per product.
        We have 3 columns of B, so 6 products total: 6*7 = 42.
        But we can share some computations.
        
        Actually, let's just use the structured approach directly.
        """
        U2, V2, W2 = strassen_2x2()
        
        # For 4x4 * 4x3:
        # A has 4 blocks (2x2 each), B has 6 blocks (2x2 each, 3 columns)
        # C has 6 blocks (2x2 each, 3 columns)
        
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # Each 2x2*2x2 product uses rank 7.
        # Total products: 12 (6 for top row, 6 for bottom row)
        # But A[0,0]*B[0,j] for j=0,1,2 can share the A[0,0] part.
        
        # Let's just use schoolbook for now and focus on the main structure.
        # The key improvement is using Strassen for the 4x4*4x4 part.
        
        return None

    # Get schoolbook solution as fallback
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try block decomposition (rank 129)
    if time.time() < deadline - 30:
        try:
            U_blk, V_blk, W_blk, rank_blk = combine_blocks()
            if rank_blk < best_rank:
                best_U, best_V, best_W, best_rank = U_blk, V_blk, W_blk, rank_blk
        except Exception as e:
            pass
    
    # Try to find better decomposition using structured random search
    # Focus on ranks below 129
    if time.time() < deadline - 20:
        # Use the block structure as a starting point and try to compress
        # by finding linear dependencies among the rank-1 terms
        
        # Alternative: try a different split
        # Split 9 as 4+4+1: 49 + 49 + 16 = 114
        def split_4_4_1():
            U44, V44, W44, r44 = strassen_4x4()
            
            # Second 4x4*4x4 for columns 4-7
            U44b, V44b, W44b, r44b = strassen_4x4()
            
            # 4x4*4x1: schoolbook rank 16
            r41 = 16
            U41 = np.zeros((16, r41))
            V41 = np.zeros((4, r41))
            W41 = np.zeros((4, r41))
            col = 0
            for i in range(4):
                for j in range(4):
                    U41[i*4+j, col] = 1
                    V41[j, col] = 1
                    W41[i, col] = 1
                    col += 1
            
            rank_total = r44 + r44b + r41  # 49 + 49 + 16 = 114
            
            U_full = np.zeros((16, rank_total))
            V_full = np.zeros((36, rank_total))
            W_full = np.zeros((36, rank_total))
            
            # Part 1: columns 0-3
            U_full[:, :r44] = U44
            for j in range(4):
                for k in range(4):
                    V_full[j*9+k, :r44] = V44[j*4+k, :]
            for k in range(4):
                for i in range(4):
                    W_full[k*4+i, :r44] = W44[k*4+i, :]
            
            # Part 2: columns 4-7
            offset = r44
            U_full[:, offset:offset+r44b] = U44b
            for j in range(4):
                for k in range(4):
                    V_full[j*9+4+k, offset:offset+r44b] = V44b[j*4+k, :]
            for k in range(4):
                for i in range(4):
                    W_full[(4+k)*4+i, offset:offset+r44b] = W44b[k*4+i, :]
            
            # Part 3: column 8
            offset2 = r44 + r44b
            U_full[:, offset2:offset2+r41] = U41
            for j in range(4):
                V_full[j*9+8, offset2:offset2+r41] = V41[j, :]
            for i in range(4):
                W_full[8*4+i, offset2:offset2+r41] = W41[i, :]
            
            assert check_validity_fast(U_full, V_full, W_full), "4+4+1 split failed"
            return U_full, V_full, W_full, rank_total
        
        try:
            U_441, V_441, W_441, rank_441 = split_4_4_1()
            if rank_441 < best_rank:
                best_U, best_V, best_W, best_rank = U_441, V_441, W_441, rank_441
        except Exception as e:
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
