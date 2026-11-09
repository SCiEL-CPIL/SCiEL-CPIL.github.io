"""Matrix multiplication 4x5 x 5x5 tensor decomposition with Strassen optimization."""
import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140  # Safety margin before 150s kill


def construct():
    n, m, p = 4, 5, 5
    deadline = time.time() + 130  # Safety margin

    def build_strassen_2x2x2():
        """
        Build Strassen's 7-term decomposition for 2x2 * 2x2.
        Input: A (2x2), B (2x2)
        Output: C = A*B (2x2)
        
        Strassen formulas:
        M1 = (A00+A11)(B00+B11)
        M2 = (A10+A11)B00
        M3 = A00(B10-B11)
        M4 = A11(B01-B10)
        M5 = (A00+A01)B11
        M6 = (A10-A00)(B00+B10)
        M7 = (A01-A11)(B01+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # For 2x2 * 2x2:
        # A indices: row i in {0,1}, col j in {0,1}
        # B indices: row j in {0,1}, col k in {0,1}
        # C indices: row i in {0,1}, col k in {0,1}
        
        # U factor: maps (i,j) -> contributions
        # V factor: maps (j,k) -> contributions  
        # W factor: maps (k,i) -> contributions
        
        # 7 terms
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        # A00: i=0,j=0 -> U[0,0]=1; A11: i=1,j=1 -> U[3,0]=1
        # B00: j=0,k=0 -> V[0,0]=1; B11: j=1,k=1 -> V[3,0]=1
        # C00: k=0,i=0 -> W[0,0]=1; C11: k=1,i=1 -> W[3,0]=1
        U[0, 0] = 1.0  # A00
        U[3, 0] = 1.0  # A11
        V[0, 0] = 1.0  # B00
        V[3, 0] = 1.0  # B11
        W[0, 0] = 1.0  # C00 (k=0,i=0)
        W[3, 0] = 1.0  # C11 (k=1,i=1)
        
        # Term 2: M2 = (A10+A11)B00
        # A10: i=1,j=0 -> U[2,1]=1; A11: i=1,j=1 -> U[3,1]=1
        # B00: j=0,k=0 -> V[0,1]=1
        # C10: k=0,i=1 -> W[2,1]=1
        U[2, 1] = 1.0  # A10
        U[3, 1] = 1.0  # A11
        V[0, 1] = 1.0  # B00
        W[2, 1] = 1.0  # C10 (k=0,i=1)
        
        # Term 3: M3 = A00(B10-B11)
        # A00: i=0,j=0 -> U[0,2]=1
        # B10: j=1,k=0 -> V[2,2]=1; B11: j=1,k=1 -> V[3,2]=-1
        # C01: k=1,i=0 -> W[1,2]=1; C11: k=1,i=1 -> W[3,2]=1
        U[0, 2] = 1.0  # A00
        V[2, 2] = 1.0  # B10
        V[3, 2] = -1.0  # B11
        W[1, 2] = 1.0  # C01 (k=1,i=0)
        W[3, 2] = 1.0  # C11 (k=1,i=1)
        
        # Term 4: M4 = A11(B01-B10)
        # A11: i=1,j=1 -> U[3,3]=1
        # B01: j=0,k=1 -> V[1,3]=1; B10: j=1,k=0 -> V[2,3]=-1
        # C00: k=0,i=0 -> W[0,3]=1; C10: k=0,i=1 -> W[2,3]=1
        U[3, 3] = 1.0  # A11
        V[1, 3] = 1.0  # B01
        V[2, 3] = -1.0  # B10
        W[0, 3] = 1.0  # C00 (k=0,i=0)
        W[2, 3] = 1.0  # C10 (k=0,i=1)
        
        # Term 5: M5 = (A00+A01)B11
        # A00: i=0,j=0 -> U[0,4]=1; A01: i=0,j=1 -> U[1,4]=1
        # B11: j=1,k=1 -> V[3,4]=1
        # C00: k=0,i=0 -> W[0,4]=-1; C01: k=1,i=0 -> W[1,4]=1
        U[0, 4] = 1.0  # A00
        U[1, 4] = 1.0  # A01
        V[3, 4] = 1.0  # B11
        W[0, 4] = -1.0  # C00 (k=0,i=0)
        W[1, 4] = 1.0  # C01 (k=1,i=0)
        
        # Term 6: M6 = (A10-A00)(B00+B10)
        # A10: i=1,j=0 -> U[2,5]=1; A00: i=0,j=0 -> U[0,5]=-1
        # B00: j=0,k=0 -> V[0,5]=1; B10: j=1,k=0 -> V[2,5]=1
        # C11: k=1,i=1 -> W[3,5]=1
        U[2, 5] = 1.0  # A10
        U[0, 5] = -1.0  # A00
        V[0, 5] = 1.0  # B00
        V[2, 5] = 1.0  # B10
        W[3, 5] = 1.0  # C11 (k=1,i=1)
        
        # Term 7: M7 = (A01-A11)(B01+B11)
        # A01: i=0,j=1 -> U[1,6]=1; A11: i=1,j=1 -> U[3,6]=-1
        # B01: j=0,k=1 -> V[1,6]=1; B11: j=1,k=1 -> V[3,6]=1
        # C00: k=0,i=0 -> W[0,6]=1; C01: k=1,i=0 -> W[1,6]=1
        U[1, 6] = 1.0  # A01
        U[3, 6] = -1.0  # A11
        V[1, 6] = 1.0  # B01
        V[3, 6] = 1.0  # B11
        W[0, 6] = 1.0  # C00 (k=0,i=0)
        W[1, 6] = 1.0  # C01 (k=1,i=0)
        
        return U, V, W

    def build_schoolbook(n, m, p):
        """Build schoolbook decomposition."""
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb), dtype=float)
        V = np.zeros((m * p, rank_sb), dtype=float)
        W = np.zeros((p * n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-9):
        """Verify tensor decomposition."""
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_mat[i * m + j, c] * V_mat[j * p + k, c] * W_mat[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def combine_decompositions(blocks, n, m, p):
        """
        Combine block decompositions into full decomposition.
        blocks: list of (i_start, j_start, k_start, i_end, j_end, k_end, U_block, V_block, W_block)
        """
        total_rank = sum(len(block) for block in blocks)
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        for (i_start, j_start, k_start, i_end, j_end, k_end, U_block, V_block, W_block) in blocks:
            rank_block = U_block.shape[1]
            # Map local indices to global
            for li in range(i_end - i_start):
                for lj in range(j_end - j_start):
                    gi = i_start + li
                    gj = j_start + lj
                    U[gi * m + gj, col:col+rank_block] = U_block[li * (j_end-j_start) + lj, :]
            for lj in range(j_end - j_start):
                for lk in range(k_end - k_start):
                    gj = j_start + lj
                    gk = k_start + lk
                    V[gj * p + gk, col:col+rank_block] = V_block[lj * (k_end-k_start) + lk, :]
            for lk in range(k_end - k_start):
                for li in range(i_end - i_start):
                    gk = k_start + lk
                    gi = i_start + li
                    W[gk * n + gi, col:col+rank_block] = W_block[lk * (i_end-i_start) + li, :]
            col += rank_block
        
        return U, V, W

    # Strategy: Use Strassen for 2x2 blocks where possible
    # n=4, m=5, p=5
    # We can partition:
    # - i: 0-1 (2 rows), 2-3 (2 rows)
    # - j: 0-1 (2 cols), 2-4 (3 cols)  
    # - k: 0-1 (2 cols), 2-4 (3 cols)
    
    # Block decomposition:
    # Block 1: i=[0,1], j=[0,1], k=[0,1] -> 2x2x2, use Strassen (rank 7)
    # Block 2: i=[0,1], j=[0,1], k=[2,4] -> 2x2x3, schoolbook (rank 12)
    # Block 3: i=[0,1], j=[2,4], k=[0,1] -> 2x3x2, schoolbook (rank 12)
    # Block 4: i=[0,1], j=[2,4], k=[2,4] -> 2x3x3, schoolbook (rank 18)
    # Block 5: i=[2,3], j=[0,1], k=[0,1] -> 2x2x2, use Strassen (rank 7)
    # Block 6: i=[2,3], j=[0,1], k=[2,4] -> 2x2x3, schoolbook (rank 12)
    # Block 7: i=[2,3], j=[2,4], k=[0,1] -> 2x3x2, schoolbook (rank 12)
    # Block 8: i=[2,3], j=[2,4], k=[2,4] -> 2x3x3, schoolbook (rank 18)
    
    # Total: 7+12+12+18+7+12+12+18 = 98
    
    # Alternative: Try to use Strassen more aggressively
    # For 2x2x2 blocks, Strassen gives rank 7 vs schoolbook 8
    
    # Let's try: partition j into [0,1] and [2,3] and [4]
    # Partition k into [0,1] and [2,3] and [4]
    
    # Blocks:
    # i=[0,1], j=[0,1], k=[0,1] -> 2x2x2 Strassen (7)
    # i=[0,1], j=[0,1], k=[2,3] -> 2x2x2 Strassen (7)
    # i=[0,1], j=[0,1], k=[4] -> 2x2x1 schoolbook (4)
    # i=[0,1], j=[2,3], k=[0,1] -> 2x2x2 Strassen (7)
    # i=[0,1], j=[2,3], k=[2,3] -> 2x2x2 Strassen (7)
    # i=[0,1], j=[2,3], k=[4] -> 2x2x1 schoolbook (4)
    # i=[0,1], j=[4], k=[0,1] -> 2x1x2 schoolbook (4)
    # i=[0,1], j=[4], k=[2,3] -> 2x1x2 schoolbook (4)
    # i=[0,1], j=[4], k=[4] -> 2x1x1 schoolbook (2)
    # i=[2,3], j=[0,1], k=[0,1] -> 2x2x2 Strassen (7)
    # i=[2,3], j=[0,1], k=[2,3] -> 2x2x2 Strassen (7)
    # i=[2,3], j=[0,1], k=[4] -> 2x2x1 schoolbook (4)
    # i=[2,3], j=[2,3], k=[0,1] -> 2x2x2 Strassen (7)
    # i=[2,3], j=[2,3], k=[2,3] -> 2x2x2 Strassen (7)
    # i=[2,3], j=[2,3], k=[4] -> 2x2x1 schoolbook (4)
    # i=[2,3], j=[4], k=[0,1] -> 2x1x2 schoolbook (4)
    # i=[2,3], j=[4], k=[2,3] -> 2x1x2 schoolbook (4)
    # i=[2,3], j=[4], k=[4] -> 2x1x1 schoolbook (2)
    
    # Total: 7+7+4+7+7+4+4+4+2+7+7+4+7+7+4+4+4+2 = 96
    
    # Build the decomposition
    blocks = []
    
    # Helper to create schoolbook block
    def schoolbook_block(n_b, m_b, p_b):
        rank_b = n_b * m_b * p_b
        U_b = np.zeros((n_b * m_b, rank_b), dtype=float)
        V_b = np.zeros((m_b * p_b, rank_b), dtype=float)
        W_b = np.zeros((p_b * n_b, rank_b), dtype=float)
        col = 0
        for i in range(n_b):
            for j in range(m_b):
                for k in range(p_b):
                    U_b[i * m_b + j, col] = 1.0
                    V_b[j * p_b + k, col] = 1.0
                    W_b[k * n_b + i, col] = 1.0
                    col += 1
        return U_b, V_b, W_b
    
    # Strassen 2x2x2
    U_s, V_s, W_s = build_strassen_2x2x2()
    
    # Define blocks
    # i partitions: [0,1], [2,3]
    # j partitions: [0,1], [2,3], [4]
    # k partitions: [0,1], [2,3], [4]
    
    i_parts = [(0, 1), (2, 3)]
    j_parts = [(0, 1), (2, 3), (4, 4)]
    k_parts = [(0, 1), (2, 3), (4, 4)]
    
    for i_start, i_end in i_parts:
        for j_start, j_end in j_parts:
            for k_start, k_end in k_parts:
                n_b = i_end - i_start
                m_b = j_end - j_start
                p_b = k_end - k_start
                
                if n_b == 2 and m_b == 2 and p_b == 2:
                    U_b, V_b, W_b = U_s.copy(), V_s.copy(), W_s.copy()
                else:
                    U_b, V_b, W_b = schoolbook_block(n_b, m_b, p_b)
                
                blocks.append((i_start, j_start, k_start, i_end, j_end, k_end, U_b, V_b, W_b))
    
    U, V, W = combine_decompositions(blocks, n, m, p)
    
    # Verify
    if not verify(U, V, W, n, m, p, tol=1e-9):
        # Fallback to schoolbook
        U, V, W = build_schoolbook(n, m, p)
        if not verify(U, V, W, n, m, p, tol=1e-9):
            raise Exception("Verification failed")
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find better decomposition with remaining time
    np.random.seed(42)
    
    # Try random search for lower rank
    for target_rank in range(best_rank - 1, max(best_rank - 20, 80), -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(2):
            if time.time() > deadline - 10:
                break
            
            # Initialize with current best structure plus perturbation
            U_t = np.zeros((n * m, target_rank), dtype=float)
            V_t = np.zeros((m * p, target_rank), dtype=float)
            W_t = np.zeros((p * n, target_rank), dtype=float)
            
            # Copy as much of best as possible
            min_rank = min(target_rank, best_rank)
            U_t[:, :min_rank] = best_U[:, :min_rank]
            V_t[:, :min_rank] = best_V[:, :min_rank]
            W_t[:, :min_rank] = best_W[:, :min_rank]
            
            # Add random perturbation to last columns
            if min_rank < target_rank:
                for c in range(min_rank, target_rank):
                    U_t[:, c] = np.random.randn(n * m) * 0.1
                    V_t[:, c] = np.random.randn(m * p) * 0.1
                    W_t[:, c] = np.random.randn(p * n) * 0.1
            
            if verify(U_t, V_t, W_t, n, m, p, tol=1e-6):
                best_rank = target_rank
                best_U, best_V, best_W = U_t.copy(), V_t.copy(), W_t.copy()
                found = True
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
