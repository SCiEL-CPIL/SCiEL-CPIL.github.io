"""Matrix multiplication 4,4,9 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

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

    def check_validity(U, V, W, tol=1e-8):
        """
        Check if the decomposition is exact.
        For each output (i, k) where i in [0,n), k in [0,p):
          sum_j sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] should be delta_{j,j'} for all j,j'
          More precisely: for each (i,k), the m x m matrix M where
          M[j, j'] = sum_t U[i*m+j, t] * V[j'*p+k, t] * W[k*n+i, t]
          should equal identity.
        """
        rank = U.shape[1]
        for i in range(n):
            for k in range(p):
                # For each output entry C[i, k] (row i of A, col k of B result)
                # We need: for all j, j', sum_t U[i*m+j, t] * V[j'*p+k, t] * W[k*n+i, t] = delta_{j,j'}
                # This means: sum_t W[k*n+i, t] * (U[i*m: (i+1)*m, t] outer V[j'*p+k, t]) ... 
                # Actually let me think more carefully.
                
                # The tensor decomposition says:
                # T[u, v, w] = sum_t U[u, t] * V[v, t] * W[w, t]
                # where u = i*m+j (A entry), v = j'*p+k (B entry), w = k*n+i (C entry)
                # For matrix mult: T[i*m+j, j'*p+k, k*n+i] = delta_{j, j'}
                
                # So for fixed (i, k):
                # sum_t U[i*m+j, t] * V[j'*p+k, t] * W[k*n+i, t] = delta_{j,j'}
                
                # Let's define for fixed (i,k):
                # For each t: U_i_j[t] = U[i*m+j, t], V_k_j'[t] = V[j'*p+k, t], W_ik[t] = W[k*n+i, t]
                # sum_t W_ik[t] * U_i_j[t] * V_k_j'[t] = delta_{j,j'}
                
                # This is: (W_ik elementwise (U_i * V_k^T)) summed over t... 
                # Actually: for each j, j': sum_t W_ik[t] * U[i*m+j,t] * V[j'*p+k,t] = delta_{j,j'}
                
                # Let A_ik = U[i*m:(i+1)*m, :]  # shape (m, rank)
                # Let B_ik = np.array([V[j*p+k, :] for j in range(m)])  # shape (m, rank), row j is V[j*p+k,:]
                # Let w_ik = W[k*n+i, :]  # shape (rank,)
                
                # Then for each j, j': sum_t w_ik[t] * A_ik[j,t] * B_ik[j',t] = delta_{j,j'}
                # This is: A_ik * diag(w_ik) * B_ik.T = I_m
                
                A_ik = U[i*m:(i+1)*m, :]  # (m, rank)
                B_ik = np.array([V[j*p+k, :] for j in range(m)])  # (m, rank)
                w_ik = W[k*n+i, :]  # (rank,)
                
                # Compute A_ik * diag(w_ik) * B_ik.T
                # = (A_ik * w_ik[None, :]) @ B_ik.T
                result = (A_ik * w_ik[None, :]) @ B_ik.T  # (m, m)
                
                expected = np.eye(m)
                if np.max(np.abs(result - expected)) > tol:
                    return False
        return True

    def strassen_2x2():
        """
        Rank-7 tensor decomposition for 2x2 matrix multiplication.
        A (2x2) times B (2x2) = C (2x2)
        Input indices: u = row*2 + col for A, v = row*2 + col for B
        Output indices: w = row*2 + col for C
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+g)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[2, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (c+d)e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = d(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (a+b)h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (c-a)(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = -1
        
        # M7 = (b-d)(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def strassen_4x4_4x9():
        """
        Use Strassen-like block decomposition for 4x4 * 4x9.
        Partition A (4x4) into 2x2 blocks: A = [[A00, A01], [A10, A11]]
        Partition B (4x9) into 2x9 blocks: B = [[B00, B01, B02], [B10, B11, B12]]
        where each block is 2x3.
        Partition C (4x9) into 2x2 blocks: C = [[C00, C01, C02], [C10, C11, C12]]
        where each block is 2x3.
        
        C[i,j] = sum over Strassen products of A-blocks * B-blocks.
        Each block product is 2x2 * 2x3, which we decompose using Strassen 2x2 * 2x2
        applied to each column of the 2x3 block (3 columns, each rank 7, total rank 21 per block product).
        
        7 block-level products * 21 rank per product = 147. That's worse than schoolbook 144.
        
        Alternative: use Strassen for 4x4*4x4 (rank 49) and schoolbook for 4x4*4x5 (rank 80).
        Total: 49 + 80 = 129. Better!
        
        Let's do that.
        """
        # Split B into B_left (4x4) and B_right (4x5)
        # C_left = A * B_left (4x4), C_right = A * B_right (4x5)
        
        # For C_left: use Strassen 4x4*4x4 with rank 49
        U2, V2, W2 = strassen_2x2()
        
        U_left = np.zeros((16, 49))
        V_left = np.zeros((16, 49))
        W_left = np.zeros((16, 49))
        
        # Strassen 4x4: partition into 2x2 blocks
        # A = [[A00, A01], [A10, A11]] (each 2x2)
        # B = [[B00, B01], [B10, B11]] (each 2x2)
        # C = [[C00, C01], [C10, C11]] (each 2x2)
        
        # 7 Strassen products at block level:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # Each block product is 2x2 * 2x2 = rank 7 (Strassen)
        # Total rank: 7*7 = 49
        
        # Block (bi, bj) of A: rows [bi*2, bi*2+1], cols [bj*2, bj*2+1]
        # Global index: row*4 + col
        # Block (bi, bj) entries: (bi*2)*4 + bj*2, (bi*2)*4 + bj*2+1, (bi*2+1)*4 + bj*2, (bi*2+1)*4 + bj*2+1
        
        # U2 indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) local
        # For block (bi, bj): local (lr,lc) -> global (bi*2+lr)*4 + (bj*2+lc)
        
        # Define helper to get global indices for a 2x2 block
        def block_indices(bi, bj, dim=4):
            """Get the 4 global indices for 2x2 block (bi, bj) in a dim x dim matrix."""
            return [(bi*2+lr)*dim + (bj*2+lc) for lr in range(2) for lc in range(2)]
        
        # Strassen block-level coefficients for each of the 7 products
        # For product k, we need:
        #   A-side: sum over blocks (bi,bj) of coeff_A[k, bi, bj] * A_block(bi,bj)
        #   B-side: sum over blocks (bk,bj) of coeff_B[k, bk, bj] * B_block(bk,bj)
        #   C-side: C_block(bi,bj) = sum_k coeff_C[k, bi, bj] * M_k(bi,bj)
        
        # Block-level Strassen:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B01-B11)
        # M4 = A11*(B10-B00)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 - M6
        
        # For each block-level product k (0..6):
        # A-side blocks and coefficients:
        # k=0 (M1): A00(+1), A11(+1)
        # k=1 (M2): A10(+1), A11(+1)
        # k=2 (M3): A00(+1)
        # k=3 (M4): A11(+1)
        # k=4 (M5): A00(+1), A01(+1)
        # k=5 (M6): A10(+1), A00(-1)
        # k=6 (M7): A01(+1), A11(-1)
        
        # B-side blocks and coefficients:
        # k=0 (M1): B00(+1), B11(+1)
        # k=1 (M2): B00(+1)
        # k=2 (M3): B01(+1), B11(-1)
        # k=3 (M4): B10(+1), B00(-1)
        # k=4 (M5): B11(+1)
        # k=5 (M6): B00(+1), B01(+1)
        # k=6 (M7): B10(+1), B11(+1)
        
        # C-side: for each k, which C blocks does it contribute to and with what coefficient?
        # k=0 (M1): C00(+1), C11(+1)
        # k=1 (M2): C10(+1), C11(-1)
        # k=2 (M3): C01(+1), C11(+1)
        # k=3 (M4): C00(+1), C10(+1)
        # k=4 (M5): C00(-1), C01(+1)
        # k=5 (M6): C11(-1)
        # k=6 (M7): C00(+1)
        
        A_blocks = {  # k: list of (bi, bj, coeff)
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
        
        # For each block-level product k, and each Strassen 2x2 product s (0..6):
        # The rank-1 term index is k*7 + s
        for k in range(7):
            for s in range(7):
                col = k * 7 + s
                # U: for each A block (bi,bj,coeff) in A_blocks[k],
                #   for each local entry l in U2[:, s]:
                #     global_idx = block_indices(bi,bj)[l]
                #     U[global_idx, col] += coeff * U2[l, s]
                for (bi, bj, coeff) in A_blocks[k]:
                    idxs = block_indices(bi, bj, dim=4)
                    for l in range(4):
                        if U2[l, s] != 0:
                            U_left[idxs[l], col] += coeff * U2[l, s]
                
                # V: for each B block (bk,bj,coeff) in B_blocks[k],
                #   for each local entry l in V2[:, s]:
                #     global_idx = block_indices(bk,bj)[l]
                #     V[global_idx, col] += coeff * V2[l, s]
                for (bk, bj, coeff) in B_blocks[k]:
                    idxs = block_indices(bk, bj, dim=4)
                    for l in range(4):
                        if V2[l, s] != 0:
                            V_left[idxs[l], col] += coeff * V2[l, s]
                
                # W: for each C block (bi,bj,coeff) in C_blocks[k],
                #   for each local entry l in W2[:, s]:
                #     global_idx = block_indices(bi,bj)[l]
                #     W[global_idx, col] += coeff * W2[l, s]
                for (bi, bj, coeff) in C_blocks[k]:
                    idxs = block_indices(bi, bj, dim=4)
                    for l in range(4):
                        if W2[l, s] != 0:
                            W_left[idxs[l], col] += coeff * W2[l, s]
        
        # Verify Strassen 4x4*4x4
        assert check_validity(U_left, V_left, W_left), "Strassen 4x4*4x4 failed"
        
        # For C_right: A (4x4) * B_right (4x5), schoolbook rank = 4*4*5 = 80
        U_right = np.zeros((16, 80))
        V_right = np.zeros((20, 80))
        W_right = np.zeros((20, 80))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U_right[i*4+j, col] = 1
                    V_right[j*5+k, col] = 1
                    W_right[k*4+i, col] = 1
                    col += 1
        
        # Combine: total rank = 49 + 80 = 129
        U = np.hstack([U_left, U_right])
        V = np.vstack([V_left, V_right])  # V_left is (16,49), V_right is (20,80)
        # Wait, V has shape (m*p, rank) = (4*9, rank) = (36, rank)
        # V_left should be (16, 49) for the 4x4 part (m=4, p_left=4)
        # V_right should be (20, 80) for the 4x5 part (m=4, p_right=5)
        # But they need to be stacked to form (36, 129)
        # V_left: rows 0..15 correspond to j*p_left + k where j in 0..3, k in 0..3
        # V_right: rows 0..19 correspond to j*p_right + k where j in 0..3, k in 0..4
        # But in the full problem, V has rows j*p + k where p=9
        # So V_right rows should be at positions j*9 + 4 + k for k in 0..4
        
        # Let me redo this more carefully.
        # Full V has shape (m*p, rank) = (36, rank)
        # V[j*p + k, t] for j in 0..3, k in 0..8
        # For the left part (k=0..3): V[j*9 + k, t] for the Strassen part
        # For the right part (k=4..8): V[j*9 + 4 + k', t] for k' in 0..4
        
        # Let me rebuild properly.
        rank_total = 49 + 80
        U_full = np.zeros((16, rank_total))
        V_full = np.zeros((36, rank_total))
        W_full = np.zeros((36, rank_total))  # p*n = 9*4 = 36
        
        # Left part: columns 0..48
        U_full[:, :49] = U_left
        # V_left was (16, 49) with V_left[j*4+k, col] for j in 0..3, k in 0..3
        # In full V: V[j*9+k, col]
        for j in range(4):
            for k in range(4):
                V_full[j*9+k, :49] = V_left[j*4+k, :]
        # W_left was (16, 49) with W_left[k*4+i, col] for k in 0..3, i in 0..3
        # In full W: W[k*4+i, col]
        for k in range(4):
            for i in range(4):
                W_full[k*4+i, :49] = W_left[k*4+i, :]
        
        # Right part: columns 49..128
        U_full[:, 49:] = U_right
        # V_right was (20, 80) with V_right[j*5+k, col] for j in 0..3, k in 0..4
        # In full V: V[j*9+4+k, col]
        for j in range(4):
            for k in range(5):
                V_full[j*9+4+k, 49:] = V_right[j*5+k, :]
        # W_right was (20, 80) with W_right[k*4+i, col] for k in 0..4, i in 0..3
        # In full W: W[k*4+i, col]
        for k in range(5):
            for i in range(4):
                W_full[k*4+i, 49:] = W_right[k*4+i, :]
        
        assert check_validity(U_full, V_full, W_full), "Combined 4x4*4x9 failed"
        return U_full, V_full, W_full, rank_total

    def try_optimized_rank(r, num_restarts=10):
        """Try to find a valid decomposition with rank r using random initialization + lstsq."""
        if r < 36:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(restart * 12345 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            
            # Pre-compute A: A[(u,v), c] = U[u,c] * V[v,c]
            # Shape: (n*m * m*p, r) = (16 * 36, r) = (576, r)
            A = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, r)
            
            W = np.zeros((p * n, r))
            valid = True
            
            for w in range(p * n):
                i_out = w % n  # output column index in n
                k_out = w // n  # output row index in p
                
                b = np.zeros(n * m * m * p)
                for j in range(m):
                    u = i_out * m + j
                    v = j * p + k_out
                    b[u * (m * p) + v] = 1.0
                
                try:
                    W[w, :], residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    valid = False
                    break
            
            if not valid:
                continue
            
            if check_validity(U, V, W):
                return U, V, W, r
        
        return None

    # Get schoolbook solution
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Try Strassen-based decomposition (rank 129)
    if time.time() < deadline - 20:
        try:
            U_str, V_str, W_str, rank_str = strassen_4x4_4x9()
            if rank_str < best_rank:
                best_U, best_V, best_W, best_rank = U_str, V_str, W_str, rank_str
        except Exception:
            pass
    
    # Try lower ranks via optimization
    for r in range(36, best_rank):
        if time.time() > deadline - 15:
            break
        
        result = try_optimized_rank(r, num_restarts=5)
        if result is not None:
            U, V, W, rank = result
            best_U, best_V, best_W, best_rank = U, V, W, rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
