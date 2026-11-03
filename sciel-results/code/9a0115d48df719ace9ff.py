"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools
from scipy.linalg import solve

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit

    def verify(U, V, W, n, m, p):
        """Vectorized verification: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        r = U.shape[1]
        # Reshape: U -> (n, m, r), V -> (m, p, r), W -> (p, n, r)
        Ur = U.reshape(n, m, r)
        Vr = V.reshape(m, p, r)
        Wr = W.reshape(p, n, r)
        # Compute tensor: T[i,j,k] = sum_r Ur[i,j,r] * Vr[j,k,r] * Wr[k,i,r]
        # Use einsum
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        return np.allclose(T, 1.0, atol=1e-9)

    def schoolbook():
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
        return U, V, W

    def greedy_reduce(U, V, W, max_time=60):
        """Try to reduce rank by removing columns and redistributing their contribution.
        For each column c, check if its contribution can be absorbed by remaining columns.
        """
        U = U.copy().astype(float)
        V = V.copy().astype(float)
        W = W.copy().astype(float)
        start = time.time()
        
        def get_tensor(U, V, W):
            r = U.shape[1]
            Ur = U.reshape(n, m, r)
            Vr = V.reshape(m, p, r)
            Wr = W.reshape(p, n, r)
            return np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        
        current_T = get_tensor(U, V, W)
        target_T = np.ones((n, m, p))
        
        while time.time() - start < max_time:
            r = U.shape[1]
            if r <= n * m:  # Can't go below trivial lower bound
                break
            
            found = False
            for c in range(r):
                if time.time() - start > max_time:
                    break
                
                # Remove column c
                U_rem = np.delete(U, c, axis=1)
                V_rem = np.delete(V, c, axis=1)
                W_rem = np.delete(W, c, axis=1)
                
                # The contribution of column c to the tensor
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                # Contribution: T_c[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
                Tc = np.einsum('ij,jk,ki->ijk', Uc, Vc, Wc)
                
                # Residual that needs to be matched by remaining columns
                residual = target_T - get_tensor(U_rem, V_rem, W_rem)
                
                # Check if residual can be expressed as sum of rank-1 tensors
                # using the existing columns. This is a bilinear system.
                # For each (i,j,k): residual[i,j,k] = sum_{r != c} U_rem[i*m+j, r] * V_rem[j*p+k, r] * W_rem[k*n+i, r]
                # But we can only modify the values, not add new columns.
                # Actually we need: the residual must be in the span of the remaining rank-1 components.
                # This is complex. Instead, try a simpler heuristic:
                # Check if residual is zero (column was redundant)
                if np.allclose(Tc, 0.0, atol=1e-12):
                    # Column c contributes nothing, remove it
                    U = U_rem
                    V = V_rem
                    W = W_rem
                    found = True
                    break
                else:
                    # Try to see if we can adjust remaining columns to absorb c's contribution
                    # This is a hard problem; skip for now
                    pass
            
            if not found:
                break
        
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Winograd-style block decomposition:
    # Embed the 3x5x6 multiplication into a 5x6x6 block structure.
    # The idea: partition rows/cols into blocks and apply low-rank decompositions
    # to the 3x3 sub-blocks using a rank-4 scheme (similar to Winograd/Fourier).
    #
    # Strategy: We can decompose the 3x5 * 5x6 product by splitting the middle
    # dimension (size 5) into blocks of size 2 and 1, or by splitting the
    # output dimensions.
    #
    # A more direct approach: use the fact that for a 3x3x3 tensor, we can
    # achieve rank 4 (the Winograd scheme for 3x3). For 3x5x6, we can:
    # - Split the 5 into 3+2 and the 6 into 3+3
    # - Apply rank-4 to the 3x3x3 blocks and Strassen to 2x2x2 blocks
    #
    # Let's implement a block decomposition:
    # Split n=3 into [3] (single block), m=5 into [3,2], p=6 into [3,3]
    # This gives blocks:
    #   (3,3,3): rank 4 (Winograd)
    #   (3,3,3): rank 4 (Winograd)
    #   (3,2,3): need to handle
    #   (3,2,3): need to handle
    #
    # Actually, let's use a cleaner approach: split m=5 into [2,3] and p=6 into [3,3]
    # Blocks:
    #   (3,2,3): use Strassen-like or direct
    #   (3,2,3): use Strassen-like or direct
    #   (3,3,3): rank 4
    #   (3,3,3): rank 4
    #
    # For (3,2,3): we can split further or use a direct rank-6 approach
    #
    # Let's try a different partition: n=3, m=5=[2,3], p=6=[2,4]
    # Blocks: (3,2,2), (3,2,4), (3,3,2), (3,3,4)
    #
    # Actually, the cleanest Winograd approach for 3x5x6:
    # Use the fact that 5 = 2+3 and 6 = 3+3
    # We'll construct the decomposition block by block.

    # Rank-4 decomposition for 3x3x3 (Winograd scheme)
    # This is based on the fact that the 3x3x3 multiplication tensor has rank 4
    # when using the Winograd Fourier transform approach.
    #
    # The 3x3x3 Winograd scheme:
    # For multiplying two 3x3 matrices, we can use 4 multiplications
    # by transforming to a different basis.
    #
    # The transformation matrices for 3x3 Winograd:
    # We use the DFT-like transform for size 3.
    # The 3x3 DFT matrix (with appropriate scaling) gives the Winograd scheme.

    def winograd_3x3():
        """Return rank-4 U, V, W for 3x3x3 matrix multiplication.
        U: (9, 4), V: (9, 4), W: (9, 4)
        Convention: U[i*3+j, r], V[j*3+k, r], W[k*3+i, r]
        """
        # The 3x3 Winograd scheme uses the following approach:
        # We embed the 3x3 multiplication into a cyclic convolution structure.
        # The rank-4 decomposition corresponds to evaluating at 4 points.
        #
        # Using the Winograd algorithm for 3x3:
        # The transform matrices are based on the 3rd roots of unity and real/imag parts.
        #
        # Let me construct this explicitly.
        # The 3x3x3 tensor T[i,j,k] = 1 for all i,j,k in {0,1,2}
        # We want sum_r U[i*3+j,r] * V[j*3+k,r] * W[k*3+i,r] = 1
        #
        # One known rank-4 decomposition uses the following structure:
        # Based on the Winograd scheme, we can use:
        # r=0: U = [1,0,0, 0,0,0, 0,0,0]^T type patterns
        #
        # Let me use a direct construction.
        # The 3x3 matrix multiplication can be decomposed using the fact that:
        # C[i,k] = sum_j A[i,j] B[j,k]
        #
        # Using the Winograd approach with 4 terms:
        # We define the following rank-1 components:
        
        # Actually, let me verify: is the rank of 3x3x3 really 4?
        # The schoolbook rank is 27. The lower bound is max(9,9,9)=9.
        # The known rank for 3x3x3 is actually not 4 - that would be for a different problem.
        # The rank of the 3x3x3 matrix multiplication tensor is at least 9 (trivial)
        # and the best known is around 23 (from Strassen-style improvements).
        #
        # Wait - I was confusing this with something else. Let me reconsider.
        # The rank of the n x n x n matrix multiplication tensor is at least n^2
        # and at most n^3. For n=3, lower bound is 9, upper bound is 27.
        # The best known rank for 3x3x3 is 23 (from various improvements).
        #
        # So a rank-4 decomposition is NOT possible for 3x3x3.
        # The Winograd scheme for 3x3 convolution is different from matrix multiplication.
        #
        # Let me use a different approach: Strassen for 2x2x2 sub-blocks.

        # For now, return schoolbook for 3x3x3
        U = np.zeros((9, 27))
        V = np.zeros((9, 27))
        W = np.zeros((9, 27))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j, col] = 1
                    V[j * 3 + k, col] = 1
                    W[k * 3 + i, col] = 1
                    col += 1
        return U, V, W

    # Strassen's rank-7 decomposition for 2x2x2
    def strassen_2x2():
        """Return rank-7 U, V, W for 2x2x2 matrix multiplication.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        # Strassen's algorithm: 7 multiplications for 2x2x2
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a11+a22)(b11+b22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (a21+a22)b11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = a11(b12-b22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = a22(b21-b11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (a11+a12)b22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (a21-a11)(b11+b12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (a12-a22)(b21+b22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # Now construct the full 3x5x6 decomposition using blocks
    # Optimal split to maximize 2x2x2 Strassen blocks:
    # n=3 -> [2,1], m=5 -> [2,2,1], p=6 -> [2,2,2]
    # This gives 2*3*3 = 18 sub-blocks, of which 1*2*3 = 6 are (2,2,2)
    # Total rank: 6*7 + remaining schoolbook blocks
    # Remaining blocks: 18 - 6 = 12 blocks with at least one dimension being 1
    # Their total schoolbook rank: 90 - 6*8 = 90 - 48 = 42
    # Total: 6*7 + 42 = 42 + 42 = 84
    
    # Let's implement this block decomposition:
    
    def construct_block_decomposition():
        """Construct the 3x5x6 decomposition using optimal block partitioning.
        Split: n=[2,1], m=[2,2,1], p=[2,2,2]
        This yields 6 (2,2,2) Strassen blocks + 12 schoolbook blocks = rank 84.
        """
        n, m, p = 3, 5, 6
        
        U_list = []
        V_list = []
        W_list = []
        
        def add_block(n_block, m_block, p_block, row_offset, col_offset_A, col_offset_B):
            """Add a block decomposition for an n_block x m_block x p_block sub-problem."""
            if n_block == 2 and m_block == 2 and p_block == 2:
                # Use Strassen rank-7
                Ub, Vb, Wb = strassen_2x2()
                rank = Ub.shape[1]
                U_adj = np.zeros((n * m, rank))
                V_adj = np.zeros((m * p, rank))
                W_adj = np.zeros((p * n, rank))
                for i in range(n_block):
                    for j in range(m_block):
                        U_adj[(row_offset + i) * m + (col_offset_A + j), :] = Ub[i * m_block + j, :]
                for j in range(m_block):
                    for k in range(p_block):
                        V_adj[(col_offset_A + j) * p + (col_offset_B + k), :] = Vb[j * p_block + k, :]
                for k in range(p_block):
                    for i in range(n_block):
                        W_adj[(col_offset_B + k) * n + (row_offset + i), :] = Wb[k * n_block + i, :]
                U_list.append(U_adj)
                V_list.append(V_adj)
                W_list.append(W_adj)
            else:
                # Schoolbook
                rank = n_block * m_block * p_block
                Ub = np.zeros((n_block * m_block, rank))
                Vb = np.zeros((m_block * p_block, rank))
                Wb = np.zeros((p_block * n_block, rank))
                col = 0
                for i in range(n_block):
                    for j in range(m_block):
                        for k in range(p_block):
                            Ub[i * m_block + j, col] = 1
                            Vb[j * p_block + k, col] = 1
                            Wb[k * n_block + i, col] = 1
                            col += 1
                U_adj = np.zeros((n * m, rank))
                V_adj = np.zeros((m * p, rank))
                W_adj = np.zeros((p * n, rank))
                for i in range(n_block):
                    for j in range(m_block):
                        U_adj[(row_offset + i) * m + (col_offset_A + j), :] = Ub[i * m_block + j, :]
                for j in range(m_block):
                    for k in range(p_block):
                        V_adj[(col_offset_A + j) * p + (col_offset_B + k), :] = Vb[j * p_block + k, :]
                for k in range(p_block):
                    for i in range(n_block):
                        W_adj[(col_offset_B + k) * n + (row_offset + i), :] = Wb[k * n_block + i, :]
                U_list.append(U_adj)
                V_list.append(V_adj)
                W_list.append(W_adj)
        
        # Partition: n=[2,1], m=[2,2,1], p=[2,2,2]
        n_parts = [2, 1]
        m_parts = [2, 2, 1]
        p_parts = [2, 2, 2]
        
        n_offsets = []
        off = 0
        for sz in n_parts:
            n_offsets.append(off)
            off += sz
        
        m_offsets = []
        off = 0
        for sz in m_parts:
            m_offsets.append(off)
            off += sz
        
        p_offsets = []
        off = 0
        for sz in p_parts:
            p_offsets.append(off)
            off += sz
        
        # Iterate over all sub-blocks
        for ni, nsz in enumerate(n_parts):
            for mi, msz in enumerate(m_parts):
                for pi, psz in enumerate(p_parts):
                    add_block(nsz, msz, psz, n_offsets[ni], m_offsets[mi], p_offsets[pi])
        
        if not U_list:
            return None
        
        U_full = np.hstack(U_list)
        V_full = np.hstack(V_list)
        W_full = np.hstack(W_list)
        
        return U_full, V_full, W_full
    
    # Try the block decomposition
    if time.time() < deadline:
        try:
            U_b, V_b, W_b = construct_block_decomposition()
            if U_b is not None:
                rank_b = U_b.shape[1]
                if verify(U_b, V_b, W_b, n, m, p):
                    if rank_b < best_rank:
                        best_U, best_V, best_W = U_b, V_b, W_b
                        best_rank = rank_b
        except Exception:
            pass
    
    # Try column-merging: check if any two columns can be combined
    # by solving for a rank reduction via null-space analysis
    if time.time() < deadline - 30 and best_rank > 80:
        try:
            def try_column_merge(U, V, W, max_time=30):
                """Try to reduce rank by checking if removing a column and
                adjusting others can still satisfy the tensor equation."""
                U = U.copy().astype(float)
                V = V.copy().astype(float)
                W = W.copy().astype(float)
                start = time.time()
                r = U.shape[1]
                
                def compute_tensor(U, V, W):
                    r = U.shape[1]
                    Ur = U.reshape(n, m, r)
                    Vr = V.reshape(m, p, r)
                    Wr = W.reshape(p, n, r)
                    return np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
                
                target = np.ones((n, m, p))
                current_T = compute_tensor(U, V, W)
                
                # Try removing each column and see if the residual is in the
                # span of remaining rank-1 tensors
                for c in range(r):
                    if time.time() - start > max_time:
                        break
                    
                    U_rem = np.delete(U, c, axis=1)
                    V_rem = np.delete(V, c, axis=1)
                    W_rem = np.delete(W, c, axis=1)
                    
                    T_rem = compute_tensor(U_rem, V_rem, W_rem)
                    residual = target - T_rem
                    
                    if np.allclose(residual, 0.0, atol=1e-10):
                        # Column was redundant
                        U = U_rem
                        V = V_rem
                        W = W_rem
                        r = U.shape[1]
                        continue
                    
                    # Try to absorb column c's contribution into remaining columns
                    # This requires solving a bilinear system - skip for now
                    # but check if residual is "small" in some sense
                    pass
                
                return U, V, W
            
            U_m, V_m, W_m = try_column_merge(best_U, best_V, best_W, max_time=30)
            if verify(U_m, V_m, W_m, n, m, p):
                rank_m = U_m.shape[1]
                if rank_m < best_rank:
                    best_U, best_V, best_W = U_m, V_m, W_m
                    best_rank = rank_m
        except Exception:
            pass
    
    return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
# EVOLVE-BLOCK-END
