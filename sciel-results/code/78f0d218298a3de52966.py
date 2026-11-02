"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            A_c = uc.reshape(n, m)
            B_c = vc.reshape(m, p)
            # W index: W[k*n+i, c] corresponds to C[i,k]
            C_c = np.zeros((n, p))
            for idx in range(p * n):
                k, i = idx // n, idx % n
                C_c[i, k] = wc[idx]
            C_computed += C_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
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

def _build_strassen_888():
    """
    Build the Strassen-Winograd rank-62 decomposition for 8x8x8 matrix multiplication.
    This uses a recursive Strassen construction:
    - 8x8 = 2x2 blocks of 4x4
    - 4x4 = 2x2 blocks of 2x2
    - 2x2 uses Strassen rank 7
    So rank = 7 * 7 * 7 = 343? No, that's too high.
    
    Actually, the known rank for 8x8x8 is 62 (Strassen-Winograd).
    Let me implement it properly using the recursive Kronecker product approach.
    
    The rank-62 algorithm for 8x8 uses:
    - 7 Strassen terms at the top level
    - Each term involves 2x2 block matrices
    - Within each block multiplication, we use a rank-7 algorithm for 4x4? No...
    
    Actually, let me think more carefully. The standard approach:
    For 8x8, we can view it as 2x2 blocks of 4x4 matrices.
    We apply Strassen to get 7 products of 4x4 matrices.
    Each 4x4 multiplication can be done with rank 49 (Strassen twice).
    So total rank = 7 * 49 = 343. That's worse than schoolbook (512).
    
    Wait, that's not right either. Let me reconsider.
    
    The known result: rank(8,8,8) = 62.
    This is achieved by a specific construction, not simply by nesting Strassen.
    
    Let me implement the Winograd-Foias algorithm or look up the explicit factors.
    For now, let me implement a recursive Strassen that gives rank 7^3 = 343,
    and then we can try to optimize.
    
    Actually, for our purposes, let's implement the recursive Strassen construction
    which gives rank 7^k for 2^k x 2^k matrices.
    For 8x8, that's 7^3 = 343.
    
    But wait, we only need 4x4x7, not 8x8x8. Let me reconsider the embedding.
    
    Strategy: Embed 4x4x7 into 8x8x8.
    - A (4x4) -> embed into top-left 4x4 of 8x8 (rest zeros)
    - B (4x7) -> embed into top-left 4x7 of 8x8 (rest zeros)  
    - C (4x7) -> extract top-left 4x7 from 8x8 result
    
    If we have a rank-R decomposition for 8x8x8, we get rank-R for 4x4x7.
    But R=343 is worse than schoolbook 112.
    
    Alternative: Use a different embedding or a different algorithm.
    
    Let me try a different approach: Use Strassen for 4x4x4 (rank 49) and find
    a better way to handle the extra 3 columns.
    
    Actually, let me implement the recursive Strassen properly and see what rank we get.
    """
    def _strassen_recursive(n):
        """
        Build Strassen decomposition for n x n x n where n = 2^k.
        Returns U, V, W of shape (n*n, rank).
        """
        if n == 2:
            # Base case: Strassen for 2x2
            rank = 7
            U = np.zeros((4, rank))
            V = np.zeros((4, rank))
            W = np.zeros((4, rank))
            
            U[:, 0] = [1, 0, 0, 1]
            V[:, 0] = [1, 0, 0, 1]
            W[:, 0] = [1, 0, 0, 1]
            
            U[:, 1] = [0, 0, 1, 1]
            V[:, 1] = [1, 0, 0, 0]
            W[:, 1] = [0, 0, 1, 0]
            
            U[:, 2] = [1, 0, 0, 0]
            V[:, 2] = [0, 1, 0, -1]
            W[:, 2] = [0, 1, 0, 0]
            
            U[:, 3] = [0, 0, 0, 1]
            V[:, 3] = [0, 0, 1, -1]
            W[:, 3] = [0, 0, 1, 0]
            
            U[:, 4] = [1, 1, 0, 0]
            V[:, 4] = [0, 0, 0, 1]
            W[:, 4] = [0, 1, 0, 0]
            
            U[:, 5] = [0, 0, 1, -1]
            V[:, 5] = [1, 1, 0, 0]
            W[:, 5] = [0, 0, 0, 1]
            
            U[:, 6] = [0, 1, 0, -1]
            V[:, 6] = [0, 0, 1, 1]
            W[:, 6] = [1, 0, 0, 0]
            
            return U, V, W
        
        # Recursive case: n x n = 2x2 blocks of (n/2) x (n/2)
        half = n // 2
        U_half, V_half, W_half = _strassen_recursive(half)
        rank_half = U_half.shape[1]
        rank_total = rank_half * rank_half
        
        U = np.zeros((n * n, rank_total))
        V = np.zeros((n * n, rank_total))
        W = np.zeros((n * n, rank_total))
        
        for outer in range(rank_half):
            u_outer = U_half[:, outer]
            v_outer = V_half[:, outer]
            w_outer = W_half[:, outer]
            
            for inner in range(rank_half):
                col = outer * rank_half + inner
                u_inner = U_half[:, inner]
                v_inner = V_half[:, inner]
                w_inner = W_half[:, inner]
                
                # For each block position
                for bi in range(2):
                    for bj in range(2):
                        coeff_u = u_outer[bi * 2 + bj]
                        if coeff_u == 0:
                            continue
                        for x in range(half):
                            for y in range(half):
                                global_idx = (2 * bi + x) * n + (2 * bj + y)
                                U[global_idx, col] += coeff_u * u_inner[x * half + y]
                        
                        coeff_v = v_outer[bi * 2 + bj]
                        if coeff_v == 0:
                            continue
                        for x in range(half):
                            for y in range(half):
                                global_idx = (2 * bi + x) * n + (2 * bj + y)
                                V[global_idx, col] += coeff_v * v_inner[x * half + y]
                        
                        coeff_w = w_outer[bi * 2 + bj]
                        if coeff_w == 0:
                            continue
                        for x in range(half):
                            for y in range(half):
                                global_idx = (2 * bi + x) * n + (2 * bj + y)
                                W[global_idx, col] += coeff_w * w_inner[x * half + y]
        
        return U, V, W
    
    return _strassen_recursive(8)

def _build_embedding_447():
    """
    Build a decomposition for 4x4x7 by embedding into 8x8x8.
    
    Embedding:
    - A (4x4) -> top-left 4x4 of 8x8
    - B (4x7) -> top-left 4x7 of 8x8  
    - C (4x7) -> top-left 4x7 of 8x8 result
    
    The 8x8x8 decomposition has rank R.
    We extract the relevant rows/columns to get a 4x4x7 decomposition.
    """
    n, m, p = 4, 4, 7
    N = 8  # Embedded size
    
    # Get the 8x8x8 Strassen decomposition
    U8, V8, W8 = _build_strassen_888()
    R = U8.shape[1]
    
    # For each term c in the 8x8x8 decomposition:
    # U8[:, c] is an 8x8 matrix (flattened to 64)
    # We need to extract the top-left 4x4 part -> 16 elements
    # V8[:, c] is an 8x8 matrix, we need top-left 4x7 -> 28 elements
    # W8[:, c] is an 8x8 matrix, we need top-left 4x7 -> 28 elements
    
    U_out = np.zeros((n * m, R))  # 16 x R
    V_out = np.zeros((m * p, R))  # 28 x R
    W_out = np.zeros((p * n, R))  # 28 x R
    
    for c in range(R):
        # Extract U: top-left 4x4 of 8x8
        # U8 is flattened as row-major: index = i*8 + j for element (i,j)
        # We want i in [0,4), j in [0,4)
        for i in range(4):
            for j in range(4):
                idx8 = i * N + j
                idx_out = i * m + j
                U_out[idx_out, c] = U8[idx8, c]
        
        # Extract V: top-left 4x7 of 8x8
        # V8 is for B (m x p in 8x8 = 8x8)
        # We want i in [0,4), j in [0,7)
        for i in range(4):
            for j in range(7):
                idx8 = i * N + j
                idx_out = i * p + j
                V_out[idx_out, c] = V8[idx8, c]
        
        # Extract W: top-left 4x7 of 8x8
        # W8 is for C, indexed as W[k*n+i] where k is col, i is row
        # In 8x8: W8[idx8] where idx8 = k*8 + i
        # We want k in [0,7), i in [0,4)
        for k in range(7):
            for i in range(4):
                idx8 = k * N + i
                idx_out = k * n + i
                W_out[idx_out, c] = W8[idx8, c]
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}

def _build_hybrid_447():
    """
    Build a hybrid decomposition for 4x4x7.
    Uses Strassen for the first 4 columns of B, and Schoolbook for the last 3.
    Rank = 49 + 48 = 97.
    """
    n, m, p = 4, 4, 7
    rank_s = 49
    rank_sb = 48
    rank_total = rank_s + rank_sb
    
    U_total = np.zeros((n * m, rank_total))
    V_total = np.zeros((m * p, rank_total))
    W_total = np.zeros((p * n, rank_total))
    
    # Part 1: Strassen for 4x4x4
    # Build recursively
    def _strassen_444():
        def _strassen_recursive(n):
            if n == 2:
                rank = 7
                U = np.zeros((4, rank))
                V = np.zeros((4, rank))
                W = np.zeros((4, rank))
                
                U[:, 0] = [1, 0, 0, 1]
                V[:, 0] = [1, 0, 0, 1]
                W[:, 0] = [1, 0, 0, 1]
                
                U[:, 1] = [0, 0, 1, 1]
                V[:, 1] = [1, 0, 0, 0]
                W[:, 1] = [0, 0, 1, 0]
                
                U[:, 2] = [1, 0, 0, 0]
                V[:, 2] = [0, 1, 0, -1]
                W[:, 2] = [0, 1, 0, 0]
                
                U[:, 3] = [0, 0, 0, 1]
                V[:, 3] = [0, 0, 1, -1]
                W[:, 3] = [0, 0, 1, 0]
                
                U[:, 4] = [1, 1, 0, 0]
                V[:, 4] = [0, 0, 0, 1]
                W[:, 4] = [0, 1, 0, 0]
                
                U[:, 5] = [0, 0, 1, -1]
                V[:, 5] = [1, 1, 0, 0]
                W[:, 5] = [0, 0, 0, 1]
                
                U[:, 6] = [0, 1, 0, -1]
                V[:, 6] = [0, 0, 1, 1]
                W[:, 6] = [1, 0, 0, 0]
                
                return U, V, W
            
            half = n // 2
            U_half, V_half, W_half = _strassen_recursive(half)
            rank_half = U_half.shape[1]
            rank_total = rank_half * rank_half
            
            U = np.zeros((n * n, rank_total))
            V = np.zeros((n * n, rank_total))
            W = np.zeros((n * n, rank_total))
            
            for outer in range(rank_half):
                u_outer = U_half[:, outer]
                v_outer = V_half[:, outer]
                w_outer = W_half[:, outer]
                
                for inner in range(rank_half):
                    col = outer * rank_half + inner
                    u_inner = U_half[:, inner]
                    v_inner = V_half[:, inner]
                    w_inner = W_half[:, inner]
                    
                    for bi in range(2):
                        for bj in range(2):
                            coeff_u = u_outer[bi * 2 + bj]
                            if coeff_u == 0:
                                continue
                            for x in range(half):
                                for y in range(half):
                                    global_idx = (2 * bi + x) * n + (2 * bj + y)
                                    U[global_idx, col] += coeff_u * u_inner[x * half + y]
                            
                            coeff_v = v_outer[bi * 2 + bj]
                            if coeff_v == 0:
                                continue
                            for x in range(half):
                                for y in range(half):
                                    global_idx = (2 * bi + x) * n + (2 * bj + y)
                                    V[global_idx, col] += coeff_v * v_inner[x * half + y]
                            
                            coeff_w = w_outer[bi * 2 + bj]
                            if coeff_w == 0:
                                continue
                            for x in range(half):
                                for y in range(half):
                                    global_idx = (2 * bi + x) * n + (2 * bj + y)
                                    W[global_idx, col] += coeff_w * w_inner[x * half + y]
            
            return U, V, W
        
        return _strassen_recursive(4)
    
    U_s, V_s, W_s = _strassen_444()
    
    U_total[:, :rank_s] = U_s
    V_total[:16, :rank_s] = V_s
    W_total[:16, :rank_s] = W_s
    
    # Part 2: Schoolbook for 4x4x3
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_total[i * 4 + j, rank_s + col] = 1
                V_global_idx = j * 7 + (4 + k)
                V_total[V_global_idx, rank_s + col] = 1
                W_global_idx = (4 + k) * 4 + i
                W_total[W_global_idx, rank_s + col] = 1
                col += 1
                
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    # Try embedding decomposition (rank 343 for 8x8x8, but let's verify)
    try:
        result = _build_embedding_447()
        rank = len(result["U"][0])
        if rank < 112:  # Better than schoolbook
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    except Exception:
        pass
    
    # Try hybrid decomposition (rank 97)
    try:
        result = _build_hybrid_447()
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    return _build_schoolbook(n, m, p)
# EVOLVE-BLOCK-END
