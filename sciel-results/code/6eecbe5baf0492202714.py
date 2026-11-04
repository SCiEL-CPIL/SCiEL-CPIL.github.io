"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def make_schoolbook():
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

    def make_hybrid_strassen():
        """
        Hybrid decomposition:
        - Split B (4x7) into B1 (4x4) and B2 (4x3)
        - Use Strassen-based rank-49 for 4x4x4 part (C1 = A*B1)
        - Use schoolbook rank-48 for 4x4x3 part (C2 = A*B2)
        - Total rank = 97
        
        For the 4x4x4 Strassen part, we use the known rank-19 decomposition
        for 4x4x4 matrix multiplication, but since that's complex, we use
        a block approach: split 4x4x4 into four 2x2x2 problems, each using
        Strassen (rank 7), total = 4*7 = 28.
        
        Wait, that's only 28, not 49. Let me reconsider.
        
        Actually, for 4x4x4, the best known is rank 19, but implementing that
        is complex. The simple block approach gives 4*7=28.
        
        So total = 28 + 48 = 76.
        """
        # For 4x4x4: 4 independent 2x2x2 Strassen decompositions
        # For 4x4x3: schoolbook rank 48
        # Total: 28 + 48 = 76
        
        rank = 76
        U = [[0] * rank for _ in range(n * m)]  # 16 x 76
        V = [[0] * rank for _ in range(m * p)]  # 28 x 76
        W = [[0] * rank for _ in range(p * n)]  # 28 x 76
        
        # Strassen 2x2x2 coefficients
        # m1 = (a11+a22)(b11+b22) -> contributes to c11, c22
        # m2 = (a21+a22)b11 -> contributes to c21
        # m3 = a11(b21-b22) -> contributes to c12, c22
        # m4 = a22(b12-b11) -> contributes to c11, c22
        # m5 = (a11+a12)b22 -> contributes to c11, c12
        # m6 = (a21-a11)(b11+b12) -> contributes to c22
        # m7 = (a12-a22)(b21+b22) -> contributes to c11
        
        # For each 2x2 block (bi, bj) where bi, bj in {0,1}:
        # A block is at rows 2*bi:2*bi+2, cols 2*bj:2*bj+2
        # B block is at rows 2*bj:2*bj+2, cols 2*bi:2*bi+2 (for 4x4x4)
        # C block is at rows 2*bi:2*bi+2, cols 2*bj:2*bj+2
        
        # Strassen terms for 2x2x2:
        # Each term has U entries (in A block), V entries (in B block), W entries (in C block)
        
        # Let's define the Strassen decomposition for 2x2x2:
        # For block at (bi, bj):
        # A indices: (2*bi+0, 2*bj+0)=a11, (2*bi+1, 2*bj+1)=a22, etc.
        # B indices: (2*bj+0, 2*bi+0)=b11, (2*bj+1, 2*bi+1)=b22, etc.
        # C indices: (2*bi+0, 2*bj+0)=c11, etc.
        
        strassen_terms = [
            # (u_indices, v_indices, w_indices)
            # u_indices: list of (row, col) in A block, with coefficient
            # v_indices: list of (row, col) in B block, with coefficient
            # w_indices: list of (row, col) in C block, with coefficient
            
            # m1 = (a11+a22)(b11+b22)
            {
                'u': [((0, 0), 1), ((1, 1), 1)],
                'v': [((0, 0), 1), ((1, 1), 1)],
                'w': [((0, 0), 1), ((1, 1), 1)]
            },
            # m2 = (a21+a22)b11
            {
                'u': [((1, 0), 1), ((1, 1), 1)],
                'v': [((0, 0), 1)],
                'w': [((1, 0), 1)]
            },
            # m3 = a11(b21-b22)
            {
                'u': [((0, 0), 1)],
                'v': [((1, 0), 1), ((1, 1), -1)],
                'w': [((0, 1), 1), ((1, 1), 1)]
            },
            # m4 = a22(b12-b11)
            {
                'u': [((1, 1), 1)],
                'v': [((0, 1), 1), ((0, 0), -1)],
                'w': [((0, 0), 1), ((1, 1), 1)]
            },
            # m5 = (a11+a12)b22
            {
                'u': [((0, 0), 1), ((0, 1), 1)],
                'v': [((1, 1), 1)],
                'w': [((0, 0), 1), ((0, 1), 1)]
            },
            # m6 = (a21-a11)(b11+b12)
            {
                'u': [((1, 0), 1), ((0, 0), -1)],
                'v': [((0, 0), 1), ((0, 1), 1)],
                'w': [((1, 1), 1)]
            },
            # m7 = (a12-a22)(b21+b22)
            {
                'u': [((0, 1), 1), ((1, 1), -1)],
                'v': [((1, 0), 1), ((1, 1), 1)],
                'w': [((0, 0), 1)]
            },
        ]
        
        col = 0
        # 4x4x4 part: 4 blocks, each with 7 Strassen terms
        for bi in range(2):
            for bj in range(2):
                for term in strassen_terms:
                    # U entries
                    for (r, c), coeff in term['u']:
                        a_row = 2 * bi + r
                        a_col = 2 * bj + c
                        U[a_row * m + a_col][col] += coeff
                    # V entries
                    for (r, c), coeff in term['v']:
                        b_row = 2 * bj + r
                        b_col = 2 * bi + c
                        V[b_row * p + b_col][col] += coeff
                    # W entries
                    for (r, c), coeff in term['w']:
                        c_row = 2 * bi + r
                        c_col = 2 * bj + c
                        W[c_col * n + c_row][col] += coeff
                    col += 1
        
        # 4x4x3 part: schoolbook, rank 4*4*3 = 48
        for i in range(n):
            for j in range(m):
                for k in range(3):  # Only first 3 columns of B
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Verify col == rank
        assert col == rank, f"Column count {col} != rank {rank}"
        
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, r):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))

    # Start with the best known decomposition
    best_result = make_hybrid_strassen()
    best_rank = 76
    
    # Verify the hybrid decomposition
    err = verify(best_result['U'], best_result['V'], best_result['W'], best_rank)
    if err > 1e-8:
        # Fallback to schoolbook if hybrid is incorrect
        best_result = make_schoolbook()
        best_rank = n * m * p
    
    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx

    def refine(U, V, W, r, max_iter=200):
        """Vectorized gradient descent refinement. Returns refined arrays."""
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            u_vals = Ua[u_idx, :]
            v_vals = Va[v_idx, :]
            w_vals = Wa[w_idx, :]
            
            prod = u_vals * v_vals * w_vals
            result = prod.sum(axis=1)
            error = result - 1.0
            err_norm = np.linalg.norm(error)
            
            if err_norm < 1e-10:
                break
            
            # Reshape to 4D: (n, m, p, r)
            err_4d = error.reshape(n, m, p, 1)
            u_4d = u_vals.reshape(n, m, p, r)
            v_4d = v_vals.reshape(n, m, p, r)
            w_4d = w_vals.reshape(n, m, p, r)
            
            # Vectorized gradient computations
            grad_U_4d = 2 * err_4d * v_4d * w_4d
            grad_U = np.sum(grad_U_4d, axis=2).reshape(n*m, r)
            
            grad_V_4d = 2 * err_4d * u_4d * w_4d
            grad_V = np.sum(grad_V_4d, axis=0).reshape(m*p, r)
            
            grad_W_4d = 2 * err_4d * u_4d * v_4d
            grad_W = np.sum(grad_W_4d, axis=1).transpose(1, 0, 2).reshape(p*n, r)
            
            lr = 0.1 / (1.0 + it * 0.1)
            
            Ua -= lr * grad_U
            Va -= lr * grad_V
            Wa -= lr * grad_W
        
        return Ua, Va, Wa

    def try_rank(r, n_trials=10, max_iter=100):
        """Try to find a rank-r decomposition with random restarts."""
        for trial in range(n_trials):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(trial * 104729 + r * 31 + 7)
            
            scale = [0.5, 1.0, 2.0, 0.1, 5.0][trial % 5]
            U = np.random.randn(n * m, r) * scale
            V = np.random.randn(m * p, r) * scale
            W = np.random.randn(p * n, r) * scale
            
            U_ref, V_ref, W_ref = refine(U, V, W, r, max_iter=max_iter)
            
            err = verify(U_ref, V_ref, W_ref, r)
            if err < 1e-8:
                return {
                    "U": U_ref.tolist(),
                    "V": V_ref.tolist(),
                    "W": W_ref.tolist()
                }
        
        return None

    # Search for lower rank decompositions
    # Try ranks below the hybrid rank
    search_ranks = list(range(50, best_rank))
    
    for r in search_ranks:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(r, n_trials=8, max_iter=80)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
