import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    max_time = 140  # Safety margin before 150s kill
    
    n, m, p = 2, 3, 4
    
    # Precompute known rank-21 construction (if we can find one)
    # For 2x3x4 matrix multiplication, the tensor rank is at most 24 (naive)
    # Strassen-based gives 22. Let's try to find better.
    # Known bounds: R(2,3,4) <= 24, and there are constructions with rank 21 or less.
    # Let's use a structured approach: decompose into overlapping 2x2x2 Strassen blocks.
    
    def build_rank_21():
        """Attempt a rank-21 construction using shared Strassen terms."""
        rank = 21
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            if col >= rank:
                return
            for idx, val in u_indices:
                if idx < n * m:
                    U[idx][col] += val
            for idx, val in v_indices:
                if idx < m * p:
                    V[idx][col] += val
            for idx, val in w_indices:
                if idx < p * n:
                    W[idx][col] += val
        
        # We'll use a different decomposition strategy.
        # The 2x3x4 tensor can be seen as:
        # Sum over j=0,1,2 of A[:,j] * B[j,:]
        # Each term is a rank-1 tensor of size 2x4
        
        # Let's try: use 7 Strassen terms for the first 2x2 block (cols 0,1 of A, rows 0,1 of B)
        # Then handle the remaining parts efficiently.
        
        # Actually, let me try a direct construction based on known results.
        # For M(2,3,4), a rank-21 construction exists.
        # Let me try to construct it systematically.
        
        # Indexing: U has 6 rows (2*3), V has 12 rows (3*4), W has 8 rows (4*2)
        # U[i*m+j] = U[i*3+j] for i in 0,1; j in 0,1,2
        # V[j*p+k] = V[j*4+k] for j in 0,1,2; k in 0,1,2,3
        # W[k*n+i] = W[k*2+i] for k in 0,1,2,3; i in 0,1
        
        # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal 1 for all i,j,k
        
        # Let me try to build this using a pattern.
        # I'll use the fact that we can share some Strassen multipliers.
        
        # Block A: rows 0-1 of A, cols 0-1 of B (2x2x2 Strassen)
        # Uses 7 terms, indices 0-6
        # A[0,0]*B[0,0] etc.
        
        # Term 0: (A00+A11)(B00+B11)
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (2, 1)], 0)
        # Term 1: (A10+A11)B00
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (2, -1)], 1)
        # Term 2: A00(B10-B00)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (0, 1)], 2)
        # Term 3: A11(B01-B10)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        # Term 4: (A00+A01)B11
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        # Term 5: (A10-A00)(B00+B01)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1)], 5)
        # Term 6: (A01-A11)(B10+B11)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block B: rows 0-1 of A, cols 2-3 of B (another 2x2x2 Strassen)
        # Uses 7 terms, indices 7-13
        # But we can potentially share some terms with Block A.
        # Let's first just use 7 more terms.
        
        # Term 7: (A00+A11)(B02+B13)
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (6, 1)], 7)
        # Term 8: (A10+A11)B02
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (6, -1)], 8)
        # Term 9: A00(B12-B02)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (4, 1)], 9)
        # Term 10: A11(B03-B12)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        # Term 11: (A00+A01)B13
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        # Term 12: (A10-A00)(B02+B03)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(6, 1)], 12)
        # Term 13: (A01-A11)(B12+B13)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Now handle the remaining parts:
        # A[0,2] * B[2,:] and A[1,2] * B[2,:]
        # These are 2x4 outer products, each of rank 1 in the A dimension but 4 in B.
        # So we need 2*4 = 8 more terms if done naively, total 7+7+8 = 22.
        # To get to 21, we need to save one term.
        
        # Let's see if we can combine some terms.
        # The outer product terms:
        # A[0,2]*B[2,0], A[0,2]*B[2,1], A[0,2]*B[2,2], A[0,2]*B[2,3]
        # A[1,2]*B[2,0], A[1,2]*B[2,1], A[1,2]*B[2,2], A[1,2]*B[2,3]
        
        # Term 14: A[0,2]*B[2,0]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        # Term 15: A[0,2]*B[2,1]
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        # Term 16: A[0,2]*B[2,2]
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        # Term 17: A[0,2]*B[2,3]
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        # Term 18: A[1,2]*B[2,0]
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        # Term 19: A[1,2]*B[2,1]
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        # Term 20: A[1,2]*B[2,2]
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        # Term 21: A[1,2]*B[2,3]
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        # That's 22 terms. We only have 21 columns. So this won't work directly.
        # The above is actually the rank-22 construction, just with different indexing.
        # Let me return None to indicate this failed.
        return None
    
    # Let's just use the rank-22 construction and try to optimize.
    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val
        
        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        return U, V, W, rank
    
    # Build the proven rank-22 construction as fallback
    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val
        
        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)
        
        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)
        
        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)
        
        return U, V, W, rank
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact using vectorized numpy."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :rank], V_arr[j*p+k, :rank] * W_arr[k*n+i, :rank])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def snap_to_half_integer(arr, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    snapped[i, j] = nearest_int
                else:
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        snapped[i, j] = nearest_half
                    else:
                        snapped[i, j] = val
        return snapped
    
    def tensor_residual(U, V, W, rank):
        """Compute the maximum residual over all tensor entries."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :rank], V_arr[j*p+k, :rank] * W_arr[k*n+i, :rank])
                    max_res = max(max_res, abs(s - 1.0))
        return max_res
    
    def try_als(rank, max_iter=300, tol=1e-12, seed=42, init_U=None, init_V=None, init_W=None):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        if init_U is not None:
            U = np.array(init_U, dtype=float).copy()
            V = np.array(init_V, dtype=float).copy()
            W = np.array(init_W, dtype=float).copy()
        else:
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
        
        for iteration in range(max_iter):
            if time.time() - start_time > max_time - 5:
                return None
            
            # Solve for W: fix U, V
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U: fix V, W
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V: fix U, W
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                U_list = [[round(float(x), 10) for x in row] for row in U]
                V_list = [[round(float(x), 10) for x in row] for row in V]
                W_list = [[round(float(x), 10) for x in row] for row in W]
                if verify(U_list, V_list, W_list, rank):
                    U_snap = snap_to_half_integer(U)
                    V_snap = snap_to_half_integer(V)
                    W_snap = snap_to_half_integer(W)
                    U_list = [[round(float(x), 10) for x in row] for row in U_snap]
                    V_list = [[round(float(x), 10) for x in row] for row in V_snap]
                    W_list = [[round(float(x), 10) for x in row] for row in W_snap]
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list
                    else:
                        return U_list, V_list, W_list
        return None
    
    def polish(U, V, W, rank, max_iter=300, tol=1e-14):
        """High-precision polish pass with tighter tolerance."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        
        for iteration in range(max_iter):
            if time.time() - start_time > max_time - 3:
                break
            
            # Solve for W
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = W[w, :]
            
            # Solve for U
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = U[u, :]
            
            # Solve for V
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = V[v, :]
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                break
        
        U_list = [[round(float(x), 12) for x in row] for row in U]
        V_list = [[round(float(x), 12) for x in row] for row in V]
        W_list = [[round(float(x), 12) for x in row] for row in W]
        if verify(U_list, V_list, W_list, rank):
            return U_list, V_list, W_list
        return None
    
    def warm_start_from_rank22(target_rank, seed):
        """Create a warm start by randomly selecting columns from rank-22 solution."""
        rng = np.random.RandomState(seed)
        U22, V22, W22, _ = build_rank_22()
        U22_arr = np.array(U22, dtype=float)
        V22_arr = np.array(V22, dtype=float)
        W22_arr = np.array(W22, dtype=float)
        
        # Strategy: select a random subset of columns, then add random linear combinations
        # to try to fill in the missing information.
        cols = rng.choice(22, size=min(target_rank, 22), replace=False)
        
        if target_rank <= 22:
            U_init = U22_arr[:, cols].copy()
            V_init = V22_arr[:, cols].copy()
            W_init = W22_arr[:, cols].copy()
        else:
            # Pad with random columns
            extra = target_rank - 22
            U_init = np.hstack([U22_arr, rng.randn(6, extra) * 0.1])
            V_init = np.hstack([V22_arr, rng.randn(12, extra) * 0.1])
            W_init = np.hstack([W22_arr, rng.randn(8, extra) * 0.1])
        
        # Add small random perturbation to escape local minima
        U_init += rng.randn(*U_init.shape) * 0.05
        V_init += rng.randn(*V_init.shape) * 0.05
        W_init += rng.randn(*W_init.shape) * 0.05
        
        return U_init, V_init, W_init
    
    # Try to find lower rank decompositions using ALS
    for rank in range(17, 22):
        if time.time() - start_time > max_time - 10:
            break
        
        best_result = None
        
        # First try warm starts from rank-22
        for seed in range(40):
            if time.time() - start_time > max_time - 5:
                break
            U_init, V_init, W_init = warm_start_from_rank22(rank, seed)
            result = try_als(rank, max_iter=200, seed=seed, init_U=U_init, init_V=V_init, init_W=W_init)
            if result is not None:
                U_list, V_list, W_list = result
                if verify(U_list, V_list, W_list, rank):
                    best_result = (U_list, V_list, W_list, rank)
                    break
        
        # If warm start didn't work, try random inits
        if best_result is None:
            for seed in range(40):
                if time.time() - start_time > max_time - 5:
                    break
                result = try_als(rank, max_iter=200, seed=seed)
                if result is not None:
                    U_list, V_list, W_list = result
                    if verify(U_list, V_list, W_list, rank):
                        best_result = (U_list, V_list, W_list, rank)
                        break
        
        if best_result is not None:
            U_list, V_list, W_list, r = best_result
            # Polish
            polished = polish(U_list, V_list, W_list, r)
            if polished is not None:
                U_list, V_list, W_list = polished
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return the proven rank-22 construction
    U, V, W, rank = build_rank_22()
    return {"U": U, "V": V, "W": W}
