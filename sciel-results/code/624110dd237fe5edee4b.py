"""Combined program for matrix multiplication 2x3x4 tensor decomposition."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin

    def verify_np(U, V, W, n, m, p, rank, tol=1e-9):
        """Vectorized verification using NumPy."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = U[i * m + j, :]
                    v_vec = V[j * p + k, :]
                    w_vec = W[k * n + i, :]
                    s = np.dot(u_vec, v_vec * w_vec)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def verify_exact(U, V, W, n, m, p, rank):
        """Exact verification using fractions."""
        Uf = [[Fraction(int(round(x * 2)), 2) for x in row] for row in U]
        Vf = [[Fraction(int(round(x * 2)), 2) for x in row] for row in V]
        Wf = [[Fraction(int(round(x * 2)), 2) for x in row] for row in W]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += Uf[i * m + j][t] * Vf[j * p + k][t] * Wf[k * n + i][t]
                    if s != 1:
                        return False
        return True

    def build_rank_22():
        """Constructs the known rank 22 decomposition programmatically."""
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

        def strassen_block(a_i, a_j, b_j, b_k, c_i, c_k, offset):
            # M1
            add_term([(a_i[0]*m + a_j[0], 1), (a_i[1]*m + a_j[1], 1)], 
                     [(b_j[0]*p + b_k[0], 1), (b_j[1]*p + b_k[1], 1)], 
                     [(b_k[0]*n + c_i[0], 1), (b_k[1]*n + c_i[1], 1)], offset + 0)
            # M2
            add_term([(a_i[1]*m + a_j[0], 1), (a_i[1]*m + a_j[1], 1)], 
                     [(b_j[0]*p + b_k[0], 1)], 
                     [(b_k[0]*n + c_i[1], 1), (b_k[1]*n + c_i[1], -1)], offset + 1)
            # M3
            add_term([(a_i[0]*m + a_j[0], 1)], 
                     [(b_j[0]*p + b_k[1], 1), (b_j[1]*p + b_k[1], -1)], 
                     [(b_k[1]*n + c_i[0], 1), (b_k[1]*n + c_i[1], 1)], offset + 2)
            # M4
            add_term([(a_i[1]*m + a_j[1], 1)], 
                     [(b_j[1]*p + b_k[0], 1), (b_j[0]*p + b_k[0], -1)], 
                     [(b_k[0]*n + c_i[0], 1), (b_k[0]*n + c_i[1], 1)], offset + 3)
            # M5
            add_term([(a_i[0]*m + a_j[0], 1), (a_i[0]*m + a_j[1], 1)], 
                     [(b_j[1]*p + b_k[1], 1)], 
                     [(b_k[0]*n + c_i[0], -1), (b_k[1]*n + c_i[0], 1)], offset + 4)
            # M6
            add_term([(a_i[1]*m + a_j[0], 1), (a_i[0]*m + a_j[0], -1)], 
                     [(b_j[0]*p + b_k[0], 1), (b_j[0]*p + b_k[1], 1)], 
                     [(b_k[1]*n + c_i[1], 1)], offset + 5)
            # M7
            add_term([(a_i[0]*m + a_j[1], 1), (a_i[1]*m + a_j[1], -1)], 
                     [(b_j[1]*p + b_k[0], 1), (b_j[1]*p + b_k[1], 1)], 
                     [(b_k[0]*n + c_i[0], 1)], offset + 6)

        # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
        strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], 0)
        
        # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
        strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3], 7)
        
        # Block 3: A[0:2, 2:3] * B[2:3, 0:4]
        offset = 14
        for i in [0, 1]:
            for k in [0, 1, 2, 3]:
                add_term([(i * m + 2, 1)], 
                         [(2 * p + k, 1)], 
                         [(k * n + i, 1)], offset)
                offset += 1

        return U, V, W, rank

    # 1. Construct and verify the known Rank 22 solution
    U22, V22, W22, rank22 = build_rank_22()
    U22_np = np.array(U22, dtype=float)
    V22_np = np.array(V22, dtype=float)
    W22_np = np.array(W22, dtype=float)
    
    best_result = None
    best_rank = rank22
    
    if verify_np(U22_np, V22_np, W22_np, n, m, p, rank22):
        if verify_exact(U22_np, V22_np, W22_np, n, m, p, rank22):
            best_result = {"U": U22, "V": V22, "W": W22}
        else:
            best_result = {"U": [[float(x) for x in row] for row in U22], 
                           "V": [[float(x) for x in row] for row in V22], 
                           "W": [[float(x) for x in row] for row in W22]}
    else:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = rank
        return best_result

    # 2. Attempt to find a lower rank using ALS and rational polishing
    def try_als(rank, max_iter=300, tol=1e-14, seed=42, init_U=None, init_V=None, init_W=None):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        if init_U is None:
            U = rng.randn(n * m, rank) * 0.5
            V = rng.randn(m * p, rank) * 0.5
            W = rng.randn(p * n, rank) * 0.5
        else:
            U = init_U.copy()
            V = init_V.copy()
            W = init_W.copy()
            U += rng.randn(n * m, rank) * 0.01
            V += rng.randn(m * p, rank) * 0.01
            W += rng.randn(p * n, rank) * 0.01

        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A_mat, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                if verify_np(U, V, W, n, m, p, rank, tol=1e-10):
                    return U, V, W
        
        if verify_np(U, V, W, n, m, p, rank, tol=1e-10):
            return U, V, W
        
        return None

    def rational_polish(U, V, W, rank, max_iters=20, seed=0):
        """Try to find a rational (half-integer) decomposition near the given solution."""
        rng = np.random.RandomState(seed)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_np(U_r, V_r, W_r, n, m, p, rank, tol=1e-8):
                if verify_exact(U_r, V_r, W_r, n, m, p, rank):
                    return U_r, V_r, W_r
            
            scale = max(0.1, 1.0 - it / max_iters)
            U = U_r + rng.randn(n * m, rank) * 0.05 * scale
            V = V_r + rng.randn(m * p, rank) * 0.05 * scale
            W = W_r + rng.randn(p * n, rank) * 0.05 * scale
            
        return None

    # Try to find Rank 21, 20, 19
    found = False
    for target_rank in [21, 20, 19]:
        if found or time.time() > deadline - 10:
            break
            
        # Strategy: Use the Rank 22 solution as a warm start by dropping columns
        # and randomizing the rest slightly.
        
        # Try dropping 1 column for rank 21
        if target_rank == 21:
            drop_count = 1
        elif target_rank == 20:
            drop_count = 2
        else:
            drop_count = 3
            
        # Generate combinations of columns to drop
        import itertools
        drops_to_try = list(itertools.combinations(range(rank22), drop_count))
        
        # Limit the number of tries to fit in time
        max_tries = 20
        for drop_tuple in drops_to_try[:max_tries]:
            if time.time() > deadline - 10:
                break
                
            idx = [i for i in range(rank22) if i not in drop_tuple]
            U_init = U22_np[:, idx]
            V_init = V22_np[:, idx]
            W_init = W22_np[:, idx]
            
            # Add noise to allow the solver to adjust
            result = try_als(target_rank, max_iter=100, seed=hash(drop_tuple)%1000, init_U=U_init, init_V=V_init, init_W=W_init)
            
            if result is not None:
                U, V, W = result
                U_r, V_r, W_r = rational_polish(U, V, W, target_rank, max_iters=10, seed=hash(drop_tuple)%1000)
                
                if U_r is not None:
                    U_list = [[float(x) for x in row] for row in U_r]
                    V_list = [[float(x) for x in row] for row in V_r]
                    W_list = [[float(x) for x in row] for row in W_r]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break

    if found:
        return best_result

    # If lower rank not found, return the verified Rank 22 solution
    return best_result
# EVOLVE-BLOCK-END
