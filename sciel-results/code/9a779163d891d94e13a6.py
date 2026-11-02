"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_to_half_integer(x):
    """Round x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _als_pass_fast(U, V, W, n, m, p, rank, max_iters=100, tol=1e-10):
    """Fast vectorized ALS to refine U, V, W."""
    prev_err = float('inf')
    for it in range(max_iters):
        # Fix V, W, optimize U
        for i in range(n):
            for j in range(m):
                A_mat = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                b_vec = np.ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Fix U, W, optimize V
        for j in range(m):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                b_vec = np.ones(n)
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Fix U, V, optimize W
        for i in range(n):
            for k in range(p):
                A_mat = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                b_vec = np.ones(m)
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err += (s - 1.0) ** 2
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _polish_high(U, V, W, n, m, p, rank, max_iters=200):
    """High-precision polish with very tight tolerance."""
    return _als_pass_fast(U, V, W, n, m, p, rank, max_iters=max_iters, tol=1e-14)

def _winograd_init(n, m, p, rank, seed=42):
    """Winograd-style initialization exploiting block structure."""
    np.random.seed(seed)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Use 2x2 Winograd blocks as base
    # For 4x4x5, we can use two 2x2 blocks in each dimension
    
    # Define Winograd transform matrices for 2x2
    # F = [[1, 0, 0, 1], [1, 1, -1, 0], [1, -1, 0, 0], [0, 1, 0, -1]]
    # M = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # G = [[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1]]
    
    F2 = np.array([[1, 0, 0, 1], [1, 1, -1, 0], [1, -1, 0, 0], [0, 1, 0, -1]], dtype=float)
    G2 = np.array([[1, 1, 0, 0], [1, -1, 0, 0], [0, 0, 1, 1], [0, 0, 1, -1]], dtype=float)
    
    # Scale to get half-integer friendly values
    scale = 0.5
    
    for r in range(rank):
        # Each rank component handles a specific pattern
        # Use structured random selection of Winograd patterns
        
        # Randomly select which 2x2 block in each dimension
        bi = np.random.randint(0, n // 2)  # block index in i
        bj = np.random.randint(0, m // 2)  # block index in j
        bk = np.random.randint(0, p // 2 + 1)  # block index in k
        
        sign = np.random.choice([-1.0, 1.0])
        
        for i in range(n):
            for j in range(m):
                # Map i, j to 2x2 block coordinates
                li = i % 2  # local i in block
                lj = j % 2  # local j in block
                # Use F2 and G2 structure
                # U[i,j] should be related to F2[li, r_mod] * G2[lj, r_mod2]
                # Simplified: use random combination
                val = 0.0
                for a in range(4):
                    for b in range(4):
                        if (a == r % 4) or (b == r % 4):
                            val += F2[li, a] * G2[lj, b] * scale
                U[i*m+j, r] = sign * val
        
        for j in range(m):
            for k in range(p):
                lj = j % 2
                lk = k % 2
                val = 0.0
                for a in range(4):
                    val += G2[lj, a] * (1.0 if lk == a % 2 else 0.0) * scale
                V[j*p+k, r] = sign * val
        
        for k in range(p):
            for i in range(n):
                lk = k % 2
                li = i % 2
                val = 0.0
                for a in range(4):
                    val += (1.0 if lk == a % 2 else 0.0) * F2[li, a] * scale
                W[k*n+i, r] = sign * val
    
    return U, V, W

def _sparse_random_init(n, m, p, rank, seed=42, density=0.3):
    """Sparse random initialization with half-integer values."""
    np.random.seed(seed)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    half_ints = [-1.0, -0.5, 0.5, 1.0]
    
    for r in range(rank):
        for idx in range(n*m):
            if np.random.random() < density:
                U[idx, r] = np.random.choice(half_ints)
        for idx in range(m*p):
            if np.random.random() < density:
                V[idx, r] = np.random.choice(half_ints)
        for idx in range(p*n):
            if np.random.random() < density:
                W[idx, r] = np.random.choice(half_ints)
    
    return U, V, W

def _try_rank(rank, n, m, p, seed=42, max_iters=80, polish_iters=100):
    """Try to find a decomposition at given rank."""
    init_methods = [
        _winograd_init,
        _sparse_random_init,
    ]
    
    for init_method in init_methods:
        U, V, W = init_method(n, m, p, rank, seed=seed)
        
        # ALS refinement
        U, V, W = _als_pass_fast(U, V, W, n, m, p, rank, max_iters=max_iters)
        
        # High-precision polish
        U, V, W = _polish_high(U, V, W, n, m, p, rank, max_iters=polish_iters)
        
        # Round to half-integers
        U_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in U])
        V_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in V])
        W_rounded = np.array([[_round_to_half_integer(x) for x in row] for row in W])
        
        # Verify
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140 second deadline (10s safety margin)
    
    # Schoolbook decomposition as guaranteed fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j][col] = 1
                V_fb[j*p+k][col] = 1
                W_fb[k*n+i][col] = 1
                col += 1
    
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook
    
    # Try ranks from 75 down to 50
    for rank in range(75, 49, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple seeds
        found = False
        for seed in range(12):
            if time.time() > deadline - 30:
                break
            
            result = _try_rank(rank, n, m, p, seed=seed, max_iters=60, polish_iters=80)
            if result is not None:
                U, V, W = result
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                found = True
                break
        
        if found:
            # Try to go lower
            continue
    
    return best_result
# EVOLVE-BLOCK-END
