"""Combined program: structured Strassen decomposition with search for lower rank."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, trials=200, seed=42):
    """Verify exact reconstruction of matrix multiplication."""
    rng = np.random.RandomState(seed)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C = A @ B
        
        C_recon = np.zeros((n, p))
        for c in range(rank):
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    C_recon[i,k] += s
        
        if not np.allclose(C, C_recon, atol=1e-9):
            return False
    return True


def _strassen_2x2x2(u_base, v_base, w_base, U, V, W, col):
    """Apply Strassen to a 2x2 x 2x2 block.
    u_base: dict mapping local (i,j) to global U index
    v_base: dict mapping local (j,k) to global V index
    w_base: dict mapping local (k,i) to global W index
    Returns new col after 7 Strassen terms.
    """
    # Local indices: a=(0,0), b=(0,1), c=(1,0), d=(1,1) for U
    # e=(0,0), f=(0,1), g=(1,0), h=(1,1) for V
    # W: (0,0), (0,1), (1,0), (1,1) for (k,i)
    
    a = u_base[(0,0)]
    b = u_base[(0,1)]
    c = u_base[(1,0)]
    d = u_base[(1,1)]
    
    e = v_base[(0,0)]
    f = v_base[(0,1)]
    g = v_base[(1,0)]
    h = v_base[(1,1)]
    
    w00 = w_base[(0,0)]  # C[0,0]
    w01 = w_base[(0,1)]  # C[0,1]
    w10 = w_base[(1,0)]  # C[1,0]
    w11 = w_base[(1,1)]  # C[1,1]
    
    strassen_steps = [
        # M1 = (a+d)(e+h)
        ({a: 1, d: 1}, {e: 1, h: 1}, {w00: 1, w11: 1}),
        # M2 = (c+d)e
        ({c: 1, d: 1}, {e: 1}, {w10: 1, w11: -1}),
        # M3 = a(f-h)
        ({a: 1}, {f: 1, h: -1}, {w01: 1, w11: 1}),
        # M4 = d(g-e)
        ({d: 1}, {g: 1, e: -1}, {w00: 1, w10: 1}),
        # M5 = (a+b)h
        ({a: 1, b: 1}, {h: 1}, {w00: -1, w01: 1}),
        # M6 = (c-a)(e+f)
        ({c: 1, a: -1}, {e: 1, f: 1}, {w11: 1}),
        # M7 = (b-d)(g+h)
        ({b: 1, d: -1}, {g: 1, h: 1}, {w00: 1}),
    ]
    
    for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
        for idx, val in u_coeffs.items():
            U[idx, col] = val
        for idx, val in v_coeffs.items():
            V[idx, col] = val
        for idx, val in w_coeffs.items():
            W[idx, col] = val
        col += 1
    
    return col


def _build_rank28():
    """Build the rank-28 decomposition using Strassen sub-blocks."""
    n, m, p = 2, 3, 5
    rank = 28
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: A[:,0] (2x1) * B[0,:] (1x5) - trivial, rank 10
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    # Part 2: A[:,1:3] (2x2) * B[1:3, 0:2] (2x2) - Strassen, rank 7
    u_base = {(0,0): 0*m+1, (0,1): 0*m+2, (1,0): 1*m+1, (1,1): 1*m+2}
    v_base = {(0,0): 1*p+0, (0,1): 1*p+1, (1,0): 2*p+0, (1,1): 2*p+1}
    w_base = {(0,0): 0*n+0, (0,1): 1*n+0, (1,0): 0*n+1, (1,1): 1*n+1}
    col = _strassen_2x2x2(u_base, v_base, w_base, U, V, W, col)
    
    # Part 3: A[:,1:3] (2x2) * B[1:3, 2:5] (2x3)
    # Sub-part 3a: B[1:3, 2:4] (2x2) - Strassen, rank 7
    u_base3a = {(0,0): 0*m+1, (0,1): 0*m+2, (1,0): 1*m+1, (1,1): 1*m+2}
    v_base3a = {(0,0): 1*p+2, (0,1): 1*p+3, (1,0): 2*p+2, (1,1): 2*p+3}
    w_base3a = {(0,0): 2*n+0, (0,1): 3*n+0, (1,0): 2*n+1, (1,1): 3*n+1}
    col = _strassen_2x2x2(u_base3a, v_base3a, w_base3a, U, V, W, col)
    
    # Sub-part 3b: B[1:3, 4] (2x1) - trivial, rank 4
    for i in range(n):
        for j_local in range(2):
            j_global = j_local + 1
            u_idx = i * m + j_global
            v_idx = j_global * p + 4
            w_idx = 4 * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _try_ilp_rank27(deadline):
    """Try to find a rank-27 decomposition using structured search."""
    n, m, p = 2, 3, 5
    r = 27
    
    # The tensor has 30 entries T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Try a parametric family based on the rank-28 solution with one column merged
    U28, V28, W28 = _build_rank28()
    U28_arr = np.array(U28, dtype=float)
    V28_arr = np.array(V28, dtype=float)
    W28_arr = np.array(W28, dtype=float)
    
    # Try removing one column and adjusting others
    # This is a heuristic approach - try each column removal
    for remove_col in range(28):
        if time.time() > deadline:
            break
        
        U_test = np.delete(U28_arr, remove_col, axis=1)
        V_test = np.delete(V28_arr, remove_col, axis=1)
        W_test = np.delete(W28_arr, remove_col, axis=1)
        
        # Check if this still works (it won't, but we can try to fix)
        # Instead, try a different approach: use small perturbations
        # For now, just return None if we can't find a valid rank-27
        pass
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, try to find a rank-27 decomposition
    result27 = None
    if time.time() < deadline - 30:
        result27 = _try_ilp_rank27(deadline - 30)
    
    if result27 is not None:
        U, V, W = result27
        if _verify_decomposition(U, V, W, n, m, p, trials=50, seed=123):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to rank-28 Strassen decomposition
    U, V, W = _build_rank28()
    
    # Verify the rank-28 solution
    if not _verify_decomposition(U, V, W, n, m, p, trials=100, seed=42):
        # This should not happen, but fall back to schoolbook
        rank = n * m * p
        U_school = [[0] * rank for _ in range(n * m)]
        V_school = [[0] * rank for _ in range(m * p)]
        W_school = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][column] = 1
                    V_school[j * p + k][column] = 1
                    W_school[k * n + i][column] = 1
                    column += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
