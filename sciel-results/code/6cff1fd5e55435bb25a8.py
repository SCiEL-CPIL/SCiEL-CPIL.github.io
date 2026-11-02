"""Program for matrix multiplication 3x4 * 4x6 with rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
n, m, p = DIMENSIONS


def _build_rank60():
    """Build a rank-60 decomposition for 3x4 x 4x6.
    
    Split the 6 output columns into two blocks of 3.
    For each block: 27 terms for 3x3x3 schoolbook + 3 terms for 3x1x3.
    Total = 2 * (27 + 3) = 60.
    """
    R = 60
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    col = 0
    for block in range(2):
        k_base = block * 3
        # 27 terms for 3x3x3 part (j in 0..2, k in k_base..k_base+2)
        for i in range(3):
            for j in range(3):
                for k_loc in range(3):
                    k = k_base + k_loc
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        # 3 terms for 3x1x3 part (j=3, k in k_base..k_base+2)
        # For 3x1x3, rank is 3 (each diagonal term is independent)
        for r in range(3):
            i = r
            k = k_base + r
            U[i * m + 3][col] = 1.0
            V[3 * p + k][col] = 1.0
            W[k * n + i][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}


def _build_schoolbook():
    """Build the schoolbook rank-72 decomposition."""
    R = n * m * p
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-6, full_zero=True):
    """Verify the decomposition exactly."""
    rank = len(U[0]) if U else 0
    
    # Check all nonzero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                if abs(val - 1.0) > tol:
                    return False
    
    # Check zero entries
    if full_zero:
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if i != i2 or j != j2 or k != k2:
                        val = 0.0
                        for r in range(rank):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if abs(val) > tol:
                            return False
    
    return True


def _try_alternating_least_squares(rank, max_iter=5000, seed=42, start_time=0.0, deadline=110.0):
    """Try to find a rank-r decomposition using alternating least squares."""
    def check_time():
        return time.time() - start_time < deadline
    
    if not check_time():
        return None
    
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Precompute all target triples
    nz = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
    
    # Build zero-triple list
    zero_triples = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                if i != i2 or j != j2 or k != k2:
                    zero_triples.append((a, b, c))
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if not check_time():
            break
        
        # Update U
        for a, b, c in nz:
            # Minimize sum over r of (U[a,r]*V[b,r]*W[c,r] - 1)^2
            # This is a quadratic in U[a,:]
            # Let v_r = V[b,r]*W[c,r]
            # Minimize sum_r (U[a,r]*v_r - 1)^2
            # d/du_r = 2*u_r*v_r^2 - 2*v_r = 0 => u_r = 1/v_r if v_r != 0
            # But we have multiple constraints, so use gradient descent
            for r in range(rank):
                v_r = V[b, r] * W[c, r]
                if abs(v_r) > 1e-12:
                    # Gradient of (sum_r U[a,r]*V[b,r]*W[c,r] - 1)^2 w.r.t U[a,r]
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    g = 2.0 * (s - 1.0) * V[b, r] * W[c, r]
                    U[a, r] -= 0.01 * g
        
        # Update V
        for a, b, c in nz:
            for r in range(rank):
                u_r = U[a, r] * W[c, r]
                if abs(u_r) > 1e-12:
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    g = 2.0 * (s - 1.0) * U[a, r] * W[c, r]
                    V[b, r] -= 0.01 * g
        
        # Update W
        for a, b, c in nz:
            for r in range(rank):
                uv_r = U[a, r] * V[b, r]
                if abs(uv_r) > 1e-12:
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    g = 2.0 * (s - 1.0) * U[a, r] * V[b, r]
                    W[c, r] -= 0.01 * g
        
        # Compute loss
        loss = 0.0
        for a, b, c in nz:
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            loss += (s - 1.0) ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if it % 500 == 0:
            U_l, V_l, W_l = best_U.tolist(), best_V.tolist(), best_W.tolist()
            if _verify(U_l, V_l, W_l, tol=1e-4, full_zero=False):
                if _verify(U_l, V_l, W_l, tol=1e-6, full_zero=True):
                    return {"U": U_l, "V": V_l, "W": W_l}
    
    return None


def construct():
    start_time = time.time()
    DEADLINE = 110.0
    
    def check_time():
        return time.time() - start_time < DEADLINE
    
    # Primary: rank-60 solution
    best = _build_rank60()
    if _verify(best["U"], best["V"], best["W"], tol=1e-6, full_zero=True):
        best_rank = 60
    else:
        # Fallback to schoolbook rank 72
        best = _build_schoolbook()
        best_rank = 72
    
    # Try to improve with numerical search
    for r in range(40, best_rank):
        if not check_time():
            break
        for seed in range(5):
            if not check_time():
                break
            res = _try_alternating_least_squares(r, max_iter=3000, seed=seed,
                                                 start_time=start_time, deadline=DEADLINE)
            if res is not None:
                if _verify(res["U"], res["V"], res["W"], tol=1e-6, full_zero=True):
                    best_rank = r
                    best = res
                    break
    
    return best
