import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    # 150s wall-clock kill; keep generous safety margin
    DEADLINE = start_time + 135.0

    def now():
        return time.time()

    def remaining():
        return DEADLINE - now()

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor T[u,v,w] where u=i*m+j, v=j*p+k, w=k*n+i
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def contract(U, V, W):
        return np.einsum('ar,br,cr->abc', U, V, W, optimize=True)

    def is_exact(U, V, W, atol=1e-7):
        prod = contract(U, V, W)
        return np.allclose(prod, T, atol=atol)

    def to_output(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W],
        }

    def snap_half(arr):
        return np.round(arr * 2) / 2

    def snap_int(arr):
        return np.round(arr)

    def snap_quarter(arr):
        return np.round(arr * 4) / 4

    def try_validate_and_snap(U, V, W):
        # Try half-integers (most common for Strassen)
        U_s, V_s, W_s = snap_half(U), snap_half(V), snap_half(W)
        if is_exact(U_s, V_s, W_s):
            return U_s, V_s, W_s
        # Try integers
        U_s, V_s, W_s = snap_int(U), snap_int(V), snap_int(W)
        if is_exact(U_s, V_s, W_s):
            return U_s, V_s, W_s
        # Try quarters
        U_s, V_s, W_s = snap_quarter(U), snap_quarter(V), snap_quarter(W)
        if is_exact(U_s, V_s, W_s):
            return U_s, V_s, W_s
        return None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P7 = (A01 - A11)(B10 + B11)
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)

        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
        return U, V, W

    # Verify baseline
    U_base, V_base, W_base = build_rank46()
    if not is_exact(U_base, V_base, W_base):
        # Fallback to trivial rank 50
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return to_output(U, V, W)

    best_rank = 46
    best_result = (U_base, V_base, W_base)

    def newton_polish(U, V, W, max_iter=10, tol=1e-12):
        """High-precision Newton-style polish."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        for _ in range(max_iter):
            if now() > DEADLINE - 5:
                break
            prod = contract(U, V, W)
            diff = prod - T
            loss = float(np.sum(diff * diff))
            if loss < tol:
                break
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            grad_norm = np.sqrt(np.sum(gradU**2) + np.sum(gradV**2) + np.sum(gradW**2))
            if grad_norm < 1e-15:
                break
            
            # Adaptive step: large step if gradient is small, small if large
            # We want to converge to 0.
            step = 0.01 * min(1.0, 1e-4 / (grad_norm + 1e-8))
            
            U -= step * gradU
            V -= step * gradV
            W -= step * gradW
        return U, V, W

    def optimize_rank(rank, seed, max_iters=100):
        """Search for a solution of a specific rank."""
        if rank >= best_rank:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialization strategies
        # 1. Drop columns from baseline
        if rank < 46:
            drop_count = 46 - rank
            drop_idx = rng.choice(46, size=drop_count, replace=False)
            keep_idx = [i for i in range(46) if i not in drop_idx]
            U = U_base[:, keep_idx].astype(float).copy()
            V = V_base[:, keep_idx].astype(float).copy()
            W = W_base[:, keep_idx].astype(float).copy()
        else:
            U = U_base[:, :rank].astype(float).copy()
            V = V_base[:, :rank].astype(float).copy()
            W = W_base[:, :rank].astype(float).copy()
            
        # Add small noise to escape local minima of the dropped solution
        U += rng.randn(*U.shape) * 0.1
        V += rng.randn(*V.shape) * 0.1
        W += rng.randn(*W.shape) * 0.1

        # Adam Optimizer
        mU = np.zeros_like(U); vU = np.zeros_like(U)
        mV = np.zeros_like(V); vV = np.zeros_like(V)
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        lr = 0.02
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        best_loss = float('inf')
        
        for t in range(max_iters):
            if now() > DEADLINE - 10:
                break
                
            prod = contract(U, V, W)
            diff = prod - T
            loss = float(np.sum(diff * diff))
            
            if loss < best_loss:
                best_loss = loss
                if loss < 1e-10:
                    # Check if we can snap
                    res = try_validate_and_snap(U, V, W)
                    if res:
                        return res
                    # Try Newton polish
                    U_p, V_p, W_p = newton_polish(U, V, W, max_iter=5)
                    res = try_validate_and_snap(U_p, V_p, W_p)
                    if res:
                        return res
                    break

            gU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            tt = t + 1
            mU = beta1*mU + (1-beta1)*gU; vU = beta2*vU + (1-beta2)*gU**2
            mV = beta1*mV + (1-beta1)*gV; vV = beta2*vV + (1-beta2)*gV**2
            mW = beta1*mW + (1-beta1)*gW; vW = beta2*vW + (1-beta2)*gW**2
            
            U -= lr*(mU/(1-beta1**tt))/(np.sqrt(vU/(1-beta2**tt))+eps)
            V -= lr*(mV/(1-beta1**tt))/(np.sqrt(vV/(1-beta2**tt))+eps)
            W -= lr*(mW/(1-beta1**tt))/(np.sqrt(vW/(1-beta2**tt))+eps)
            
            lr *= 0.995
            
            # Periodic Newton polish to help convergence
            if t % 20 == 19 and loss < 1e-4:
                U, V, W = newton_polish(U, V, W, max_iter=3)

        # Final check
        res = try_validate_and_snap(U, V, W)
        if res:
            return res
            
        return None

    # Search for lower ranks
    seed = 0
    for rank in [45, 44, 43, 42, 41, 40]:
        if now() > DEADLINE - 20:
            break
        if rank >= best_rank:
            continue
            
        # Budget allocation
        # We try a few seeds for each rank
        found = None
        for s in range(5):
            if now() > DEADLINE - 15:
                break
            found = optimize_rank(rank, seed)
            seed += 1
            if found is not None:
                break
                
        if found is not None:
            U_r, V_r, W_r = found
            if is_exact(U_r, V_r, W_r):
                best_rank = rank
                best_result = (U_r, V_r, W_r)
                # If we found a lower rank, we can continue to try even lower
                # But if we found 40, we might want to stop or try 39?
                # Given time, we stick to the loop.

    U_f, V_f, W_f = best_result
    return to_output(U_f, V_f, W_f)

# EVOLVE-BLOCK-END
