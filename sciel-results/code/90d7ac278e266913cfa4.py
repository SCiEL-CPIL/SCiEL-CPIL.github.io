import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110

    # Schoolbook decomposition as fallback
    def schoolbook():
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

    best = schoolbook()
    best_rank = n * m * p

    # Define the target tensor
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[(a, b, c)] = 1

    def verify_fast(U_np, V_np, W_np, rank):
        T = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U_np[:, r], V_np[:, r], W_np[:, r])
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T[a, b, c] - 1) > 1e-9:
                        return False
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target:
                        if abs(T[a, b, c]) > 1e-9:
                            return False
        return True

    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        return verify_fast(U_np, V_np, W_np, rank)

    def verify_rounded(U, V, W, rank, tol=1e-8):
        """Verify after rounding to nearest 0.5"""
        U_r = np.round(np.array(U, dtype=float) * 2) / 2
        V_r = np.round(np.array(V, dtype=float) * 2) / 2
        W_r = np.round(np.array(W, dtype=float) * 2) / 2
        return verify_fast(U_r, V_r, W_r, rank)

    def flatten_params(U, V, W, rank):
        """Flatten U, V, W into a single parameter vector"""
        return np.concatenate([
            np.array(U, dtype=float).flatten(),
            np.array(V, dtype=float).flatten(),
            np.array(W, dtype=float).flatten()
        ])

    def unflatten_params(params, rank):
        """Unflatten parameter vector back into U, V, W"""
        um_size = n * m * rank
        vm_size = m * p * rank
        U = params[:um_size].reshape(n * m, rank)
        V = params[um_size:um_size + vm_size].reshape(m * p, rank)
        W = params[um_size + vm_size:].reshape(p * n, rank)
        return U, V, W

    def tensor_residual(params, rank):
        """Compute residual for tensor decomposition"""
        U, V, W = unflatten_params(params, rank)
        # Compute the tensor
        T = np.zeros((n * m, m * p, p * n))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        # Compute residual
        residual = np.zeros_like(T)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    residual[a, b, c] = T[a, b, c] - 1
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in target:
                        residual[a, b, c] = T[a, b, c]
        return residual.flatten()

    def strassen_2x4x4():
        """Strassen-based rank 28 decomposition for 2x4x4"""
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # Strassen terms for 2x2x2
        strassen_terms = [
            # M1: (X11+X22)(Y11+Y22) -> contributes to Z11, Z22
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2: (X21+X22)Y11 -> contributes to Z21, Z22
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # M3: X11(Y12-Y22) -> contributes to Z12
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
            # M4: X22(Y21-Y11) -> contributes to Z21
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(1,0,1)]),
            # M5: (X11+X12)Y22 -> contributes to Z12, Z22
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,1,-1), (1,1,1)]),
            # M6: (X21-X11)(Y11+Y12) -> contributes to Z21
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,0,1)]),
            # M7: (X12-X22)(Y21+Y22) -> contributes to Z12
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,1,1)]),
        ]

        # 4 blocks: A11*B11->C11, A11*B12->C12, A12*B21->C11, A12*B22->C12
        blocks = [
            (0, 0, 0, 0, 0, 0),  # A11*B11->C11
            (0, 0, 0, 2, 2, 0),  # A11*B12->C12
            (0, 2, 2, 0, 0, 0),  # A12*B21->C11
            (0, 2, 2, 2, 2, 0),  # A12*B22->C12
        ]

        col = 0
        for a_ro, a_co, b_ro, b_co, c_ro, c_co in blocks:
            for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                for (ar, ac, coeff_u) in u_coeffs:
                    global_ar = a_ro + ar
                    global_ac = a_co + ac
                    U[global_ar * m + global_ac, col] += coeff_u
                for (br, bc, coeff_v) in v_coeffs:
                    global_br = b_ro + br
                    global_bc = b_co + bc
                    V[global_br * p + global_bc, col] += coeff_v
                for (cr, cc, coeff_w) in w_coeffs:
                    global_cr = c_ro + cr
                    global_cc = c_co + cc
                    W[global_cr * n + global_cc, col] += coeff_w
                col += 1

        return {"U": U, "V": V, "W": W}

    def try_rank_optimization(rank, initial_params=None, max_iters=50000):
        """Try to find a valid decomposition of given rank using optimization"""
        if time.time() > deadline - 5:
            return None
        
        if initial_params is None:
            # Initialize with schoolbook-like random
            initial_params = np.random.randn(n * m * rank + m * p * rank + p * n * rank) * 0.5
        else:
            initial_params = np.array(initial_params, dtype=float)
        
        def objective(params):
            return np.sum(tensor_residual(params, rank) ** 2)
        
        try:
            result = least_squares(
                objective,
                initial_params,
                method='trf',
                max_nfev=max_iters,
                x_scale='jac',
                ftol=1e-12,
                gtol=1e-12,
                xtol=1e-12
            )
        except Exception:
            return None
        
        if time.time() > deadline - 3:
            return None
        
        U, V, W = unflatten_params(result.x, rank)
        
        # Try rounding to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if verify_fast(U_r, V_r, W_r, rank):
            return {"U": U_r, "V": V_r, "W": W_r}
        
        # Try without rounding
        if verify_fast(U, V, W, rank):
            return {"U": U, "V": W, "W": W_r}
        
        return None

    # First try Strassen rank 28
    if time.time() < deadline - 30:
        strassen = strassen_2x4x4()
        if verify(strassen["U"], strassen["V"], strassen["W"], 28):
            best = strassen
            best_rank = 28
        else:
            # Try to fix Strassen with optimization
            params = flatten_params(strassen["U"], strassen["V"], strassen["W"], 28)
            fixed = try_rank_optimization(28, params, max_iters=20000)
            if fixed is not None:
                best = fixed
                best_rank = 28

    # Try to find rank 24 decomposition
    if best_rank > 24 and time.time() < deadline - 15:
        # Use a structured initialization
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            # Initialize with a combination of schoolbook terms
            params = np.zeros(n * m * 24 + m * p * 24 + p * n * 24)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col = (i * m + j) * 4 + (j * 4 + k) % 24
                        if col < 24:
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            params[a * 24 + col] = 1.0
                            params[(n * m * 24) + b * 24 + col] = 1.0
                            params[(n * m * 24 + m * p * 24) + c * 24 + col] = 1.0
            
            result = try_rank_optimization(24, params, max_iters=30000)
            if result is not None:
                best = result
                best_rank = 24
                break

    # Try rank 22
    if best_rank > 22 and time.time() < deadline - 10:
        params = np.random.randn(n * m * 22 + m * p * 22 + p * n * 22) * 0.3
        result = try_rank_optimization(22, params, max_iters=40000)
        if result is not None:
            best = result
            best_rank = 22

    return best
# EVOLVE-BLOCK-END
