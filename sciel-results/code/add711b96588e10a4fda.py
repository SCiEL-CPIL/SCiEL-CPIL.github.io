import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    size_A = n * m
    size_B = m * p
    size_C = p * n
    
    T = np.zeros((size_A, size_B, size_C), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * p + k
                T[a, b, c] = 1.0

    def solve_U(U, V, W):
        # U shape: (size_A, R)
        # For each a, solve M_a U_a = T_a
        # M_a has shape (size_B * size_C, R)
        # M_a[ (b,c), r ] = V[b, r] * W[c, r]
        R = U.shape[1]
        for a in range(size_A):
            # Construct M_a
            M_a = np.zeros((size_B * size_C, R))
            for b in range(size_B):
                for c in range(size_C):
                    idx = b * size_C + c
                    for r in range(R):
                        M_a[idx, r] = V[b, r] * W[c, r]
            T_a = T[a, :, :].flatten()
            # Solve M_a U_a = T_a
            # Use least squares
            U_a, _, _, _ = np.linalg.lstsq(M_a, T_a, rcond=None)
            U[a, :] = U_a
        return U

    def solve_V(U, V, W):
        R = V.shape[1]
        for b in range(size_B):
            M_b = np.zeros((size_A * size_C, R))
            for a in range(size_A):
                for c in range(size_C):
                    idx = a * size_C + c
                    for r in range(R):
                        M_b[idx, r] = U[a, r] * W[c, r]
            T_b = T[:, b, :].transpose(1, 0).flatten() # T[a,b,c] -> T_b[a,c]
            # Note: T[:, b, :] has shape (size_A, size_C). Flatten order C (row-major) means a varies slow?
            # np.flatten() is row-major. T[:, b, :] is (A, C).
            # We want M_b[idx, r] where idx corresponds to (a,c).
            # T_b should match the flattening of T[:, b, :].
            V_b, _, _, _ = np.linalg.lstsq(M_b, T_b, rcond=None)
            V[b, :] = V_b
        return V

    def solve_W(U, V, W):
        R = W.shape[1]
        for c in range(size_C):
            M_c = np.zeros((size_A * size_B, R))
            for a in range(size_A):
                for b in range(size_B):
                    idx = a * size_B + b
                    for r in range(R):
                        M_c[idx, r] = U[a, r] * V[b, r]
            T_c = T[:, :, c].transpose(1, 0).flatten() # T[a,b,c] -> T_c[a,b]
            W_c, _, _, _ = np.linalg.lstsq(M_c, T_c, rcond=None)
            W[c, :] = W_c
        return W

    def check(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_calc, T, atol=1e-9)

    def refine(U, V, W):
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        if check(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        # Try integer rounding
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if check(U_int, V_int, W_int):
            return U_int, V_int, W_int
        return None

    def search_rank(R, time_limit=10):
        start_time = time.time()
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
            U = np.random.randn(size_A, R)
            V = np.random.randn(size_B, R)
            W = np.random.randn(size_C, R)
            
            for it in range(50):
                if time.time() - start_time > time_limit:
                    break
                U = solve_U(U, V, W)
                V = solve_V(U, V, W)
                W = solve_W(U, V, W)
                
                # Check convergence or exactness
                if check(U, V, W):
                    return U, V, W
            # Try to refine
            res = refine(U, V, W)
            if res:
                return res
        return None

    # Try ranks
    for R in range(15, 25):
        res = search_rank(R, time_limit=5)
        if res:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback
    rank = n * m * p
    U = np.zeros((size_A, rank), dtype=int)
    V = np.zeros((size_B, rank), dtype=int)
    W = np.zeros((size_C, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = i * p + k
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
