import numpy as np
import time

n, m, p = 2, 5, 6


def construct():
    start_time = time.time()
    deadline = start_time + 140

    def check_decomposition(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def try_strassen_embedding():
        """
        Embed 2x5x6 into a 4x4x4 tensor space and try to use a rank-16 
        4x4x4 decomposition projected down. This is heuristic and may not
        yield a valid lower-rank decomposition, but we attempt it.
        
        The idea: 2x5x6 multiplication tensor T_{ijk} = 1 if the indices are
        consistent with matrix multiplication. We can embed this into a larger
        tensor and see if a known low-rank decomposition of the larger tensor
        restricts to our tensor.
        
        In practice, for this specific case, we try to find a rank-R decomposition
        by random search in a structured subspace inspired by Strassen's method.
        """
        if time.time() > deadline - 30:
            return None

        # Strassen's algorithm for 2x2x2 uses rank 7 instead of 8.
        # For larger matrices, recursive Strassen can help, but the asymptotic
        # rank is n^log2(7). For 2x5x6, this is not directly applicable in a
        # simple way.
        
        # Instead, let's try a more direct approach: random search with
        # structured initialization based on Kronecker products of 2x2x2 rank-7.
        
        # Strassen's 2x2x2 rank-7 decomposition:
        # M1 = (A11-A22)*(B22-B11)
        # M2 = (A11+A12)*B22
        # M3 = (A21+A22)*B11
        # M4 = A11*(B11-B22)
        # M5 = A22*(B22-B11)
        # M6 = (A11-A21)*(B11+B12)
        # M7 = (A21-A22)*(B21+B22)
        # C11 = M1 + M4 + M5 - M7
        # C12 = M2 + M3
        # C21 = M6 + M7
        # C22 = M1 - M4 + M2 + M5
        
        # This is for 2x2x2. We can try to use this as a building block.
        
        # For 2x5x6, we can partition the matrices into 2x2 blocks where possible.
        # But 5 and 6 are not powers of 2, so this is tricky.
        
        # Let's try a different approach: use the fact that the rank of the
        # tensor is at most the product of the ranks of the constituent matrices
        # under Kronecker product structure.
        
        # Actually, let's just do a more intensive random search with better
        # initialization and more restarts, since the Strassen embedding is
        # not straightforward for non-power-of-2 dimensions.
        
        pass

    def try_rank_optimization(R, num_restarts=50):
        """
        Try to find a rank-R decomposition using Nelder-Mead optimization
        on the sum of squared residuals.
        """
        if time.time() > deadline - 30:
            return None

        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R

        def objective(x):
            U = x[:size_U].reshape((n * m, R))
            V = x[size_U:size_U + size_V].reshape((m * p, R))
            W = x[size_U + size_V:].reshape((p * n, R))

            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                        residual += (val - 1.0) ** 2
            return residual

        from scipy.optimize import minimize

        for restart in range(num_restarts):
            if time.time() > deadline - 30:
                return None

            # Random initialization
            x0 = np.random.randn(size_U + size_V + size_W) * 0.5

            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
            except Exception:
                continue

            if result.fun < 1e-10:
                U = result.x[:size_U].reshape((n * m, R))
                V = result.x[size_U:size_U + size_V].reshape((m * p, R))
                W = result.x[size_U + size_V:].reshape((p * n, R))

                # Snap to half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2

                if check_decomposition(U_int, V_int, W_int):
                    return U_int, V_int, W_int

                # Try rounding to integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)

                if check_decomposition(U_int, V_int, W_int):
                    return U_int, V_int, W_int

        return None

    # Try ranks from 59 down to 50
    for R in range(59, 49, -1):
        if time.time() > deadline - 40:
            break
        result = try_rank_optimization(R, num_restarts=20)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
