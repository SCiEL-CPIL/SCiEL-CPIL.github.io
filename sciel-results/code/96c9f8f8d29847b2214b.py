"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 9

    def verify(U, V, W):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}

    # Build rank-32 decomposition: 4 Strassen blocks (7 each) + 1 schoolbook (4) = 32
    def build_rank32():
        rank = 32
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(k0, k1):
            """Add Strassen 2x2x2 decomposition for columns k0, k1 of B."""
            nonlocal col
            # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
            # V indices: j*p+k, so row0: k0, k1; row1: p+k0, p+k1
            # W indices: k*n+i, so col0: 2*k0, 2*k0+1; col1: 2*k1, 2*k1+1

            # M1 = (A00+A11)(B00+B11) -> C[0,k0] += 1, C[1,k1] += 1
            U[0][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            # M2 = (A10+A11)B00 -> C[1,k0] += 1, C[1,k1] -= 1
            U[2][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
            col += 1

            # M3 = A00(B01-B11) -> C[0,k1] += 1, C[1,k1] += 1
            U[0][col] = 1
            V[0 * p + k1][col] = 1; V[1 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            # M4 = A11(B10-B00) -> C[0,k0] += 1, C[1,k0] += 1
            U[3][col] = 1
            V[1 * p + k0][col] = 1; V[0 * p + k0][col] = -1
            W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
            col += 1

            # M5 = (A00+A01)B11 -> C[0,k0] -= 1, C[0,k1] += 1
            U[0][col] = 1; U[1][col] = 1
            V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
            col += 1

            # M6 = (A10-A00)(B00+B01) -> C[1,k1] += 1
            U[2][col] = 1; U[0][col] = -1
            V[0 * p + k0][col] = 1; V[0 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1

            # M7 = (A01-A11)(B10+B11) -> C[0,k0] += 1
            U[1][col] = 1; U[3][col] = -1
            V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1

        def add_trivial(k):
            """Add schoolbook 2x2x1 decomposition for column k of B."""
            nonlocal col
            U[0][col] = 1; V[0 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[1][col] = 1; V[1 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[2][col] = 1; V[0 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1
            U[3][col] = 1; V[1 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1

        add_strassen(0, 1)
        add_strassen(2, 3)
        add_strassen(4, 5)
        add_strassen(6, 7)
        add_trivial(8)

        assert col == 32, f"Expected col=32, got {col}"
        return {"U": U, "V": V, "W": W}

    # Build rank-33 decomposition: 3 blocks of (Strassen-7 + schoolbook-4) = 33
    def build_rank33():
        rank = 33
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen_block(k0, k1):
            nonlocal col
            U[0][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            U[2][col] = 1; U[3][col] = 1
            V[0 * p + k0][col] = 1
            W[k0 * n + 1][col] = 1; W[k1 * n + 1][col] = -1
            col += 1

            U[0][col] = 1
            V[0 * p + k1][col] = 1; V[1 * p + k1][col] = -1
            W[k1 * n + 0][col] = 1; W[k1 * n + 1][col] = 1
            col += 1

            U[3][col] = 1
            V[1 * p + k0][col] = 1; V[0 * p + k0][col] = -1
            W[k0 * n + 0][col] = 1; W[k0 * n + 1][col] = 1
            col += 1

            U[0][col] = 1; U[1][col] = 1
            V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = -1; W[k1 * n + 0][col] = 1
            col += 1

            U[2][col] = 1; U[0][col] = -1
            V[0 * p + k0][col] = 1; V[0 * p + k1][col] = 1
            W[k1 * n + 1][col] = 1
            col += 1

            U[1][col] = 1; U[3][col] = -1
            V[1 * p + k0][col] = 1; V[1 * p + k1][col] = 1
            W[k0 * n + 0][col] = 1
            col += 1

        def add_trivial_col(k):
            nonlocal col
            U[0][col] = 1; V[0 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[1][col] = 1; V[1 * p + k][col] = 1; W[k * n + 0][col] = 1; col += 1
            U[2][col] = 1; V[0 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1
            U[3][col] = 1; V[1 * p + k][col] = 1; W[k * n + 1][col] = 1; col += 1

        for block in range(3):
            k0 = block * 3
            k1 = block * 3 + 1
            k2 = block * 3 + 2
            add_strassen_block(k0, k1)
            add_trivial_col(k2)

        assert col == 33, f"Expected col=33, got {col}"
        return {"U": U, "V": V, "W": W}

    def try_rank31_search(time_limit=50.0):
        """
        Try to find a rank-31 decomposition by optimizing from rank-32.
        Uses scipy minimize with a smooth epigraph formulation.
        """
        start = time.time()
        result32 = build_rank32()
        U32 = np.array(result32["U"], dtype=float)
        V32 = np.array(result32["V"], dtype=float)
        W32 = np.array(result32["W"], dtype=float)

        # Precompute index pairs for tensor entries
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    indices.append((i * m + j, j * p + k, k * n + i))

        def compute_residuals(U, V, W):
            """Compute residuals for all tensor entries."""
            residuals = np.zeros(len(indices))
            for idx, (ui, vi, wi) in enumerate(indices):
                residuals[idx] = np.sum(U[ui] * V[vi] * W[wi]) - 1.0
            return residuals

        def objective_and_grad(x):
            """
            Objective: smooth epigraph of max(|residuals|) + penalty for non-zero eliminated column.
            x = [U_flat (4*31), V_flat (18*31), W_flat (18*31), t]
            """
            U = x[0:4*31].reshape(4, 31)
            V = x[4*31:4*31+18*31].reshape(18, 31)
            W = x[4*31+18*31:4*31+18*31+18*31].reshape(18, 31)
            t = x[-1]

            residuals = compute_residuals(U, V, W)

            # Smooth max via log-sum-exp
            alpha = 50.0
            max_res = np.max(residuals)
            if max_res > 1e-10:
                smooth_max = (1.0/alpha) * np.log(np.sum(np.exp(alpha * residuals)))
            else:
                smooth_max = max_res

            # Penalty for the last column (which we want to drive to zero)
            col_penalty = np.sum(U[:, 30]**2) + np.sum(V[:, 30]**2) + np.sum(W[:, 30]**2)

            obj = smooth_max + 0.01 * col_penalty + 0.1 * t**2

            # Gradient
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)

            if max_res > 1e-10:
                weights = np.exp(alpha * residuals)
                weights /= np.sum(weights)
            else:
                weights = np.zeros(len(indices))
                weights[np.argmax(residuals)] = 1.0

            for idx, (ui, vi, wi) in enumerate(indices):
                w = weights[idx]
                dU[ui] += w * V[vi] * W[wi]
                dV[vi] += w * U[ui] * W[wi]
                dW[wi] += w * U[ui] * V[vi]

            dU[:, 30] += 0.02 * U[:, 30]
            dV[:, 30] += 0.02 * V[:, 30]
            dW[:, 30] += 0.02 * W[:, 30]

            grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten(), [0.2 * t]])

            return obj, grad

        # Start from rank-32 with last column zeroed and perturbed
        np.random.seed(123)
        best_obj = float('inf')
        best_x = None

        for attempt in range(20):
            if time.time() - start > time_limit:
                break

            x0 = np.concatenate([
                U32[:, :31].flatten(),
                V32[:, :31].flatten(),
                W32[:, :31].flatten(),
                [0.0]
            ])

            # Add small perturbation
            x0[:-1] += np.random.randn(len(x0)-1) * 0.05

            try:
                res = minimize(objective_and_grad, x0, method='L-BFGS-B',
                              options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue

            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x

            if time.time() - start > time_limit - 5:
                break

        if best_x is None:
            return None

        U_opt = best_x[0:4*31].reshape(4, 31)
        V_opt = best_x[4*31:4*31+18*31].reshape(18, 31)
        W_opt = best_x[4*31+18*31:4*31+18*31+18*31].reshape(18, 31)

        # Check if solution is good enough
        residuals = compute_residuals(U_opt, V_opt, W_opt)
        max_res = np.max(np.abs(residuals))

        if max_res < 1e-6:
            # Convert to list of lists with integer/half-integer rounding
            U_list = [[0]*31 for _ in range(4)]
            V_list = [[0]*31 for _ in range(18)]
            W_list = [[0]*31 for _ in range(18)]

            for i in range(4):
                for c in range(31):
                    val = U_opt[i, c]
                    # Round to nearest half-integer
                    U_list[i][c] = round(val * 2) / 2
            for i in range(18):
                for c in range(31):
                    val = V_opt[i, c]
                    V_list[i][c] = round(val * 2) / 2
            for i in range(18):
                for c in range(31):
                    val = W_opt[i, c]
                    W_list[i][c] = round(val * 2) / 2

            # Verify the rounded solution
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # First, build and verify the rank-32 solution
    result32 = build_rank32()
    if not verify(result32["U"], result32["V"], result32["W"]):
        # Fallback to schoolbook
        return schoolbook()

    # Try to find rank-31 via scipy optimization
    if time.time() < deadline - 20:
        result31 = try_rank31_search(time_limit=50.0)
        if result31 is not None:
            return result31

    # Return the verified rank-32 solution
    return result32
# EVOLVE-BLOCK-END
