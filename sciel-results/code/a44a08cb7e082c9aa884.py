import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    num_u = n * m
    num_v = m * p
    num_w = p * n

    # Precompute index mappings
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i * m + j)
                v_indices.append(j * p + k)
                w_indices.append(k * n + i)
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)

    def error_func(params, R):
        U = params[:R * num_u].reshape(R, num_u).T
        V = params[R * num_u:R * (num_u + num_v)].reshape(R, num_v).T
        W = params[R * (num_u + num_v):].reshape(R, num_w).T

        # Vectorized computation
        # For each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this as:
        # For each r, compute U[u_idx] * V[v_idx] * W[w_idx] and sum over r
        # More efficiently: for each r, compute element-wise product and sum
        # Or: reshape and use einsum-like operations
        
        # Let's do it in a loop over r for clarity, but vectorized over (i,j,k)
        total = np.zeros(n * m * p)
        for r in range(R):
            # U[u_indices, r] * V[v_indices, r] * W[w_indices, r]
            total += U[u_indices, r] * V[v_indices, r] * W[w_indices, r]
        
        return total - 1.0

    def verify_solution(U, V, W, R):
        """Check if the solution is valid (sum equals 1 for all i,j,k)"""
        errors = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    errors[idx] = val - 1.0
                    idx += 1
        return np.max(np.abs(errors)) < 1e-9, np.max(np.abs(errors))

    def snap_to_half_integers(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        return U_snap, V_snap, W_snap

    # Target ranks to try
    target_ranks = [41, 40, 39, 38]

    for R in target_ranks:
        if time.time() > deadline - 5:
            break

        total_vars = R * (num_u + num_v + num_w)

        # Try multiple random initializations
        max_restarts = 30
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break

            np.random.seed(10000 + R * 100 + restart)

            # Different initialization strategies
            if restart % 3 == 0:
                # Small random
                x0 = np.random.randn(total_vars) * 0.1
            elif restart % 3 == 1:
                # Medium random
                x0 = np.random.randn(total_vars) * 0.5
            else:
                # Sparse random
                x0 = np.zeros(total_vars)
                num_nonzero = min(total_vars // 2, 150)
                nonzero_idx = np.random.choice(total_vars, size=num_nonzero, replace=False)
                x0[nonzero_idx] = np.random.randn(num_nonzero) * 0.3

            try:
                res = least_squares(error_func, x0, args=(R,), method='lm',
                                   max_nfev=5000, ftol=1e-14, xtol=1e-14)
            except Exception:
                continue

            if res.cost < 1e-12:
                U = res.x[:R * num_u].reshape(R, num_u).T
                V = res.x[R * num_u:R * (num_u + num_v)].reshape(R, num_v).T
                W = res.x[R * (num_u + num_v):].reshape(R, num_w).T

                # Try to snap to half-integers
                U_snap, V_snap, W_snap = snap_to_half_integers(U, V, W)

                # Check if snapped solution is still valid
                valid_snapped, max_err = verify_solution(U_snap, V_snap, W_snap, R)
                if valid_snapped:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }

                # If snapping fails but the original is very close to half-integers,
                # check if the original solution itself is valid
                valid_orig, max_err_orig = verify_solution(U, V, W, R)
                if valid_orig and max_err_orig < 1e-10:
                    # Check proximity to half-integers
                    if (np.max(np.abs(U - U_snap)) < 1e-8 and
                        np.max(np.abs(V - V_snap)) < 1e-8 and
                        np.max(np.abs(W - W_snap)) < 1e-8):
                        return {
                            "U": U_snap.tolist(),
                            "V": V_snap.tolist(),
                            "W": W_snap.tolist()
                        }

    # Fallback to schoolbook rank 42
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
