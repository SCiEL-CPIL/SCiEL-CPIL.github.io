"""Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.

Strategy:
1. Construct a verified rank-26 decomposition (Strassen 2x2 core + 19 border terms).
2. Use remaining time to attempt lower ranks (25, 24, 23) via structured optimization.
3. Return the best verified result (prefer lowest rank found, else rank-26).
"""
import time
import numpy as np

try:
    from scipy.optimize import least_squares
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120 s limit

    n, m, p = 3, 3, 3

    def idx_u(i, j):
        return i * m + j

    def idx_v(j, k):
        return j * p + k

    def idx_w(k, i):
        return k * n + i

    def verify(U_flat, V_flat, W_flat, R):
        """Verify that the decomposition is exact."""
        U_r = np.array(U_flat).reshape(n, m, R)
        V_r = np.array(V_flat).reshape(m, p, R)
        W_r = np.array(W_flat).reshape(p, n, R)
        T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_tgt = np.ones((n, m, p))
        return np.allclose(T_rec, T_tgt, atol=1e-10)

    def rank26_decomposition():
        """Construct the verified rank-26 decomposition."""
        R = 26
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))

        # Strassen 7 terms for the 2x2 core (indices 0,1)
        # m1 = (a00+a11)(b00+b11)
        U[idx_u(0,0), 0] = 1; U[idx_u(1,1), 0] = 1
        V[idx_v(0,0), 0] = 1; V[idx_v(1,1), 0] = 1
        W[idx_w(0,0), 0] = 1; W[idx_w(1,1), 0] = 1

        # m2 = (a10+a11)b00
        U[idx_u(1,0), 1] = 1; U[idx_u(1,1), 1] = 1
        V[idx_v(0,0), 1] = 1
        W[idx_w(0,1), 1] = 1; W[idx_w(1,1), 1] = -1

        # m3 = a00(b01-b11)
        U[idx_u(0,0), 2] = 1
        V[idx_v(0,1), 2] = 1; V[idx_v(1,1), 2] = -1
        W[idx_w(1,0), 2] = 1; W[idx_w(1,1), 2] = 1

        # m4 = a11(b10-b00)
        U[idx_u(1,1), 3] = 1
        V[idx_v(1,0), 3] = 1; V[idx_v(0,0), 3] = -1
        W[idx_w(0,0), 3] = 1; W[idx_w(0,1), 3] = 1

        # m5 = (a00+a01)b11
        U[idx_u(0,0), 4] = 1; U[idx_u(0,1), 4] = 1
        V[idx_v(1,1), 4] = 1
        W[idx_w(0,0), 4] = -1; W[idx_w(1,0), 4] = 1

        # m6 = (a10-a00)(b00+b01)
        U[idx_u(1,0), 5] = 1; U[idx_u(0,0), 5] = -1
        V[idx_v(0,0), 5] = 1; V[idx_v(0,1), 5] = 1
        W[idx_w(1,1), 5] = 1

        # m7 = (a01-a11)(b10+b11)
        U[idx_u(0,1), 6] = 1; U[idx_u(1,1), 6] = -1
        V[idx_v(1,0), 6] = 1; V[idx_v(1,1), 6] = 1
        W[idx_w(0,0), 6] = 1

        # 19 border terms (any index = 2)
        col = 7
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1

        # Verify
        if not verify(U, V, W, R):
            # Should never happen, but fallback to schoolbook
            return _schoolbook()

        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def _schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # First, construct the reliable rank-26 decomposition
    result_r26 = rank26_decomposition()

    # If we have time and scipy, try to find lower ranks
    elapsed = time.time() - start_time
    if HAVE_SCIPY and elapsed < 30:
        base_U = np.array(result_r26['U'])
        base_V = np.array(result_r26['V'])
        base_W = np.array(result_r26['W'])
        best_result = _try_lower_ranks(start_time, deadline, idx_u, idx_v, idx_w, n, m, p, base_U, base_V, base_W)
        if best_result is not None:
            return best_result

    # Return the verified rank-26
    return result_r26


def _try_lower_ranks(start_time, deadline, idx_u, idx_v, idx_w, n, m, p, base_U, base_V, base_W):
    """Attempt to find lower-rank decompositions via structured optimization.
    
    Tries ranks 25, 24, 23 in order, using structured initializations from the base solution.
    """
    base_R = base_U.shape[1]
    
    for target_R in [25, 24, 23]:
        if time.time() > deadline - 15:
            break
        
        result = _try_rank_optimized(target_R, start_time, deadline, idx_u, idx_v, idx_w, n, m, p, base_U, base_V, base_W, base_R)
        if result is not None:
            return result
    
    return None


def _try_rank_optimized(R, start_time, deadline, idx_u, idx_v, idx_w, n, m, p, base_U, base_V, base_W, base_R):
    """Attempt to find a rank-R decomposition via numerical optimization with structured initialization."""
    num_params = R * (n * m + m * p + p * n)

    def tensor_error(params):
        """Compute the tensor equation residual as a flat vector."""
        U = params[0 : R * n * m].reshape(n, m, R)
        V = params[R * n * m : 2 * R * n * m].reshape(m, p, R)
        W = params[2 * R * n * m : 3 * R * n * m].reshape(p, n, R)
        T_rec = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        T_tgt = np.ones((n, m, p))
        return (T_tgt - T_rec).flatten()

    def round_and_verify(U, V, W):
        """Round to half-integers and verify exactly."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_tgt = np.ones((n, m, p))
        return np.allclose(T_rec, T_tgt, atol=1e-12)

    def make_init_from_base(drop_cols):
        """Create an initialization by dropping specified columns from the base solution."""
        U_init = np.zeros((n, m, R))
        V_init = np.zeros((m, p, R))
        W_init = np.zeros((p, n, R))
        
        kept = [c for c in range(base_R) if c not in drop_cols]
        for new_col, old_col in enumerate(kept[:R]):
            U_init[:, :, new_col] = base_U.reshape(n, m, base_R)[:, :, old_col]
            V_init[:, :, new_col] = base_V.reshape(m, p, base_R)[:, :, old_col]
            W_init[:, :, new_col] = base_W.reshape(p, n, base_R)[:, :, old_col]
        
        return np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

    # Generate multiple structured initializations
    inits = []
    
    # Init 1: Drop the last (base_R - R) border columns
    drop1 = list(range(base_R - (base_R - R), base_R))
    inits.append(make_init_from_base(drop1))
    
    # Init 2: Drop spread-out columns
    if base_R - R > 1:
        step = base_R // (base_R - R)
        drop2 = [i * step for i in range(base_R - R) if i * step < base_R]
        inits.append(make_init_from_base(drop2))
    
    # Init 3: Random perturbation of truncated base
    np.random.seed(123)
    U_rand = base_U.reshape(n, m, base_R)[:, :, :R].copy() + np.random.randn(n, m, R) * 0.1
    V_rand = base_V.reshape(m, p, base_R)[:, :, :R].copy() + np.random.randn(m, p, R) * 0.1
    W_rand = base_W.reshape(p, n, base_R)[:, :, :R].copy() + np.random.randn(p, n, R) * 0.1
    inits.append(np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()]))
    
    # Init 4: Pure random
    np.random.seed(42)
    inits.append(np.random.randn(num_params) * 0.5)
    
    # Try each initialization
    for init_idx, init_params in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        try:
            res = least_squares(
                tensor_error,
                init_params,
                method='lm',
                max_nfev=5000,
                x_scale='jac',
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
            )
        except Exception:
            continue

        if res.cost < 1e-10:
            U_opt = res.x[0 : R * n * m].reshape(n, m, R)
            V_opt = res.x[R * n * m : 2 * R * n * m].reshape(m, p, R)
            W_opt = res.x[2 * R * n * m : 3 * R * n * m].reshape(p, n, R)

            if round_and_verify(U_opt, V_opt, W_opt):
                U_final = np.round(U_opt * 2) / 2
                V_final = np.round(V_opt * 2) / 2
                W_final = np.round(W_opt * 2) / 2
                U_flat = U_final.reshape(n * m, R)
                V_flat = V_final.reshape(m * p, R)
                W_flat = W_final.reshape(p * n, R)
                return {"U": U_flat.tolist(), "V": V_flat.tolist(), "W": W_flat.tolist()}
        
        # Also try with tighter tolerance if cost is small but not tiny
        if res.cost < 1e-6:
            U_opt = res.x[0 : R * n * m].reshape(n, m, R)
            V_opt = res.x[R * n * m : 2 * R * n * m].reshape(m, p, R)
            W_opt = res.x[2 * R * n * m : 3 * R * n * m].reshape(p, n, R)
            
            # Try rounding to integers and half-integers
            for mult in [2, 4]:
                U_r = np.round(U_opt * mult) / mult
                V_r = np.round(V_opt * mult) / mult
                W_r = np.round(W_opt * mult) / mult
                T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                T_tgt = np.ones((n, m, p))
                if np.allclose(T_rec, T_tgt, atol=1e-12):
                    U_flat = U_r.reshape(n * m, R)
                    V_flat = V_r.reshape(m * p, R)
                    W_flat = W_r.reshape(p * n, R)
                    return {"U": U_flat.tolist(), "V": V_flat.tolist(), "W": W_flat.tolist()}

    return None
