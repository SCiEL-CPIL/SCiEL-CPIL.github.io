import numpy as np
import time
from scipy.optimize import least_squares, minimize

_DIMS = (3, 3, 6)
_DEADLINE = 145.0  # safety margin before 150s kill

def _make_residual(n, m, p, R):
    """Return residual(x) for tensor decomposition of T[i,j,k]=1."""
    def residual(x):
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:2*n*m*R].reshape(m*p, R)
        W = x[2*n*m*R:].reshape(p*n, R)
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return (T - 1.0).ravel()
    return residual

def _polish(x0, residual, max_nfev=5000):
    """High-precision polish using tight tolerances."""
    try:
        res = least_squares(
            residual, x0, method='trf', x_scale='jac',
            max_nfev=max_nfev, ftol=1e-18, xtol=1e-18, gtol=1e-18
        )
        return res
    except Exception:
        return None

def _check_half_int(U, V, W, n, m, p, R):
    """Verify U,V,W (as half-integers) give exact T[i,j,k]=1."""
    Ui = np.round(U * 2).astype(np.int64)
    Vi = np.round(V * 2).astype(np.int64)
    Wi = np.round(W * 2).astype(np.int64)
    U_t = Ui.reshape(n, m, R)
    V_t = Vi.reshape(m, p, R)
    W_t = Wi.reshape(p, n, R)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    return np.all(T == 8)  # 2^3 * 1

def construct():
    t0 = time.time()
    n, m, p = _DIMS
    # Schoolbook rank
    R_school = n * m * p  # 54

    def _elapsed():
        return time.time() - t0

    def _remaining():
        return _DEADLINE - _elapsed()

    def _schoolbook():
        R = R_school
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W, R

    def _kronecker_3x3x3():
        """
        Use known rank-23 decomposition for 3x3x3 (Strassen-like).
        For 3x3x6, we can do a Kronecker product with a 1x2 decomposition.
        This gives rank 23 * 2 = 46.
        Actually, let's use a simpler approach: split the 6 columns into two 3-column blocks.
        Each 3x3x3 block can be done in rank 23 (or use Strassen rank 7 for 2x2x2... no).
        
        Actually, for 3x3x3, the best known is 23. But let's use a simpler construction.
        We can view 3x3x6 as two independent 3x3x3 multiplications if we split k into two groups of 3.
        But that would give rank 2*23 = 46 if we use the rank-23 decomp for each.
        
        Let me use a different approach: use the fact that we can do 3x3x3 with rank 23
        (Landsberg's decomposition). But implementing that is complex.
        
        Alternative: use Strassen's algorithm for 2x2x2 (rank 7) and pad to 3x3.
        That gives rank 7 + 3 + 3 = 13 for 3x3x3? No, that's not right.
        
        Let me just use a simple block approach:
        Split the 6 k-values into two groups of 3.
        For each group, use a 3x3x3 decomposition.
        If I use rank 9 (schoolbook) for each, total is 18.
        But that's worse than the full schoolbook of 54? No, 18 < 54.
        Wait, schoolbook for 3x3x6 is 3*3*6 = 54.
        If I split into two 3x3x3 blocks, each needs 3*3*3=27, total 54. Same.
        
        Let me try a smarter split. What if I use rank 23 for each 3x3x3?
        I need to implement a rank-23 decomposition for 3x3x3.
        This is complex. Let me try a different approach.
        
        Actually, let me try using the fact that we can share some factors.
        For 3x3x6, we can think of it as multiplying a 3x3 matrix by two 3x3 matrices side by side.
        If we use the same U factors for both, we can save on U.
        But V and W would need to be different for each block.
        
        Let me try: use rank 23 for the "shared" part and add extra for the second block.
        This is getting complicated. Let me just try random search more aggressively.
        """
        pass

    # Try ranks from lower bounds up to schoolbook-1
    # Theoretical lower bound for 3x3x6: at least max(3*3, 3*6, 6*3) = 18
    # But tensor rank lower bound is higher. For 3x3x3 it's 19, so for 3x3x6 maybe ~38?
    # Let's be practical and try a range.
    
    best_result = None
    best_rank = R_school

    # Try various ranks, starting from lower values
    ranks_to_try = list(range(24, 54))
    ranks_to_try.sort()

    for R in ranks_to_try:
        if _remaining() < 20.0:  # Need at least 20s for final check and fallback
            break

        n_vars = (n*m + m*p + p*n) * R
        residual = _make_residual(n, m, p, R)

        found = False
        n_restarts = 0
        # More restarts for smaller ranks
        max_restarts = 120 if R <= 30 else 80 if R <= 40 else 50

        while not found and n_restarts < max_restarts:
            if _remaining() < 10.0:
                break
            n_restarts += 1

            # Different initialization strategies
            if n_restarts == 1:
                # Structured: start from a scaled-down schoolbook pattern
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                # Distribute schoolbook entries across R columns
                for idx in range(min(n*m*p, R)):
                    i, j, k = idx // (m*p), (idx // p) % m, idx % p
                    U[i*m + j, idx] = 1.0
                    V[j*p + k, idx] = 1.0
                    W[k*n + i, idx] = 1.0
                x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            elif n_restarts == 2:
                # Random with moderate scale
                scale = 0.5 + 0.5 * np.random.rand()
                x0 = np.random.uniform(-scale, scale, n_vars)
            elif n_restarts == 3:
                # Sparse random
                x0 = np.random.uniform(-1, 1, n_vars)
                mask = np.random.rand(n_vars) < 0.3
                x0[~mask] = 0.0
            elif n_restarts == 4:
                # Near-schoolbook perturbation
                U0, V0, W0, _ = _schoolbook()
                R_eff = min(R, 54)
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                U[:, :R_eff] = U0[:, :R_eff]
                V[:, :R_eff] = V0[:, :R_eff]
                W[:, :R_eff] = W0[:, :R_eff]
                U += 0.1 * np.random.randn(n*m, R)
                V += 0.1 * np.random.randn(m*p, R)
                W += 0.1 * np.random.randn(p*n, R)
                x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            elif n_restarts == 5:
                # Block-structured initialization
                # Assume R is divisible by 3, use 3 blocks of R/3
                R_block = R // 3
                U = np.zeros((n*m, R))
                V = np.zeros((m*p, R))
                W = np.zeros((p*n, R))
                for b in range(3):
                    start = b * R_block
                    end = min(start + R_block, R)
                    if end > start:
                        # Each block handles a subset of (i,j,k)
                        for idx in range(end - start):
                            i = (idx + b * 18) % 3
                            j = (idx // 3 + b * 6) % 3
                            k = (idx // 9 + b * 2) % 6
                            U[i*m + j, start + idx] = 1.0
                            V[j*p + k, start + idx] = 1.0
                            W[k*n + i, start + idx] = 1.0
                x0 = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            else:
                # Various random scales and structures
                scale = 0.3 + 1.5 * np.random.rand()
                x0 = np.random.uniform(-scale, scale, n_vars)
                # Occasionally add structure
                if np.random.rand() < 0.3:
                    # Block-diagonal-like structure
                    block_size = max(1, R // 6)
                    for b in range(6):
                        start = b * block_size
                        end = min(start + block_size, R)
                        if end > start:
                            x0[start:end] += np.random.uniform(-0.5, 0.5, end - start)

            try:
                # First optimization pass
                res = least_squares(
                    residual, x0, method='trf', x_scale='jac',
                    max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10
                )

                # If close enough, polish
                if res.cost < 1e-4:
                    res2 = _polish(res.x, residual, max_nfev=2000)
                    if res2 is not None and res2.cost < res.cost:
                        res = res2

                # Check if we have a good solution
                if res.cost < 1e-8:
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R:2*n*m*R].reshape(m*p, R)
                    W_sol = res.x[2*n*m*R:].reshape(p*n, R)

                    # Round to nearest half-integer
                    U_round = np.round(U_sol * 2) / 2.0
                    V_round = np.round(V_sol * 2) / 2.0
                    W_round = np.round(W_sol * 2) / 2.0

                    # Verify exactness
                    if _check_half_int(U_round, V_round, W_round, n, m, p, R):
                        found = True
                        best_rank = R
                        best_result = (U_round, V_round, W_round, R)
                        break
                    else:
                        # Try rounding to nearest integer too
                        U_int = np.round(U_sol).astype(float)
                        V_int = np.round(V_sol).astype(float)
                        W_int = np.round(W_sol).astype(float)
                        Ui = np.round(U_int * 2).astype(np.int64)
                        Vi = np.round(V_int * 2).astype(np.int64)
                        Wi = np.round(W_int * 2).astype(np.int64)
                        U_t = Ui.reshape(n, m, R)
                        V_t = Vi.reshape(m, p, R)
                        W_t = Wi.reshape(p, n, R)
                        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
                        if np.all(T == 8):
                            found = True
                            best_rank = R
                            best_result = (U_int, V_int, W_int, R)
                            break

            except Exception:
                pass

        if found:
            break

    # If we found a better rank, use it; otherwise schoolbook
    if best_result is not None and best_rank < R_school:
        U, V, W, R = best_result
        # Final verification
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        if np.allclose(T, 1.0, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    U, V, W, R = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
