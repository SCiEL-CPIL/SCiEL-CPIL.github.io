"""Matrix multiplication 2x3x6 - robust rank search with rational snapping."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    n, m, p = 2, 3, 6
    N = n * m  # 6
    M = m * p  # 18
    P = p * n  # 12

    def schoolbook():
        rank = n * m * p
        U = np.zeros((N, rank))
        V = np.zeros((M, rank))
        W = np.zeros((P, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    def verify_exact(U, V, W, n_s, m_s, p_s):
        """Verify decomposition using exact integer arithmetic."""
        r = U.shape[1]
        # Convert to integers by multiplying by 2 (assuming half-integers)
        # We check if 2*U, 2*V, 2*W are integers and if the equation holds.
        # Equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
        # Let U' = 2U, V' = 2V, W' = 2W. Then sum_r (U'/2)(V'/2)(W'/2) = 1 => sum_r U'V'W' = 8.
        
        U_int = np.round(U * 2)
        V_int = np.round(V * 2)
        W_int = np.round(W * 2)
        
        # Check if they are integers
        if not np.allclose(U_int, U * 2, atol=1e-9): return False
        if not np.allclose(V_int, V * 2, atol=1e-9): return False
        if not np.allclose(W_int, W * 2, atol=1e-9): return False
        
        U_int = U_int.astype(np.int64)
        V_int = V_int.astype(np.int64)
        W_int = W_int.astype(np.int64)
        
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    total = 0
                    for c in range(r):
                        total += U_int[i * m_s + j, c] * V_int[j * p_s + k, c] * W_int[k * n_s + i, c]
                    if total != 8:
                        return False
        return True

    def try_rank(rank, max_seeds=5):
        """Try to find a rank-r decomposition."""
        size_U = N * rank
        size_V = M * rank
        size_W = P * rank
        total_size = size_U + size_V + size_W

        for seed in range(max_seeds):
            if time.time() > deadline - 5:
                return None

            np.random.seed(seed * 1000 + rank)
            
            # Initialize with random values, normalized
            x0 = np.random.randn(total_size)
            U0 = x0[:size_U].reshape(N, rank)
            V0 = x0[size_U:size_U + size_V].reshape(M, rank)
            W0 = x0[size_U + size_V:].reshape(P, rank)
            
            # Normalize columns to prevent scale issues
            for c in range(rank):
                nU = np.linalg.norm(U0[:, c])
                nV = np.linalg.norm(V0[:, c])
                nW = np.linalg.norm(W0[:, c])
                if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                    scale = (nU * nV * nW) ** (1.0/3.0)
                    U0[:, c] /= scale
                    V0[:, c] /= scale
                    W0[:, c] /= scale
            
            x = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])

            # Gauss-Newton / Levenberg-Marquardt
            # Residuals: R_{i,j,k} = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1
            # We use scipy if available, otherwise a simple gradient step with line search might be too slow.
            # Since scipy is allowed, we use it for robustness.
            try:
                import scipy.optimize as opt
            except ImportError:
                return None

            def fun(x):
                U = x[:size_U].reshape(N, rank)
                V = x[size_U:size_U + size_V].reshape(M, rank)
                W = x[size_U + size_V:].reshape(P, rank)
                residuals = np.zeros(n * m * p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(rank):
                                val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                            residuals[idx] = val - 1.0
                            idx += 1
                return residuals

            try:
                res = opt.least_squares(fun, x, method='lm', max_nfev=300)
                err = np.sum(res.fun**2)
                
                if err < 1e-12:
                    U_opt = res.x[:size_U].reshape(N, rank)
                    V_opt = res.x[size_U:size_U + size_V].reshape(M, rank)
                    W_opt = res.x[size_U + size_V:].reshape(P, rank)
                    
                    # Snap to half-integers
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    if verify_exact(U_snap, V_snap, W_snap, n, m, p):
                        return U_snap, V_snap, W_snap
                        
                    # Try integers
                    U_int = np.round(U_opt)
                    V_int = np.round(V_opt)
                    W_int = np.round(W_opt)
                    if verify_exact(U_int, V_int, W_int, n, m, p):
                        return U_int, V_int, W_int
                        
            except Exception:
                pass

        return None

    # Search for lower ranks
    # Theoretical lower bound is max(nm, mp, np) = 18.
    # We try from 18 up to best_rank - 1.
    # Given the time limit, we prioritize lower ranks but limit seeds for higher ranks.
    for rank in range(18, best_rank):
        if time.time() > deadline - 10:
            break
            
        # Reduce seeds for higher ranks to save time
        seeds = 10 if rank < 25 else 5
        
        result = try_rank(rank, max_seeds=seeds)
        if result:
            U_final, V_final, W_final = result
            if rank < best_rank:
                best_rank = rank
                best_result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                # If we found a rank close to the lower bound, we can stop early
                if rank <= 19:
                    break

    return best_result
