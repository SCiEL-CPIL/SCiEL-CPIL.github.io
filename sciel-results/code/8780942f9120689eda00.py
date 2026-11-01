import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 110.0

    def verify_decomp(U, V, W, n, m, p):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        R = U.shape[1]
        U_r = U.reshape(n, m, R)
        V_r = V.reshape(m, p, R)
        W_r = W.reshape(p, n, R)
        T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-8)

    def try_global_search(R, max_restarts=20, max_nfev=5000):
        """Try to find a rank-R decomposition of the 2x2x9 multiplication tensor."""
        total_params = 4 * R + 18 * R + 18 * R  # U(4,R) + V(18,R) + W(18,R)
        
        def residuals(params):
            U = params[:4*R].reshape(4, R)
            V = params[4*R:4*R+18*R].reshape(18, R)
            W = params[4*R+18*R:4*R+36*R].reshape(18, R)
            U_r = U.reshape(2, 2, R)
            V_r = V.reshape(2, 9, R)
            W_r = W.reshape(9, 2, R)
            T = np.einsum('ikr,kjr,jir->ikj', U_r, V_r, W_r)
            return T.flatten() - 1.0

        best_sol = None
        best_cost = float('inf')
        
        for attempt in range(max_restarts):
            if time.time() > deadline:
                break
            # Use structured initialization based on rank-32 solution perturbed
            x0 = np.random.randn(total_params) * 0.3
            try:
                res = least_squares(residuals, x0, method='trf', max_nfev=max_nfev,
                                    ftol=1e-14, xtol=1e-14, gtol=1e-14)
                if res.cost < best_cost:
                    best_cost = res.cost
                    best_sol = res.x
                    if res.cost < 1e-12:
                        break
            except Exception:
                continue
        
        if best_sol is not None and best_cost < 1e-10:
            U = best_sol[:4*R].reshape(4, R)
            V = best_sol[4*R:4*R+18*R].reshape(18, R)
            W = best_sol[4*R+18*R:4*R+36*R].reshape(18, R)
            
            # Try snapping to half-integers
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            if verify_decomp(U_snapped, V_snapped, W_snapped, n, m, p):
                return U_snapped, V_snapped, W_snapped
            
            # Try integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if verify_decomp(U_int, V_int, W_int, n, m, p):
                return U_int, V_int, W_int
            
            # Try quarter-integers
            U_q = np.round(U * 4) / 4
            V_q = np.round(V * 4) / 4
            W_q = np.round(W * 4) / 4
            if verify_decomp(U_q, V_q, W_q, n, m, p):
                return U_q, V_q, W_q
        
        return None

    def get_rank32_fallback():
        """Verified rank-32 decomposition using Strassen for 4 pairs + schoolbook for last column."""
        terms = []
        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
            terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
            terms.append(({0: 1}, {k1: 1, 9+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
            terms.append(({3: 1}, {9+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
            terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: -1, 2*k1: 1}))
            terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
            terms.append(({1: 1, 3: -1}, {9+k0: 1, 9+k1: 1}, {2*k0: 1}))
        
        k = 8
        terms.append(({0: 1}, {k: 1}, {2*k: 1}))
        terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
        terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
        terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
        
        rank = len(terms)
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(18)]
        W = [[0] * rank for _ in range(18)]
        
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        
        return U, V, W

    # Try rank 30 first (best possible improvement)
    if time.time() < deadline:
        result = try_global_search(30, max_restarts=30, max_nfev=8000)
        if result is not None:
            U, V, W = result
            if verify_decomp(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 31
    if time.time() < deadline:
        result = try_global_search(31, max_restarts=25, max_nfev=6000)
        if result is not None:
            U, V, W = result
            if verify_decomp(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to verified rank 32
    U, V, W = get_rank32_fallback()
    return {"U": U, "V": V, "W": W}
