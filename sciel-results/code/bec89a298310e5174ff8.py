"""Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.

Strategy:
1. Construct a verified rank-26 decomposition (Strassen 2x2 core + 19 border terms).
2. Use remaining time to attempt lower ranks (25, 24, 23) via structured optimization.
3. Return the best verified result (prefer lowest rank found, else rank-26).
"""
import time
import numpy as np
from itertools import product

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

    def _try_rank23_structured():
        """Try to find a rank-23 decomposition using the known structure.
        
        The 3x3x3 matmul tensor T[i,j,k] = 1 for all i,j,k in {0,1,2}.
        We use 7 Strassen terms for the 2x2 core, plus 16 additional terms.
        The additional terms must satisfy:
          sum_{r=7}^{22} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1 - sum_{r=0}^{6} U[i,j,r]*V[j,k,r]*W[k,i,r]
        for all i,j,k.
        
        For the 2x2 core (i,j,k in {0,1}), the RHS is 0 (Strassen handles it).
        For border positions, the RHS is the correction needed.
        """
        R = 23
        # Build the Strassen part
        U_s = np.zeros((n, m, 7))
        V_s = np.zeros((m, p, 7))
        W_s = np.zeros((p, n, 7))
        
        U_s[0,0,0]=1; U_s[1,1,0]=1
        V_s[0,0,0]=1; V_s[1,1,0]=1
        W_s[0,0,0]=1; W_s[1,1,0]=1
        
        U_s[1,0,1]=1; U_s[1,1,1]=1
        V_s[0,0,1]=1
        W_s[0,1,1]=1; W_s[1,1,1]=-1
        
        U_s[0,0,2]=1
        V_s[0,1,2]=1; V_s[1,1,2]=-1
        W_s[1,0,2]=1; W_s[1,1,2]=1
        
        U_s[1,1,3]=1
        V_s[1,0,3]=1; V_s[0,0,3]=-1
        W_s[0,0,3]=1; W_s[0,1,3]=1
        
        U_s[0,0,4]=1; U_s[0,1,4]=1
        V_s[1,1,4]=1
        W_s[0,0,4]=-1; W_s[1,0,4]=1
        
        U_s[1,0,5]=1; U_s[0,0,5]=-1
        V_s[0,0,5]=1; V_s[0,1,5]=1
        W_s[1,1,5]=1
        
        U_s[0,1,6]=1; U_s[1,1,6]=-1
        V_s[1,0,6]=1; V_s[1,1,6]=1
        W_s[0,0,6]=1
        
        # Compute the border correction needed
        # T_needed[i,j,k] = 1 - sum_r U_s[i,j,r]*V_s[j,k,r]*W_s[k,i,r]
        T_needed = np.ones((n, m, p))
        T_strassen = np.einsum('ijr,jkr,kir->ijk', U_s, V_s, W_s)
        T_needed -= T_strassen
        
        # For the 2x2 core, T_needed should be 0
        # For border positions, T_needed is what the 16 additional terms must produce
        
        # The 16 additional terms: each is a rank-1 tensor U_a ⊗ V_a ⊗ W_a
        # We need: sum_{a=0}^{15} U_a[i,j] * V_a[j,k] * W_a[k,i] = T_needed[i,j,k]
        
        # Strategy: Use structured initialization where each additional term
        # is supported on a specific subset of border positions.
        
        # For each border position (i,j,k) where at least one index is 2,
        # we need the sum of products to equal T_needed[i,j,k].
        
        # Let's try a direct numerical optimization approach with good initialization.
        n_add = R - 7  # 16 additional terms
        
        def tensor_error_add(params):
            """Residual for the additional 16 terms only."""
            U_a = params[0 : n_add*n*m].reshape(n, m, n_add)
            V_a = params[n_add*n*m : 2*n_add*n*m].reshape(m, p, n_add)
            W_a = params[2*n_add*n*m : 3*n_add*n*m].reshape(p, n, n_add)
            T_rec = np.einsum('ijr,jkr,kir->ijk', U_a, V_a, W_a)
            return (T_needed - T_rec).flatten()
        
        # Generate structured initializations
        inits = []
        
        # Init 1: Each additional term covers one "type" of border position
        # Type A: i=2, j<2, k<2 (3 positions: (2,0,0),(2,1,0),(2,0,1),(2,1,1)) - actually 4
        # Type B: i<2, j=2, k<2 (4 positions)
        # Type C: i<2, j<2, k=2 (4 positions)  
        # Type D: two indices = 2 (3 positions: (2,2,0),(2,0,2),(0,2,2))
        # Type E: three indices = 2 (1 position: (2,2,2))
        # Total: 4+4+4+3+1 = 16
        
        # Assign each of the 16 terms to cover one border position type
        border_positions = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i < 2 and j < 2 and k < 2:
                        continue
                    border_positions.append((i,j,k))
        
        # Sort border positions by "type" for structured init
        def border_type(i,j,k):
            return (i==2, j==2, k==2)
        
        sorted_bp = sorted(border_positions, key=lambda x: (border_type(*x), x))
        
        # Init 1: sparse - each term initialized to cover its assigned position
        U_init = np.zeros((n, m, n_add))
        V_init = np.zeros((m, p, n_add))
        W_init = np.zeros((p, n, n_add))
        for a, (i,j,k) in enumerate(sorted_bp):
            U_init[i,j,a] = 1.0
            V_init[j,k,a] = 1.0
            W_init[k,i,a] = 1.0
        inits.append(np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()]))
        
        # Init 2: Random perturbation
        np.random.seed(42)
        U_r = U_init.copy() + np.random.randn(n, m, n_add) * 0.5
        V_r = V_init.copy() + np.random.randn(m, p, n_add) * 0.5
        W_r = W_init.copy() + np.random.randn(p, n, n_add) * 0.5
        inits.append(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
        
        # Init 3: Pure random
        np.random.seed(123)
        inits.append(np.random.randn(3 * n_add * n * m) * 0.3)
        
        # Init 4: Sparse with half-integer values
        U_init2 = np.zeros((n, m, n_add))
        V_init2 = np.zeros((m, p, n_add))
        W_init2 = np.zeros((p, n, n_add))
        for a, (i,j,k) in enumerate(sorted_bp):
            U_init2[i,j,a] = 0.5
            V_init2[j,k,a] = 0.5
            W_init2[k,i,a] = 2.0  # 0.5*0.5*2 = 0.5, need to check
        inits.append(np.concatenate([U_init2.flatten(), V_init2.flatten(), W_init2.flatten()]))
        
        best_result = None
        for init_idx, init_params in enumerate(inits):
            if time.time() > deadline - 20:
                break
            
            try:
                res = least_squares(
                    tensor_error_add,
                    init_params,
                    method='lm',
                    max_nfev=8000,
                    x_scale='jac',
                    ftol=1e-14,
                    xtol=1e-14,
                    gtol=1e-14,
                )
            except Exception:
                continue
            
            if res.cost < 1e-12:
                U_a = res.x[0 : n_add*n*m].reshape(n, m, n_add)
                V_a = res.x[n_add*n*m : 2*n_add*n*m].reshape(m, p, n_add)
                W_a = res.x[2*n_add*n*m : 3*n_add*n*m].reshape(p, n, n_add)
                
                # Try rounding to half-integers
                for mult in [2, 4, 1]:
                    U_r = np.round(U_a * mult) / mult
                    V_r = np.round(V_a * mult) / mult
                    W_r = np.round(W_a * mult) / mult
                    
                    T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    if np.allclose(T_rec, T_needed, atol=1e-12):
                        # Build full decomposition
                        U_full = np.hstack([U_s, U_r])
                        V_full = np.hstack([V_s, V_r])
                        W_full = np.hstack([W_s, W_r])
                        
                        U_flat = U_full.reshape(n*m, R)
                        V_flat = V_full.reshape(m*p, R)
                        W_flat = W_full.reshape(p*n, R)
                        
                        if verify(U_flat, V_flat, W_flat, R):
                            return {"U": U_flat.tolist(), "V": V_flat.tolist(), "W": W_flat.tolist()}
        
        return None
    
    def _try_rank_optimized(R, start_time, deadline, idx_u, idx_v, idx_w, n, m, p, base_U, base_V, base_W, base_R):
        """Attempt to find a rank-R decomposition via numerical optimization."""
        num_params = R * (n * m + m * p + p * n)

        def tensor_error(params):
            U = params[0 : R * n * m].reshape(n, m, R)
            V = params[R * n * m : 2 * R * n * m].reshape(m, p, R)
            W = params[2 * R * n * m : 3 * R * n * m].reshape(p, n, R)
            T_rec = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            T_tgt = np.ones((n, m, p))
            return (T_tgt - T_rec).flatten()

        def round_and_verify(U, V, W):
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            T_rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
            T_tgt = np.ones((n, m, p))
            return np.allclose(T_rec, T_tgt, atol=1e-12)

        def make_init_from_base(drop_cols):
            U_init = np.zeros((n, m, R))
            V_init = np.zeros((m, p, R))
            W_init = np.zeros((p, n, R))
            kept = [c for c in range(base_R) if c not in drop_cols]
            for new_col, old_col in enumerate(kept[:R]):
                U_init[:, :, new_col] = base_U.reshape(n, m, base_R)[:, :, old_col]
                V_init[:, :, new_col] = base_V.reshape(m, p, base_R)[:, :, old_col]
                W_init[:, :, new_col] = base_W.reshape(p, n, base_R)[:, :, old_col]
            return np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

        inits = []
        drop1 = list(range(base_R - (base_R - R), base_R))
        inits.append(make_init_from_base(drop1))
        
        if base_R - R > 1:
            step = base_R // (base_R - R)
            drop2 = [i * step for i in range(base_R - R) if i * step < base_R]
            inits.append(make_init_from_base(drop2))
        
        np.random.seed(123)
        U_rand = base_U.reshape(n, m, base_R)[:, :, :R].copy() + np.random.randn(n, m, R) * 0.1
        V_rand = base_V.reshape(m, p, base_R)[:, :, :R].copy() + np.random.randn(m, p, R) * 0.1
        W_rand = base_W.reshape(p, n, base_R)[:, :, :R].copy() + np.random.randn(p, n, R) * 0.1
        inits.append(np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()]))
        
        np.random.seed(42)
        inits.append(np.random.randn(num_params) * 0.5)
        
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
        
            if res.cost < 1e-6:
                U_opt = res.x[0 : R * n * m].reshape(n, m, R)
                V_opt = res.x[R * n * m : 2 * R * n * m].reshape(m, p, R)
                W_opt = res.x[2 * R * n * m : 3 * R * n * m].reshape(p, n, R)
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

    # First, construct the reliable rank-26 decomposition
    result_r26 = rank26_decomposition()

    # Try rank-23 structured search first (most promising)
    if HAVE_SCIPY and time.time() < deadline - 30:
        r23_result = _try_rank23_structured()
        if r23_result is not None:
            return r23_result

    # Try other lower ranks if time permits
    elapsed = time.time() - start_time
    if HAVE_SCIPY and elapsed < 60:
        base_U = np.array(result_r26['U'])
        base_V = np.array(result_r26['V'])
        base_W = np.array(result_r26['W'])
        for target_R in [25, 24]:
            if time.time() > deadline - 15:
                break
            result = _try_rank_optimized(target_R, start_time, deadline, idx_u, idx_v, idx_w, n, m, p, base_U, base_V, base_W, 26)
            if result is not None:
                return result

    # Return the verified rank-26
    return result_r26
