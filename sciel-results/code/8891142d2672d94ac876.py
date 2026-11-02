import numpy as np
import time
import itertools
from scipy import linalg

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Searches for rank-20 decomposition via elimination from Strassen-21,
    with fallback to rank-21 Strassen.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12

    # Build the target tensor T[a,b,c]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Strassen steps for 2x2 multiplication
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]

    # Build Strassen-21 decomposition
    def build_strassen_21():
        rank = 21
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for g in range(3):
            k0, k1 = 2*g, 2*g+1
            v_idx = [0*6+k0, 0*6+k1, 1*6+k0, 1*6+k1]
            w_idx = [k0*2+0, k1*2+0, k0*2+1, k1*2+1]
            for s in range(7):
                r = 7*g + s
                step = strassen_steps[s]
                for li in range(4):
                    if step['u'][li] != 0:
                        U[li, r] = step['u'][li]
                    if step['v'][li] != 0:
                        V[v_idx[li], r] = step['v'][li]
                    if step['w'][li] != 0:
                        W[w_idx[li], r] = step['w'][li]
        return U, V, W

    U21, V21, W21 = build_strassen_21()

    # Verify Strassen-21
    T_check = np.einsum('ar,br,cr->abc', U21, V21, W21)
    if not np.allclose(T_check, T, atol=1e-10):
        # Fallback to schoolbook
        rank_s = 24
        Us = np.zeros((nm, rank_s))
        Vs = np.zeros((mp, rank_s))
        Ws = np.zeros((pn, rank_s))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    Us[a, col] = 1
                    Vs[b, col] = 1
                    Ws[c, col] = 1
                    col += 1
        return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

    # Try to find rank-20 by removing one product from Strassen-21
    # and checking if the residual is rank-1 (absorbable)
    def try_remove_product(removal_idx):
        """
        Try removing product at index removal_idx from the 21-product Strassen.
        Check if T - sum_{r != removal_idx} U[:,r] V[:,r] W[:,r] is rank-1.
        If so, we have a rank-20 decomposition.
        """
        # Compute residual
        residual = T.copy()
        for r in range(21):
            if r == removal_idx:
                continue
            residual -= np.einsum('a,b,c->abc', U21[:, r], V21[:, r], W21[:, r])

        # Check if residual is rank-1: all 2x2 minors of any matricization are zero
        # Matricize as (nm*mp) x pn
        R = residual.reshape(nm * mp, pn)
        # Rank-1 means all 2x2 minors vanish
        # Use SVD to check
        try:
            svals = np.linalg.svd(R, compute_uv=False)
        except:
            return None
        if len(svals) >= 2 and svals[1] > 1e-8:
            return None
        if len(svals) == 1:
            if svals[0] < 1e-12:
                # Residual is zero - we have rank 20 (one product was redundant)
                # But that means the 21st product contributed nothing
                pass
            # Residual is rank-1
            # Extract the rank-1 vectors
            s, U_mat, Vt = np.linalg.svd(R, compute_uv=True)
            if s[0] < 1e-12:
                # Zero residual
                pass
            u_vec = U_mat[:, 0] * s[0]
            w_vec = Vt[0, :]
            # Reshape u_vec to (nm, mp) for the U and V parts
            # Actually we need to decompose u_vec into U_part and V_part
            # u_vec = U_part * V_part (outer product)
            # u_vec is (nm*mp), we need U_part (nm) and V_part (mp) such that
            # u_vec = U_part ⊗ V_part
            # This is only possible if u_vec is itself rank-1 as a matrix
            U_mat_res = u_vec.reshape(nm, mp)
            # Check if U_mat_res is rank-1
            svals2 = np.linalg.svd(U_mat_res, compute_uv=False)
            if len(svals2) >= 2 and svals2[1] > 1e-8:
                return None
            if svals2[0] < 1e-12:
                # Zero - the removed product was actually needed
                return None
            # Extract
            s2, U2_mat, V2t = np.linalg.svd(U_mat_res, compute_uv=True)
            u_part = U2_mat[:, 0] * s2[0]
            v_part = V2t[0, :]
            w_part = w_vec

            # Build rank-20 decomposition
            U20 = np.zeros((nm, 20))
            V20 = np.zeros((mp, 20))
            W20 = np.zeros((pn, 20))
            col = 0
            for r in range(21):
                if r == removal_idx:
                    continue
                U20[:, col] = U21[:, r]
                V20[:, col] = V21[:, r]
                W20[:, col] = W21[:, r]
                col += 1
            # Add the residual rank-1 product
            U20[:, col] = u_part
            V20[:, col] = v_part
            W20[:, col] = w_part

            # Verify
            T20 = np.einsum('ar,br,cr->abc', U20, V20, W20)
            if np.allclose(T20, T, atol=1e-8):
                return (U20, V20, W20)
            return None
        elif len(svals) == 0:
            return None
        return None

    # Try removing each of the 21 products
    found_rank20 = None
    for idx in range(21):
        if time.time() > deadline - 30:
            break
        result = try_remove_product(idx)
        if result is not None:
            found_rank20 = result
            break

    if found_rank20 is not None:
        U20, V20, W20 = found_rank20
        return {"U": U20.tolist(), "V": V20.tolist(), "W": W20.tolist()}

    # If no rank-20 found via simple removal, try a more general search:
    # Use optimization to find a rank-20 decomposition by adjusting the
    # U, V, W matrices from a perturbed Strassen-21 starting point.
    if time.time() < deadline - 30:
        rng = np.random.RandomState(42)
        best_rank20 = None
        best_err = float('inf')
        
        # Strategy: start from Strassen-21, randomly drop one product,
        # then optimize the remaining 20 U,V,W to best approximate T.
        # If the optimal error is ~0, we have a valid rank-20 decomposition.
        
        for drop_idx in range(21):
            if time.time() > deadline - 35:
                break
            
            # Initialize with the 20 non-dropped Strassen products
            U_init = np.zeros((nm, 20))
            V_init = np.zeros((mp, 20))
            W_init = np.zeros((pn, 20))
            col = 0
            for r in range(21):
                if r == drop_idx:
                    continue
                U_init[:, col] = U21[:, r]
                V_init[:, col] = V21[:, r]
                W_init[:, col] = W21[:, r]
                col += 1
            
            # Add small random perturbation to break symmetry
            for _ in range(3):
                U_init += rng.randn(nm, 20) * 0.01
                V_init += rng.randn(mp, 20) * 0.01
                W_init += rng.randn(pn, 20) * 0.01
            
            # Define objective: ||T - sum_r U[:,r] V[:,r] W[:,r]||_F^2
            def objective(params):
                U_opt = params[:nm*20].reshape(nm, 20)
                V_opt = params[nm*20:nm*20+mp*20].reshape(mp, 20)
                W_opt = params[nm*20+mp*20:].reshape(pn, 20)
                T_approx = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt)
                return np.sum((T - T_approx)**2)
            
            # Use scipy minimize with L-BFGS-B
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import minimize
                result = minimize(objective, params_init, method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                
                if result.fun < 1e-10:
                    # Found a valid rank-20 decomposition!
                    U_opt = result.x[:nm*20].reshape(nm, 20)
                    V_opt = result.x[nm*20:nm*20+mp*20].reshape(mp, 20)
                    W_opt = result.x[nm*20+mp*20:].reshape(pn, 20)
                    
                    # Verify
                    T20 = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt)
                    if np.allclose(T20, T, atol=1e-8):
                        return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
                    elif result.fun < best_err:
                        best_err = result.fun
                        best_rank20 = (U_opt, V_opt, W_opt)
            except Exception:
                pass
        
        # If we found a near-optimal but not exact solution, check if it's close enough
        if best_rank20 is not None and best_err < 1e-6:
            U_opt, V_opt, W_opt = best_rank20
            T20 = np.einsum('ar,br,cr->abc', U_opt, V_opt, W_opt)
            if np.allclose(T20, T, atol=1e-6):
                return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    
    # Return Strassen-21 as the best we have
    return {"U": U21.tolist(), "V": V21.tolist(), "W": W21.tolist()}
