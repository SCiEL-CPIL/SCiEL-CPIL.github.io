import numpy as np
import time
from scipy.optimize import least_squares
from scipy.linalg import null_space

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0

    def check_decomposition(U, V, W, atol=1e-9):
        """Check if U, V, W exactly decompose target."""
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)

    def round_to_half_int(M):
        return np.round(M * 2) / 2

    def try_rank_als(r, num_restarts=80):
        """
        Try to find a rank r decomposition using Alternating Least Squares (ALS).
        """
        if time.time() > deadline:
            return None

        best_U, best_V, best_W = None, None, None
        best_err = float('inf')

        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break

            # Multiple initialization strategies
            if restart % 4 == 0:
                # Random initialization
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
            elif restart % 4 == 1:
                # Schoolbook-based initialization with perturbation
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            col = (i * m + j) * p + k
                            if col < r:
                                U[i * m + j, col] = 1.0
                                V[j * p + k, col] = 1.0
                                W[k * n + i, col] = 1.0
                # Add small random perturbation to break symmetry
                U += np.random.randn(nm, r) * 0.05
                V += np.random.randn(mp, r) * 0.05
                W += np.random.randn(pn, r) * 0.05
            elif restart % 4 == 2:
                # Sparse random initialization
                U = np.random.randn(nm, r)
                V = np.random.randn(mp, r)
                W = np.random.randn(pn, r)
                mask_u = np.random.rand(nm, r) < 0.3
                mask_v = np.random.rand(mp, r) < 0.3
                mask_w = np.random.rand(pn, r) < 0.3
                U[mask_u] = 0
                V[mask_v] = 0
                W[mask_w] = 0
            else:
                # Structured initialization based on matrix multiplication structure
                U = np.zeros((nm, r))
                V = np.zeros((mp, r))
                W = np.zeros((pn, r))
                # Try to exploit the fact that we're doing 2x3 * 3x4
                # Group by i (row of first matrix)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # Assign to a column based on (i, j, k) pattern
                            col = (i * m + j) * p + k
                            if col < r:
                                U[i * m + j, col] = 1.0
                                V[j * p + k, col] = 1.0
                                W[k * n + i, col] = 1.0
                # Add structured perturbation
                for col in range(r):
                    U[:, col] += np.random.randn(nm) * 0.1
                    V[:, col] += np.random.randn(mp) * 0.1
                    W[:, col] += np.random.randn(pn) * 0.1

            # ALS Optimization with adaptive learning
            for it in range(80):
                if time.time() > deadline - 5:
                    break

                # Update U: For each row a, solve A_U @ U[a,:] = T_flat[a,:]
                # A_U[:, k] = kron(W[:, k], V[:, k])
                A_U = np.zeros((mp * pn, r))
                for k in range(r):
                    A_U[:, k] = np.kron(W[:, k], V[:, k])
                
                T_flat = target.reshape(nm, mp * pn)
                for a in range(nm):
                    try:
                        u_a, _, _, _ = np.linalg.lstsq(A_U, T_flat[a], rcond=None)
                        U[a, :] = u_a
                    except Exception:
                        pass

                # Update V
                A_V = np.zeros((nm * pn, r))
                for k in range(r):
                    A_V[:, k] = np.kron(W[:, k], U[:, k])
                
                T_reshaped_V = np.zeros((mp, nm * pn))
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            T_reshaped_V[b, a * pn + c] = target[a, b, c]
                
                for b in range(mp):
                    try:
                        v_b, _, _, _ = np.linalg.lstsq(A_V, T_reshaped_V[b], rcond=None)
                        V[b, :] = v_b
                    except Exception:
                        pass

                # Update W
                A_W = np.zeros((nm * mp, r))
                for k in range(r):
                    A_W[:, k] = np.kron(V[:, k], U[:, k])
                
                T_reshaped_W = np.zeros((pn, nm * mp))
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            T_reshaped_W[c, a * mp + b] = target[a, b, c]
                
                for c in range(pn):
                    try:
                        w_c, _, _, _ = np.linalg.lstsq(A_W, T_reshaped_W[c], rcond=None)
                        W[c, :] = w_c
                    except Exception:
                        pass

            # Check convergence and rounding
            if time.time() > deadline - 5:
                break

            T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
            err = np.sum((T_approx - target)**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    # Try rounding to half-integers
                    U_r = round_to_half_int(U)
                    V_r = round_to_half_int(V)
                    W_r = round_to_half_int(W)
                    
                    if check_decomposition(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    elif check_decomposition(U, V, W):
                        return U, V, W

        # Final polish with least_squares if we got close
        if best_err < 1e-6 and best_U is not None:
            params = np.concatenate([best_U.flatten(), best_V.flatten(), best_W.flatten()])
            
            def residuals(params):
                U_p = params[:r * nm].reshape(nm, r)
                V_p = params[r * nm:r * (nm + mp)].reshape(mp, r)
                W_p = params[r * (nm + mp):].reshape(pn, r)
                T_approx = np.einsum('ak,bk,ck->abc', U_p, V_p, W_p)
                return (T_approx - target).flatten()

            try:
                res = least_squares(residuals, params, method='lm', max_nfev=1000)
                U_p = res.x[:r * nm].reshape(nm, r)
                V_p = res.x[r * nm:r * (nm + mp)].reshape(mp, r)
                W_p = res.x[r * (nm + mp):].reshape(pn, r)
                
                U_r = round_to_half_int(U_p)
                V_r = round_to_half_int(V_p)
                W_r = round_to_half_int(W_p)
                
                if check_decomposition(U_r, V_r, W_r):
                    return U_r, V_r, W_r
                elif check_decomposition(U_p, V_p, W_p):
                    return U_p, V_p, W_p
            except Exception:
                pass

        return None

    def try_rank_nullspace(r):
        """
        Try to find a rank r decomposition using null space analysis.
        The constraint is: for each (a,b,c) with target[a,b,c]=1,
        sum_k U[a,k]*V[b,k]*W[c,k] = 1
        and for all other (a,b,c), sum_k U[a,k]*V[b,k]*W[c,k] = 0
        """
        if time.time() > deadline:
            return None
        
        # For a given rank r, we have 3*r unknowns
        # The number of constraints is nm*mp*pn = 6*12*8 = 576
        # But only 24 are non-zero, so we have 24 active constraints + 552 zero constraints
        
        # This is a nonlinear system. Let's try a different approach:
        # Fix W to be random, then solve for U and V
        # T[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]
        # For fixed W, this is bilinear in U and V
        
        best_U, best_V, best_W = None, None, None
        best_err = float('inf')
        
        for restart in range(30):
            if time.time() > deadline - 10:
                break
            
            # Random W
            W = np.random.randn(pn, r)
            
            # For fixed W, we need:
            # sum_k U[a,k]*V[b,k]*W[c,k] = target[a,b,c]
            # This can be written as: sum_k (U[a,k]*W[c,k]) * V[b,k] = target[a,b,c]
            # Let X[a,c,k] = U[a,k]*W[c,k], then sum_k X[a,c,k]*V[b,k] = target[a,b,c]
            # This is still nonlinear. Let's try alternating updates.
            
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            
            for it in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each a, sum_k U[a,k]*(V[b,k]*W[c,k]) = target[a,b,c]
                # Let M[b,c,k] = V[b,k]*W[c,k], then sum_k U[a,k]*M[b,c,k] = target[a,b,c]
                # Reshape: for each a, we have a linear system in U[a,:]
                M_flat = np.zeros((mp*pn, r))
                for k in range(r):
                    M_flat[:, k] = np.kron(W[:, k], V[:, k])
                
                T_flat = target.reshape(nm, mp*pn)
                for a in range(nm):
                    try:
                        U[a, :] = np.linalg.lstsq(M_flat, T_flat[a], rcond=None)[0]
                    except:
                        pass
                
                # Update V: for each b, sum_k V[b,k]*(U[a,k]*W[c,k]) = target[a,b,c]
                M_flat_v = np.zeros((nm*pn, r))
                for k in range(r):
                    M_flat_v[:, k] = np.kron(W[:, k], U[:, k])
                
                T_reshaped_v = np.zeros((mp, nm*pn))
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            T_reshaped_v[b, a*pn+c] = target[a, b, c]
                
                for b in range(mp):
                    try:
                        V[b, :] = np.linalg.lstsq(M_flat_v, T_reshaped_v[b], rcond=None)[0]
                    except:
                        pass
                
                # Update W: for each c, sum_k W[c,k]*(U[a,k]*V[b,k]) = target[a,b,c]
                M_flat_w = np.zeros((nm*mp, r))
                for k in range(r):
                    M_flat_w[:, k] = np.kron(V[:, k], U[:, k])
                
                T_reshaped_w = np.zeros((pn, nm*mp))
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            T_reshaped_w[c, a*mp+b] = target[a, b, c]
                
                for c in range(pn):
                    try:
                        W[c, :] = np.linalg.lstsq(M_flat_w, T_reshaped_w[c], rcond=None)[0]
                    except:
                        pass
            
            T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
            err = np.sum((T_approx - target)**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    U_r = round_to_half_int(U)
                    V_r = round_to_half_int(V)
                    W_r = round_to_half_int(W)
                    if check_decomposition(U_r, V_r, W_r):
                        return U_r, V_r, W_r
                    elif check_decomposition(U, V, W):
                        return U, V, W
        
        return None

    # Try ranks from 20 to 23
    for r in [20, 21, 22, 23]:
        if time.time() > deadline - 15:
            break
        
        # Try ALS first
        res = try_rank_als(r, num_restarts=60)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try null space approach
        res = try_rank_nullspace(r)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 24 (Schoolbook)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
