import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def snap_half_int(arr):
        return np.round(arr * 2) / 2.0

    def verify(U, V, W, tol=1e-10):
        R = U.shape[1]
        T_comp = np.zeros((nm, mp, pn))
        for r in range(R):
            T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(T_comp - T)) < tol

    def verify_strict(U, V, W):
        """Check exact decomposition with half-integer entries"""
        R = U.shape[1]
        # Check all entries are half-integers
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=1e-9):
                return False
        T_comp = np.zeros((nm, mp, pn))
        for r in range(R):
            T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T_comp, T, atol=1e-9)

    def polish_solution(U, V, W, R, max_iters=50):
        """High-precision polish using Gauss-Newton style updates"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            # Compute residual
            T_comp = np.zeros((nm, mp, pn))
            for r in range(R):
                T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            residual = T_comp - T
            cost = np.sum(residual**2)
            
            if cost < 1e-20:
                break
            
            # Gauss-Newton: solve for updates
            # For U: minimize ||T - sum_r u_r v_r w_r||^2
            # Gradient wrt U[a, r] = -sum_{b,c} residual[a,b,c] V[b,r] W[c,r]
            # Hessian approx: sum_{b,c} V[b,r]W[c,r] V[b,r']W[c,r'] for r,r'
            
            # Solve for U update
            A_U = np.zeros((nm, R, R))
            for a in range(nm):
                for r in range(R):
                    for rp in range(R):
                        A_U[a, r, rp] = np.sum(V[:, r] * W[:, rp] * (V[:, r] * W[:, rp]))
                # Actually this is wrong. Let me use the proper normal equations.
            
            # Better: use lstsq on the linearized system
            # For each a, we have: sum_r delta_U[a,r] * (V[:,r] * W[:,r]).flatten() = -residual[a,:,:].flatten()
            for a in range(nm):
                A_a = (V[:, None, :] * W[None, :, :]).reshape(-1, R)
                b_a = -residual[a, :, :].reshape(-1)
                try:
                    delta_U_a = np.linalg.lstsq(A_a, b_a, rcond=None)[0]
                    U[a, :] += delta_U_a
                except:
                    pass
            
            # For V
            for b in range(mp):
                B_b = (U[:, None, :] * W[None, :, :]).reshape(-1, R)
                c_b = -residual[:, b, :].reshape(-1)
                try:
                    delta_V_b = np.linalg.lstsq(B_b, c_b, rcond=None)[0]
                    V[b, :] += delta_V_b
                except:
                    pass
            
            # For W
            for c in range(pn):
                C_c = (U[:, None, :] * V[None, :, :]).reshape(-1, R)
                d_c = -residual[:, :, c].reshape(-1)
                try:
                    delta_W_c = np.linalg.lstsq(C_c, d_c, rcond=None)[0]
                    W[c, :] += delta_W_c
                except:
                    pass
        
        # Snap to half-integers
        U_s = snap_half_int(U)
        V_s = snap_half_int(V)
        W_s = snap_half_int(W)
        
        if verify_strict(U_s, V_s, W_s):
            return U_s, V_s, W_s, 0.0
        
        # Try without snapping if close enough
        if verify(U, V, W, tol=1e-10):
            return U, V, W, cost
        
        return None, None, None, cost

    def solve_als(U_init, V_init, W_init, R, max_iters=100):
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        prev_cost = np.inf
        for it in range(max_iters):
            # Solve for U
            for a in range(nm):
                A_a = (V[:, None, :] * W[None, :, :]).reshape(-1, R)
                t_a = T[a, :, :].reshape(-1)
                U[a, :] = np.linalg.lstsq(A_a, t_a, rcond=None)[0]
            
            # Solve for V
            for b in range(mp):
                B_b = (U[:, None, :] * W[None, :, :]).reshape(-1, R)
                t_b = T[:, b, :].reshape(-1)
                V[b, :] = np.linalg.lstsq(B_b, t_b, rcond=None)[0]
            
            # Solve for W
            for c in range(pn):
                C_c = (U[:, None, :] * V[None, :, :]).reshape(-1, R)
                t_c = T[:, :, c].reshape(-1)
                W[c, :] = np.linalg.lstsq(C_c, t_c, rcond=None)[0]
            
            # Calculate cost
            T_comp = np.zeros((nm, mp, pn))
            for r in range(R):
                T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            cost = np.sum((T_comp - T)**2)
            
            if cost < 1e-15:
                break
            if abs(prev_cost - cost) < 1e-14:
                break
            prev_cost = cost
        
        # Polish
        U_p, V_p, W_p, cost_p = polish_solution(U, V, W, R, max_iters=30)
        if U_p is not None:
            return U_p, V_p, W_p, cost_p
        
        # Try snapping
        U_s = snap_half_int(U)
        V_s = snap_half_int(V)
        W_s = snap_half_int(W)
        
        if verify_strict(U_s, V_s, W_s):
            return U_s, V_s, W_s, 0.0
        
        return None, None, None, cost

    def generate_structured_init(R, seed):
        """Generate initialization based on known structure of matrix multiplication tensor"""
        np.random.seed(seed)
        
        # Strategy: Start with a perturbed schoolbook-like structure
        # The schoolbook has 24 terms. We need to combine them.
        
        # Create base random factors
        U_base = np.random.randn(nm, R)
        V_base = np.random.randn(mp, R)
        W_base = np.random.randn(pn, R)
        
        # Scale
        U_base /= np.sqrt(np.mean(U_base**2) + 1e-10)
        V_base /= np.sqrt(np.mean(V_base**2) + 1e-10)
        W_base /= np.sqrt(np.mean(W_base**2) + 1e-10)
        
        return U_base, V_base, W_base

    def generate_mixed_init(R, seed):
        """Mix of random and structured initialization"""
        np.random.seed(seed)
        
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        W = np.random.randn(pn, R)
        
        # Add some structure: make certain entries more likely to be nonzero
        # Based on the support of the tensor
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    # Boost entries that participate in the tensor
                    col = seed % R
                    U[a, col] += 0.5
                    V[b, col] += 0.5
                    W[c, col] += 0.5
        
        U /= np.sqrt(np.mean(U**2) + 1e-10)
        V /= np.sqrt(np.mean(V**2) + 1e-10)
        W /= np.sqrt(np.mean(W**2) + 1e-10)
        
        return U, V, W

    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try ranks from 16 to 23 (16 is the conjectured minimal rank for 2x3x4)
    for R in [16, 17, 18, 19, 20, 21, 22, 23]:
        if time.time() > deadline - 30:
            break
            
        found = False
        # More restarts for lower ranks
        if R <= 17:
            num_restarts = 30
        elif R <= 19:
            num_restarts = 20
        else:
            num_restarts = 10
            
        for restart in range(num_restarts):
            if time.time() > deadline - 30:
                break
            
            seed = 42 + R * 1000 + restart
            
            if restart % 2 == 0:
                U_init, V_init, W_init = generate_structured_init(R, seed)
            else:
                U_init, V_init, W_init = generate_mixed_init(R, seed)
            
            U, V, W, cost = solve_als(U_init, V_init, W_init, R, max_iters=80)
            if U is not None:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback: Schoolbook rank 24
    R = n * m * p
    U = np.zeros((nm, R))
    V = np.zeros((mp, R))
    W = np.zeros((pn, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
