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
    
    def verify(U, V, W, tol=1e-8):
        R = U.shape[1]
        T_comp = np.zeros((nm, mp, pn))
        for r in range(R):
            T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(T_comp - T)) < tol
    
    def snap_half_int(arr):
        return np.round(arr * 2) / 2.0
    
    def solve_als_fast(U, V, W, R, T, max_iters=50, deadline=None):
        """Fast ALS with pre-computed design matrices."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Pre-compute design matrices
        # For U: A_U[(b,c), r] = V[b,r] * W[c,r], shape (mp*pn, R)
        # For V: A_V[(a,c), r] = U[a,r] * W[c,r], shape (nm*pn, R)
        # For W: A_W[(a,b), r] = U[a,r] * V[b,r], shape (nm*mp, R)
        
        for it in range(max_iters):
            if deadline and time.time() > deadline:
                break
                
            # Solve for U
            A_U = (V[:, None, :] * W[None, :, :]).reshape(-1, R)
            for a in range(nm):
                t_a = T[a, :, :].reshape(-1)
                U[a, :], _, _, _ = np.linalg.lstsq(A_U, t_a, rcond=None)
            
            # Solve for V
            A_V = (U[:, None, :] * W[None, :, :]).reshape(-1, R)
            for b in range(mp):
                t_b = T[:, b, :].reshape(-1)
                V[b, :], _, _, _ = np.linalg.lstsq(A_V, t_b, rcond=None)
            
            # Solve for W
            A_W = (U[:, None, :] * V[None, :, :]).reshape(-1, R)
            for c in range(pn):
                t_c = T[:, :, c].reshape(-1)
                W[c, :], _, _, _ = np.linalg.lstsq(A_W, t_c, rcond=None)
            
            # Check convergence
            if it % 5 == 0:
                T_comp = np.zeros((nm, mp, pn))
                for r in range(R):
                    T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                cost = np.max(np.abs(T_comp - T))
                if cost < 1e-10:
                    break
        
        return U, V, W
    
    def polish(U, V, W, R, T, deadline=None):
        """High-precision polish using alternating least squares with tighter tolerance."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(30):
            if deadline and time.time() > deadline:
                break
                
            # Solve for U with high precision
            A_U = (V[:, None, :] * W[None, :, :]).reshape(-1, R)
            for a in range(nm):
                t_a = T[a, :, :].reshape(-1)
                U[a, :], _, _, _ = np.linalg.lstsq(A_U, t_a, rcond=1e-15)
            
            # Solve for V
            A_V = (U[:, None, :] * W[None, :, :]).reshape(-1, R)
            for b in range(mp):
                t_b = T[:, b, :].reshape(-1)
                V[b, :], _, _, _ = np.linalg.lstsq(A_V, t_b, rcond=1e-15)
            
            # Solve for W
            A_W = (U[:, None, :] * V[None, :, :]).reshape(-1, R)
            for c in range(pn):
                t_c = T[:, :, c].reshape(-1)
                W[c, :], _, _, _ = np.linalg.lstsq(A_W, t_c, rcond=1e-15)
        
        return U, V, W
    
    def generate_structured_init(R, seed):
        """Generate initialization based on schoolbook structure with perturbations."""
        np.random.seed(seed)
        
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        
        # Base schoolbook columns
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < R:
                        U[i*m+j, col] = 1.0
                        V[j*p+k, col] = 1.0
                        W[k*n+i, col] = 1.0
                        col += 1
        
        # Add random perturbations to remaining columns
        for r in range(col, R):
            U[:, r] = np.random.randn(nm) * 0.5
            V[:, r] = np.random.randn(mp) * 0.5
            W[:, r] = np.random.randn(pn) * 0.5
        
        # Also perturb the base columns slightly
        for r in range(min(col, R)):
            U[:, r] += np.random.randn(nm) * 0.1
            V[:, r] += np.random.randn(mp) * 0.1
            W[:, r] += np.random.randn(pn) * 0.1
        
        return U, V, W
    
    def generate_random_init(R, seed):
        """Pure random initialization."""
        np.random.seed(seed)
        U = np.random.randn(nm, R)
        V = np.random.randn(mp, R)
        W = np.random.randn(pn, R)
        
        # Scale to reasonable magnitude
        U /= np.linalg.norm(U) * np.sqrt(nm)
        V /= np.linalg.norm(V) * np.sqrt(mp)
        W /= np.linalg.norm(W) * np.sqrt(pn)
        
        return U, V, W
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Try ranks from 20 to 23
    for R in [20, 21, 22, 23]:
        if time.time() > deadline:
            break
        
        found = False
        
        # Number of restarts: more for lower ranks
        if R == 20:
            num_restarts = 30
        elif R == 21:
            num_restarts = 15
        else:
            num_restarts = 8
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Alternate between structured and random initialization
            if restart % 2 == 0:
                U_init, V_init, W_init = generate_structured_init(R, seed=1000 + R * 100 + restart)
            else:
                U_init, V_init, W_init = generate_random_init(R, seed=2000 + R * 100 + restart)
            
            # Phase 1: Coarse ALS
            U, V, W = solve_als_fast(U_init, V_init, W_init, R, T, max_iters=30, deadline=deadline)
            
            if time.time() > deadline:
                break
            
            # Check if we're close
            T_comp = np.zeros((nm, mp, pn))
            for r in range(R):
                T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            cost = np.max(np.abs(T_comp - T))
            
            if cost < 1e-6:
                # Phase 2: Polish
                U, V, W = polish(U, V, W, R, T, deadline=deadline)
                
                # Verify
                if verify(U, V, W, tol=1e-8):
                    # Try snapping to half-integers
                    U_s = snap_half_int(U)
                    V_s = snap_half_int(V)
                    W_s = snap_half_int(W)
                    
                    if verify(U_s, V_s, W_s, tol=1e-8):
                        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                    else:
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
