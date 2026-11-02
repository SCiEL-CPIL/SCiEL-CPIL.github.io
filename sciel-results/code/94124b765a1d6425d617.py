import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 36
    
    # Build the target tensor T flattened
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We represent T as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Trivial solution as fallback
    U_triv = np.zeros((n * m, trivial_rank))
    V_triv = np.zeros((m * p, trivial_rank))
    W_triv = np.zeros((p * n, trivial_rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, idx] = 1.0
                V_triv[j * p + k, idx] = 1.0
                W_triv[k * n + i, idx] = 1.0
                idx += 1

    deadline = time.time() + 110  # Safety margin
    
    # Pre-compute flattened T views
    T_flat_U = T.reshape(n * m, m * p * p * n)
    T_flat_V = T.reshape(m * p, n * m * p * n)
    T_flat_W = T.reshape(p * n, n * m * m * p)
    
    def polish_solution(U, V, W, r, max_iters=50):
        """High-precision polish pass with tighter tolerance"""
        for it in range(max_iters):
            if time.time() > deadline:
                return None, None, None, None
            
            # Update U
            D_U = np.einsum('br,cr->rbc', V, W).reshape(r, m * p * p * n)
            try:
                U_new = np.linalg.lstsq(D_U, T_flat_U.T, rcond=1e-12)[0].T
            except np.linalg.LinAlgError:
                return None, None, None, None
                
            # Update V
            D_V = np.einsum('ar,cr->rac', U_new, W).reshape(r, n * m * p * n)
            try:
                V_new = np.linalg.lstsq(D_V, T_flat_V.T, rcond=1e-12)[0].T
            except np.linalg.LinAlgError:
                return None, None, None, None

            # Update W
            D_W = np.einsum('ar,br->rab', U_new, V_new).reshape(r, n * m * m * p)
            try:
                W_new = np.linalg.lstsq(D_W, T_flat_W.T, rcond=1e-12)[0].T
            except np.linalg.LinAlgError:
                return None, None, None, None

            U, V, W = U_new, V_new, W_new
            
            # Check convergence with tighter tolerance
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.linalg.norm(T - T_approx)
            
            if residual < 1e-10:
                return U, V, W, residual
        
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        residual = np.linalg.norm(T - T_approx)
        return U, V, W, residual
    
    def try_round_and_verify(U, V, W, r):
        """Try rounding to half-integers and verify"""
        # Try different rounding strategies
        for scale in [2, 4, 8]:
            U_rounded = np.round(U * scale) / scale
            V_rounded = np.round(V * scale) / scale
            W_rounded = np.round(W * scale) / scale
            
            T_approx_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
            residual_rounded = np.linalg.norm(T - T_approx_rounded)
            
            if residual_rounded < 1e-8:
                return U_rounded, V_rounded, W_rounded, scale
        
        return None, None, None, None
    
    # Try to find a lower rank decomposition
    # We will try ranks from 35 down to 28
    best_result = None
    best_rank = trivial_rank
    
    for r in range(35, 27, -1):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        num_trials = 60 if r >= 33 else 40
        
        for trial in range(num_trials):
            if time.time() > deadline:
                break
                
            # Different initialization strategies
            if trial % 4 == 0:
                # Strategy 1: Random subset of trivial columns
                perm = np.random.permutation(trivial_rank)
                idxs = perm[:r]
                U = U_triv[:, idxs].copy()
                V = V_triv[:, idxs].copy()
                W = W_triv[:, idxs].copy()
                U += np.random.randn(n * m, r) * 0.05
                V += np.random.randn(m * p, r) * 0.05
                W += np.random.randn(p * n, r) * 0.05
            elif trial % 4 == 1:
                # Strategy 2: Random Gaussian initialization
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
            elif trial % 4 == 2:
                # Strategy 3: Mixed - some trivial, some random
                perm = np.random.permutation(trivial_rank)
                idxs = perm[:r//2]
                U = U_triv[:, idxs].copy()
                V = V_triv[:, idxs].copy()
                W = W_triv[:, idxs].copy()
                
                U_rand = np.random.randn(n * m, r - r//2) * 0.5
                V_rand = np.random.randn(m * p, r - r//2) * 0.5
                W_rand = np.random.randn(p * n, r - r//2) * 0.5
                
                U = np.hstack([U, U_rand])
                V = np.hstack([V, V_rand])
                W = np.hstack([W, W_rand])
            else:
                # Strategy 4: Structured initialization based on tensor structure
                # Initialize with patterns that respect the matrix multiplication structure
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                
                # Create structured patterns
                for col in range(r):
                    # Each column gets a structured pattern
                    i_idx = col % n
                    j_idx = (col // n) % m
                    k_idx = (col // (n * m)) % p
                    
                    # Set entries based on the pattern
                    U[i_idx * m + j_idx, col] = 1.0
                    V[j_idx * p + k_idx, col] = 1.0
                    W[k_idx * n + i_idx, col] = 1.0
                
                # Add random perturbation to break symmetry
                U += np.random.randn(n * m, r) * 0.1
                V += np.random.randn(m * p, r) * 0.1
                W += np.random.randn(p * n, r) * 0.1
            
            # ALS Iterations
            for it in range(100):
                if time.time() > deadline:
                    break
                    
                # Update U
                D_U = np.einsum('br,cr->rbc', V, W).reshape(r, m * p * p * n)
                try:
                    U_new = np.linalg.lstsq(D_U, T_flat_U.T, rcond=1e-10)[0].T
                except np.linalg.LinAlgError:
                    continue
                    
                # Update V
                D_V = np.einsum('ar,cr->rac', U_new, W).reshape(r, n * m * p * n)
                try:
                    V_new = np.linalg.lstsq(D_V, T_flat_V.T, rcond=1e-10)[0].T
                except np.linalg.LinAlgError:
                    continue

                # Update W
                D_W = np.einsum('ar,br->rab', U_new, V_new).reshape(r, n * m * m * p)
                try:
                    W_new = np.linalg.lstsq(D_W, T_flat_W.T, rcond=1e-10)[0].T
                except np.linalg.LinAlgError:
                    continue

                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                residual = np.linalg.norm(T - T_approx)
                
                if residual < 1e-8:
                    # Found a solution! Try to polish and round
                    U_polished, V_polished, W_polished, res_polished = polish_solution(U, V, W, r, max_iters=50)
                    
                    if U_polished is not None:
                        U_rounded, V_rounded, W_rounded, scale = try_round_and_verify(U_polished, V_polished, W_polished, r)
                        
                        if U_rounded is not None:
                            return {
                                "U": U_rounded.tolist(),
                                "V": V_rounded.tolist(),
                                "W": W_rounded.tolist()
                            }
                    
                    # If polishing didn't help, try direct rounding
                    U_rounded, V_rounded, W_rounded, scale = try_round_and_verify(U, V, W, r)
                    if U_rounded is not None:
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }

    # Fallback to trivial
    return {
        "U": U_triv.tolist(),
        "V": V_triv.tolist(),
        "W": W_triv.tolist()
    }
