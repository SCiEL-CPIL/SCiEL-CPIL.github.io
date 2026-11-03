"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Build the target tensor T
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_indices.append((a, b, c))
                
    T_target = np.zeros((n * m, m * p, p * n))
    for a, b, c in target_indices:
        T_target[a, b, c] = 1.0

    def check_and_snap(U, V, W, r):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        U_snap = np.round(U_np * 2) / 2
        V_snap = np.round(V_np * 2) / 2
        W_snap = np.round(W_np * 2) / 2
        
        T_recon = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
        
        if not np.allclose(T_recon, T_target, atol=1e-9):
            return None
            
        U_out = [[float(x) for x in row] for row in U_snap]
        V_out = [[float(x) for x in row] for row in V_snap]
        W_out = [[float(x) for x in row] for row in W_snap]
        
        return {"U": U_out, "V": V_out, "W": W_out}

    # Fallback schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_school
    
    # Linear code approach: exploit n=2 structure
    # For each k, outputs C[0,k] and C[1,k] are dot products of length 5
    # Try to find shared intermediate products across k values
    
    def try_structured_rank(r, seed=42):
        if time.time() > deadline:
            return None
            
        np.random.seed(seed)
        
        # Strategy: assume each rank term is "localized" to specific (i,j,k) patterns
        # Initialize with a structured pattern based on grouping by k
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Structured init: for each k, allocate some rank terms
        # Each k needs to produce 2 outputs from 5 products
        # Try: for each k, use 2 terms that compute (A[0,j]+A[1,j])*B[j,k] and (A[0,j]-A[1,j])*B[j,k]
        # That's 10 terms for 5 k's, but we can share across k if we use bilinear forms in j
        
        # Simpler: random structured init
        for idx in range(r):
            # Randomly assign this term to contribute to certain (i,j,k) combinations
            # Use a sparse pattern
            for i in range(n):
                for j in range(m):
                    if np.random.rand() < 0.3:
                        U[i*m+j, idx] = np.random.choice([-1, 0, 1])
            for j in range(m):
                for k in range(p):
                    if np.random.rand() < 0.3:
                        V[j*p+k, idx] = np.random.choice([-1, 0, 1])
            for k in range(p):
                for i in range(n):
                    if np.random.rand() < 0.3:
                        W[k*n+i, idx] = np.random.choice([-1, 0, 1])
        
        def loss(params):
            size_U = (n * m) * r
            size_V = (m * p) * r
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]
            
            U_p = U_flat.reshape((n * m), r)
            V_p = V_flat.reshape((m * p), r)
            W_p = W_flat.reshape((p * n), r)
            
            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            err = np.sum((T_recon - T_target) ** 2)
            return err
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-14})
        except Exception:
            return None
            
        if res.fun < 1e-10:
            size_U = (n * m) * r
            size_V = (m * p) * r
            U_sol = res.x[:size_U].reshape((n * m), r)
            V_sol = res.x[size_U:size_U + size_V].reshape((m * p), r)
            W_sol = res.x[size_U + size_V:].reshape((p * n), r)
            
            result = check_and_snap(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), r)
            if result:
                return result
        return None

    def try_deterministic_rank(r):
        """Try to find a rank-r decomposition using the linear structure.
        
        For n=2, we can write:
        C[0,k] = sum_j A[0,j]*B[j,k]
        C[1,k] = sum_j A[1,j]*B[j,k]
        
        Introduce intermediates: for each j,k, let P[j,k] = A[0,j]*B[j,k] and Q[j,k] = A[1,j]*B[j,k]
        That's 50 products. To reduce, we can use:
        R[j,k] = (A[0,j]+A[1,j])*B[j,k] = P[j,k]+Q[j,k]
        S[j,k] = (A[0,j]-A[1,j])*B[j,k] = P[j,k]-Q[j,k]
        Then P[j,k] = (R[j,k]+S[j,k])/2, Q[j,k] = (R[j,k]-S[j,k])/2
        Still 10 per k, no savings.
        
        Real savings: use bilinear forms that span multiple j's.
        E.g., T1 = (A[0,0]+A[0,1])*B[0,k] + A[0,2]*B[2,k] + ... 
        This is getting complex. Let's try a different structured approach.
        """
        if time.time() > deadline:
            return None
        
        # Try a specific structured pattern:
        # For each k, use 2 rank terms:
        # Term 1: U has 1's at positions (0,0),(0,1),(0,2),(0,3),(0,4) for row 0
        #          V has 1's at positions (0,k),(1,k),(2,k),(3,k),(4,k)
        #          W has 1 at (k,0)
        # Term 2: similar for row 1
        # That's 10 terms for 5 k's. But we can try to merge.
        
        # Alternative: use 4 terms per k with Hadamard-like structure
        # For each k:
        #   t1: U=[1,1,0,0,0] (row0), V=[1,1,0,0,0] (col k), W=[1,0]
        #   t2: U=[1,-1,0,0,0], V=[1,1,0,0,0], W=[1,0]  -- this gives A[0,0]*B[0,k] and -A[0,1]*B[0,k]... no
        # This is tricky. Let's just try optimization with multiple structured inits.
        
        return None
    
    # Try ranks from 45 down, with structured and random inits
    ranks_to_try = list(range(45, 30, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        found = False
        
        # Try structured initialization
        for seed in [42, 123, 777, 999]:
            if time.time() > deadline:
                break
            result = try_structured_rank(r, seed)
            if result:
                best_result = result
                best_rank = r
                found = True
                break
        
        if found:
            break
        
        # Also try the original schoolbook-based init for this rank
        if time.time() > deadline:
            break
        np.random.seed(42)
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        for idx in range(min(r, n*m*p)):
            i = idx // (m * p)
            j = (idx // p) % m
            k = idx % p
            U[i * m + j, idx] = 1.0
            V[j * p + k, idx] = 1.0
            W[k * n + i, idx] = 1.0
        
        for idx in range(min(r, n*m*p), r):
            U[:, idx] = np.random.randn(n * m) * 0.5
            V[:, idx] = np.random.randn(m * p) * 0.5
            W[:, idx] = np.random.randn(p * n) * 0.5
        
        def loss(params):
            size_U = (n * m) * r
            size_V = (m * p) * r
            U_flat = params[:size_U]
            V_flat = params[size_U:size_U + size_V]
            W_flat = params[size_U + size_V:]
            
            U_p = U_flat.reshape((n * m), r)
            V_p = V_flat.reshape((m * p), r)
            W_p = W_flat.reshape((p * n), r)
            
            T_recon = np.einsum('ar,br,cr->abc', U_p, V_p, W_p)
            err = np.sum((T_recon - T_target) ** 2)
            return err
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 3000, 'ftol': 1e-14})
        except Exception:
            pass
        else:
            if res.fun < 1e-10:
                size_U = (n * m) * r
                size_V = (m * p) * r
                U_sol = res.x[:size_U].reshape((n * m), r)
                V_sol = res.x[size_U:size_U + size_V].reshape((m * p), r)
                W_sol = res.x[size_U + size_V:].reshape((p * n), r)
                
                result = check_and_snap(U_sol.tolist(), V_sol.tolist(), W_sol.tolist(), r)
                if result:
                    best_result = result
                    best_rank = r
                    break

    return best_result
# EVOLVE-BLOCK-END
