"""Matrix multiplication 3x3 x 3x7 rank minimization."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    for idx in range(rank0):
        i = idx // (m*p)
        j = (idx // p) % m
        k = idx % p
        U0[i*m+j, idx] = 1
        V0[j*p+k, idx] = 1
        W0[k*n+i, idx] = 1
    
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = rank0
    
    def compute_residual(U, V, W):
        """Compute max absolute residual"""
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T_recon = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.max(np.abs(T_recon - 1.0))
    
    def verify(U, V, W, tol=1e-9):
        return compute_residual(U, V, W) < tol
    
    def structured_init(target_rank, seed=42):
        """Structured initialization using block structure"""
        np.random.seed(seed)
        r = target_rank
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        
        # Use a block structure: divide r columns into groups
        # For each (i,k) pair, we need sum_j contributions to equal 1
        # Initialize with random but structured values
        
        # Strategy: for each column c, assign it to an (i,k) block
        # and make it contribute primarily to that block
        for c in range(r):
            # Assign to a random (i,k) pair
            i = c % n
            k = (c // n) % p
            # Add random weights
            scale = np.sqrt(1.0 / 3.0)  # since m=3
            U[i*m:(i+1)*m, c] = np.random.randn(m) * scale
            V[:, k, c] if False else None  # placeholder
            # Actually let's do proper initialization
            pass
        
        # Simpler: random initialization with proper scaling
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Normalize columns
        for c in range(r):
            nu = np.linalg.norm(U[:, c])
            nv = np.linalg.norm(V[:, c])
            nw = np.linalg.norm(W[:, c])
            if nu * nv * nw > 1e-15:
                s = (nu * nv * nw) ** (1.0/3.0)
                U[:, c] /= s
                V[:, c] /= s
                W[:, c] /= s
        
        return U, V, W
    
    def optimize_rank(target_rank, max_iters=500, seed=42):
        """ALS optimization with smoother convergence"""
        np.random.seed(seed)
        r = target_rank
        
        # Initialize
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Normalize
        for c in range(r):
            nu = np.linalg.norm(U[:, c])
            nv = np.linalg.norm(V[:, c])
            nw = np.linalg.norm(W[:, c])
            if nu * nv * nw > 1e-15:
                s = (nu * nv * nw) ** (1.0/3.0)
                U[:, c] /= s
                V[:, c] /= s
                W[:, c] /= s
        
        best_res = compute_residual(U, V, W)
        U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    B = np.zeros((n, r))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    C = np.zeros((m, r))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check improvement
            res = compute_residual(U, V, W)
            if res < best_res - 1e-12:
                best_res = res
                U_best, V_best, W_best = U.copy(), V.copy(), W.copy()
            
            # Early stop if converged
            if best_res < 1e-10:
                break
        
        return U_best, V_best, W_best, best_res
    
    # Try various target ranks from low to high
    for target_rank in range(24, 64):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds
        found = False
        for seed in [42, 123, 456, 789]:
            if time.time() > deadline - 5:
                break
            
            U_opt, V_opt, W_opt, res = optimize_rank(target_rank, max_iters=300, seed=seed)
            
            if res < 1e-9:
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Final polish: try to reduce rank by removing near-zero columns
    if best_rank < rank0:
        r = best_rank
        for c in range(r - 1, -1, -1):
            if np.max(np.abs(best_U[:, c])) < 1e-12 and \
               np.max(np.abs(best_V[:, c])) < 1e-12 and \
               np.max(np.abs(best_W[:, c])) < 1e-12:
                mask = np.ones(r, dtype=bool)
                mask[c] = False
                best_U = best_U[:, mask]
                best_V = best_V[:, mask]
                best_W = best_W[:, mask]
                best_rank -= 1
    
    # Convert to list format
    r = best_U.shape[1]
    U_list = [[float(best_U[i, c]) for c in range(r)] for i in range(n*m)]
    V_list = [[float(best_V[i, c]) for c in range(r)] for i in range(m*p)]
    W_list = [[float(best_W[i, c]) for c in range(r)] for i in range(p*n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
