"""Matrix multiplication 2x3 x 3x3: rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 3]

# Deadline guard: 150s wall-clock limit; keep 10s safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _time_left():
    return _DEADLINE - time.time()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x3 matrix multiplication.
    
    The tensor T has indices (i, j, j', k, i', k') where:
      i in [0,1], j in [0,2], j' in [0,2], k in [0,2], i' in [0,1], k' in [0,2]
      T[i,j,j',k,i',k'] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    We seek rank r such that T = sum_{c=0}^{r-1} u_c ⊗ v_c ⊗ w_c where:
      u_c has shape (2,3) = 6 entries (A-indices i,j)
      v_c has shape (3,3) = 9 entries (B-indices j',k)
      w_c has shape (3,2) = 6 entries (C-indices k',i')  [note: W[k*n+i, r] means W index is k'*n+i]
    
    Actually per the interface:
      U has shape (n*m, rank) = (6, rank)
      V has shape (m*p, rank) = (9, rank)
      W has shape (p*n, rank) = (6, rank)
      W[k*n+i, c] where n=2, so W index = k*2 + i
    
    The decomposition satisfies:
      For all (i,j,j',k,i',k'):
        T[i,j,j',k,i',k'] = sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
    
    Schoolbook gives rank 18. The known minimum rank for 2x3 * 3x3 is 16.
    We'll use a structured search: try rank 16 first with a parameterized family,
    then fall back to rank 17, then 18.
    """
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (9, rank)
    # W: (p*n, rank) = (6, rank), W[k*n+i, c]
    
    def build_tensor():
        """Build the 6x9x6 tensor T[i,j,j',k, k',i'] = delta_{i,i'}delta_{j,j'}delta_{k,k'}"""
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for i2 in range(n):
                            for k2 in range(p):
                                if i == i2 and j == j2 and k == k2:
                                    T[i*m+j, j2*p+k, k2*n+i2] = 1.0
        return T
    
    T = build_tensor()
    
    def verify(U, V, W):
        """Check if U ⊗ V ⊗ W reconstructs T."""
        # U: (6, r), V: (9, r), W: (6, r)
        r = U.shape[1]
        # Reconstruct: T_recon[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
        # Use einsum: 'ad,bd,cd->abc'
        recon = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.allclose(recon, T, atol=1e-10)
    
    def to_lists(U, V, W):
        """Convert numpy arrays to lists for the return interface."""
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Known lower bound: max(n*m, m*p, p*n, (n*m*p)/(n+m-1)) ... 
    # For 2x3*3x3: lower bounds suggest rank >= 16
    # Schoolbook: rank = 18
    
    # Strategy: Use a parameterized Strassen-like family for rank 16.
    # The 2x3 * 3x3 problem can be seen as multiplying [A1 A2; A3 A4 A5 A6] (2x3) by [B] (3x3).
    # 
    # A known rank-16 decomposition exists. Let's use a structured search.
    # 
    # We parameterize U, V, W with a small number of free parameters and solve.
    
    # Approach: For a given rank r, set up the system of equations.
    # T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
    # This is a system of 6*9*6 = 324 equations in r*21 unknowns (6r + 9r + 6r = 21r).
    # For r=16: 336 unknowns, 324 equations. Overdetermined by 12.
    # For r=17: 357 unknowns, 324 equations.
    # For r=18: 378 unknowns, 324 equations.
    
    # We'll use a structured initialization and iterative refinement.
    
    def try_rank(r, n_restarts=50, max_iter=200):
        """Try to find a decomposition of rank r using iterative methods."""
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(n_restarts):
            if _time_left() < 5.0:
                break
            
            # Initialize with random small values
            rng = np.random.RandomState(42 + restart)
            U = rng.randn(n*m, r) * 0.5
            V = rng.randn(m*p, r) * 0.5
            W = rng.randn(p*n, r) * 0.5
            
            # Iteratively solve: fix two factors, solve for the third
            for it in range(max_iter):
                if _time_left() < 5.0:
                    break
                
                # Fix V and W, solve for U:
                # For each a: T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                # Rearrange: T[a,:,:] = U[a,:] @ (V * W^T)^T? 
                # T[a,b,c] = sum_d U[a,d] * (V[b,d] * W[c,d])
                # Let M = V * W^T  (9 x 6 matrix where M[b,c] = sum_d V[b,d]*W[c,d])
                # Then T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                # This is not simply U @ M because the product is element-wise in d.
                # Actually: T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                # For fixed b,c: T[a,b,c] = sum_d U[a,d] * (V[b,d]*W[c,d])
                # So T[:,b,c] = U @ (V[:,b]*W[:,c])  ... no
                # T[a,b,c] for all a: vector T[:,b,c] = U @ v where v[d] = V[b,d]*W[c,d]
                # This gives a linear system in U[a,:] for each (b,c)
                
                # Alternative: For fixed a, T[a,:,:] (9x6) = sum_d U[a,d] * (V[:,d] ⊗ W[:,d])^T reshaped
                # Let's use: for each d, the contribution to T from factor d is U[:,d] ⊗ V[:,d] ⊗ W[:,d]
                # So T = sum_d U[:,d] ⊗ V[:,d] ⊗ W[:,d]
                
                # To solve for U given V,W:
                # T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                # Multiply both sides by V[b',d] W[c',d] and sum over b',c' with appropriate weights?
                # Better: treat as a 3D linear system.
                
                # For U: fix V,W. For each (b,c), T[:,b,c] = U @ (V[:,b] * W[:,c] ... ) no.
                # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
                # Let phi_{b,c}[d] = V[b,d] * W[c,d]
                # Then T[:,b,c] = U @ phi_{b,c}  (where phi is a vector, U @ phi means sum_d U[a,d]*phi[d])
                # So for each (b,c): T[:,b,c] = U * phi_{b,c}
                # Stacking all (b,c): we have a linear system.
                
                # Collect equations: for each (b,c), we get n*m equations (one per a)
                # Total: (m*p)*(p*n) = 54 equations in 6*r = 6r unknowns for U
                # For r=16: 96 unknowns, 54 equations. Underdetermined!
                # So we need to solve least squares.
                
                # Build the system: A_U @ U_vec = T_vec
                # U_vec is (6r,). Each entry U[a,d] is unknown.
                # Equation for (a,b,c): sum_d U[a,d] V[b,d] W[c,d] = T[a,b,c]
                
                # Let's build the matrix A_U of shape (54, 6r)
                A_U = np.zeros((m*p * p*n, n*m * r))
                rhs_U = np.zeros(m*p * p*n)
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        for a in range(n*m):
                            for d in range(r):
                                A_U[idx, a*r + d] = V[b, d] * W[c, d]
                            rhs_U[idx] = T[a, b, c]
                            idx += 1
                
                # Solve least squares
                U_new, _, _, _ = np.linalg.lstsq(A_U, rhs_U, rcond=None)
                U = U_new.reshape(n*m, r)
                
                # Fix U and W, solve for V:
                # T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                # For each (a,c): T[a,:,c] = V @ (U[:,a] * W[:,c]) ... 
                # T[a,b,c] = sum_d V[b,d] U[a,d] W[c,d]
                # phi_{a,c}[d] = U[a,d] * W[c,d]
                # T[a,:,c] = V @ phi_{a,c}
                # System: (n*m * p*n) equations in (m*p * r) unknowns
                A_V = np.zeros((n*m * p*n, m*p * r))
                rhs_V = np.zeros(n*m * p*n)
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        for b in range(m*p):
                            for d in range(r):
                                A_V[idx, b*r + d] = U[a, d] * W[c, d]
                            rhs_V[idx] = T[a, b, c]
                            idx += 1
                
                V_new, _, _, _ = np.linalg.lstsq(A_V, rhs_V, rcond=None)
                V = V_new.reshape(m*p, r)
                
                # Fix U and V, solve for W:
                # T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                # For each (a,b): T[a,b,:] = W @ (U[:,a] * V[:,b])
                # phi_{a,b}[d] = U[a,d] * V[b,d]
                # T[a,b,:] = W @ phi_{a,b}
                A_W = np.zeros((n*m * m*p, p*n * r))
                rhs_W = np.zeros(n*m * m*p)
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            for d in range(r):
                                A_W[idx, c*r + d] = U[a, d] * V[b, d]
                            rhs_W[idx] = T[a, b, c]
                            idx += 1
                
                W_new, _, _, _ = np.linalg.lstsq(A_W, rhs_W, rcond=None)
                W = W_new.reshape(p*n, r)
                
                # Compute error
                recon = np.einsum('ad,bd,cd->abc', U, V, W)
                err = np.linalg.norm(recon - T)
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-10:
                    break
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10 and best_U is not None:
            return to_lists(best_U, best_V, best_W), best_err
        return None, best_err
    
    # Try rank 16 first
    if _time_left() > 20:
        result, err = try_rank(16, n_restarts=100, max_iter=300)
        if result is not None:
            return result
    
    # Try rank 17
    if _time_left() > 15:
        result, err = try_rank(17, n_restarts=100, max_iter=300)
        if result is not None:
            return result
    
    # Fall back to schoolbook rank 18
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return to_lists(U, V, W)
