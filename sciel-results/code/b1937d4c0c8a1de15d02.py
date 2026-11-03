import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Build the target tensor T[i, j, k] = 1 if output (i,k) depends on input (i,j) and (j,k)
    # T has shape (n*m, m*p, p*n) in the flattened index convention
    # Actually, the tensor for matrix multiplication: T[a_idx, b_idx, c_idx] = 1 
    # where a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
    # for all valid i, j, k
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                T[a_idx, b_idx, c_idx] = 1
    
    # Current best: rank 50 from Strassen
    best_rank = 50
    best_U = None
    best_V = None
    best_W = None
    
    # Generate the rank-50 decomposition (same as before)
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(U_entry, V_entry, W_entry):
        U_col = [0] * (n * m)
        for idx, val in U_entry.items():
            U_col[idx] = val
        V_col = [0] * (m * p)
        for idx, val in V_entry.items():
            V_col[idx] = val
        W_col = [0] * (p * n)
        for idx, val in W_entry.items():
            W_col[idx] = val
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    for block_start in range(0, 8, 2):
        k0 = block_start
        k1 = block_start + 1
        add_term({0*m+0: 1, 1*m+1: 1}, {0*p+k0: 1, 1*p+k1: 1}, {k0*n+0: 1, k1*n+1: 1})
        add_term({1*m+0: 1, 1*m+1: 1}, {0*p+k0: 1}, {k0*n+1: 1, k1*n+1: -1})
        add_term({0*m+0: 1}, {0*p+k1: 1, 1*p+k1: -1}, {k1*n+0: 1, k1*n+1: 1})
        add_term({1*m+1: 1}, {1*p+k0: 1, 0*p+k0: -1}, {k0*n+0: 1, k0*n+1: 1})
        add_term({0*m+0: 1, 0*m+1: 1}, {1*p+k1: 1}, {k0*n+0: -1, k1*n+0: 1})
        add_term({1*m+0: 1, 0*m+0: -1}, {0*p+k0: 1, 0*p+k1: 1}, {k1*n+1: 1})
        add_term({0*m+1: 1, 1*m+1: -1}, {1*p+k0: 1, 1*p+k1: 1}, {k0*n+0: 1})
    
    k = 8
    add_term({0*m+0: 1}, {0*p+k: 1}, {k*n+0: 1})
    add_term({0*m+1: 1}, {1*p+k: 1}, {k*n+0: 1})
    add_term({1*m+0: 1}, {0*p+k: 1}, {k*n+1: 1})
    add_term({1*m+1: 1}, {1*p+k: 1}, {k*n+1: 1})
    
    for i in range(n):
        for k in range(p):
            add_term({i*m+2: 1}, {2*p+k: 1}, {k*n+i: 1})
    
    rank = len(U_cols)
    U = np.array([[U_cols[r][i] for r in range(rank)] for i in range(n*m)], dtype=float)
    V = np.array([[V_cols[r][j] for r in range(rank)] for j in range(m*p)], dtype=float)
    W = np.array([[W_cols[r][k] for r in range(rank)] for k in range(p*n)], dtype=float)
    
    # Verify
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    if np.allclose(reconstructed, T, atol=1e-10):
        best_rank = rank
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions
    # Strategy: try rank 49, 48, ... down to some minimum
    # Use random initialization + gradient-free optimization (Nelder-Mead or similar)
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W exactly decompose T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=tol)
    
    def snap_to_half_integers(arr, tol=1e-6):
        """Try to snap values to nearest half-integer"""
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None
    
    # Try rank 49
    target_rank = best_rank - 1
    while target_rank >= 30 and time.time() < deadline:
        # Try multiple random initializations
        found = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W with random values
            U_r = np.random.randn(n*m, target_rank) * 0.5
            V_r = np.random.randn(m*p, target_rank) * 0.5
            W_r = np.random.randn(p*n, target_rank) * 0.5
            
            # Simple iterative refinement: minimize ||T - sum_r U_r[:,r] ⊗ V_r[:,r] ⊗ W_r[:,r]||
            # Use a few iterations of ALS (alternating least squares)
            for iteration in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                for r in range(target_rank):
                    # T - sum_{r' != r} U[:,r'] ⊗ V[:,r'] ⊗ W[:,r']
                    residual = T.copy()
                    for r2 in range(target_rank):
                        if r2 != r:
                            residual -= np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
                    # residual[a,b,c] = U[a,r] * V[b,r] * W[c,r]
                    # For each a: residual[a,:,:] = U[a,r] * (V[:,r] ⊗ W[:,r])
                    # U[a,r] = <residual[a,:,:], V[:,r] ⊗ W[:,r]> / ||V[:,r] ⊗ W[:,r]||^2
                    vw = np.outer(V[:, r], W[:, r])
                    denom = np.sum(vw * vw)
                    if denom > 1e-12:
                        U[a, r] = np.sum(residual[a, :, :] * vw) / denom
                    # Wait, this is wrong. Let me fix.
                    # residual[a,b,c] should equal U[a,r]*V[b,r]*W[c,r]
                    # Sum over b,c: residual[a,b,c] * V[b,r] * W[c,r] = U[a,r] * sum_{b,c} V[b,r]^2 * W[c,r]^2
                    # So U[a,r] = sum_{b,c} residual[a,b,c]*V[b,r]*W[c,r] / (sum_b V[b,r]^2 * sum_c W[c,r]^2)
                    sv = np.sum(V[:, r] ** 2)
                    sw = np.sum(W[:, r] ** 2)
                    denom = sv * sw
                    if denom > 1e-12:
                        U[:, r] = np.sum(residual * np.outer(np.ones(n*m), np.outer(V[:, r], W[:, r])), axis=(1,2)) / denom
                        # That's not right either. Let me think more carefully.
                        # sum_{b,c} residual[a,b,c] * V[b,r] * W[c,r] for each a
                        U[:, r] = np.einsum('abc,b,c->a', residual, V[:, r], W[:, r]) / denom
                
                # Update V
                for r in range(target_rank):
                    residual = T.copy()
                    for r2 in range(target_rank):
                        if r2 != r:
                            residual -= np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
                    su = np.sum(U[:, r] ** 2)
                    sw = np.sum(W[:, r] ** 2)
                    denom = su * sw
                    if denom > 1e-12:
                        V[:, r] = np.einsum('abc,a,c->b', residual, U[:, r], W[:, r]) / denom
                
                # Update W
                for r in range(target_rank):
                    residual = T.copy()
                    for r2 in range(target_rank):
                        if r2 != r:
                            residual -= np.einsum('a,b,c->abc', U[:, r2], V[:, r2], W[:, r2])
                    su = np.sum(U[:, r] ** 2)
                    sv = np.sum(V[:, r] ** 2)
                    denom = su * sv
                    if denom > 1e-12:
                        W[:, r] = np.einsum('abc,a,b->c', residual, U[:, r], V[:, r]) / denom
            
            # Check convergence
            if check_decomposition(U_r, V_r, W_r, tol=1e-6):
                # Try to snap to half-integers
                U_snap = snap_to_half_integers(U_r, tol=1e-4)
                V_snap = snap_to_half_integers(V_r, tol=1e-4)
                W_snap = snap_to_half_integers(W_r, tol=1e-4)
                if U_snap is not None and V_snap is not None and W_snap is not None:
                    if check_decomposition(U_snap, V_snap, W_snap, tol=1e-8):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        found = True
                        break
                else:
                    # Use the floating point solution if it's very close
                    if check_decomposition(U_r, V_r, W_r, tol=1e-10):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_r, V_r, W_r
                        found = True
                        break
        
        if found:
            target_rank -= 1
        else:
            break
    
    # Return the best decomposition found
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback: return the rank-50 decomposition
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
