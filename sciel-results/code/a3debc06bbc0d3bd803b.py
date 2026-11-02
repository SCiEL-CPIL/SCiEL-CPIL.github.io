import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 12
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find a rank-12 solution
    # Target tensor T has 1s at (i*m+j, j*p+k, k*n+i)
    # We want U, V, W of shape (nm, R), (mp, R), (pn, R)
    # such that sum_c U[:,c] outer V[:,c] outer W[:,c] == T
    
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    def compute_residual(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Residual is the tensor difference
        # T_flat is a vector of length nm*mp*pn? No, too big.
        # We only care about the non-zero entries of T.
        # T has 1s at specific indices.
        # Let's compute the sum for those indices.
        # Index mapping:
        # u_idx = i*m + j
        # v_idx = j*p + k
        # w_idx = k*n + i
        # We need sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c] == 1
        # And for all other (u,v,w), the sum should be 0?
        # Actually, the tensor decomposition defines the full tensor.
        # If we only match the non-zero entries of T, we might have "noise" in the zero entries.
        # But the rank of the tensor is defined by the decomposition of the specific tensor T.
        # T is zero elsewhere.
        # So we must ensure the sum is 0 for all (u,v,w) not in the support of T.
        # This is a huge number of constraints.
        # However, if we minimize the Frobenius norm of the difference between 
        # the computed tensor and T, we cover all entries.
        
        # Efficient computation:
        # Tensor C = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # C[u, v, w] = sum_c U[u,c] V[v,c] W[w,c]
        
        # We can compute this using einsum or loops.
        # Given small dimensions, loops are fine.
        # nm=6, mp=12, pn=8. Total 576 entries.
        # R=12.
        
        C = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # C += uc[:, None, None] * vc[None, :, None] * wc[None, None, :]
            C += np.einsum('u,v,w->uvw', uc, vc, wc)
        
        # Target T
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        
        return np.linalg.norm(C - T)

    def snap_to_half_int(arr):
        # Snap to nearest 0.5
        return np.round(arr * 2) / 2.0

    def verify(U, V, W):
        # Check if entries are half-integers
        if not np.allclose(U * 2, np.round(U * 2)): return False
        if not np.allclose(V * 2, np.round(V * 2)): return False
        if not np.allclose(W * 2, np.round(W * 2)): return False
        
        # Check if decomposition is exact
        C = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            C += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
                    
        return np.allclose(C, T, atol=1e-6)

    best_score = float('inf')
    best_solution = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize with random values
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Run ALS for several iterations
        for iteration in range(300):
            if time.time() > deadline:
                break
            
            # Update each factor using alternating least squares
            for c in range(target_rank):
                # Update U[:, c]
                residual = T.copy()
                for c2 in range(target_rank):
                    if c2 != c:
                        residual -= np.einsum('u,v,w->uvw', U[:, c2], V[:, c2], W[:, c2])
                
                A = np.zeros((m*p * p*n, n*m))
                b = np.zeros(m*p * p*n)
                idx = 0
                for v in range(m*p):
                    for w in range(p*n):
                        factor = V[v, c] * W[w, c]
                        for u in range(n*m):
                            A[idx, u] = factor
                        b[idx] = residual[:, v, w]
                        idx += 1
                
                U[:, c] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                if time.time() > deadline:
                    break
                
                # Update V[:, c]
                residual = T.copy()
                for c2 in range(target_rank):
                    if c2 != c:
                        residual -= np.einsum('u,v,w->uvw', U[:, c2], V[:, c2], W[:, c2])
                
                A = np.zeros((n*m * p*n, m*p))
                b = np.zeros(n*m * p*n)
                idx = 0
                for u in range(n*m):
                    for w in range(p*n):
                        factor = U[u, c] * W[w, c]
                        for v in range(m*p):
                            A[idx, v] = factor
                        b[idx] = residual[u, :, w]
                        idx += 1
                
                V[:, c] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                if time.time() > deadline:
                    break
                
                # Update W[:, c]
                residual = T.copy()
                for c2 in range(target_rank):
                    if c2 != c:
                        residual -= np.einsum('u,v,w->uvw', U[:, c2], V[:, c2], W[:, c2])
                
                A = np.zeros((n*m * m*p, p*n))
                b = np.zeros(n*m * m*p)
                idx = 0
                for u in range(n*m):
                    for v in range(m*p):
                        factor = U[u, c] * V[v, c]
                        for w in range(p*n):
                            A[idx, w] = factor
                        b[idx] = residual[u, v, :]
                        idx += 1
                
                W[:, c] = np.linalg.lstsq(A, b, rcond=None)[0]
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
            
            # Check if we're close to a solution
            res = compute_residual(U, V, W)
            if res < best_score:
                best_score = res
                # Try snapping to half-integers
                U_snap = snap_to_half_int(U)
                V_snap = snap_to_half_int(V)
                W_snap = snap_to_half_int(W)
                if verify(U_snap, V_snap, W_snap):
                    best_solution = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    return best_solution
            
            if res < 1e-10:
                U_snap = snap_to_half_int(U)
                V_snap = snap_to_half_int(V)
                W_snap = snap_to_half_int(W)
                if verify(U_snap, V_snap, W_snap):
                    best_solution = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    return best_solution
    
    if best_solution:
        return best_solution
    else:
        return fallback

# EVOLVE-BLOCK-END
