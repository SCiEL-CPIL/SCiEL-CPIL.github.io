import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor support
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                support.append((u, v, w))
    
    target_rank = 12
    
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    start_time = time.time()
    deadline = 110.0
    
    def compute_tensor(U, V, W):
        """Compute the full tensor from factor matrices."""
        R = U.shape[1]
        C = np.zeros((nm, mp, pn))
        for c in range(R):
            C += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
        return C
    
    def build_target():
        T = np.zeros((nm, mp, pn))
        for (u, v, w) in support:
            T[u, v, w] = 1.0
        return T
    
    T = build_target()
    
    def residual_norm(U, V, W):
        C = compute_tensor(U, V, W)
        return np.linalg.norm(C - T)
    
    def verify_exact(U, V, W, atol=1e-8):
        """Check if U,V,W exactly decompose T with half-integer entries."""
        # Check half-integer constraint
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=atol):
                return False
        C = compute_tensor(U, V, W)
        return np.allclose(C, T, atol=atol)
    
    def snap_half(x):
        return np.round(x * 2) / 2.0
    
    # Strategy: Try structured initializations based on pairing support entries
    # Each rank-1 component covers some subset of support entries
    # Try all ways to partition 24 entries into 12 groups of 2
    
    best_solution = None
    best_residual = float('inf')
    
    # First, try a gradient descent approach with good initialization
    # Use the fact that we can initialize each rank-1 component to cover 2 support entries
    
    def try_als(U0, V0, W0, max_iters=200, tol=1e-10):
        """Run ALS optimization."""
        U, V, W = U0.copy(), V0.copy(), W0.copy()
        R = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update each factor
            for c in range(R):
                # Update U[:, c]
                # C_target - sum_{c'!=c} U_c' ⊗ V_c' ⊗ W_c' = U_c ⊗ V_c ⊗ W_c (approximately)
                # For fixed V_c, W_c: residual_u = T - sum_{c'!=c} U_c'⊗V_c'⊗W_c'
                # We want: U_c ⊗ (V_c ⊗ W_c) ≈ residual
                # This is: for all (v,w): sum_u U_c[u] * (V_c[v]*W_c[w]) = residual[u,v,w]
                # => U_c = argmin ||A @ U_c - b|| where A[v*w, u] = V_c[v]*W_c[w], b[v*w] = residual[:,v,w].ravel()
                
                res = T.copy()
                for c2 in range(R):
                    if c2 != c:
                        res -= np.einsum('u,v,w->uvw', U[:, c2], V[:, c2], W[:, c2])
                
                # Build linear system for U[:,c]
                # A has shape (mp*pn, nm), b has shape (mp*pn,)
                VW = np.outer(V[:, c], W[:, c])  # shape (mp*pn,)
                A = np.tile(VW[:, None], (1, nm))  # shape (mp*pn, nm)
                b = res.reshape(mp * pn)
                
                try:
                    U[:, c] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    break
                
                if time.time() > deadline:
                    break
                
                # Update V[:, c]
                res = T.copy()
                for c2 in range(R):
                    if c2 != c:
                        res -= np.einsum('u,v,w->uvw', U[:, c2], V[:, c2], W[:, c2])
                
                UW = np.outer(U[:, c], W[:, c])  # shape (nm*pn,)
                A = np.tile(UW[:, None], (1, mp))  # shape (nm*pn, mp)
                b = res.reshape(nm * pn)
                
                try:
                    V[:, c] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    break
                
                if time.time() > deadline:
                    break
                
                # Update W[:, c]
                res = T.copy()
                for c2 in range(R):
                    if c2 != c:
                        res -= np.einsum('u,v,w->uvw', U[:, c2], V[:, c2], W[:, c2])
                
                UV = np.outer(U[:, c], V[:, c])  # shape (nm*mp,)
                A = np.tile(UV[:, None], (1, pn))  # shape (nm*mp, pn)
                b = res.reshape(nm * mp)
                
                try:
                    W[:, c] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    break
                
                if time.time() > deadline:
                    break
            
            # Check convergence
            r = residual_norm(U, V, W)
            if r < best_residual:
                best_residual = r
                # Try snapping
                Us, Vs, Ws = snap_half(U), snap_half(V), snap_half(W)
                if verify_exact(Us, Vs, Ws):
                    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}, r
            
            if r < tol:
                Us, Vs, Ws = snap_half(U), snap_half(V), snap_half(W)
                if verify_exact(Us, Vs, Ws):
                    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}, r
            
            # Also check if current (unsnapped) is close enough that snapping works
            if r < 1e-6:
                Us, Vs, Ws = snap_half(U), snap_half(V), snap_half(W)
                if verify_exact(Us, Vs, Ws, atol=1e-4):
                    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}, r
        
        return None, residual_norm(U, V, W)
    
    # Generate structured initializations
    # Each rank-1 component should cover 2 support entries
    # For a pair ((u1,v1,w1), (u2,v2,w2)), we need:
    # U_c[u1]*V_c[v1]*W_c[w1] = 1
    # U_c[u2]*V_c[v2]*W_c[w2] = 1
    # And zero elsewhere (approximately)
    
    def make_init_from_pairs(pairs):
        """Create initial U,V,W from a list of 12 pairs of support entries."""
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        for c, (e1, e2) in enumerate(pairs):
            u1, v1, w1 = e1
            u2, v2, w2 = e2
            
            # Simple initialization: set the relevant entries
            U[u1, c] = 1.0
            U[u2, c] = 1.0
            V[v1, c] = 1.0
            V[v2, c] = 1.0
            W[w1, c] = 1.0
            W[w2, c] = 1.0
        
        return U, V, W
    
    # Try several random pairings
    np.random.seed(42)
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        # Random pairing of 24 support entries into 12 pairs
        indices = list(range(len(support)))
        np.random.shuffle(indices)
        pairs = []
        for i in range(0, 24, 2):
            pairs.append((support[indices[i]], support[indices[i+1]]))
        
        U0, V0, W0 = make_init_from_pairs(pairs)
        
        # Add small random perturbation to avoid degeneracy
        U0 += np.random.randn(nm, target_rank) * 0.1
        V0 += np.random.randn(mp, target_rank) * 0.1
        W0 += np.random.randn(pn, target_rank) * 0.1
        
        sol, res = try_als(U0, V0, W0, max_iters=100)
        if sol:
            best_solution = sol
            break
    
    # Also try pure random initializations
    if best_solution is None:
        for trial in range(50):
            if time.time() > deadline:
                break
            
            U0 = np.random.randn(nm, target_rank)
            V0 = np.random.randn(mp, target_rank)
            W0 = np.random.randn(pn, target_rank)
            
            sol, res = try_als(U0, V0, W0, max_iters=100)
            if sol:
                best_solution = sol
                break
    
    if best_solution:
        return best_solution
    else:
        return fallback
