"""Matrix multiplication 2x4x8 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 16
    
    # The multiplication tensor: T[i,j,k] = 1 for each (i,j,k)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    # and 0 for "wrong" combinations... actually no.
    # 
    # The tensor for matrix multiplication is:
    # T[a,b,c] where a indexes A (n*m), b indexes B (m*p), c indexes C (n*p)
    # T[a,b,c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k
    # T[a,b,c] = 0 otherwise
    #
    # A rank-1 decomposition: U[a,c'] * V[b,c'] * W[c,c']
    # We need sum over c' of U[a,c']*V[b,c']*W[c,c'] = T[a,b,c]
    
    # Build the target tensor
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find low-rank decomposition
    # Strategy: use randomized initialization and gradient-free search
    # Or use the fact that for small sizes, we can try structured decompositions
    
    # First, let's try a known approach: use the Strassen-like idea
    # For 2x2x2, Strassen gives rank 7 vs schoolbook 8
    # For our case, we need to find the minimum rank
    
    # Let's use an iterative approach:
    # 1. Start with a random low-rank factorization
    # 2. Use alternating least squares to minimize the error
    
    best_rank = None
    best_factors = None
    
    # Try ranks from lower bound upward
    # Lower bound: max(nm, mp, pn) = 16 (but this is weak)
    # Actually, for matrix mult, the tensor rank is at least max(nm, mp, pn)
    # But we can do better with structure
    
    # Let's try a more clever approach: 
    # The tensor T can be seen as a 3-way array. We want to find
    # U (nm x r), V (mp x r), W (pn x r) such that
    # T[a,b,c] = sum_c' U[a,c'] * V[b,c'] * W[c,c']
    
    # This is equivalent to: for each c', the outer product U[:,c'] ⊗ V[:,c'] ⊗ W[:,c']
    # sums to T.
    
    # Let's use a greedy approach:
    # At each step, find the best rank-1 approximation to the residual
    
    def find_best_rank1(residual, max_iter=50):
        """Find best rank-1 approximation (u,v,w) to residual tensor."""
        # Use iterative method: fix two, optimize the third
        # Initialize
        u = np.random.randn(nm)
        v = np.random.randn(mp)
        w = np.random.randn(pn)
        
        best_err = np.inf
        best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        for it in range(max_iter):
            # Update u: minimize ||residual - u ⊗ v ⊗ w||_F
            # This is: u = (sum over b,c residual[a,b,c]*v[b]*w[c]) / (sum (v^2)*(w^2))
            num_u = np.einsum('abc,b,c->a', residual, v, w)
            den_u = np.sum(v**2) * np.sum(w**2)
            if den_u > 1e-12:
                u = num_u / den_u
            
            # Update v
            num_v = np.einsum('abc,a,c->b', residual, u, w)
            den_v = np.sum(u**2) * np.sum(w**2)
            if den_v > 1e-12:
                v = num_v / den_v
            
            # Update w
            num_w = np.einsum('abc,a,b->c', residual, u, v)
            den_w = np.sum(u**2) * np.sum(v**2)
            if den_w > 1e-12:
                w = num_w / den_w
            
            # Compute error
            approx = np.einsum('a,b,c->abc', u, v, w)
            err = np.sum((residual - approx)**2)
            if err < best_err:
                best_err = err
                best_u, best_v, best_w = u.copy(), v.copy(), w.copy()
        
        return best_u, best_v, best_w, best_err
    
    # Greedy decomposition
    for target_rank in [16, 17, 18, 19, 20, 22, 24, 28, 32, 36, 40, 48, 56, 64]:
        if time.time() - start_time > deadline - 30:
            break
        
        residual = T.copy()
        Us = []
        Vs = []
        Ws = []
        
        for c in range(target_rank):
            if time.time() - start_time > deadline - 30:
                break
            
            u, v, w, err = find_best_rank1(residual, max_iter=30)
            Us.append(u)
            Vs.append(v)
            Ws.append(w)
            residual -= np.einsum('a,b,c->abc', u, v, w)
        
        # Check if residual is small enough
        final_err = np.sum(residual**2)
        if final_err < 1e-6:
            # We have a valid decomposition!
            U_mat = np.column_stack(Us)
            V_mat = np.column_stack(Vs)
            W_mat = np.column_stack(Ws)
            
            # Try to snap to half-integers
            # Scale: the entries should be in {0, ±0.5, ±1}
            # Check if they're close to half-integers
            U_snap = np.round(U_mat * 2) / 2
            V_snap = np.round(V_mat * 2) / 2
            W_snap = np.round(W_mat * 2) / 2
            
            # Verify the snapped version
            T_check = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
            snap_err = np.sum((T - T_check)**2)
            
            if snap_err < 1e-6:
                best_rank = target_rank
                best_factors = (U_snap, V_snap, W_snap)
                break
            else:
                # Use the non-snapped version if it's valid
                T_check2 = np.einsum('ac,bc,dc->abd', U_mat, V_mat, W_mat)
                err2 = np.sum((T - T_check2)**2)
                if err2 < 1e-6:
                    best_rank = target_rank
                    best_factors = (U_mat, V_mat, W_mat)
                    break
    
    if best_factors is None:
        # Fallback to schoolbook
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
    
    U_mat, V_mat, W_mat = best_factors
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
