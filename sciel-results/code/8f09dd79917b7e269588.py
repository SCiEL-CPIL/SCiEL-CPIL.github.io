"""Matrix multiplication 4x4x9 rank-minimal exact tensor decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Shapes
    U_shape = (n * m,)  # 16
    V_shape = (m * p,)  # 36
    W_shape = (p * n,)  # 36
    
    # The target tensor: T[a, b, c] = 1 if there exist i,j,k with
    # a = i*m + j, b = j*p + k, c = k*n + i
    # Otherwise 0.
    # We need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Build the list of (a, b, c) triples where T=1
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    triple_set = set(triples)
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    for col, (a, b, c) in enumerate(triples):
        U_school[a, col] = 1
        V_school[b, col] = 1
        W_school[c, col] = 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is exact."""
        r = U.shape[1]
        # For each triple, check sum_r U[a,r]*V[b,r]*W[c,r] = 1
        for (a, b, c) in triples:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1.0) > tol:
                return False
        # Check that for non-triple entries, the sum is 0
        # We need to check all (a,b,c) not in triples
        # Total entries: 16*36*36 = 20736, triples: 144
        # So 20592 non-triple entries to check
        # This is expensive but necessary for correctness
        # Use vectorized approach: build the full tensor and check
        # Actually, let's use a smarter approach
        # For each r, the contribution to (a,b,c) is U[a,r]*V[b,r]*W[c,r]
        # We can compute the full tensor as sum over r of outer products
        
        # Efficient check: compute the full multiplication tensor
        # T_computed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # We need T_computed = T_target
        
        # Build target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for (a, b, c) in triples:
            T_target[a, b, c] = 1.0
        
        # Compute computed tensor
        # For each r, add U[:,r] outer V[:,r] outer W[:,r]
        # This is expensive for large r, but we can do it incrementally
        T_computed = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            T_computed += np.einsum('i,j,k->ijk', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        
        if not np.allclose(T_computed, T_target, atol=tol):
            return False
        
        return True
    
    def check_decomposition_fast(U, V, W, tol=1e-9):
        """Faster check using only the constraint equations."""
        r = U.shape[1]
        # Check all 144 triple constraints
        for (a, b, c) in triples:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - 1.0) > tol:
                return False
        
        # Check a sample of non-triple constraints
        # There are 20592 non-triple entries
        # We'll check all of them using vectorized operations
        # For each (a,b,c) not in triples, we need sum_r U[a,r]*V[b,r]*W[c,r] = 0
        
        # Vectorized: for each a,b, compute the vector of values for all c
        # val[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = (U[a,:] * V[b,:]) dot W[:,c] for each c
        # Let's compute this for all a,b at once
        # UV[a,b,r] = U[a,r]*V[b,r]
        UV = np.einsum('ar,br->abr', U, V)  # shape (16, 36, r)
        # Now val[a,b,c] = sum_r UV[a,b,r] * W[c,r]
        # = UV @ W.T for each a,b
        # val = np.einsum('abr,cr->abc', UV, W)  # shape (16, 36, 36)
        
        # This might be memory intensive: 16*36*36*r floats
        # For r up to 100, that's 16*36*36*100 = 2,073,600 floats ~ 16MB, OK
        val = np.einsum('abr,cr->abc', UV, W)
        
        # Check that all non-triple entries are ~0
        # Create mask for triples
        mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for (a, b, c) in triples:
            mask[a, b, c] = True
        
        # Check non-triple entries
        non_triple_vals = val[~mask]
        if np.any(np.abs(non_triple_vals) > tol):
            return False
        
        return True
    
    # Try to find lower rank decompositions using ALS
    # Lower bound: max(n*m, m*p, p*n) = max(16, 36, 36) = 36
    
    def als_optimize(U, V, W, n_iters=100, tol=1e-12):
        """Alternating least squares optimization."""
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each a, solve least squares
            # We need sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
            # For fixed a, this is: sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c]
            # Let M_a[b,c,r] = V[b,r]*W[c,r]
            # Then U[a,:] is the solution to: sum_r U[a,r]*M_a[b,c,r] = T[a,b,c]
            # This is a linear system in U[a,:]
            
            # For each a:
            # Flatten M_a to get a matrix of shape (m*p*p*n, r)
            # And flatten T[a,:,:] to get a vector of shape (m*p*p*n,)
            # Then solve: M_a_flat @ U[a,:] = T_a_flat
            
            for a in range(n*m):
                # Build the design matrix for this a
                # M_a[b,c,r] = V[b,r] * W[c,r]
                # We need to solve for U[a,r] such that
                # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all b,c
                
                # This is equivalent to:
                # For each (b,c), the equation is: U[a,:] dot (V[b,:]*W[c,:]) = T[a,b,c]
                
                # Let's build the matrix A of shape (m*p*p*n, r)
                # where A[b*p*n+c, r] = V[b,r]*W[c,r]
                # and the target vector y of shape (m*p*p*n,)
                # where y[b*p*n+c] = T[a,b,c]
                
                A = np.zeros((m*p*p*n, r))
                y = np.zeros(m*p*p*n)
                
                idx = 0
                for b in range(m*p):
                    for c in range(p*n):
                        A[idx, :] = V[b, :] * W[c, :]
                        # T[a,b,c] = 1 if (a,b,c) in triples, else 0
                        y[idx] = 1.0 if (a, b, c) in triple_set else 0.0
                        idx += 1
                
                # Solve least squares: A @ u = y
                # Use np.linalg.lstsq
                try:
                    u_new, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    U[a, :] = u_new
                except:
                    pass
            
            # Similarly update V
            for b in range(m*p):
                A = np.zeros((n*m*p*n, r))
                y = np.zeros(n*m*p*n)
                idx = 0
                for a in range(n*m):
                    for c in range(p*n):
                        A[idx, :] = U[a, :] * W[c, :]
                        y[idx] = 1.0 if (a, b, c) in triple_set else 0.0
                        idx += 1
                try:
                    v_new, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    V[b, :] = v_new
                except:
                    pass
            
            # Similarly update W
            for c in range(p*n):
                A = np.zeros((n*m*m*p, r))
                y = np.zeros(n*m*m*p)
                idx = 0
                for a in range(n*m):
                    for b in range(m*p):
                        A[idx, :] = U[a, :] * V[b, :]
                        y[idx] = 1.0 if (a, b, c) in triple_set else 0.0
                        idx += 1
                try:
                    w_new, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                    W[c, :] = w_new
                except:
                    pass
        
        return U, V, W
    
    # Try ranks from lower bound up
    for target_rank in range(36, min(60, rank_school)):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(2):
            if time.time() > deadline - 20:
                break
            
            rng = np.random.RandomState(target_rank * 100 + attempt)
            U = rng.randn(n*m, target_rank) * 0.5
            V = rng.randn(m*p, target_rank) * 0.5
            W = rng.randn(p*n, target_rank) * 0.5
            
            # Run ALS
            U, V, W = als_optimize(U, V, W, n_iters=50)
            
            # Check if we got a valid decomposition
            if check_decomposition_fast(U, V, W, tol=1e-9):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Convert to half-integer if possible
    def try_half_integer(U, V, W):
        """Try to round to half-integers and verify."""
        U_h = np.round(U * 2) / 2
        V_h = np.round(V * 2) / 2
        W_h = np.round(W * 2) / 2
        if check_decomposition_fast(U_h, V_h, W_h, tol=1e-9):
            return U_h, V_h, W_h
        return None
    
    result = try_half_integer(best_U, best_V, best_W)
    if result is not None:
        best_U, best_V, best_W = result
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
