import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = 3, 4, 6
    nm, mp, pn = n*m, m*p, p*n  # 12, 24, 18
    
    # Build the target tensor T as a 3D array T[a,b,c]
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try to find a low-rank decomposition
    # Start with a random rank and try to minimize the error
    # Use a greedy/iterative approach
    
    best_rank = n * m * p  # 72
    best_U = None
    best_V = None
    best_W = None
    
    # First, get schoolbook as baseline
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-6):
        """Check if U, V, W form a valid decomposition."""
        r = U.shape[1]
        # Compute the tensor and compare with T
        # T_computed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # For efficiency, check only non-zero entries of T
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check that off-diagonal entries are zero
        # This is expensive, so do a random sample
        rng = np.random.RandomState(42)
        for _ in range(2000):
            a = rng.randint(nm)
            b = rng.randint(mp)
            c = rng.randint(pn)
            # Check if this should be zero
            # a = i*m+j, b = j'*p+k, c = k'*n+i'
            i = a // m
            j = a % m
            jp = b // p
            k = b % p
            kn = c // n
            ip = c % n
            # This should be 1 only if j == jp and k == kn and i == ip
            if not (j == jp and k == kn and i == ip):
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        return True
    
    # Try Strassen-like optimizations
    # For 2x2x2 blocks, we can save 1 rank per block
    # 3x4x6: we can identify 2x2x2 blocks
    
    # Let's try a direct numerical search for rank < 72
    # Use a random initialization and iterate
    
    for target_rank in range(71, 50, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(5):
            if time.time() > deadline - 30:
                break
            
            rng = np.random.RandomState(attempt * 1000 + target_rank)
            U = rng.randn(nm, target_rank) * 0.1
            V = rng.randn(mp, target_rank) * 0.1
            W = rng.randn(pn, target_rank) * 0.1
            
            # Iterate to satisfy the equations
            # Use a simple alternating least squares approach
            for iteration in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Update U: for each a, solve for U[a,:]
                # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # This is nonlinear, so use gradient descent
                # Gradient w.r.t. U[a,r] = sum_{b,c: T[a,b,c]!=0} T[a,b,c]*V[b,r]*W[c,r] - sum_{b,c} U[a,r]*V[b,r]*W[c,r]*V[b,r]*W[c,r]
                # This is getting complex. Let me use a simpler approach.
                
                # Actually, let me use a different strategy: fix two matrices and solve for the third linearly
                # If V and W are fixed, then for each a:
                # sum_r U[a,r] * (V[b,r]*W[c,r]) = T[a,b,c] for all b,c
                # This is a linear system in U[a,r]
                
                # For each a, build the matrix M_a where M_a[(b,c), r] = V[b,r]*W[c,r]
                # Then U[a,:] = M_a^{-1} * T[a,:,:] (if M_a is square and invertible)
                
                # But M_a has shape (mp*pn, target_rank) = (24*18, target_rank) = (432, target_rank)
                # And we need to solve for U[a,:] of size target_rank
                # This is overdetermined, so use least squares
                
                for a in range(nm):
                    # Build M_a
                    M_a = np.zeros((mp * pn, target_rank))
                    for b in range(mp):
                        for c in range(pn):
                            M_a[b * pn + c, :] = V[b, :] * W[c, :]
                    # Target vector
                    t_vec = T[a, :, :].flatten()
                    # Solve least squares
                    try:
                        U[a, :] = np.linalg.lstsq(M_a, t_vec, rcond=None)[0]
                    except:
                        pass
                
                # Similarly update V and W
                for b in range(mp):
                    M_b = np.zeros((nm * pn, target_rank))
                    for a in range(nm):
                        for c in range(pn):
                            M_b[a * pn + c, :] = U[a, :] * W[c, :]
                    t_vec = T[:, b, :].flatten()
                    try:
                        V[b, :] = np.linalg.lstsq(M_b, t_vec, rcond=None)[0]
                    except:
                        pass
                
                for c in range(pn):
                    M_c = np.zeros((nm * mp, target_rank))
                    for a in range(nm):
                        for b in range(mp):
                            M_c[a * mp + b, :] = U[a, :] * V[b, :]
                    t_vec = T[:, :, c].flatten()
                    try:
                        W[c, :] = np.linalg.lstsq(M_c, t_vec, rcond=None)[0]
                    except:
                        pass
                
                # Check convergence
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.sum(U[a, :] * V[b, :] * W[c, :])
                            err += (val - 1.0) ** 2
                if err < 1e-8:
                    found = True
                    break
            
            if found:
                break
        
        if found:
            # Round to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify(U_rounded, V_rounded, W_rounded, tol=1e-6):
                best_rank = target_rank
                best_U = U_rounded
                best_V = V_rounded
                best_W = W_rounded
                break
    
    if best_U is None:
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
        best_rank = n * m * p
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
