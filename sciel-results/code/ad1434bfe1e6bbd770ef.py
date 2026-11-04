import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W, T, atol=1e-7):
    """Check exact decomposition."""
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=atol)

def _try_rank_structured(n, m, p, rank, T, deadline, seed=0):
    """
    Try to find a rank-r decomposition where each term has support on
    a single (i,j,k) coordinate. The tensor is:
    T[a,b,c] = sum_r u[a,r] * v[b,r] * w[c,r]
    
    If term r has i_r, j_r, k_r, then:
    u[i_r*m+j_r, r] = a_r (nonzero), u[other, r] = 0
    v[j_r*p+k_r, r] = b_r (nonzero), v[other, r] = 0
    w[k_r*n+i_r, r] = c_r (nonzero), w[other, r] = 0
    
    Then T[i*m+j, j*p+k, k*n+i] = sum_{r: i_r=i, j_r=j, k_r=k} a_r*b_r*c_r
    
    This means: for each (i,j,k) with T=1, we need at least one term
    covering it, and the product of weights must sum to 1.
    For (i,j,k) with T=0, no term should cover it (or weights cancel).
    """
    # Since T is 0-1 (indicator of valid (i,j,k) triples), and all entries
    # are non-negative, the simplest valid decomposition is:
    # one term per (i,j,k) with T=1, each with a_r=b_r=c_r=1
    # That gives rank = n*m*p = 40 (schoolbook).
    #
    # To go lower, we need terms that cover multiple (i,j,k) points.
    # A term (i_r, j_r, k_r) only contributes to T[i_r*m+j_r, j_r*p+k_r, k_r*n+i_r].
    # So each term covers exactly ONE (i,j,k) point.
    # Therefore, we need at least n*m*p terms... unless we allow
    # terms where the indices don't align perfectly.
    #
    # Wait - that's the key constraint. Let me reconsider.
    # T[a,b,c] is nonzero only when a=i*m+j, b=j*p+k, c=k*n+i for some (i,j,k).
    # A rank-1 term u⊗v⊗w has u[a,r]*v[b,r]*w[c,r] nonzero when
    # u[a,r]≠0, v[b,r]≠0, w[c,r]≠0 simultaneously.
    # If u has support {a1,...,ak}, v has support {b1,...,bl}, w has support {c1,...,cm},
    # the term contributes to all (a,b,c) in the Cartesian product of supports.
    #
    # For the term to only contribute to valid (i,j,k) points, we need:
    # for all a in supp(u), b in supp(v), c in supp(w):
    #   a=i*m+j, b=j*p+k, c=k*n+i for some consistent (i,j,k)
    #
    # This is very restrictive. Let's think about it differently.
    #
    # Actually, the term CAN contribute to invalid points as long as the
    # total sum over all terms gives 0 there. With non-negative weights this
    # is impossible. With signed weights, it's possible but hard to find.
    
    # Let me try a different approach: use the fact that the problem is
    # small and do a more careful search.
    
    # Approach: For each rank r, try to find U, V, W via alternating
    # least squares with multiple random restarts, checking for exactness.
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    T_flat_a = T.reshape(nm, mp * pn)  # T[a, b*pn+c]
    T_flat_b = T.transpose(1, 0, 2).reshape(mp, nm * pn)  # T[b, a*pn+c]
    T_flat_c = T.transpose(2, 0, 1).reshape(pn, nm * mp)  # T[c, a*mp+b]
    
    rng = np.random.default_rng(seed)
    
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        # Random initialization
        U = rng.standard_normal((nm, rank))
        V = rng.standard_normal((mp, rank))
        W = rng.standard_normal((pn, rank))
        
        # Normalize columns to avoid scale issues
        for r in range(rank):
            norm = np.linalg.norm(U[:, r]) * np.linalg.norm(V[:, r]) * np.linalg.norm(W[:, r])
            if norm > 1e-10:
                scale = norm ** (1.0 / 3.0)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        # ALS iterations
        converged = False
        for it in range(300):
            if time.time() > deadline:
                break
            
            # Update U: for each a, solve min ||T[a] - sum_r U[a,r] * (v_r ⊗ w_r)||
            # M_U[r, idx] = V[idx//pn, r] * W[idx%pn, r]
            M_U = (V[:, :, None] * W[None, :, :]).transpose(1, 0, 2).reshape(rank, mp * pn)
            try:
                # Solve M_U @ U[a,:] = T[a,:].flatten() for each a
                # Equivalent to: U = lstsq(M_U.T, T_flat_a.T).T
                Y = np.linalg.lstsq(M_U.T, T_flat_a.T, rcond=None)[0]
                U = Y.T
            except:
                break
            
            # Update V
            M_V = (U[:, :, None] * W[None, :, :]).transpose(1, 0, 2).reshape(rank, nm * pn)
            try:
                Y = np.linalg.lstsq(M_V.T, T_flat_b.T, rcond=None)[0]
                V = Y.T
            except:
                break
            
            # Update W
            M_W = (U[:, :, None] * V[None, :, :]).transpose(1, 0, 2).reshape(rank, nm * mp)
            try:
                Y = np.linalg.lstsq(M_W.T, T_flat_c.T, rcond=None)[0]
                W = Y.T
            except:
                break
            
            # Check convergence
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.max(np.abs(T_recon - T))
            if err < 1e-10:
                converged = True
                break
        
        if converged:
            # Verify and round
            if _verify(U, V, W, T):
                # Try rounding to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify(U_r, V_r, W_r, T):
                    return U_r, V_r, W_r
                else:
                    return U, V, W
    
    return None

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    nm = n * m
    mp = m * p
    pn = p * n
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
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    deadline = time.time() + 135  # Safety margin
    
    # Try ranks from low to high
    # For 2x4x5, schoolbook is 40. Let's try to do better.
    # Known lower bounds suggest rank >= 25 or so for this size.
    
    best_result = None
    
    # Try structured ranks
    for rank in range(24, 41):
        if time.time() > deadline:
            break
        
        result = _try_rank_structured(n, m, p, rank, T, deadline, seed=rank)
        if result is not None:
            best_result = result
            break  # Found a valid decomposition at this rank
    
    if best_result is None:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
