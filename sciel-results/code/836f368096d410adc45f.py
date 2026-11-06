import numpy as np
import time

def construct():
    N = 24
    deadline = time.time() + 140.0  # Safety margin
    sqrt_N = np.sqrt(N)
    
    # Precompute V
    M = 2048  # Higher resolution for better coarse search
    theta = 2.0 * np.pi * np.arange(M) / M
    V = np.exp(1j * np.outer(theta, np.arange(N)))
    
    def full_score(c):
        """Compute the accurate score with golden-section refinement."""
        vals = V @ c
        mags = np.abs(vals)
        top_k = 10
        idx = np.argsort(mags)[-top_k:]
        best = 0.0
        for i in idx:
            m_int = int(i)
            dtheta = 2.0 * np.pi / M
            a = theta[m_int] - dtheta
            b = theta[m_int] + dtheta
            
            # Golden section search for max |P(t)|
            invphi = (np.sqrt(5) - 1) / 2
            t1 = a + (1 - invphi) * (b - a)
            t2 = a + invphi * (b - a)
            
            # Helper to evaluate P at t
            # c is fixed during this refinement
            def eval_p(t):
                return np.sum(c * np.exp(1j * np.arange(N) * t))
                
            f1 = np.abs(eval_p(t1))
            f2 = np.abs(eval_p(t2))
            
            for _ in range(50):
                if f1 < f2:
                    b = t2
                    t2 = t1
                    f2 = f1
                    t1 = a + (1 - invphi) * (b - a)
                    f1 = np.abs(eval_p(t1))
                else:
                    a = t1
                    t1 = t2
                    f1 = f2
                    t2 = a + invphi * (b - a)
                    f2 = np.abs(eval_p(t2))
                    
            tm = (a + b) / 2.0
            fm = np.abs(eval_p(tm))
            if fm > best:
                best = fm
        return best / sqrt_N

    def fast_score(c, vals):
        """Fast approximate score using grid max."""
        return np.max(np.abs(vals)) / sqrt_N

    def local_search_fast(c):
        """Local search using fast_score for comparisons."""
        c = c.copy()
        vals = V @ c
        best_score = fast_score(c, vals)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            best_i = -1
            best_new_score = best_score
            for i in range(N):
                # Flip i
                # new_vals = vals - 2 * c[i] * V[:, i]
                new_vals = vals - 2 * c[i] * V[:, i]
                new_score = fast_score(c, new_vals)
                
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_i = i
                    
            if best_i != -1:
                # Apply flip
                c[best_i] = -c[best_i]
                vals = vals - 2 * c[best_i] * V[:, best_i] # Wait, c[best_i] is now flipped
                # Correction:
                # old_c = c[best_i] before flip
                # new_vals = vals - 2 * old_c * V[:, best_i]
                # We computed new_vals using c[i] (old) in the loop?
                # In the loop: new_vals = vals - 2 * c[i] * V[:, i]. Here c[i] is the CURRENT c[i] (old).
                # So new_vals is correct.
                # Now we update c and vals.
                # c[best_i] is flipped in place.
                # vals is updated to new_vals.
                c[best_i] = -c[best_i]
                vals = new_vals
                best_score = best_new_score
                improved = True
                
        return c

    # Initial guesses
    starts = []
    # Golay-RS
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    starts.append(np.array(p[:N]))
    
    # Random
    np.random.seed(42)
    for _ in range(10):
        starts.append(np.random.choice([-1, 1], N))
        
    # All 1
    starts.append(np.ones(N, dtype=int))
    
    best_global_c = None
    best_global_score = float('inf')
    
    for start in starts:
        if time.time() > deadline - 10.0:
            break
        c_fast = local_search_fast(start)
        s_full = full_score(c_fast)
        if s_full < best_global_score:
            best_global_score = s_full
            best_global_c = c_fast.copy()
            
    # ILS Loop
    current_c = best_global_c.copy()
    current_score = best_global_score
    
    k_pert = 3
    
    while time.time() < deadline - 5.0:
        # Perturb
        c_pert = current_c.copy()
        idxs = np.random.choice(N, k_pert, replace=False)
        c_pert[idxs] = -c_pert[idxs]
        
        # Local Search
        c_new = local_search_fast(c_pert)
        s_new = full_score(c_new)
        
        # Acceptance
        if s_new < current_score:
            current_c = c_new
            current_score = s_new
            if s_new < best_global_score:
                best_global_score = s_new
                best_global_c = current_c.copy()
            # Reduce perturbation on success
            k_pert = max(2, k_pert - 1)
        else:
            # Increase perturbation on failure occasionally
            if np.random.rand() < 0.2:
                k_pert = min(10, k_pert + 1)
                
    return list(best_global_c)
