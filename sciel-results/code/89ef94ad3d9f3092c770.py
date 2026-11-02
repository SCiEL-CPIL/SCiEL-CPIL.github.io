import numpy as np
import time

N = 50

def compute_score(h):
    h = np.asarray(h, dtype=float)
    if len(h) != N:
        return 0.0
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    c = np.correlate(h, h, mode='full')
    sup_c = np.max(c)
    if sup_c < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_c)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_h = np.zeros(N)
    best_score = 0.0

    def try_candidate(h):
        nonlocal best_h, best_score
        h = np.asarray(h, dtype=float)
        if np.any(h < -1e-12):
            h = np.maximum(h, 0.0)
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
        return s

    # 1. Baseline: 2-element
    h2 = np.zeros(N)
    h2[0] = 2.0 / 3.0
    h2[1] = 1.0 / 3.0
    try_candidate(h2)

    # 2. Baseline: uniform
    h_uni = np.ones(N) / N
    try_candidate(h_uni)

    # 3. Baseline: delta
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    try_candidate(h_delta)

    # 4. Sparse spike patterns (Sidon-like sets)
    # We want sets where all pairwise differences are unique to keep autoconvolution peak low relative to L2.
    # A simple way to generate good sparse sets is to use quadratic residues or just random search for small K.
    
    def find_good_sparse_set(K, trials=2000):
        """Find a set of K indices that maximizes the ratio for a 0/1 vector."""
        best_set = None
        best_r = 0.0
        # Try some structured sets first
        # Quadratic residues mod p
        for p in [53, 59, 61]:
            residues = set((i*i) % p for i in range(p))
            # Map to [0, N)
            indices = [r for r in residues if r < N]
            if len(indices) >= K:
                # Take first K
                h = np.zeros(N)
                for i in indices[:K]:
                    h[i] = 1.0
                r = compute_score(h)
                if r > best_r:
                    best_r = r
                    best_set = h.copy()
        
        # Random search
        np.random.seed(123)
        for _ in range(trials):
            indices = np.random.choice(N, K, replace=False)
            h = np.zeros(N)
            h[indices] = 1.0
            r = compute_score(h)
            if r > best_r:
                best_r = r
                best_set = h.copy()
        return best_set, best_r

    for K in [3, 4, 5, 6, 7, 8, 10, 12, 15]:
        if time.time() > deadline - 20:
            break
        h_sp, r_sp = find_good_sparse_set(K, trials=1000)
        if h_sp is not None:
            try_candidate(h_sp)

    # 5. Coordinate Descent Refinement
    # This is the core refinement machinery. It takes a candidate and tries to improve it
    # by adjusting individual coordinates.
    
    def coordinate_descent(h, max_iter=200, fine=False):
        """Cyclic coordinate descent for maximizing the ratio."""
        h = h.copy()
        current_score = compute_score(h)
        
        # Determine step sizes based on scale
        scale = np.sum(h)
        if scale < 1e-10:
            return h
            
        if fine:
            deltas = [-0.01, -0.005, -0.001, 0.0, 0.001, 0.005, 0.01]
            factors = [0.99, 1.0, 1.01]
            max_iter = 500
        else:
            deltas = [-0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
            factors = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
            max_iter = 100

        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                old = h[i]
                best_local = current_score
                best_val = old
                
                # Try additive changes
                for delta in deltas:
                    h[i] = max(0.0, old + delta)
                    s = compute_score(h)
                    if s > best_local:
                        best_local = s
                        best_val = h[i]
                
                # Try multiplicative changes
                for factor in factors:
                    h[i] = max(0.0, old * factor)
                    s = compute_score(h)
                    if s > best_local:
                        best_local = s
                        best_val = h[i]
                
                h[i] = best_val
                if best_local > current_score:
                    current_score = best_local
                    improved = True
            
            if not improved:
                break
        
        return h

    # Refine the best sparse sets found so far
    # We don't have access to the specific h_sp from the loop easily without storing them,
    # so let's re-run a few key ones or just refine the current best_h and a few known good structures.
    
    # Let's refine the best found so far
    if time.time() < deadline - 10:
        h_refined = coordinate_descent(best_h, max_iter=100)
        try_candidate(h_refined)
        
    # Refine specific good structures
    # The 2-element solution is often a local optimum, but maybe we can perturb it?
    # Let's try refining a few random sparse sets with coordinate descent.
    
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        K = np.random.randint(3, 10)
        indices = np.random.choice(N, K, replace=False)
        h_rand = np.zeros(N)
        h_rand[indices] = 1.0
        # Random weights
        h_rand[indices] = np.random.uniform(0.5, 1.5, K)
        
        h_ref = coordinate_descent(h_rand, max_iter=50)
        try_candidate(h_ref)

    # 6. Gradient-based / Multiplicative Update Approach (from Inspiration 2)
    # This helps escape local minima of coordinate descent by pushing mass away from the peak of the autoconvolution.
    
    def multiplicative_update(h_init, max_iters=2000, eta_init=0.05):
        h = h_init.copy()
        if np.sum(h) < 1e-10:
            return h
        h = h / np.sum(h)
        
        eta = eta_init
        best_local_h = h.copy()
        best_local_score = compute_score(h)
        
        for i in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            conv = np.convolve(h, h, mode='full')
            sup_conv = np.max(conv)
            
            if sup_conv < 1e-15:
                break
                
            k_star = np.argmax(conv)
            
            # Gradient of peak convolution
            grad_h = np.zeros(N)
            for i_idx in range(N):
                val = 0.0
                idx1 = k_star - i_idx
                if 0 <= idx1 < N:
                    val += h[idx1]
                idx2 = i_idx - k_star
                if 0 <= idx2 < N:
                    val += h[idx2]
                grad_h[i_idx] = val
            
            # Multiplicative update
            update = np.exp(-eta * grad_h)
            h = h * update
            
            s = np.sum(h)
            if s < 1e-15:
                break
            h = h / s
            
            r = compute_score(h)
            if r > best_local_score:
                best_local_score = r
                best_local_h = h.copy()
            else:
                # Decay eta if no improvement
                eta *= 0.95
                if eta < 1e-4:
                    break
        
        return best_local_h

    # Apply multiplicative update to the current best
    if time.time() < deadline - 15:
        h_mu = multiplicative_update(best_h)
        try_candidate(h_mu)
        
        # And then refine the result of multiplicative update with coordinate descent
        if time.time() < deadline - 5:
            h_final = coordinate_descent(h_mu, max_iter=200, fine=True)
            try_candidate(h_final)

    # 7. Final Polish: High precision coordinate descent on the absolute best
    if time.time() < deadline - 2:
        h_polish = coordinate_descent(best_h, max_iter=500, fine=True)
        try_candidate(h_polish)

    return list(best_h)
