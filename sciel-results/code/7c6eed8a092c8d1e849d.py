import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _generate_sidon_set(n, k):
    """
    Greedy generation of a Sidon-like set of size k from {0, ..., n-1}.
    A Sidon set has distinct pairwise sums.
    """
    S = []
    sums = set()
    # Start with 0
    S.append(0)
    sums.add(0) # 0+0? No, usually distinct pairs or a+b with a<=b. 
    # Let's stick to the standard: a+b = c+d => {a,b}={c,d}.
    # For a set S, we want all sums s_i + s_j (i<=j) to be unique.
    
    # Reset for clarity
    S = [0]
    sums = {0}
    
    for i in range(1, n):
        if len(S) >= k:
            break
        
        # Check if adding i creates duplicate sums
        valid = True
        new_sums = set()
        
        for s in S:
            val = s + i
            if val in sums:
                valid = False
                break
            new_sums.add(val)
        
        # Also need to check if i+i creates a duplicate?
        # The loop above covers s=i if i was in S, but i is new.
        # The loop covers s in S (old elements).
        # We also need to check i+i.
        if valid:
            val = 2 * i
            if val in sums:
                valid = False
            else:
                new_sums.add(val)
        
        if valid:
            sums.update(new_sums)
            S.append(i)
            
    return S

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit (10s buffer)
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # 1. Parametric search for good initializations
    # Gaussian
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(1, N/2, 15):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # Triangular
    for c in np.linspace(2, N-2, 15):
        for w in np.linspace(1, N, 15):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()
    
    # 2. Sidon-like sparse initializations
    # These have low autoconvolution peaks relative to their L2 norm
    for k in range(3, 15):
        for _ in range(10): # Multiple random restarts for Sidon generation (though greedy is deterministic based on start)
            if time.time() > deadline: break
            # Greedy Sidon starting from 0 is deterministic.
            # To get variety, we can try shifting or permuting?
            # Or just rely on the optimization to refine the weights.
            # Let's just generate the standard Sidon set for this k.
            sidon_idx = _generate_sidon_set(N, k)
            if len(sidon_idx) < 2:
                continue
            h_sidon = np.zeros(N)
            h_sidon[sidon_idx] = 1.0
            # Add small noise to avoid exact symmetries or numerical issues
            h_sidon += np.random.rand(N) * 0.01
            s = _score_raw(h_sidon)
            if s > best_score:
                best_score = s
                best_h = h_sidon.copy()
                
            # Also try a shifted Sidon set
            for shift in [1, 2, 3]:
                if shift + sidon_idx[-1] >= N: continue
                h_shift = np.zeros(N)
                h_shift[sidon_idx + shift] = 1.0
                h_shift += np.random.rand(N) * 0.01
                s = _score_raw(h_shift)
                if s > best_score:
                    best_score = s
                    best_h = h_shift.copy()

    # 3. Random sparse initialization
    for _ in range(100):
        if time.time() > deadline: break
        h_rand = np.zeros(N)
        num_peaks = np.random.randint(5, 15)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_rand[indices] = 1.0
        # Add some noise
        h_rand += np.random.rand(N) * 0.1
        s = _score_raw(h_rand)
        if s > best_score:
            best_score = s
            best_h = h_rand.copy()

    # Optimization using SLSQP on h directly
    # Objective: Maximize L2^2 / (L1 * AutoMax)
    # Scale invariant? No.
    # But if we fix L1=1, we maximize L2^2 / AutoMax.
    # Let's optimize with L1=1 constraint.
    
    def neg_obj_h(h):
        l1 = np.sum(h)
        if l1 == 0: return 0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0: return 0
        return -(l2_sq / max_conv) # Assuming L1=1

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
        {'type': 'ineq', 'fun': lambda h: h}
    ]
    bounds = [(0.0, 1.0)] * N
    
    # Collect initializations for optimization
    inits = []
    
    # 1. Best found so far
    h_start = best_h / np.sum(best_h)
    inits.append(h_start)
    
    # 2. Uniform
    inits.append(np.ones(N) / N)
    
    # 3. Gaussians
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N/2, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g)
            
    # 4. Triangles
    for c in np.linspace(5, N-5, 5):
        for w in np.linspace(1, N, 5):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Sparse (Sidon-like)
    for k in range(5, 12):
        sidon_idx = _generate_sidon_set(N, k)
        if len(sidon_idx) >= 2:
            h_s = np.zeros(N)
            h_s[sidon_idx] = 1.0
            h_s /= np.sum(h_s)
            inits.append(h_s)
            
            # Shifted
            for shift in [1, 2]:
                if shift + sidon_idx[-1] < N:
                    h_s2 = np.zeros(N)
                    h_s2[sidon_idx + shift] = 1.0
                    h_s2 /= np.sum(h_s2)
                    inits.append(h_s2)
        
    # 6. Random smooth
    for _ in range(20):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r /= np.sum(h_r)
        inits.append(h_r)

    # 7. Random sparse
    for _ in range(20):
        h_s = np.zeros(N)
        k = np.random.randint(5, 15)
        idx = np.random.choice(N, k, replace=False)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)

    # Optimize from each init
    for h_init in inits:
        if time.time() > deadline - 5: break
        
        if np.sum(h_init) == 0: continue
        h_init = h_init / np.sum(h_init)
        
        try:
            res = minimize(neg_obj_h, h_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if res.success:
                h_candidate = res.x
                # Ensure non-negativity
                h_candidate = np.maximum(h_candidate, 0)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-14})
            if res.success:
                h_candidate = res.x
                h_candidate = np.maximum(h_candidate, 0)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
            
    return list(best_h)
