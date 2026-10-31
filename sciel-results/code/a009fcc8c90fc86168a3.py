import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 115.0 # 120s limit, 5s margin

def compute_ratio(h):
    h = np.array(h)
    if np.any(h < -1e-9):
        return -1.0
    h = np.clip(h, 0, None)
    
    l1 = np.sum(h)
    if l1 < 1e-12:
        return -1.0
        
    l2_sq = np.sum(h**2)
    
    # Linear convolution
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(h_conv))
    
    if max_conv < 1e-12:
        return -1.0
        
    return (l2_sq / (l1 * max_conv))

def is_sidon(arr):
    """Check if a set of integers is a Sidon set (B2 set)."""
    arr = sorted(set(arr))
    n = len(arr)
    sums = {}
    for i in range(n):
        for j in range(i, n):
            s = arr[i] + arr[j]
            if s in sums:
                return False
            sums[s] = (i, j)
    return True

def find_sidon_set_greedy(n, max_val=1000):
    """Find a Sidon set of size n using a greedy algorithm."""
    sidon = []
    for i in range(max_val):
        candidate = sidon + [i]
        if is_sidon(candidate):
            sidon.append(i)
            if len(sidon) == n:
                break
    return sidon

def local_search_swap(h, max_iters=1000, time_limit=10.0):
    """
    Perform local search by swapping values at two indices.
    This helps escape local minima in the continuous optimization by exploring discrete permutations of support weights.
    """
    h = h.copy()
    l1 = np.sum(h)
    if l1 < 1e-12:
        return h
    
    best_val = compute_ratio(h)
    start_time = time.time()
    no_improve = 0
    
    for _ in range(max_iters):
        if time.time() > time_limit:
            break
        if time.time() > DEADLINE - 5:
            break
            
        # Pick two random indices to swap
        i, j = np.random.choice(N, 2, replace=False)
        
        if i == j:
            continue
            
        # Try swapping
        h_new = h.copy()
        h_new[i], h_new[j] = h[j], h[i]
        
        val_new = compute_ratio(h_new)
        
        if val_new > best_val + 1e-12:
            h = h_new
            best_val = val_new
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 100:
                # Restart with random perturbation? Or just break?
                # Let's try a small random perturbation to escape
                h += np.random.randn(N) * 1e-3
                h = np.clip(h, 0, None)
                if np.sum(h) < 1e-12:
                    break
                best_val = compute_ratio(h)
                no_improve = 0
                
    return h

def optimize_sidon_weights(sidon_indices, time_limit=5.0):
    """
    Optimize weights on the given Sidon set indices.
    Returns the best h vector found.
    """
    if not sidon_indices:
        return np.zeros(N)
        
    K = len(sidon_indices)
    # Initial guess: uniform weights
    w0 = np.ones(K) / K
    
    def neg_ratio_sidon(w):
        # w are the weights for the Sidon indices
        # Scale invariance: we can fix sum(w)=1 or just let it vary.
        # Let's fix sum(w)=1 to simplify? No, the ratio is scale invariant.
        # We can just minimize -log(ratio) on the raw weights, but that might drift.
        # Better: normalize w to sum 1 before computing ratio?
        # R(h) is scale invariant. So we can fix ||h||_1 = 1.
        w_norm = w / np.sum(w)
        h = np.zeros(N)
        h[sidon_indices] = w_norm
        r = compute_ratio(h)
        if r <= 0:
            return 1000.0
        return -np.log(r)

    best_h = np.zeros(N)
    best_val = -1.0
    start_time = time.time()
    
    # Try multiple random restarts
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > start_time + time_limit:
            break
        if time.time() > DEADLINE - 2:
            break
            
        if i == 0:
            w_start = w0.copy()
        else:
            w_start = np.random.dirichlet(np.ones(K))
            
        try:
            # Nelder-Mead is good for low dimensions and non-smooth objectives
            res = minimize(neg_ratio_sidon, w_start, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.success or res.fun < 1000:
                w_opt = res.x
                w_opt = np.clip(w_opt, 0, None)
                if np.sum(w_opt) == 0:
                    continue
                w_opt = w_opt / np.sum(w_opt)
                
                h_opt = np.zeros(N)
                h_opt[sidon_indices] = w_opt
                val_opt = compute_ratio(h_opt)
                
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass
            
    return best_h

def optimize_basis_coefficients(num_modes=10, time_limit=10.0):
    """
    Optimize h using a basis of low-frequency DCT modes.
    This encourages smooth, spectrally flat solutions.
    h = B @ alpha, where B is N x num_modes.
    We optimize alpha >= 0.
    """
    # Construct DCT basis (Type II)
    # We use the first 'num_modes' modes.
    # Standard DCT-II: C[k, n] = cos(pi * k * (2n + 1) / (2N))
    n = np.arange(N)
    k = np.arange(num_modes)
    B = np.cos(np.pi * np.outer(k, 2*n + 1) / (2*N))
    
    # Normalize columns of B to have unit L2 norm for better conditioning?
    # Or just leave them. Let's normalize to unit L2 to keep scale manageable.
    B_norm = np.linalg.norm(B, axis=0)
    B = B / B_norm
    
    # Objective: Maximize ratio(h) where h = B @ alpha
    # We use log-space for alpha to ensure positivity?
    # Or just bound alpha >= 0.
    # Let's use L-BFGS-B with bounds alpha >= 0.
    
    def neg_log_ratio_basis(alpha):
        h = B @ alpha
        # Clip small negatives due to numerical error
        h = np.clip(h, 0, None)
        # Normalize h to L1 = 1 for scale invariance stability in optimization?
        # The ratio is scale invariant, so we can fix sum(h)=1.
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 1000.0
        h_norm = h / l1
        r = compute_ratio(h_norm)
        if r <= 0:
            return 1000.0
        return -np.log(r)
    
    best_h = np.zeros(N)
    best_val = -1.0
    start_time = time.time()
    
    # Try different numbers of modes? The function signature takes num_modes.
    # We will call this with a few different num_modes in construct().
    
    n_restarts = 10
    for i in range(n_restarts):
        if time.time() > start_time + time_limit:
            break
        if time.time() > DEADLINE - 2:
            break
            
        if i == 0:
            # Uniform alpha
            alpha_start = np.ones(num_modes) / num_modes
        else:
            # Random Dirichlet-like start (positive)
            alpha_start = np.random.dirichlet(np.ones(num_modes))
            
        try:
            # Bounds: alpha >= 0
            bounds = [(0, None)] * num_modes
            res = minimize(neg_log_ratio_basis, alpha_start, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.success or res.fun < 1000:
                alpha_opt = np.clip(res.x, 0, None)
                h_opt = B @ alpha_opt
                h_opt = np.clip(h_opt, 0, None)
                if np.sum(h_opt) > 0:
                    h_opt = h_opt / np.sum(h_opt)
                    val_opt = compute_ratio(h_opt)
                    if val_opt > best_val:
                        best_val = val_opt
                        best_h = h_opt
        except Exception:
            pass
            
    return best_h

def construct():
    start_time = time.time()
    
    # Initial candidates
    candidates = []
    
    # 1. Uniform
    h_uni = np.ones(N) / N
    candidates.append(h_uni)
    
    # 2. Spike
    h_spike = np.zeros(N)
    h_spike[N//2] = 1.0
    candidates.append(h_spike)
    
    # 3. Gaussian-like
    x = np.linspace(-1, 1, N)
    h_gauss = np.exp(-5 * x**2)
    h_gauss /= np.sum(h_gauss)
    candidates.append(h_gauss)
    
    # 4. Cosine bump
    h_cos = 0.5 * (1 + np.cos(np.pi * x))
    h_cos /= np.sum(h_cos)
    candidates.append(h_cos)
    
    # 5. Sidon set based initialization
    # Find a Sidon set of size roughly sqrt(N) or as large as possible within [0, N-1]
    # Greedy Sidon set in [0, N-1]
    sidon_indices = []
    for i in range(N):
        candidate = sidon_indices + [i]
        if is_sidon(candidate):
            sidon_indices.append(i)
    
    # Create h based on Sidon indices
    h_sidon = np.zeros(N)
    if sidon_indices:
        h_sidon[sidon_indices] = 1.0 / np.sqrt(len(sidon_indices))
    candidates.append(h_sidon)
    
    # Also try a "sparse" random initialization
    # Pick K random indices
    for K in [5, 10, 20]:
        h_sparse = np.zeros(N)
        indices = np.random.choice(N, K, replace=False)
        h_sparse[indices] = 1.0 / np.sqrt(K)
        candidates.append(h_sparse)

    best_h = candidates[0]
    best_val = compute_ratio(best_h)
    
    for h in candidates[1:]:
        val = compute_ratio(h)
        if val > best_val:
            best_val = val
            best_h = h
            
    # Optimization
    # Minimize -log(ratio)
    # Use log-transform for positivity
    def neg_log_ratio(params):
        h = np.exp(params)
        h_norm = h / np.sum(h)
        r = compute_ratio(h_norm)
        if r <= 0:
            return 1000.0
        return -np.log(r)
        
    # Random restarts
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > DEADLINE - 2:
            break
            
        if i == 0:
            x0 = np.log(best_h + 1e-6)
        elif i == 1 and sidon_indices:
            # Start from Sidon
            x0 = np.log(h_sidon + 1e-6)
        else:
            # Random sparse start
            K = np.random.randint(3, 15)
            h_start = np.zeros(N)
            indices = np.random.choice(N, K, replace=False)
            h_start[indices] = 1.0 / np.sqrt(K)
            x0 = np.log(h_start + 1e-6)
            
        try:
            res = minimize(neg_log_ratio, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12})
            if res.success:
                h_opt = np.exp(res.x)
                h_opt = h_opt / np.sum(h_opt)
                val_opt = compute_ratio(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass

        # Apply local search swap to the current best
        if time.time() > DEADLINE - 2:
            continue
            
        # Limit local search time
        remaining = DEADLINE - 2 - time.time()
        if remaining < 1.0:
            continue
            
        h_swapped = local_search_swap(best_h, max_iters=500, time_limit=min(remaining, 5.0))
        val_swapped = compute_ratio(h_swapped)
        if val_swapped > best_val:
            best_val = val_swapped
            best_h = h_swapped
            
    # Dedicated Sidon Set Optimization
    # This is crucial because the general optimizer might drift away from the sparse Sidon structure
    if time.time() < DEADLINE - 10 and sidon_indices:
        h_sidon_opt = optimize_sidon_weights(sidon_indices, time_limit=10.0)
        val_sidon_opt = compute_ratio(h_sidon_opt)
        if val_sidon_opt > best_val:
            best_val = val_sidon_opt
            best_h = h_sidon_opt

    # NEW: Basis Expansion Optimization
    # Try different numbers of DCT modes
    if time.time() < DEADLINE - 15:
        for num_modes in [5, 10, 15]:
            if time.time() > DEADLINE - 5:
                break
            h_basis_opt = optimize_basis_coefficients(num_modes=num_modes, time_limit=5.0)
            val_basis_opt = compute_ratio(h_basis_opt)
            if val_basis_opt > best_val:
                best_val = val_basis_opt
                best_h = h_basis_opt

    # NEW: Final Polish
    # Run a high-precision L-BFGS-B on the best solution found so far
    if time.time() < DEADLINE - 2:
        # Convert best_h to log-space for positivity constraint
        # Ensure best_h is strictly positive for log
        h_polish_start = best_h + 1e-6
        x0 = np.log(h_polish_start)
        
        try:
            res = minimize(neg_log_ratio, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.success or res.fun < 1000:
                h_opt = np.exp(res.x)
                h_opt = h_opt / np.sum(h_opt)
                val_opt = compute_ratio(h_opt)
                if val_opt > best_val:
                    best_val = val_opt
                    best_h = h_opt
        except Exception:
            pass

    # Final check
    if time.time() > DEADLINE - 5:
         pass

    # Ensure non-negative
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
         best_h = np.ones(N) / N
         
    return list(best_h)
