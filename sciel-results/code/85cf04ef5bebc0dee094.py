import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 115.0  # 120s limit, 5s margin

def compute_ratio(h):
    h = np.array(h, dtype=float)
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
    # We want to support N points. A Sidon set of size N has distinct sums.
    # If we place mass on a Sidon set, the autoconvolution peaks might be more spread out or have lower max relative to L2?
    # Actually, for non-negative h, the peak is at lag 0 (sum h_i^2).
    # Wait, the ratio is ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    # For non-negative h, ||h*h||_inf = (h*h)_0 = ||h||_2^2.
    # So the ratio is always 1 for non-negative h?
    # Let's check.
    # h_conv = np.convolve(h, h, 'full').
    # (h*h)_k = sum_i h_i h_{k-i}.
    # For k=0, sum_i h_i h_{-i}. Since h is defined on 0..N-1, h_{-i}=0 for i>0.
    # So (h*h)_0 = h_0^2.
    # Wait, standard convolution definition: (f*g)[n] = sum_m f[m] g[n-m].
    # If h is supported on [0, N-1], then h[m] is 0 for m<0 or m>=N.
    # (h*h)[0] = sum_m h[m] h[-m]. Only m=0 contributes. So (h*h)[0] = h[0]^2.
    # (h*h)[1] = h[0]h[1] + h[1]h[0] = 2 h[0]h[1].
    # ...
    # (h*h)[N-1] = sum_{m=0}^{N-1} h[m] h[N-1-m].
    # The maximum of h*h is NOT necessarily at lag 0.
    # For example, if h is a delta at N/2, h*h is a delta at N.
    # If h is uniform, h*k is a triangle peaking at the center.
    # For uniform h of length N, peak is N * (1/N)^2 = 1/N.
    # ||h||_2^2 = N * (1/N)^2 = 1/N.
    # ||h||_1 = 1.
    # Ratio = (1/N) / (1 * 1/N) = 1.
    #
    # If h is a spike at center:
    # h[k] = 1 if k=N/2 else 0.
    # ||h||_2^2 = 1.
    # ||h||_1 = 1.
    # h*h is a spike at 2*(N/2) = N.
    # ||h*h||_inf = 1.
    # Ratio = 1 / (1 * 1) = 1.
    #
    # Is it possible to get > 1?
    # Ratio = ||h||_2^2 / (||h||_1 * max_k (h*h)_k).
    # Since h >= 0, (h*h)_k >= 0.
    # Also (h*h)_0 = h_0^2.
    # By Cauchy-Schwarz, (h*h)_k = sum h_i h_{k-i} <= sqrt(sum h_i^2) sqrt(sum h_{k-i}^2) = ||h||_2^2.
    # So max_k (h*h)_k <= ||h||_2^2.
    # Therefore Ratio >= ||h||_2^2 / (||h||_1 * ||h||_2^2) = 1/||h||_1.
    # If we normalize ||h||_1 = 1, then Ratio >= 1.
    # Wait, the inequality is max_k (h*h)_k <= ||h||_2^2.
    # So 1/Ratio = ||h||_1 * max_k (h*h)_k / ||h||_2^2 <= ||h||_1.
    # So Ratio >= 1/||h||_1.
    # If we fix ||h||_1 = 1, Ratio >= 1.
    #
    # Can we make Ratio > 1?
    # We need max_k (h*h)_k < ||h||_2^2 / ||h||_1.
    # If ||h||_1 = 1, we need max_k (h*h)_k < ||h||_2^2.
    # This requires that the peak of the autoconvolution is strictly less than the L2 norm squared.
    # This happens if the mass is spread out such that no single lag k has a large sum of products.
    # The "spike" case has max_k (h*h)_k = ||h||_2^2 (at the lag corresponding to the double index).
    # The "uniform" case has max_k (h*h)_k = 1/N * (1/N) * N = 1/N? No.
    # Uniform h_i = 1/N.
    # ||h||_2^2 = N * (1/N)^2 = 1/N.
    # (h*h)_k = sum h_i h_{k-i}.
    # For k in [0, N-1], the number of non-zero terms is N-k (if k>=0).
    # Actually, for full convolution of length N vectors:
    # (h*h)_k = sum_{i=max(0, k-(N-1))}^{min(N-1, k)} h_i h_{k-i}.
    # For uniform h_i = c.
    # (h*h)_k = c^2 * (number of terms).
    # Max number of terms is N (at k=N-1).
    # So max (h*h)_k = N c^2 = N (1/N)^2 = 1/N.
    # Ratio = (1/N) / (1 * 1/N) = 1.
    #
    # It seems for non-negative h, the ratio is always 1?
    # Let's check the inequality again.
    # max_k (h*h)_k <= ||h||_2^2 is true.
    # But is it possible that max_k (h*h)_k is much smaller?
    # If h is a Sidon set, the sums i+j are distinct.
    # This doesn't directly help with the magnitude of the convolution.
    #
    # Let's try a random vector.
    #
    # Wait, the previous program scored 1.0819.
    # The seed program (all 1s) scored 1.0.
    # So it is possible to get > 1.
    #
    # Let's look at the inspiration.
    # "The seed program returns all 1s, which gives a ratio of 1.0"
    # "Combined program: Sidon-set structured initialization + L-BFGS-B refinement" scored 1.1484.
    #
    # So Sidon sets are key.
    # Why?
    # If h is supported on a Sidon set S, then for any k, the number of pairs (i, j) in S such that i+j=k is at most 1 (if k is in 2S) or 0.
    # Wait, Sidon set means all sums i+j (i<=j) are distinct.
    # So for any k, there is at most one pair (i,j) with i+j=k and i<=j.
    # If i!=j, then (h*h)_k = 2 h_i h_j.
    # If i=j, then (h*h)_k = h_i^2.
    # So (h*h)_k is either 0, h_i^2, or 2 h_i h_j.
    #
    # If we set h_i = 1/sqrt(|S|) for i in S, and 0 otherwise.
    # ||h||_1 = 1/sqrt(|S|).
    # ||h||_2^2 = |S| * (1/|S|) = 1.
    # max_k (h*h)_k = max(1/|S|, 2/|S|) = 2/|S| (if there are distinct pairs).
    # Ratio = 1 / (1/sqrt(|S|) * 2/|S|) = |S| / (2 sqrt(|S|)) = sqrt(|S|) / 2.
    # For N=50, if we can fit a Sidon set of size 50, we get sqrt(50)/2 ~ 3.5?
    # No, we have N=50 positions. We can choose which positions to use.
    # If we use all 50 positions, and they form a Sidon set, then |S|=50.
    # But can 50 integers form a Sidon set? Yes, e.g., powers of 2, or quadratic residues.
    # But the indices are 0..49.
    # A Sidon set of size 50 requires a range of at least 50^2 = 2500?
    # No, the minimum size of a Sidon set in [1, M] is roughly sqrt(M).
    # So to have a Sidon set of size 50, we need M ~ 2500.
    # But our indices are only 0..49.
    # So we CANNOT have a Sidon set of size 50 in [0, 49].
    # The maximum size of a Sidon set in [0, 49] is roughly sqrt(49) = 7.
    #
    # So we can only support ~7 non-zero entries if we want them to be a Sidon set.
    # If we use 7 entries, |S|=7.
    # Ratio ~ sqrt(7)/2 ~ 1.32.
    #
    # What if we use more entries but not a perfect Sidon set?
    # The optimization should find the best mix.
    #
    # Let's implement the Sidon initialization as suggested.
    
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

    # Final check
    if time.time() > DEADLINE - 5:
         pass

    # Ensure non-negative
    best_h = np.clip(best_h, 0, None)
    if np.sum(best_h) == 0:
         best_h = np.ones(N) / N
         
    return list(best_h)
