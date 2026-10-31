import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 50

def compute_ratio(h):
    """
    Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0.
    """
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h)
    if l1 < 1e-300:
        return 0.0
    
    l2_sq = np.sum(h * h)
    
    # Efficient autoconvolution using FFT
    n_fft = 1 << (2 * N - 1).bit_length()
    H = np.fft.rfft(h, n=n_fft)
    hh = np.fft.irfft(H * H, n=n_fft)
    hh_max = np.max(hh)
    
    if hh_max < 1e-300:
        return 0.0
        
    return l2_sq / (l1 * hh_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Strategy:
    # 1. The objective function R(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf) is scale invariant.
    # 2. We suspect the optimal solution is sparse or has a specific structure (like a Sidon set).
    # 3. We use a greedy search to find a good support set S (subset of {0..N-1}).
    # 4. For a fixed support S, we optimize the weights w_i (for i in S) using log-space L-BFGS-B.
    #    Let h_i = exp(z_i) for i in S, and h_i = 0 otherwise.
    #    This ensures h_i > 0 automatically.
    # 5. We iterate: pick a candidate support, optimize weights, evaluate.
    #    Then try to improve the support by swapping elements or adding/removing.
    
    # Helper to compute ratio from log-weights on a support
    def ratio_from_logs(z, support):
        h = np.zeros(N)
        h[support] = np.exp(z)
        return compute_ratio(h)
    
    # Helper to optimize weights for a given support
    def optimize_weights(support, z0, time_limit):
        if len(support) == 0:
            return np.array([]), 0.0
        
        def obj(z):
            if time.time() > time_limit:
                raise Exception("Time limit exceeded")
            h = np.zeros(N)
            h[support] = np.exp(z)
            r = compute_ratio(h)
            return -r
        
        # Initial guess: all weights equal (z=0)
        z_init = np.zeros(len(support))
        
        try:
            res = minimize(obj, z_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                return res.x, -res.fun
            else:
                # Fallback to initial if optimization fails
                return z_init, -obj(z_init)
        except Exception:
            return z_init, -obj(z_init)

    # Greedy Support Search
    # Start with a small random support
    rng = np.random.default_rng(42)
    
    best_score = 0.0
    best_support = []
    best_z = []
    
    # Try different support sizes
    # Based on inspiration, sparse supports might be good.
    # Let's try sizes from 5 to 25.
    
    current_support = set()
    
    # Initialize with a random set of size 10
    initial_size = 10
    current_support = set(rng.choice(N, size=initial_size, replace=False))
    
    # Optimize initial support
    z, score = optimize_weights(list(current_support), np.zeros(initial_size), deadline - 10)
    best_score = score
    best_support = list(current_support)
    best_z = z
    
    # Iterative improvement:
    # 1. Try adding a random element
    # 2. Try removing a random element
    # 3. Try swapping a random element with another
    
    max_iterations = 500
    iter_count = 0
    
    while time.time() < deadline - 5.0 and iter_count < max_iterations:
        iter_count += 1
        support_list = list(current_support)
        n_sup = len(support_list)
        
        # Determine move type
        move_type = rng.random()
        
        if move_type < 0.3:
            # Add element
            if n_sup < 30:
                candidates = [i for i in range(N) if i not in current_support]
                if not candidates:
                    continue
                new_idx = rng.choice(candidates)
                new_support = set(current_support)
                new_support.add(new_idx)
                # Optimize with new support
                # Use previous z as warm start (padding with 0 for new element)
                z_new = np.zeros(len(new_support))
                # Map old z to new indices
                old_indices = sorted(current_support)
                new_indices = sorted(new_support)
                for i, idx in enumerate(old_indices):
                    if idx in new_indices:
                        pos_in_new = new_indices.index(idx)
                        z_new[pos_in_new] = best_z[old_indices.index(idx)]
                
                z_opt, score_opt = optimize_weights(list(new_support), z_new, deadline - 5.0)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_support = list(new_support)
                    best_z = z_opt
                    current_support = new_support
                # Accept move if it's not too bad (simulated annealing style? or just greedy?)
                # Let's stick to greedy for now to avoid getting stuck in local minima of the support.
                # But if we only accept improvements, we might get stuck.
                # Let's try a simple hill climbing with random restarts if stuck.
                # For now, greedy.
                
        elif move_type < 0.6:
            # Remove element
            if n_sup > 3:
                remove_idx = rng.choice(support_list)
                new_support = set(current_support)
                new_support.remove(remove_idx)
                new_indices = sorted(new_support)
                
                # Construct z_new
                z_new = np.zeros(len(new_support))
                old_indices = sorted(current_support)
                for i, idx in enumerate(new_indices):
                    pos_in_old = old_indices.index(idx)
                    z_new[i] = best_z[pos_in_old]
                
                z_opt, score_opt = optimize_weights(list(new_support), z_new, deadline - 5.0)
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_support = list(new_support)
                    best_z = z_opt
                    current_support = new_support
                    
        else:
            # Swap element
            if n_sup > 0:
                old_idx = rng.choice(support_list)
                candidates = [i for i in range(N) if i not in current_support]
                if candidates:
                    new_idx = rng.choice(candidates)
                    new_support = set(current_support)
                    new_support.remove(old_idx)
                    new_support.add(new_idx)
                    new_indices = sorted(new_support)
                    
                    z_new = np.zeros(len(new_support))
                    old_indices = sorted(current_support)
                    for i, idx in enumerate(new_indices):
                        if idx in old_indices:
                            pos_in_old = old_indices.index(idx)
                            z_new[i] = best_z[pos_in_old]
                        else:
                            z_new[i] = 0.0 # New element starts at weight 1 (log=0)
                    
                    z_opt, score_opt = optimize_weights(list(new_support), z_new, deadline - 5.0)
                    
                    if score_opt > best_score:
                        best_score = score_opt
                        best_support = list(new_support)
                        best_z = z_opt
                        current_support = new_support

    # Final reconstruction
    if best_score <= 0:
        # Fallback to uniform
        return [1.0] * N
        
    h_final = np.zeros(N)
    h_final[best_support] = np.exp(best_z)
    
    # Ensure no underflow/overflow issues by scaling? 
    # Ratio is scale invariant, so we can just return as is.
    # But very small numbers might be bad for floating point precision in evaluator?
    # Let's scale so max(h) = 1.0
    max_h = np.max(h_final)
    if max_h > 0:
        h_final = h_final / max_h
        
    return list(h_final)

if __name__ == "__main__":
    h = construct()
    print(compute_ratio(h))
