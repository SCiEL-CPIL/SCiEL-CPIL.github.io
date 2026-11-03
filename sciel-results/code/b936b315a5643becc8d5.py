import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s limit
    
    N = 100
    target_sum = N / 2.0
    
    # Linear correlation: C(k) = sum_i h[i] * (1 - h[i+k]) for k = 1..N-1
    # We want to minimize max_k C(k)
    
    def get_correlations_linear(h):
        g = 1.0 - h
        c = np.correlate(h, g, mode='full')
        # Exclude shift 0 (index N-1)
        # c[:N-1] are negative shifts, c[N:] are positive shifts
        c_nonzero = np.concatenate([c[:N-1], c[N:]])
        return c_nonzero
    
    def max_overlap(h):
        c = get_correlations_linear(h)
        return np.max(c)
    
    # Smooth max using log-sum-exp with temperature
    def smooth_max(c, temp=0.05):
        scaled = c / temp
        m = np.max(scaled)
        if m > 700:  # Avoid overflow
            return m * temp
        return temp * (m + np.log(np.sum(np.exp(scaled - m))))
    
    def objective(h, t=None):
        """Epigraph formulation: minimize t subject to C(k) <= t for all k"""
        c = get_correlations_linear(h)
        if t is None:
            return smooth_max(c)
        # Penalty for violating t
        viol = np.maximum(c - t, 0)
        return t + 100.0 * np.sum(viol ** 2)
    
    # Projection onto feasible set: 0 <= h_i <= 1, sum(h) = N/2
    def project(h):
        h = np.clip(h, 0, 1)
        # Adjust sum
        for _ in range(20):
            diff = target_sum - np.sum(h)
            if abs(diff) < 1e-12:
                break
            h += diff / N
            h = np.clip(h, 0, 1)
        return h
    
    # Swendsen-Wang style: identify clusters of similar heights
    def find_clusters(h, threshold=0.1):
        """Find contiguous blocks where heights are within threshold of each other"""
        clusters = []
        i = 0
        while i < N:
            j = i
            base_val = h[i]
            while j < N and abs(h[j] - base_val) < threshold:
                j += 1
            if j > i:
                clusters.append((i, j))
            i = j
        return clusters
    
    def cluster_move(h, clusters, rng):
        """Perform a non-local move on a cluster: shift all values in the cluster"""
        if not clusters:
            return h.copy()
        
        # Pick a random cluster
        idx = rng.integers(0, len(clusters))
        i, j = clusters[idx]
        length = j - i
        
        # Generate a new value for the entire cluster
        # The cluster will be set to a uniform value, then projected
        h_new = h.copy()
        
        # Strategy: shift the cluster's values by a random amount
        # This preserves the "block" structure while exploring
        shift = rng.uniform(-0.2, 0.2)
        h_new[i:j] += shift
        
        # Project back to feasible set
        h_new = project(h_new)
        return h_new
    
    def cluster_flip(h, clusters, rng):
        """Flip a cluster: replace values with 1 - values (antiferromagnetic flip)"""
        if not clusters:
            return h.copy()
        
        idx = rng.integers(0, len(clusters))
        i, j = clusters[idx]
        
        h_new = h.copy()
        h_new[i:j] = 1.0 - h[i:j]
        h_new = project(h_new)
        return h_new
    
    # Initialize
    rng = np.random.default_rng(42)
    
    # Start with constant solution
    h_curr = np.full(N, 0.5)
    best_h = h_curr.copy()
    best_score = max_overlap(best_h)
    
    # Simulated annealing with cluster moves
    T_init = 0.1
    T_final = 0.001
    n_iterations = 5000
    
    for it in range(n_iterations):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Temperature schedule
        progress = it / n_iterations
        T = T_init * (T_final / T_init) ** progress
        
        # Find clusters
        clusters = find_clusters(h_curr, threshold=0.05)
        
        # Choose move type
        move_type = rng.random()
        if move_type < 0.4:
            h_prop = cluster_move(h_curr, clusters, rng)
        elif move_type < 0.7:
            h_prop = cluster_flip(h_curr, clusters, rng)
        else:
            # Single-point perturbation
            h_prop = h_curr.copy()
            i = rng.integers(0, N)
            h_prop[i] += rng.uniform(-0.1, 0.1)
            h_prop = project(h_prop)
        
        # Acceptance criterion
        score_curr = max_overlap(h_curr)
        score_prop = max_overlap(h_prop)
        
        delta = score_prop - score_curr
        if delta < 0 or rng.random() < np.exp(-delta / T):
            h_curr = h_prop
        
        # Track best
        curr_score = max_overlap(h_curr)
        if curr_score < best_score:
            best_score = curr_score
            best_h = h_curr.copy()
    
    # Final refinement: gradient descent on the smooth objective
    if time.time() - start_time < time_limit - 10:
        h = best_h.copy()
        lr = 0.005
        v = np.zeros(N)
        momentum = 0.9
        
        for it in range(2000):
            if time.time() - start_time > time_limit - 3:
                break
            
            # Numerical gradient of smooth_max objective
            eps = 1e-5
            grad = np.zeros(N)
            c_base = get_correlations_linear(h)
            base_val = smooth_max(c_base)
            
            for i in range(N):
                h_plus = h.copy()
                h_plus[i] = min(1.0, h[i] + eps)
                h_minus = h.copy()
                h_minus[i] = max(0.0, h[i] - eps)
                
                # Renormalize to maintain sum
                h_plus = project(h_plus)
                h_minus = project(h_minus)
                
                c_plus = get_correlations_linear(h_plus)
                c_minus = get_correlations_linear(h_minus)
                
                val_plus = smooth_max(c_plus)
                val_minus = smooth_max(c_minus)
                
                grad[i] = (val_plus - val_minus) / (2 * eps)
            
            # Update
            v = momentum * v - lr * grad
            h = h + v
            h = project(h)
            
            # Check if improved
            curr_score = max_overlap(h)
            if curr_score < best_score:
                best_score = curr_score
                best_h = h.copy()
    
    # Ensure final solution is valid
    best_h = project(best_h)
    
    return best_h
# EVOLVE-BLOCK-END
