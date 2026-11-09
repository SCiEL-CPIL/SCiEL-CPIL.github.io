import numpy as np
import time
import sys

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Strategy: 
    1. Vectorized O(n^4) probability calculation.
    2. Global random search over variable support sizes (n=2 to n=8).
    3. "Heavy-tail" prior for support points (log-uniform) to explore large ranges.
    4. Simplex prior for weights.
    5. Deadline guard to ensure safe return.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s limit

    def calc_prob_fast(support, weights):
        """
        Vectorized calculation of P[X1+X2+X3 < 2*X4].
        support, weights: 1D arrays of length n.
        """
        n = len(support)
        if n == 0:
            return 0.0
        
        # Weights are assumed to sum to 1. 
        # To be safe against floating point drift in random generation, 
        # we can normalize here or ensure in generation.
        # The task says w_i = weights[i] / sum(weights).
        # So we can just use the raw weights and normalize at the end, 
        # or normalize now. Let's normalize now for correctness in the metric.
        w_sum = np.sum(weights)
        if w_sum <= 1e-9:
            return 0.0
        w_norm = weights / w_sum
        
        # Meshgrid for X1, X2, X3
        # X1, X2, X3 are independent.
        # We need sum of weights for all tuples (i,j,k) such that s_i+s_j+s_k < 2*s_l.
        
        # Efficient approach:
        # For each l (X4), compute the probability that X1+X2+X3 < 2*s_l.
        # This is the CDF of the sum of 3 i.i.d. variables at 2*s_l.
        
        p_total = 0.0
        s = support
        w = w_norm
        
        # Precompute all pairwise sums? No, n is small (<=8). 
        # n^3 is at most 512. n is at most 8. 
        # Total ops ~ 8 * 512 = 4096 per evaluation. Very fast.
        
        # Let's use broadcasting to compute all sums of 3.
        # s1, s2, s3 shapes: (n,1,1), (1,n,1), (1,1,n) -> sum shape (n,n,n)
        # Weights: w1, w2, w3 shapes: (n,1,1), (1,n,1), (1,1,n) -> prod shape (n,n,n)
        
        s_1 = s[:, np.newaxis, np.newaxis]
        s_2 = s[np.newaxis, :, np.newaxis]
        s_3 = s[np.newaxis, np.newaxis, :]
        
        sum_3 = s_1 + s_2 + s_3
        
        w_1 = w[:, np.newaxis, np.newaxis]
        w_2 = w[np.newaxis, :, np.newaxis]
        w_3 = w[np.newaxis, np.newaxis, :]
        
        prod_w = w_1 * w_2 * w_3
        
        # Now iterate over X4 (l)
        for l in range(n):
            threshold = 2.0 * s[l]
            # Sum of weights where sum_3 < threshold
            mask = sum_3 < threshold
            prob_sum_lt = np.sum(prod_w * mask)
            p_total += w[l] * prob_sum_lt
            
        return p_total

    def generate_random_measure(n_points, rng):
        """
        Generate a random measure with n_points support points.
        Uses log-uniform for support (to favor wide ranges) and Dirichlet for weights.
        """
        # Support: 
        # We want points in [0, inf). 
        # Log-uniform is tricky for 0. 
        # Let's pick points from [1e-3, 1e5] log-uniformly, then sort.
        # Or, pick ratios? 
        # Let's try: s_1 = 1.0, s_i = s_{i-1} * exp(U(0, 10)).
        # This ensures s_1 < s_2 < ... and allows for huge jumps.
        
        s = np.zeros(n_points)
        s[0] = 1.0
        for i in range(1, n_points):
            # Log-uniform step
            # Exp of uniform(0, 10) gives range [1, 22000]
            # This might be too wide or too narrow. 
            # Let's use a mixture: sometimes close, sometimes far.
            # Or just simple: sort random log-uniform points.
            pass
            
        # Better: Generate n-1 random log-uniform points in [1e-2, 1e5] and sort.
        # Add 0? No, 0 is not required, but often optimal. 
        # The current best has 0. 
        # Let's force one point to be 0? 
        # If we force 0, we lose generality if non-0 is better. 
        # But 0 is very likely optimal because it makes the LHS small.
        # Let's include 0 with high probability.
        
        if rng.random() < 0.8:
            s[0] = 0.0
            # Generate n-1 other points
            # Log-uniform in [1e-3, 1e5]
            logs = rng.uniform(np.log(1e-3), np.log(1e5), n_points - 1)
            s[1:] = np.sort(np.exp(logs))
        else:
            # No 0
            logs = rng.uniform(np.log(1e-3), np.log(1e5), n_points)
            s = np.sort(np.exp(logs))
            
        # Weights: Dirichlet(1) is uniform on simplex.
        # Maybe Dirichlet(alpha) with alpha < 1 to favor sparse weights?
        # Or alpha > 1 to favor balanced?
        # Let's use Dirichlet(1.0) for uniform coverage, 
        # but also try Dirichlet(0.5) for sparsity in some runs.
        
        if rng.random() < 0.5:
            alpha = 1.0
        else:
            alpha = 0.5
            
        w = rng.dirichlet(alpha * np.ones(n_points))
        
        return s, w

    best_score = -np.inf
    best_s = None
    best_w = None
    
    rng = np.random.default_rng(42)
    
    # We will iterate through different n and random samples.
    # n=2, 3, 4, 5, 6, 7, 8
    
    # To maximize coverage within time, we'll run a loop.
    # Each evaluation is ~1ms. 
    # 590s / 1ms = 590,000 evaluations.
    # We can do a lot of random restarts.
    
    # Let's structure the search:
    # Phase 1: Broad random search over n=2..8.
    # Phase 2: Local refinement around best found (optional if time permits).
    
    # For simplicity and robustness, just do massive random search.
    # If we find a high score, we can try to refine it.
    
    # Let's track the best few to potentially refine.
    
    count = 0
    while time.time() < deadline:
        # Choose n
        # Weighted probability towards smaller n? 
        # n=4 is current best. n=3 is simple.
        # Let's pick n uniformly from 2 to 8.
        n = rng.integers(2, 9)
        
        s, w = generate_random_measure(n, rng)
        
        p = calc_prob_fast(s, w)
        
        if p > best_score:
            best_score = p
            best_s = s.copy()
            best_w = w.copy()
            
            # If we find a very high score, we might want to try refining it?
            # For now, just keep searching.
            
        count += 1
        
        # Check time every 1000 iterations to avoid overhead
        if count % 1000 == 0:
            if time.time() > deadline - 5.0:
                break

    # If we have time, try a simple local refinement on the best found
    if time.time() < deadline - 10.0 and best_s is not None:
        # Simple coordinate descent / random perturbation
        current_s = best_s
        current_w = best_w
        current_score = best_score
        n = len(current_s)
        
        # Try 1000 perturbations
        for i in range(1000):
            if time.time() > deadline - 5.0:
                break
                
            # Perturb support
            new_s = current_s.copy()
            # Randomly pick a point and perturb it
            idx = rng.integers(0, n)
            # Log-normal perturbation
            factor = np.exp(rng.normal(0, 0.5))
            new_s[idx] *= factor
            # Ensure non-negative and sorted? 
            # If we break sorting, we need to re-sort and re-associate weights.
            # This is tricky. 
            # Better: perturb in the sorted space?
            # Let's just reject if not sorted or if negative.
            if np.any(new_s < 0) or np.any(np.diff(new_s) < 0):
                continue
                
            # Perturb weights
            new_w = current_w.copy()
            # Add small noise and renormalize
            noise = rng.normal(0, 0.1, n)
            new_w += noise
            new_w = np.maximum(new_w, 0)
            new_w_sum = np.sum(new_w)
            if new_w_sum < 1e-9:
                continue
            new_w /= new_w_sum
            
            p = calc_prob_fast(new_s, new_w)
            if p > current_score:
                current_score = p
                current_s = new_s
                current_w = new_w
                best_score = p
                best_s = current_s.copy()
                best_w = current_w.copy()

    # Fallback
    if best_s is None:
        # Return a safe default
        return np.array([0.0, 1.0, 10.0]), np.array([0.5, 0.3, 0.2])
        
    # Final validity check
    if np.any(best_s < -1e-9) or np.any(best_w < -1e-9) or np.sum(best_w) <= 1e-9:
        return np.array([0.0, 1.0, 10.0]), np.array([0.5, 0.3, 0.2])
        
    # Ensure float64
    best_s = best_s.astype(np.float64)
    best_w = best_w.astype(np.float64)
    
    return best_s, best_w

if __name__ == "__main__":
    s, w = beat_the_average_measure()
    print(s, w)
    print("Score:", calc_prob_fast(s, w))
