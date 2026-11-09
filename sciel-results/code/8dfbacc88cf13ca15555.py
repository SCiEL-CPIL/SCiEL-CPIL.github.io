# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def entropy_from_counts(counts):
    """Calculate Shannon entropy in bits from counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def compute_score(P):
    """
    Compute the ratio D/M.
    P is a 10x10 array where P[y, x] is the probability.
    """
    # Check validity
    if P.shape != (10, 10):
        return 0.0
    if not np.all(np.isfinite(P)):
        return 0.0
    if np.any(P < -1e-9):
        return 0.0
    total_sum = np.sum(P)
    if total_sum <= 1e-9:
        return 0.0
    
    # Normalize
    P = P / total_sum
    
    # Extract x and y indices
    # x is column index, y is row index
    # P[y, x]
    y_idx, x_idx = np.indices((10, 10))
    
    # Calculate values for each slope
    # D = H_{-1}: distribution of x - y
    vals_d = x_idx - y_idx
    # M components: H_0 (x), H_1 (x+y), H_2 (x+2y)
    vals_h0 = x_idx
    vals_h1 = x_idx + y_idx
    vals_h2 = x_idx + 2 * y_idx
    
    # Pool cells with equal values and compute entropy
    def entropy_of_projection(vals, P):
        # Get unique values and sum probabilities for each
        unique_vals = np.unique(vals)
        probs = []
        for v in unique_vals:
            mask = (vals == v)
            prob = np.sum(P[mask])
            if prob > 0:
                probs.append(prob)
        if len(probs) == 0:
            return 0.0
        probs = np.array(probs)
        probs = probs / probs.sum()  # ensure normalized
        probs = probs[probs > 0]
        if len(probs) == 0:
            return 0.0
        return -np.sum(probs * np.log2(probs))
    
    D = entropy_of_projection(vals_d, P)
    H0 = entropy_of_projection(vals_h0, P)
    H1 = entropy_of_projection(vals_h1, P)
    H2 = entropy_of_projection(vals_h2, P)
    M = max(H0, H1, H2)
    
    if M <= 1e-9:
        return 0.0
    
    return D / M

def generate_random_distribution(seed):
    """Generate a random distribution with structure to favor high D/M ratio."""
    rng = np.random.RandomState(seed)
    n = 10
    P = np.ones((n, n), dtype=np.float64)
    
    # Strategy: emphasize cells where x-y takes extreme values (more spread)
    # while keeping x, x+y, x+2y relatively concentrated
    
    # Add random noise
    P += rng.dirichlet(np.ones(n*n)).reshape(n, n)
    
    # Emphasize corners to spread x-y
    # x-y ranges from -9 to 9
    # High |x-y| at corners: (0,9), (9,0)
    P[0, 9] *= 2.0  # y=0, x=9: x-y=9
    P[9, 0] *= 2.0  # y=9, x=0: x-y=-9
    
    # Suppress center to reduce entropy of x, x+y, x+2y
    center_mask = (y_idx >= 3) & (y_idx <= 6) & (x_idx >= 3) & (x_idx <= 6)
    P[center_mask] *= 0.3
    
    P = np.clip(P, 0, None)
    total = P.sum()
    if total > 0:
        P = P / total
    
    return P

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    start_time = time.time()
    deadline = 620.0  # 620s to leave safety margin before 630s kill
    
    # Start with uniform
    n = 10
    best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    best_score = compute_score(best_P)
    
    # Precompute indices
    y_idx, x_idx = np.indices((n, n))
    
    # Try structured random distributions
    for seed in range(5000):
        if time.time() > deadline:
            break
        
        P = generate_random_distribution(seed)
        score = compute_score(P)
        
        if score > best_score:
            best_score = score
            best_P = P.copy()
    
    # Local optimization: gradient-free search
    if time.time() < deadline - 10:
        current_P = best_P.copy()
        current_score = best_score
        step_size = 0.1
        
        for iteration in range(2000):
            if time.time() > deadline:
                break
            
            # Random perturbation
            delta = np.random.randn(n, n) * step_size
            new_P = current_P + delta
            new_P = np.clip(new_P, 0, None)
            
            total = new_P.sum()
            if total <= 1e-9:
                continue
            new_P = new_P / total
            
            new_score = compute_score(new_P)
            
            if new_score > current_score:
                current_P = new_P
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_P = current_P.copy()
                step_size = min(step_size * 1.1, 0.5)
            else:
                step_size = max(step_size * 0.9, 0.01)
    
    # Final validation and normalization
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total > 0:
        best_P = best_P / total
    else:
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    
    return best_P

# EVOLVE-BLOCK-END
