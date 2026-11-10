# EVOLVE-BLOCK-START
import numpy as np
import time

def _shannon_entropy_bits(probs):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _compute_ratios(P):
    """
    Compute D = H_{-1} and M = max(H_0, H_1, H_2) for a given 10x10 matrix P.
    P[y, x] corresponds to probability of (x, y).
    Returns (D, M, ratio) where ratio = D/M if M > 1e-9 else 0.
    """
    n = P.shape[0]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[i,j] = j, Y[i,j] = i? No.
    # Let's be careful. P is (y, x). So P[y, x].
    # We need distributions of:
    # H_{-1}: x - y
    # H_0: x
    # H_1: x + y
    # H_2: x + 2y
    
    # Flatten P to 100 elements
    P_flat = P.flatten()
    
    # Indices for x and y
    # x ranges 0..9, y ranges 0..9
    # In P[y, x], the first index is y, second is x.
    # Flatten order in numpy is row-major (C-order).
    # So P[y, x] -> index y*n + x.
    
    # Let's create arrays for x and y values corresponding to each flattened index
    # idx = y*10 + x
    # y = idx // 10
    # x = idx % 10
    idx = np.arange(n*n)
    y_vals = idx // n
    x_vals = idx % n
    
    # Values for each slope
    val_m1 = x_vals - y_vals       # x - y
    val_0  = x_vals                # x
    val_1  = x_vals + y_vals       # x + y
    val_2  = x_vals + 2*y_vals     # x + 2y
    
    # Compute histograms (distributions) for each
    # Since values are integers, we can use bincount or unique.
    # Range of x-y: -9 to 9 (19 values)
    # Range of x: 0 to 9 (10 values)
    # Range of x+y: 0 to 18 (19 values)
    # Range of x+2y: 0 to 27 (28 values)
    
    def get_entropy_from_values(vals, probs):
        # vals: array of integer values
        # probs: array of probabilities corresponding to vals
        # We need to sum probabilities for equal values, then compute entropy.
        # Using np.unique with return_inverse
        unique_vals, inverse = np.unique(vals, return_inverse=True)
        # Sum probabilities for each unique value
        dist = np.zeros(len(unique_vals))
        np.add.at(dist, inverse, probs)
        # Normalize? No, P already sums to 1. But due to floating point, let's check.
        # Actually, P is normalized, so sum(dist) should be 1.
        return _shannon_entropy_bits(dist)
    
    H_m1 = get_entropy_from_values(val_m1, P_flat)
    H_0  = get_entropy_from_values(val_0, P_flat)
    H_1  = get_entropy_from_values(val_1, P_flat)
    H_2  = get_entropy_from_values(val_2, P_flat)
    
    D = H_m1
    M = max(H_0, H_1, H_2)
    
    if M <= 1e-9:
        return D, M, 0.0
    ratio = D / M
    return D, M, ratio

def _gradient_ascent_step(P, step_size, lr=0.1):
    """
    Perform one step of gradient ascent on the ratio D/M.
    Uses numerical differentiation or analytical gradient if possible.
    For simplicity and robustness, we'll use a simple coordinate ascent or
    a small perturbation method if analytical gradient is too complex.
    
    Actually, let's try to compute the gradient analytically or use a simple
    stochastic gradient approach.
    
    Given the complexity of the entropy function (which is non-linear and
    involves max), a simple approach is to use a small perturbation in a
    random direction and keep it if it improves the score.
    """
    # This is a placeholder for a more sophisticated optimizer.
    # For now, let's implement a simple random search / hill climbing.
    pass

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds limit, keep margin
    
    # Initialize with uniform distribution
    P = np.ones((n, n), dtype=np.float64) / (n * n)
    _, _, best_score = _compute_ratios(P)
    best_P = P.copy()
    
    # Try a few heuristic initializations
    # 1. Diagonal concentration (x = y)
    P_diag = np.zeros((n, n))
    for i in range(n):
        P_diag[i, i] = 1.0 / n
    _, _, score_diag = _compute_ratios(P_diag)
    if score_diag > best_score:
        best_P = P_diag.copy()
        best_score = score_diag
    
    # 2. Anti-diagonal concentration (x + y = constant)
    # Let's try x + y = 9
    P_anti = np.zeros((n, n))
    for i in range(n):
        P_anti[i, n-1-i] = 1.0 / n
    _, _, score_anti = _compute_ratios(P_anti)
    if score_anti > best_score:
        best_P = P_anti.copy()
        best_score = score_anti
    
    # 3. Random sparse distributions
    rng = np.random.default_rng(42)
    for _ in range(50):
        P_rand = np.zeros((n, n))
        num_points = rng.integers(5, 20)
        points = rng.choice(n*n, size=num_points, replace=False)
        P_rand[points // n, points % n] = 1.0 / num_points
        _, _, score_rand = _compute_ratios(P_rand)
        if score_rand > best_score:
            best_P = P_rand.copy()
            best_score = score_rand
    
    # 4. Gradient ascent / Hill climbing
    # We will use a simple coordinate ascent or random direction ascent.
    # Given the non-smoothness of max, we might use a soft-max surrogate or
    # just do local search.
    
    # Let's try a simple local search: perturb a random cell and adjust others to maintain sum=1.
    # Or better, use a smooth parameterization.
    
    # Let's use a simple iterative improvement:
    # Pick a random cell, increase its probability, and decrease others proportionally.
    # This is a simple form of stochastic gradient ascent.
    
    current_P = best_P.copy()
    current_score = best_score
    step_size = 0.01
    
    while time.time() < deadline - 10:
        # Random direction
        direction = rng.normal(size=(n, n))
        # Ensure the direction sums to 0 to maintain normalization
        direction -= direction.mean()
        
        # Try a few step sizes
        improved = False
        for lr in [0.01, 0.005, 0.001]:
            new_P = current_P + lr * step_size * direction
            new_P = np.clip(new_P, 0, None)
            total = new_P.sum()
            if total <= 1e-9:
                continue
            new_P /= total
            
            _, _, new_score = _compute_ratios(new_P)
            if new_score > current_score + 1e-9:
                current_P = new_P
                current_score = new_score
                if current_score > best_score:
                    best_P = current_P.copy()
                    best_score = current_score
                improved = True
                break
        
        if not improved:
            # Try a smaller step or a different direction
            # If stuck, try a larger perturbation
            direction = rng.normal(size=(n, n)) * 0.1
            direction -= direction.mean()
            new_P = current_P + direction
            new_P = np.clip(new_P, 0, None)
            total = new_P.sum()
            if total > 1e-9:
                new_P /= total
                _, _, new_score = _compute_ratios(new_P)
                if new_score > current_score:
                    current_P = new_P
                    current_score = new_score
                    if current_score > best_score:
                        best_P = current_P.copy()
                        best_score = current_score
    
    # Final polish: try to refine the best solution with very small steps
    # This is a simple local optimization
    for _ in range(1000):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n):
            for j in range(n):
                # Try moving probability from (i, j) to a random other cell
                k = rng.integers(0, n*n)
                ki, kj = k // n, k % n
                if ki == i and kj == j:
                    continue
                
                delta = 0.001
                if current_P[i, j] < delta:
                    continue
                
                new_P = current_P.copy()
                new_P[i, j] -= delta
                new_P[ki, kj] += delta
                
                _, _, new_score = _compute_ratios(new_P)
                if new_score > current_score + 1e-9:
                    current_P = new_P
                    current_score = new_score
                    if current_score > best_score:
                        best_P = current_P.copy()
                        best_score = current_score
                    improved = True
                    break
            if improved:
                break
    
    # Ensure final validity
    best_P = np.clip(best_P, 0, None)
    total = best_P.sum()
    if total <= 1e-9:
        # Fallback to uniform
        best_P = np.ones((n, n), dtype=np.float64) / (n * n)
    else:
        best_P /= total
    
    return best_P

# EVOLVE-BLOCK-END
