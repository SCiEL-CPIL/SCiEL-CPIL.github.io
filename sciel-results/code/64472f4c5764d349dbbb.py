# EVOLVE-BLOCK-START
import numpy as np
import time


def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    deadline = time.time() + 600.0 - 30.0  # 30s safety margin

    # Precompute projection values
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')  # X[y, x], Y[y, x]
    # Actually meshgrid with indexing='xy' gives X[i,j]=x[j], Y[i,j]=y[i]
    # We need P[y, x] so let's be explicit
    xs = np.arange(n)
    ys = np.arange(n)
    X_grid, Y_grid = np.meshgrid(xs, ys, indexing='ij')  # X_grid[x, y], Y_grid[x, y]
    # But P is indexed [y, x], so:
    # P[y, x] corresponds to coordinate (x, y)
    # Let's just compute for each (y, x)
    
    # Projection values for each cell (y, x):
    # x - y (for D)
    # x (for M, r=0)
    # x + y (for M, r=1)
    # x + 2*y (for M, r=2)
    
    def compute_ratio(P_flat):
        """Compute D/M ratio from a flat array of 100 weights."""
        P = P_flat.reshape(10, 10)  # P[y, x]
        if P.sum() <= 1e-15:
            return -1e18
        P = P / P.sum()
        
        # Get all (y, x) indices
        ys_idx, xs_idx = np.where(P > 1e-15)
        if len(ys_idx) == 0:
            return -1e18
        
        probs = P[ys_idx, xs_idx]
        x_vals = xs_idx
        y_vals = ys_idx
        
        # D: entropy of x - y
        d_vals = x_vals - y_vals
        # M components: x, x+y, x+2y
        m0_vals = x_vals
        m1_vals = x_vals + y_vals
        m2_vals = x_vals + 2 * y_vals
        
        def shannon_entropy(values, probabilities):
            if len(values) == 0:
                return 0.0
            # Pool by value
            unique_vals, inverse = np.unique(values, return_inverse=True)
            counts = np.zeros(len(unique_vals))
            np.add.at(counts, inverse, probabilities)
            counts = counts / counts.sum()
            nonzero = counts > 1e-15
            if not np.any(nonzero):
                return 0.0
            return -np.sum(counts[nonzero] * np.log2(counts[nonzero]))
        
        D = shannon_entropy(d_vals, probs)
        M0 = shannon_entropy(m0_vals, probs)
        M1 = shannon_entropy(m1_vals, probs)
        M2 = shannon_entropy(m2_vals, probs)
        M = max(M0, M1, M2)
        
        if M <= 1e-9:
            return -1e18
        
        return D / M
    
    # Strategy: Start with a good initial distribution and refine
    # Key idea: weight cells to make x-y spread out but x concentrated
    
    # Initial guess: weight by how "spread out" x-y is
    # For each cell (y, x), the value d = x - y
    # We want d to be uniform, so weight cells with rare d values more
    
    # Better: use a parametric family
    # P[y, x] proportional to exp(alpha * (x - y)^2 + beta * x + gamma * y)
    # or simpler: concentrate on a diagonal band
    
    # Let's try: weight = exp(-lambda * (x - c)^2) * exp(mu * (x - y))
    # This concentrates x near c and spreads x-y
    
    best_P = np.ones((10, 10)) / 100.0
    best_ratio = compute_ratio(best_P.flatten())
    
    # Grid search over parameters
    # Try different center positions and spreads
    for c in np.arange(-1, 11, 0.5):
        if time.time() > deadline:
            break
        for sigma in np.arange(0.5, 6.0, 0.25):
            if time.time() > deadline:
                break
            for mu in np.arange(-2.0, 2.5, 0.25):
                if time.time() > deadline:
                    break
                # P[y, x] = exp(-((x - c)^2) / (2*sigma^2)) * exp(mu * (x - y))
                weights = np.zeros((10, 10))
                for yi in range(10):
                    for xi in range(10):
                        w = np.exp(-((xi - c)**2) / (2 * sigma**2)) * np.exp(mu * (xi - yi))
                        weights[yi, xi] = w
                if weights.sum() <= 1e-15:
                    continue
                ratio = compute_ratio(weights.flatten())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_P = weights.copy()
    
    # Local refinement around best
    if time.time() < deadline:
        # Try perturbations
        for _ in range(5000):
            if time.time() > deadline:
                break
            perturbed = best_P.copy()
            # Randomly adjust some cells
            for _ in range(3):
                yi = np.random.randint(0, 10)
                xi = np.random.randint(0, 10)
                factor = np.random.uniform(0.5, 1.5)
                perturbed[yi, xi] *= factor
            if perturbed.sum() <= 1e-15:
                continue
            ratio = compute_ratio(perturbed.flatten())
            if ratio > best_ratio:
                best_ratio = ratio
                best_P = perturbed.copy()
    
    # Normalize and clip
    P = best_P.copy()
    P = np.clip(P, 0, None)
    total = P.sum()
    if total <= 1e-15:
        P = np.ones((10, 10)) / 100.0
    else:
        P = P / total
    
    return P


# EVOLVE-BLOCK-END
