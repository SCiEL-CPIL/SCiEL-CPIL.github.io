"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 330s limit
    
    # Flatten X for 1D processing
    X = np.asarray(X_train, dtype=float).reshape(-1).flatten()
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)
    
    # Sort training data by X for efficient neighbor lookups
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Precompute cumulative sums of powers of X and Y for fast windowed polynomial regression
    # We will use a local linear model: y = a * x + b
    # We need sums: sum(1), sum(x), sum(y), sum(x*y), sum(x^2) within the window
    # To make this efficient, we use prefix sums.
    # prefix_sum[i] = sum of first i elements
    
    prefix_sum_1 = np.zeros(n + 1)
    prefix_sum_x = np.zeros(n + 1)
    prefix_sum_y = np.zeros(n + 1)
    prefix_sum_xy = np.zeros(n + 1)
    prefix_sum_x2 = np.zeros(n + 1)
    
    prefix_sum_1[1:] = np.cumsum(np.ones(n))
    prefix_sum_x[1:] = np.cumsum(X_sorted)
    prefix_sum_y[1:] = np.cumsum(y_sorted)
    prefix_sum_xy[1:] = np.cumsum(X_sorted * y_sorted)
    prefix_sum_x2[1:] = np.cumsum(X_sorted ** 2)
    
    # Determine bandwidth based on local density
    # Since data is 1D and uniform-ish, we can use a fixed number of neighbors or a distance threshold.
    # Adaptive bandwidth: use the distance to the k-th nearest neighbor.
    # Let's pick k based on n. For n=200, k=10 seems reasonable for local linearity.
    k_neighbors = max(5, n // 20)  # At least 5, up to 10% of data
    
    def get_window_indices(x_query, k):
        """Get indices of k nearest neighbors to x_query in sorted array."""
        # Find insertion point
        idx = np.searchsorted(X_sorted, x_query)
        
        # Expand window to find k neighbors
        # We need to check left and right
        left = max(0, idx - k)
        right = min(n, idx + k)
        
        # If we have more than k points in [left, right), we might need to trim
        # But for a simple local linear fit, taking a symmetric window around the insertion point
        # is a good heuristic. Let's take a window of size 2*k centered at idx.
        
        # Better approach: just take k points closest to x_query.
        # Since they are sorted, we can take a window [idx-k, idx+k] and then sort by distance if needed,
        # but for local linear regression, the order doesn't matter for the least squares solution,
        # only the set of points matters.
        
        # Let's just take a window of size 2*k around the insertion point.
        # This is an approximation of k-NN but much faster.
        lo = max(0, idx - k)
        hi = min(n, idx + k)
        
        # If the window is too small (e.g. near boundaries), expand it
        # Actually, let's just ensure we have at least k points if possible.
        if hi - lo < k:
            # Try to expand
            if lo == 0:
                hi = min(n, hi + (k - (hi - lo)))
            elif hi == n:
                lo = max(0, lo - (k - (hi - lo)))
            else:
                # Expand both sides
                expand = (k - (hi - lo)) // 2
                lo = max(0, lo - expand)
                hi = min(n, hi + (k - (hi - lo) - expand))
                
        return lo, hi

    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).flatten()
        predictions = np.zeros_like(X_pred)
        
        for i, xq in enumerate(X_pred):
            lo, hi = get_window_indices(xq, k_neighbors)
            
            # If window is empty or too small, fall back to global linear
            if hi - lo < 2:
                # Global linear fit fallback
                # We pre-computed global sums
                n_global = n
                sum_1 = n_global
                sum_x = prefix_sum_x[n]
                sum_y = prefix_sum_y[n]
                sum_xy = prefix_sum_xy[n]
                sum_x2 = prefix_sum_x2[n]
                
                A = np.array([[sum_x2, sum_x], [sum_x, sum_1]])
                b = np.array([sum_xy, sum_y])
                try:
                    coeffs = np.linalg.solve(A, b)
                    a, b_const = coeffs
                    predictions[i] = a * xq + b_const
                except np.linalg.LinAlgError:
                    predictions[i] = np.mean(y)
                continue
            
            # Extract sums for the window [lo, hi)
            sum_1 = prefix_sum_1[hi] - prefix_sum_1[lo]
            sum_x = prefix_sum_x[hi] - prefix_sum_x[lo]
            sum_y = prefix_sum_y[hi] - prefix_sum_y[lo]
            sum_xy = prefix_sum_xy[hi] - prefix_sum_xy[lo]
            sum_x2 = prefix_sum_x2[hi] - prefix_sum_x2[lo]
            
            # Local linear regression: minimize sum((y_i - (a*x_i + b))^2)
            # Normal equations:
            # [sum(x^2)  sum(x) ] [a]   [sum(xy)]
            # [sum(x)    sum(1) ] [b] = [sum(y) ]
            
            A = np.array([[sum_x2, sum_x], [sum_x, sum_1]])
            b = np.array([sum_xy, sum_y])
            
            # Check for singularity
            det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
            if abs(det) < 1e-10:
                # Fallback to local mean
                predictions[i] = sum_y / sum_1
            else:
                try:
                    coeffs = np.linalg.solve(A, b)
                    a, b_const = coeffs
                    predictions[i] = a * xq + b_const
                except np.linalg.LinAlgError:
                    predictions[i] = sum_y / sum_1
        
        return predictions

    return predict
# EVOLVE-BLOCK-END
