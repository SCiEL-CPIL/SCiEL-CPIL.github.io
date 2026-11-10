import numpy as np
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # Start with Keich construction
    positions = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    # Area computation function
    def compute_area(x):
        y_vals = (np.arange(20000) + 0.5) / 20000.0
        w = (1.0 - y_vals) / n
        a = (np.arange(n) + 1) / n
        # L_i(y) = x_i + a_i * y
        L = x[:, np.newaxis] + a[:, np.newaxis] * y_vals[np.newaxis, :]
        L_sorted = np.sort(L, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        union_len = w[np.newaxis, :] + np.sum(np.minimum(gaps, w[np.newaxis, :]), axis=0)
        return np.mean(union_len)
    
    # Gradient via finite differences (sparse)
    def compute_grad(x):
        grad = np.zeros(n)
        eps = 1e-7
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (compute_area(x_plus) - compute_area(x_minus)) / (2 * eps)
        return grad
    
    # L-BFGS-like optimization with line search
    deadline = time.time() + 600.0
    margin = 30.0  # seconds before deadline to stop
    
    best_x = positions.copy()
    best_area = compute_area(best_x)
    
    # Simple gradient descent with adaptive step size
    lr = 0.01
    momentum = 0.9
    v = np.zeros(n)
    
    iteration = 0
    while time.time() < deadline - margin:
        iteration += 1
        
        # Compute gradient
        grad = compute_grad(best_x)
        
        # Update with momentum
        v = momentum * v - lr * grad
        new_x = best_x + v
        
        # Check validity
        if not np.all(np.isfinite(new_x)):
            v *= 0.5
            continue
        
        new_area = compute_area(new_x)
        
        if new_area < best_area:
            best_x = new_x
            best_area = new_area
            # Try to increase learning rate slightly
            if iteration % 100 == 0:
                lr *= 1.05
        else:
            # Reduce learning rate
            lr *= 0.9
            v *= 0.5
        
        # If learning rate is too small, try random restarts
        if lr < 1e-8:
            # Add small random perturbation
            best_x += np.random.randn(n) * 1e-4
            lr = 0.001
            v = np.zeros(n)
    
    return best_x
