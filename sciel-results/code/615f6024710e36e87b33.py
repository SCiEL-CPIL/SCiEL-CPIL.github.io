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
    
    M = 20000
    y_vals = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    w_vals = (1.0 - y_vals) / n  # shape (M,)
    
    def compute_area(x):
        """Compute union area using exact formula."""
        L = x[:, np.newaxis] + a[:, np.newaxis] * y_vals[np.newaxis, :]
        L_sorted = np.sort(L, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        union_len = w_vals[np.newaxis, :] + np.sum(np.minimum(gaps, w_vals[np.newaxis, :]), axis=0)
        return np.mean(union_len)
    
    def compute_area_and_grad_soft(x, beta=50.0):
        """
        Compute area and gradient using soft-min smoothing for min(gaps, w).
        soft_min(a, b) ≈ a + b - soft_max(a, b) where soft_max uses log-sum-exp.
        Actually, min(a,b) = (a+b - |a-b|)/2, and |a-b| ≈ soft_abs.
        Alternatively: min(a,b) ≈ -soft_max(-a, -b)
        soft_max(u, v) = (1/beta) * log(exp(beta*u) + exp(beta*v))
        """
        L = x[:, np.newaxis] + a[:, np.newaxis] * y_vals[np.newaxis, :]
        L_sorted = np.sort(L, axis=0)
        # Track which index was at each sorted position
        idx_sorted = np.argsort(L, axis=0)  # shape (n, M)
        
        gaps = np.diff(L_sorted, axis=0)  # shape (n-1, M)
        
        # soft_min(gaps, w) where w is shape (1, M)
        # min(g, w) = (g + w - |g - w|) / 2
        # |g - w| ≈ soft_abs(g - w) = (1/beta) * log(1 + exp(2*beta*(g-w)))  [for positive]
        # More stably: soft_abs(z) = z*sign(z) + softplus(-|z|) but let's use:
        # soft_min(g, w) = g + w - soft_max(g, w)
        # soft_max(g, w) = max(g,w) + (1/beta)*log(1 + exp(-beta*|g-w|))
        # This is getting complex. Let's use a simpler approach:
        # min(g, w) ≈ (g + w - soft_abs(g-w)) / 2
        # soft_abs(z) = (1/beta)*log(1 + exp(2*beta*z)) for z>0, symmetric
        # Actually simpler: soft_min(g,w) = -soft_max(-g, -w)
        # soft_max(u,v) = (1/beta)*log(exp(beta*u) + exp(beta*v))
        
        # Let's compute soft_max(gaps, w) element-wise
        # gaps shape (n-1, M), w shape (M,)
        w_expanded = w_vals[np.newaxis, :]  # (1, M)
        
        # soft_max(g, w) = max + (1/beta)*log1p(exp(-beta*|g-w|))
        g = gaps  # (n-1, M)
        w_b = w_expanded  # (1, M)
        
        max_gw = np.maximum(g, w_b)
        min_gw = np.minimum(g, w_b)
        diff_abs = max_gw - min_gw  # |g - w|
        
        # soft_max = max + (1/beta)*log1p(exp(-beta*diff_abs))
        # For numerical stability, when beta*diff_abs is large, exp(-beta*diff_abs) ≈ 0
        beta_diff = beta * diff_abs
        # log1p(exp(-x)) for x > 0: use -softplus(-x) where softplus(x)=log(1+exp(x))
        # Actually: log1p(exp(-x)) = -log1p(exp(-x))... no.
        # log(1 + exp(-x)) = -log(1 + exp(x)) + ... no.
        # Let's just compute it carefully.
        # For x >= 0: log(1+exp(-x)) is well-behaved, max is log(2) at x=0
        # For large x, it's ~exp(-x)
        
        # Use stable computation
        log_term = np.zeros_like(beta_diff)
        mask = beta_diff < 30  # where exp(-beta_diff) is not negligible
        log_term[mask] = np.log1p(np.exp(-beta_diff[mask]))
        # For beta_diff >= 30, log_term ≈ 0 (since exp(-30) ~ 1e-13)
        
        soft_max_gw = max_gw + log_term / beta
        soft_min_gw = g + w_b - soft_max_gw
        
        union_len = w_b + np.sum(soft_min_gw, axis=0)
        area = np.mean(union_len)
        
        # Gradient computation
        # d(area)/d(x_i) = mean over y of d(union_len)/d(x_i)
        # union_len(y) = w(y) + sum_j soft_min(g_j(y), w(y))
        # g_j(y) = L_sorted[j+1](y) - L_sorted[j](y)
        # d(g_j)/d(x_i) = d(L_sorted[j+1])/d(x_i) - d(L_sorted[j])/d(x_i)
        # L_sorted[j](y) = x_{idx_sorted[j](y)} + a_{idx_sorted[j](y)} * y
        # So d(L_sorted[j])/d(x_i) = 1 if idx_sorted[j](y) == i, else 0
        
        # d(soft_min(g, w))/dg = 1 - d(soft_max(g,w))/dg
        # d(soft_max(g,w))/dg = 1 if g > w else 0 (in the smooth limit, sigmoid)
        # More precisely: d(soft_max(g,w))/dg = 1 / (1 + exp(beta*(w-g))) = sigmoid(beta*(g-w))
        
        # So d(soft_min(g,w))/dg = 1 - sigmoid(beta*(g-w)) = sigmoid(beta*(w-g))
        
        # d(soft_min(g,w))/dw = -d(soft_max(g,w))/dw = -sigmoid(beta*(w-g))
        # But w doesn't depend on x, so only dg term matters for gradient wrt x.
        
        # d(union_len)/d(x_i) = sum_j d(soft_min(g_j,w))/dg_j * d(g_j)/d(x_i)
        # = sum_j sigmoid(beta*(w - g_j)) * (1[i == idx_sorted[j+1](y)] - 1[i == idx_sorted[j](y)])
        
        # This is expensive to compute directly. Let's use a different approach.
        # For each y, we can compute the contribution to each x_i.
        
        # Alternative: use the fact that the gradient can be computed via
        # the sorted indices.
        
        # d(union_len(y))/d(x_i) = sum_{j=0}^{n-2} sigmoid(beta*(w(y) - g_j(y))) * (delta_{i, idx_sorted[j+1](y)} - delta_{i, idx_sorted[j](y)})
        
        # Let's compute this efficiently.
        # For each y (column), we have idx_sorted[:, y] which gives the permutation.
        # We need for each i, the sum over j of sigmoid(...) * (1[i==idx_sorted[j+1]] - 1[i==idx_sorted[j]])
        
        # This is like: for each position j in sorted order, the coefficient is sigmoid(beta*(w - g_j))
        # And it contributes +coeff to x_{idx_sorted[j+1]} and -coeff to x_{idx_sorted[j]}
        
        # Let's build the gradient using scatter operations.
        grad = np.zeros(n)
        
        # sigmoid_beta = sigmoid(beta * (w - g))
        # w - g = w_b - g, shape (n-1, M)
        zw = beta * (w_b - g)
        # sigmoid(zw) = 1 / (1 + exp(-zw))
        # Stable sigmoid
        sig = np.zeros_like(zw)
        pos = zw > 0
        sig[pos] = 1.0 / (1.0 + np.exp(-zw[pos]))
        sig[~pos] = np.exp(zw[~pos]) / (1.0 + np.exp(zw[~pos]))
        
        # Now, for each y (column m), and each j (row 0 to n-2):
        # grad[idx_sorted[j+1, m]] += sig[j, m]
        # grad[idx_sorted[j, m]] -= sig[j, m]
        
        # We can use np.add.at for this.
        # idx_sorted shape (n, M), sig shape (n-1, M)
        
        # For the +1 contributions: idx_sorted[1:, :] with values sig
        # For the -1 contributions: idx_sorted[:-1, :] with values -sig
        
        np.add.at(grad, idx_sorted[1:].ravel(), sig.ravel())
        np.add.at(grad, idx_sorted[:-1].ravel(), -sig.ravel())
        
        # Average over M heights
        grad /= M
        
        return area, grad
    
    # L-BFGS-like optimization with line search
    deadline = time.time() + 630.0
    margin = 30.0  # seconds before deadline to stop
    
    best_x = positions.copy()
    best_area = compute_area(best_x)
    
    # Phase 1: Soft-min gradient descent with annealing
    beta = 50.0
    lr = 0.01
    momentum = 0.9
    v = np.zeros(n)
    
    iteration = 0
    while time.time() < deadline - margin - 60:  # reserve 60s for polish
        iteration += 1
        
        # Compute area and gradient using soft-min
        area_soft, grad = compute_area_and_grad_soft(best_x, beta)
        
        # Update with momentum
        v = momentum * v - lr * grad
        new_x = best_x + v
        
        # Check validity
        if not np.all(np.isfinite(new_x)):
            v *= 0.5
            continue
        
        # Compute true area for acceptance
        new_area = compute_area(new_x)
        
        if new_area < best_area:
            best_x = new_x
            best_area = new_area
            # Increase learning rate slightly every 100 iterations
            if iteration % 100 == 0:
                lr *= 1.05
        else:
            # Reduce learning rate
            lr *= 0.9
            v *= 0.5
        
        # Increase beta to sharpen
        if iteration % 50 == 0:
            beta *= 1.1
            beta = min(beta, 500.0)
        
        # If learning rate is too small, try random restarts
        if lr < 1e-8:
            best_x += np.random.randn(n) * 1e-4
            lr = 0.001
            v = np.zeros(n)
    
    # Phase 2: Final polish with exact subgradient
    # Use subgradient method for the nonsmooth objective
    lr = 0.001
    momentum = 0.9
    v = np.zeros(n)
    best_x_polish = best_x.copy()
    best_area_polish = best_area
    
    # Compute exact subgradient
    def compute_subgrad(x):
        """Compute a subgradient of the union area."""
        L = x[:, np.newaxis] + a[:, np.newaxis] * y_vals[np.newaxis, :]
        L_sorted = np.sort(L, axis=0)
        idx_sorted = np.argsort(L, axis=0)
        gaps = np.diff(L_sorted, axis=0)
        
        # Subgradient of min(g, w): 1 if g < w, 0 if g > w, [0,1] if g == w
        # Use 0.5 for equality (or just check)
        subgrad_coeff = np.where(gaps < w_vals[np.newaxis, :], 1.0, 
                                 np.where(gaps > w_vals[np.newaxis, :], 0.0, 0.5))
        
        grad = np.zeros(n)
        np.add.at(grad, idx_sorted[1:].ravel(), subgrad_coeff.ravel())
        np.add.at(grad, idx_sorted[:-1].ravel(), -subgrad_coeff.ravel())
        grad /= M
        return grad
    
    polish_iters = 0
    while time.time() < deadline - margin:
        polish_iters += 1
        
        grad = compute_subgrad(best_x_polish)
        v = momentum * v - lr * grad
        new_x = best_x_polish + v
        
        if not np.all(np.isfinite(new_x)):
            v *= 0.5
            continue
        
        new_area = compute_area(new_x)
        
        if new_area < best_area_polish:
            best_x_polish = new_x
            best_area_polish = new_area
            if polish_iters % 50 == 0:
                lr *= 1.05
        else:
            lr *= 0.9
            v *= 0.5
        
        if lr < 1e-10:
            break
    
    if best_area_polish < best_area:
        best_x = best_x_polish
        best_area = best_area_polish
    
    return best_x
