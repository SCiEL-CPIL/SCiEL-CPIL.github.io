# EVOLVE-BLOCK-START
import numpy as np


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimize the union area of 64 Kakeya needle triangles using a
    custom projected subgradient method with Nesterov acceleration
    and iterate averaging.
    """
    n = 64
    M = 20000
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64

    # Precompute heights
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    w = (1.0 - y) / n             # shape (M,)

    # L_i at height y: L[i, m] = x[i] + a[i] * y[m]
    A = np.outer(a, y)  # shape (n, M)
    indices_m = np.arange(M)[None, :]  # shape (1, M) for broadcasting

    def compute_stats(x):
        """
        Compute the area and the subgradient in a single pass.
        Returns (area, grad).
        """
        L = x[:, None] + A  # (n, M)
        order = np.argsort(L, axis=0)  # (n, M)
        
        # Gather sorted L values
        L_sorted = L[order, indices_m]  # (n, M)
        
        # Compute gaps
        gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)
        
        # Compute area
        overlaps = np.minimum(gaps, w[None, :])
        union_lengths = w + overlaps.sum(axis=0)
        area = union_lengths.mean()
        
        # Compute subgradient
        # mask[j, m] is True if gap[j, m] < w[m]
        mask = gaps < w[None, :]  # (n-1, M), boolean
        
        # pos_grad[j, m] is the subgradient for the element at sorted position j
        # pos_grad[j] = mask[j-1] (if j>0) - mask[j] (if j<n-1)
        # We can construct this efficiently:
        # Start with zeros
        pos_grad = np.zeros((n, M))
        
        # Add +1 where mask is true, shifted by 1 down
        pos_grad[1:, :] += mask.astype(float)
        
        # Subtract 1 where mask is true
        pos_grad[:-1, :] -= mask.astype(float)
        
        # Scatter back to original indices
        # grad[i, m] = pos_grad[order[i, m], m]
        grad = pos_grad[order, indices_m]
        
        # Average over heights
        grad_mean = grad.mean(axis=1)
        
        return area, grad_mean

    # Initialize with the Keich construction
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)

    # Optimization parameters
    n_iters = 5000
    beta = 0.9  # Nesterov momentum
    
    # Initialize x_avg for averaging
    x_avg = np.zeros(n)
    
    best_x = x.copy()
    best_area, _ = compute_stats(x)
    
    v = np.zeros(n)
    
    # Estimate step size constant C
    # Using theoretical bound: gamma_t = C / sqrt(t)
    # C approx R / G
    # Let's start with a heuristic C and adjust if needed, or just use a fixed schedule
    # Based on previous runs, C around 0.05-0.1 works well
    C = 0.1
    
    for it in range(n_iters):
        # Compute stats at current x
        area, g = compute_stats(x)
        
        # Nesterov update
        v = beta * v + (1 - beta) * g
        gamma = C / np.sqrt(it + 1)
        x_new = x - gamma * v
        
        # Update average
        x_avg = x_avg + x_new / (it + 1)
        
        # Check if x_new is better
        # We need to evaluate x_new. 
        # To save time, we could evaluate every k-th iteration, 
        # but for accuracy, we evaluate every time.
        area_new, _ = compute_stats(x_new)
        
        if area_new < best_area:
            best_area = area_new
            best_x = x_new.copy()
            
        x = x_new
        
        # Occasionally check if the average is better
        if it % 100 == 0 and it > 0:
            area_avg, _ = compute_stats(x_avg)
            if area_avg < best_area:
                best_area = area_avg
                best_x = x_avg.copy()
                
        if it % 1000 == 0:
            print(f"Iter {it}: best_area={best_area:.6f}, x_new_area={area_new:.6f}")

    # Final check on average
    area_avg, _ = compute_stats(x_avg)
    if area_avg < best_area:
        best_x = x_avg.copy()
        
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
