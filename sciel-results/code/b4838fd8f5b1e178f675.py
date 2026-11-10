# EVOLVE-BLOCK-START
import numpy as np
import time

def _calc_area(x):
    """
    Compute the area of the union of 64 triangles defined by base positions x.
    x: shape (64,)
    Returns: scalar area
    """
    n = 64
    # Use the exact quadrature points specified in the task
    y = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    
    # L_i(y) = x_i + a_i * y
    # Shape: (64, 20000)
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1 - y) / n  # Shape: (20000,)
    
    # Sort L along axis 0 for each y
    L_sorted = np.sort(L, axis=0)
    
    # Gaps between consecutive sorted L's
    gaps = np.diff(L_sorted, axis=0)  # Shape: (63, 20000)
    
    # Union length = w + sum(min(gaps, w))
    contrib = np.minimum(gaps, w[None, :])
    union_len = w + contrib.sum(axis=0)
    area = union_len.mean()
    
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # Safety margin, evaluator kills at 630s
    
    n = 64
    k = int(np.log2(n))
    
    # Initialize with Keich construction as a strong starting point
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    best_x = x.copy()
    best_area = _calc_area(best_x)
    
    # --- Stage 1: Coarse Grid Search / Random Restart ---
    # We perform a limited number of random perturbations to find a better basin.
    # Since the problem is 64D, pure random search is inefficient, but 
    # perturbation from the Keich solution is good.
    
    rng = np.random.default_rng(42)
    
    # We will use a simple Simulated Annealing approach on the full 64D vector.
    # Initial temperature high enough to explore, cooling schedule exponential.
    
    T = 0.05  # Initial temperature (scale of perturbations)
    T_min = 1e-6
    alpha = 0.999 # Cooling rate
    
    current_x = best_x.copy()
    current_area = best_area
    
    # We allocate time dynamically. 
    # Let's reserve 10% of time for final polishing if needed, but SA is the main driver.
    # We run until time is up, keeping the best found.
    
    # To speed up evaluation, we can potentially reduce quadrature points for intermediate steps?
    # The task says "computed by quadrature" with specific y_m. The evaluator uses the full 20000.
    # If we use fewer points for the search, we might converge to a local min that is bad on the full grid.
    # However, 20000 points * 64 triangles is fast in numpy. 
    # _calc_area takes ~5-10ms. 
    # 600s / 0.01s = 60,000 iterations. This is plenty.
    
    # Let's optimize the area calculation slightly by reusing arrays if possible, 
    # but numpy sort is the bottleneck.
    
    # We will use a standard SA loop.
    # Perturbation: Gaussian noise.
    
    # To handle the non-smoothness, we can also try coordinate-wise moves, 
    # but full-vector Gaussian is simpler and often effective for this type of metric.
    
    # Let's add a "reheat" mechanism if stuck.
    stuck_count = 0
    max_stuck = 500
    
    while time.time() < deadline - 5:
        # Generate perturbation
        # Scale noise by current temperature
        noise = rng.normal(0, T, size=n)
        x_new = current_x + noise
        
        # Evaluate
        area_new = _calc_area(x_new)
        
        if area_new < current_area:
            current_x = x_new
            current_area = area_new
            stuck_count = 0
            
            if area_new < best_area:
                best_area = area_new
                best_x = x_new.copy()
        else:
            # Accept with probability exp(-delta / T)
            delta = area_new - current_area
            if delta > 0:
                # Avoid overflow in exp
                if delta / T > 100:
                    accept = False
                else:
                    accept = np.random.rand() < np.exp(-delta / T)
            else:
                accept = True
            
            if accept:
                current_x = x_new
                current_area = area_new
                stuck_count = 0
            else:
                stuck_count += 1
        
        # Cooling
        T *= alpha
        
        # Reheat if stuck
        if stuck_count > max_stuck:
            T = max(T, 0.01) # Reset to a moderate temperature
            stuck_count = 0
            
        if T < T_min:
            # If temperature is too low, we might be stuck in a local min.
            # Restart from best_x with a fresh temperature?
            # Or just break if we've been running for a while and T is tiny.
            # Let's restart from best_x to explore a new region.
            if np.linalg.norm(current_x - best_x) < 1e-5:
                 current_x = best_x + rng.normal(0, 0.05, size=n)
                 current_area = _calc_area(current_x)
                 T = 0.05
                 stuck_count = 0
            else:
                 T = T_min

    # Final check
    final_area = _calc_area(best_x)
    if final_area < best_area:
        best_area = final_area
        # best_x is already the best seen, but ensure consistency
        pass

    # Ensure valid output
    if not np.all(np.isfinite(best_x)):
        # Fallback to initial Keich
        return np.array(positions, dtype=np.float64)
        
    return best_x

# EVOLVE-BLOCK-END
