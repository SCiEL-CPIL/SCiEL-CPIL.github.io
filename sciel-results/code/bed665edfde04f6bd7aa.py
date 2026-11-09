# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area_exact(x, n=64):
    """
    Compute the exact area of the union of triangles by identifying
    breakpoints in y where the overlap structure changes.
    """
    a = (np.arange(n) + 1) / n
    w0 = 1.0 / n
    
    # Collect all critical y values
    # 1. L_i = L_j => x_i + a_i y = x_j + a_j y => y = (x_j - x_i) / (a_i - a_j)
    # 2. L_i + w = L_j => x_i + a_i y + w0(1-y) = x_j + a_j y
    #    => (a_i - a_j + w0) y = x_j - x_i - w0
    #    => y = (x_j - x_i - w0) / (a_i - a_j + w0)
    
    y_crits = [0.0, 1.0]
    
    for i in range(n):
        for j in range(i + 1, n):
            # Case 1: Left edges cross
            denom1 = a[i] - a[j]
            if abs(denom1) > 1e-12:
                y1 = (x[j] - x[i]) / denom1
                if 0.0 < y1 < 1.0:
                    y_crits.append(y1)
            
            # Case 2: Right edge of i touches left edge of j
            # L_i + w = L_j
            denom2 = a[i] - a[j] + w0
            if abs(denom2) > 1e-12:
                y2 = (x[j] - x[i] - w0) / denom2
                if 0.0 < y2 < 1.0:
                    y_crits.append(y2)
            
            # Case 3: Left edge of i touches right edge of j
            # L_i = L_j + w
            # x_i + a_i y = x_j + a_j y + w0(1-y)
            # (a_i - a_j + w0) y = x_j + w0 - x_i
            if abs(denom2) > 1e-12: # Note: a_i - a_j + w0 is same denominator
                y3 = (x[j] + w0 - x[i]) / denom2
                if 0.0 < y3 < 1.0:
                    y_crits.append(y3)

    y_crits = np.array(y_crits)
    y_crits = np.unique(y_crits)
    y_crits.sort()
    
    # Integrate over each interval
    area = 0.0
    num_intervals = len(y_crits) - 1
    
    for k in range(num_intervals):
        y_start = y_crits[k]
        y_end = y_crits[k + 1]
        
        if y_end <= y_start:
            continue
            
        # Evaluate union length at midpoint of this interval
        # Since U(y) is linear on this interval (no topology changes),
        # the integral is (y_end - y_start) * U(mid)
        y_mid = (y_start + y_end) / 2.0
        
        # Calculate L_i at y_mid
        L = x + a * y_mid
        w = w0 * (1.0 - y_mid)
        
        # Sort L
        L_sorted = np.sort(L)
        
        # Calculate union length
        # The union length is sum of lengths of disjoint intervals.
        # We can iterate through sorted L and merge.
        # Or use the formula: w + sum(min(gap, w)) is NOT correct for arbitrary overlaps if gaps are not considered in order?
        # Wait, the formula in the prompt:
        # "With the L_i sorted, the union length is w + sum over consecutive pairs of min(L_{j+1} - L_j, w)."
        # This formula assumes that if L_{j+1} - L_j < w, they overlap, and the added length is the gap.
        # If L_{j+1} - L_j >= w, they are disjoint, and the added length is w.
        # So union_len = w + sum_{j=0}^{n-2} min(L_{j+1} - L_j, w)
        
        gaps = np.diff(L_sorted)
        union_len = w + np.sum(np.minimum(gaps, w))
        
        area += (y_end - y_start) * union_len
        
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 610  # Safety margin
    
    # 1. Generate Keich Initial Point
    def _keich_initial(n=64):
        k = int(np.log2(n))
        j = np.arange(1, k + 1)
        x0 = np.zeros(n)
        for i in range(n):
            # Binary representation of i, padded to k bits
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
        return x0

    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area_exact(best_x)
    
    # 2. Optimization Loop
    # We will use Nelder-Mead and L-BFGS-B alternating with basin hopping.
    
    def objective(x):
        return _compute_area_exact(x)
    
    # Gradient approximation for L-BFGS-B
    # Since exact area is fast, we can afford finite differences
    # But Nelder-Mead is derivative-free and robust to non-smoothness.
    # Let's stick to Nelder-Mead primarily, as the objective is non-smooth.
    
    np.random.seed(42)
    
    # Strategy 1: Initial Nelder-Mead polish
    result = minimize(
        objective, best_x, method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    if result.fun < best_area:
        best_area = result.fun
        best_x = result.x.copy()
    
    # Strategy 2: Basin Hopping
    # Perturb best_x, then polish with Nelder-Mead.
    for i in range(50):  # Max 50 hops
        if time.time() > deadline - 30:
            break
            
        # Perturbation scale decreases over time
        scale = 0.05 * (0.95 ** i)
        x_perturbed = best_x + np.random.normal(0, scale, size=n)
        
        # Polish the perturbed point
        try:
            result = minimize(
                objective, x_perturbed, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
            
        # Check time
        if time.time() > deadline - 60:
            break

    # Strategy 3: Final high-precision polish
    if time.time() < deadline - 10:
        result = minimize(
            objective, best_x, method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
            
    # Final verification
    final_area = _compute_area_exact(best_x)
    if final_area <= 0 or not np.all(np.isfinite(best_x)):
        best_x = x0
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
