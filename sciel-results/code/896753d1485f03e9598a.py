import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

def _compute_area(x):
    """
    Computes the area of the union of the 64 triangles.
    x: array of shape (64,)
    """
    n = 64
    # Precomputed constants
    y = _Y_VALUES  # Shape (1, 20000)
    a = _A_VALUES  # Shape (n, 1)
    
    # L_i(y) = x_i + a_i * y
    L = x.reshape(-1, 1) + a * y  # Shape (n, 20000)
    
    # w(y) = (1 - y) / n
    w = _W_VALUES  # Shape (1, 20000)
    
    # Sort L along axis 0
    L_sorted = np.sort(L, axis=0)  # Shape (n, 20000)
    
    # Differences between consecutive sorted L's
    D = L_sorted[1:, :] - L_sorted[:-1, :]  # Shape (n-1, 20000)
    
    # Union length at each y: w + sum(min(D_j, w))
    min_diffs = np.minimum(D, w)
    
    # Sum over triangles (axis 0)
    union_lengths = w[0, :] + np.sum(min_diffs, axis=0)  # Shape (20000,)
    
    # Area is mean of union lengths
    area = np.mean(union_lengths)
    return area

# Precompute constants
_N = 64
_M = 20000
_A_VALUES = ((np.arange(_N) + 1) / _N).reshape(-1, 1)
_Y_VALUES = ((np.arange(_M) + 0.5) / _M).reshape(1, -1)
_W_VALUES = (1.0 - _Y_VALUES) / _N

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    start_time = time.time()
    deadline = start_time + 580  # 580s limit to be safe with 630s total
    
    def _run_optimization(x0, max_time=None):
        """Run L-BFGS-B + coordinate descent from x0"""
        if max_time is None:
            max_time = deadline
        res = minimize(_compute_area, x0, method='L-BFGS-B', 
                       options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100})
        x_best = res.x
        area_best = res.fun
        
        if time.time() > max_time:
            return x_best, area_best
        
        # Coordinate Descent Polishing
        improved = True
        max_rounds = 30
        round_num = 0
        while improved and (time.time() < max_time) and round_num < max_rounds:
            improved = False
            round_num += 1
            for i in range(n):
                if time.time() > max_time:
                    return x_best, area_best
                
                def f_1d(xi, _i=i, _x=x_best):
                    x_temp = _x.copy()
                    x_temp[_i] = xi
                    return _compute_area(x_temp)
                
                window = 0.2 if round_num < 3 else 0.05
                res_i = minimize_scalar(f_1d, bounds=(x_best[i] - window, x_best[i] + window), 
                                        method='bounded', options={'xatol': 1e-11})
                
                if res_i.success:
                    if abs(res_i.x - x_best[i]) > 1e-13:
                        x_best[i] = res_i.x
                        improved = True
                    if res_i.fun < area_best:
                        area_best = res_i.fun
        
        # Final fine-tuning
        if time.time() < max_time - 10:
            for _ in range(5):
                if time.time() > max_time:
                    break
                for i in range(n):
                    if time.time() > max_time:
                        return x_best, area_best
                    def f_1d_fine(xi, _i=i, _x=x_best):
                        x_temp = _x.copy()
                        x_temp[_i] = xi
                        return _compute_area(x_temp)
                    
                    res_i = minimize_scalar(f_1d_fine, bounds=(x_best[i] - 0.005, x_best[i] + 0.005), 
                                            method='bounded', options={'xatol': 1e-13})
                    if res_i.success and res_i.fun < area_best:
                        x_best[i] = res_i.x
                        area_best = res_i.fun
        
        return x_best, area_best
    
    # Generate multiple initial points
    initial_points = []
    
    # 1. Keich Construction
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    initial_points.append(np.array(positions, dtype=np.float64))
    
    # 2. Linear spread
    initial_points.append(np.linspace(-0.5, 0.5, n))
    
    # 3. Clustered at 0
    initial_points.append(np.zeros(n))
    
    # 4. Random perturbations of Keich
    rng = np.random.default_rng(42)
    for trial in range(5):
        x_rand = initial_points[0] + rng.normal(0, 0.1, n)
        initial_points.append(x_rand)
    
    # 5. Antipodal: x_i = -x_{n-1-i}
    x_keich = initial_points[0]
    x_anti = np.zeros(n)
    for i in range(n // 2):
        x_anti[i] = x_keich[i]
        x_anti[n - 1 - i] = -x_keich[i]
    if n % 2 == 1:
        x_anti[n // 2] = 0.0
    initial_points.append(x_anti)
    
    # Run optimization from each initial point, allocating time proportionally
    best_x = None
    best_area = np.inf
    
    num_starts = len(initial_points)
    time_per_start = (deadline - start_time) / num_starts * 0.8  # 80% of time for multi-start
    
    for idx, x0 in enumerate(initial_points):
        if time.time() > deadline - 30:
            break
        
        # Allocate more time to earlier (more promising) starts
        if idx == 0:
            t_alloc = time_per_start * 2.0
        elif idx == 1:
            t_alloc = time_per_start * 1.5
        else:
            t_alloc = time_per_start
        
        t_alloc = min(t_alloc, deadline - start_time - 30)
        if t_alloc < 10:
            t_alloc = 10
        
        x_result, area_result = _run_optimization(x0, max_time=start_time + t_alloc)
        
        if area_result < best_area:
            best_area = area_result
            best_x = x_result
    
    # Use remaining time for final polishing of the best solution
    if best_x is not None and time.time() < deadline - 10:
        x_result, area_result = _run_optimization(best_x, max_time=deadline - 10)
        if area_result < best_area:
            best_area = area_result
            best_x = x_result
    
    return best_x
