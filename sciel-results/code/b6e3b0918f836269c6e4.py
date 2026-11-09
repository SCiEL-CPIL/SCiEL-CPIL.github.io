import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    # Precompute constants
    a = np.arange(1, n + 1) / n
    y_vals = (np.arange(20000) + 0.5) / 20000.0
    w_vals = (1 - y_vals) / n
    
    def compute_area(x):
        # x shape (n,)
        # L shape (n, 20000)
        # L_i = x_i + a_i * y
        L = x[:, np.newaxis] + a[:, np.newaxis] * y_vals[np.newaxis, :]
        
        # Sort L along axis 0 for each y
        # np.argsort is O(n log n) per y, or O(n^2) total if done naively.
        # n=64, 20000 y's. 64*20000 = 1.28M elements.
        # Sorting 20000 arrays of size 64.
        # This might be slow in Python loops, but vectorized sort is fast.
        idx = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, idx, axis=0)
        
        # Calculate union length
        # w is same for all i at a given y.
        # Union length = w + sum_{j=0}^{n-2} min(L_sorted[j+1] - L_sorted[j], w)
        diffs = L_sorted[1:] - L_sorted[:-1]
        # min(diffs, w)
        # w shape (20000,), diffs shape (63, 20000)
        # Need to broadcast w to (63, 20000)
        w_b = w_vals[np.newaxis, :]
        gaps = np.minimum(diffs, w_b)
        
        # Sum over axis 0 (the triangles)
        # union_len shape (20000,)
        union_len = w_vals + np.sum(gaps, axis=0)
        
        area = np.mean(union_len)
        return area

    def gradient(x):
        # Finite difference gradient is slow.
        # Let's try to derive or use central differences if needed.
        # For now, rely on L-BFGS-B finite differences or just use 'Nelder-Mead' if gradient is hard.
        # L-BFGS-B with finite differences for 64 vars is 128 function evals per step.
        # If function takes 10ms, that's 1.28s per step. Too slow.
        
        # We need a faster area calculation or analytical gradient.
        # Or use a coarser y grid for optimization, and fine grid for final eval?
        # The prompt says "computed by quadrature... at each height y_m...".
        # The evaluator uses the fine grid. We should optimize on the fine grid or a surrogate.
        
        # Let's check the speed of compute_area.
        # n=64, M=20000.
        # L creation: 64*20000 = 1.28M floats.
        # Sort: 20000 * 64 log 64.
        # This is probably ~0.1s - 0.5s in Python/NumPy.
        # 128 evals * 0.2s = 25s per step.
        # 100 steps = 2500s. Too slow.
        
        # We MUST use a coarser grid for optimization, or an analytical gradient.
        # Analytical gradient:
        # Area = (1/M) sum_m U(y_m)
        # U(y) = w(y) + sum_{j} min(L_{(j+1)}(y) - L_{(j)}(y), w(y))
        # Let S(y) be the sorted order of L at y.
        # This order changes at critical y values.
        # Between critical y's, the order is constant.
        # We can identify the critical y's?
        # L_i(y) = x_i + a_i y.
        # L_i(y) = L_j(y) => x_i + a_i y = x_j + a_j y => y = (x_j - x_i) / (a_i - a_j).
        # There are O(n^2) critical y's. 64*63/2 = 2016 critical points.
        # We can partition [0,1] into intervals where the order is constant.
        # In each interval, U(y) is a linear function of x?
        # L_i is linear in x. min(linear, w) is piecewise linear.
        # So in a fixed order region, if the min switches don't happen, it's linear.
        # But min(L_{j+1}-L_j, w) switches when L_{j+1}-L_j = w.
        # This adds more critical points.
        
        # Alternative: Use a coarser grid for optimization (e.g., 200 points) and refine.
        # Or use 'Nelder-Mead' which is derivative-free but uses many function evals.
        # Nelder-Mead on 64 dims is very slow.
        
        # Let's try to speed up the area calculation.
        # Can we avoid sorting?
        # The union of intervals is the same regardless of order, but the formula relies on sorting.
        # We can compute the union length by merging intervals.
        # But sorting is the bottleneck.
        
        # Let's try to use the "Keich" structure as a strong prior.
        # The Keich solution is likely very close to optimal.
        # Maybe a few steps of Nelder-Mead or L-BFGS-B with a COARSE grid will improve it.
        
        # Plan:
        # 1. Generate Keich positions.
        # 2. Define a fast area function using M=100 or M=500.
        # 3. Optimize using L-BFGS-B (finite diff) on the coarse grid.
        # 4. Polish the result using the fine grid (M=20000) with a few steps of L-BFGS-B?
        #    Or just use the coarse grid result if time is tight.
        #    The score is based on the fine grid area.
        #    If the coarse grid is a good surrogate, the fine grid area should be lower.
        
        # Let's estimate the improvement.
        # The "Knowledge" says 1.097 is possible.
        
        pass

    # Keich Construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x_keich = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x_keich[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
        
    # Fast Area Calculation for Optimization
    M_opt = 200 # Coarse grid
    y_opt = (np.arange(M_opt) + 0.5) / M_opt
    w_opt = (1 - y_opt) / n
    
    def area_coarse(x):
        L = x[:, np.newaxis] + a[:, np.newaxis] * y_opt[np.newaxis, :]
        idx = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, idx, axis=0)
        diffs = L_sorted[1:] - L_sorted[:-1]
        w_b = w_opt[np.newaxis, :]
        gaps = np.minimum(diffs, w_b)
        union_len = w_opt + np.sum(gaps, axis=0)
        return np.mean(union_len)

    # Optimization
    start_time = time.time()
    timeout = 600 # seconds
    safety_margin = 30
    
    best_x = x_keich
    best_area_fine = compute_area(x_keich)
    
    # Try a few steps of L-BFGS-B on coarse grid
    # Since gradient is expensive, let's use 'Nelder-Mead' with limited maxiter?
    # Or 'L-BFGS-B' with analytical gradient if we can derive it?
    # Let's stick to L-BFGS-B with finite differences on coarse grid.
    # 64 vars, 129 evals/step.
    # area_coarse is fast (M=200).
    
    try:
        res = minimize(
            area_coarse, 
            x_keich, 
            method='L-BFGS-B', 
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        if res.success:
            best_x = res.x
            # Evaluate on fine grid
            a_fine = compute_area(best_x)
            if a_fine < best_area_fine:
                best_area_fine = a_fine
    except Exception as e:
        pass
        
    # Try a few random restarts if time permits
    np.random.seed(42)
    for _ in range(5):
        if time.time() - start_time > timeout - safety_margin:
            break
        x_rand = np.random.uniform(0, 1, n)
        # Sort x_rand? No, just random.
        try:
            res = minimize(
                area_coarse, 
                x_rand, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            if res.success:
                a_fine = compute_area(res.x)
                if a_fine < best_area_fine:
                    best_area_fine = a_fine
                    best_x = res.x
        except Exception as e:
            pass

    # Final polish on fine grid?
    # L-BFGS-B on fine grid is slow.
    # Maybe just return best_x.
    
    return best_x
