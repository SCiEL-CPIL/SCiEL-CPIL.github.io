import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimizes the base positions x_i for 64 Kakeya needle triangles to minimize
    the area of their union.
    
    Uses a fast vectorized area computation with coordinate descent and
    Nelder-Mead refinement.
    """
    n = 64
    M = 20000
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1 - y) / n

    def compute_area(x):
        """
        Computes the area of the union of the triangles.
        x: shape (n,)
        """
        # L_i(y) = x_i + a_i * y
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        min_gaps = np.minimum(gaps, w[:, None])
        union_len = w + np.sum(min_gaps, axis=1)
        return np.mean(union_len)

    # Precompute for faster coordinate descent
    # For coordinate descent, we need to efficiently update one x_i at a time
    
    def compute_area_fast(x):
        """Same as compute_area but slightly optimized."""
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = np.diff(L_sorted, axis=1)
        min_gaps = np.minimum(gaps, w[:, None])
        union_len = w + np.sum(min_gaps, axis=1)
        return np.mean(union_len)

    # Initializations
    inits = []
    
    # Aligned: all converge to (0.5, 1)
    x_aligned = 0.5 - a
    inits.append(x_aligned.copy())
    
    # Converging to 0
    x_conv0 = -a.copy()
    inits.append(x_conv0)
    
    # Keich
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    inits.append(np.array(positions, dtype=np.float64))
    
    # Random
    np.random.seed(42)
    inits.append(np.random.uniform(-0.5, 0.5, n))

    # Evaluate all inits
    best_x = inits[0].copy()
    best_val = compute_area(best_x)
    for x0 in inits[1:]:
        val = compute_area(x0)
        if val < best_val:
            best_val = val
            best_x = x0.copy()

    start_time = time.time()
    deadline = start_time + 600
    safety_margin = 30

    # Coordinate descent: for each coordinate, do a 1D minimization
    # This is efficient because we only need to sort 64 values 20000 times
    # per coordinate update, and we can cache the sorted order from other coords
    
    def coordinate_descent(x, max_rounds=10, tol=1e-10):
        """Perform coordinate descent on the area function."""
        x = x.copy()
        prev_val = compute_area(x)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - safety_margin:
                break
            
            improved = False
            for i in range(n):
                if time.time() > deadline - safety_margin:
                    break
                
                # 1D minimization for x[i]
                # Try a range of values around current x[i]
                # The function is piecewise linear in x[i], so we can find
                # the minimum by checking breakpoints
                
                x_old = x[i]
                
                # Sample a range of values
                # The optimal x[i] is likely within [-1, 1]
                # Use golden section or ternary search on a bounded interval
                lo = x_old - 0.5
                hi = x_old + 0.5
                
                # Ternary search for 1D piecewise linear function
                # Since it's piecewise linear, the minimum is at a breakpoint
                # or at the boundary of the interval
                # Let's just do a fine grid search + refinement
                
                # Quick grid search
                grid = np.linspace(lo, hi, 100)
                best_local_val = prev_val
                best_local_x = x_old
                
                for gx in grid:
                    x[i] = gx
                    val = compute_area(x)
                    if val < best_local_val:
                        best_local_val = val
                        best_local_x = gx
                
                # Refine around best_local_x
                if best_local_x != x_old:
                    lo2 = best_local_x - (hi - lo) / 100
                    hi2 = best_local_x + (hi - lo) / 100
                    grid2 = np.linspace(lo2, hi2, 50)
                    for gx in grid2:
                        x[i] = gx
                        val = compute_area(x)
                        if val < best_local_val:
                            best_local_val = val
                            best_local_x = gx
                    
                    x[i] = best_local_x
                    if abs(best_local_val - prev_val) > tol:
                        improved = True
                        prev_val = best_local_val
                else:
                    x[i] = x_old
            
            if not improved:
                break
        
        return x, prev_val

    # Run coordinate descent from best init
    x_cd, val_cd = coordinate_descent(best_x, max_rounds=5)
    if val_cd < best_val:
        best_x = x_cd
        best_val = val_cd

    # Nelder-Mead refinement
    if time.time() < deadline - 60:
        nm_options = {
            'maxiter': 200,
            'xatol': 1e-7,
            'fatol': 1e-9,
            'adaptive': True
        }
        try:
            res = minimize(
                compute_area,
                best_x,
                method='Nelder-Mead',
                options=nm_options
            )
            if res.fun < best_val:
                best_x = res.x
                best_val = res.fun
        except Exception:
            pass

    # Final coordinate descent pass
    if time.time() < deadline - 30:
        x_cd2, val_cd2 = coordinate_descent(best_x, max_rounds=3)
        if val_cd2 < best_val:
            best_x = x_cd2
            best_val = val_cd2

    if not np.all(np.isfinite(best_x)):
        return inits[1].copy()  # Fallback to aligned

    return best_x
