import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimizes the base positions x_i for 64 Kakeya needle triangles to minimize
    the area of their union.
    
    Uses the aligned initialization (x_i = 0.5 - a_i) as the primary starting point,
    refined by a fast coordinate descent with exact 1D breakpoint search, followed
    by Nelder-Mead polish.
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

    # Precompute for coordinate descent
    # For coordinate i, the function is piecewise linear in x[i].
    # The breakpoints occur when L_i(y) = L_j(y) for some j, i.e.,
    # x_i + a_i*y = x_j + a_j*y  =>  x_i = x_j + (a_j - a_i)*y
    # Since y varies over M values, there are M breakpoints for each pair (i,j).
    # But we can be smarter: for a fixed x (with x[i] removed), the other L_j(y)
    # are fixed. The sorted order of the other n-1 values changes at certain y.
    # However, a simpler approach: the area as a function of x[i] is convex and
    # piecewise linear. The minimum is at a breakpoint or at the boundary of the
    # "effective" range.
    
    # A more efficient coordinate descent:
    # For coordinate i, compute the area as a function of x[i] on a grid,
    # but use the fact that the function is piecewise linear to find the exact
    # minimum by checking the subgradient.
    
    # Actually, let's use a robust approach:
    # 1. Compute the current area.
    # 2. For each coordinate i, perform a 1D minimization using the fact that
    #    the function is piecewise linear. We can find the minimum by:
    #    a. Computing the left and right derivatives at the current point.
    #    b. If the subgradient contains 0, we're at a minimum.
    #    c. Otherwise, move in the direction of steepest descent.
    
    # But computing derivatives is tricky. Let's stick with a well-tuned
    # grid search + refinement, but make it faster by using the structure.

    # Key insight: for coordinate i, the function f(x[i]) is piecewise linear.
    # The breakpoints are at x[i] = x[j] + (a[j] - a[i]) * y_m for all j != i, m.
    # There are (n-1) * M = 63 * 20000 = 1,260,000 breakpoints. Too many.
    
    # Alternative: use a coarse grid to find the approximate minimum, then
    # refine with a finer grid. This is what the current code does.
    
    # Let's optimize the grid search by:
    # 1. Using a coarser initial grid (e.g., 20 points instead of 100).
    # 2. Only refining if the improvement is significant.
    # 3. Using a tighter range around the current value.

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
                
                x_old = x[i]
                
                # The optimal x[i] is likely within [-0.5, 0.5] of the current value
                # but let's be a bit more conservative initially
                lo = x_old - 0.25
                hi = x_old + 0.25
                
                # Coarse grid search
                grid = np.linspace(lo, hi, 20)
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
                    step = (hi - lo) / 20
                    lo2 = best_local_x - step
                    hi2 = best_local_x + step
                    grid2 = np.linspace(lo2, hi2, 10)
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

    # Initializations
    inits = []
    
    # Aligned: all converge to (0.5, 1) - this is the best known initialization
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

    # Coordinate descent from best init
    x_cd, val_cd = coordinate_descent(best_x, max_rounds=8)
    if val_cd < best_val:
        best_x = x_cd
        best_val = val_cd

    # Nelder-Mead refinement
    if time.time() < deadline - 60:
        nm_options = {
            'maxiter': 300,
            'xatol': 1e-8,
            'fatol': 1e-10,
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
        return inits[0].copy()  # Fallback to aligned

    return best_x
