import numpy as np
import time
from scipy.optimize import differential_evolution
import sys

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    frac = 1.0 / N
    start_time = time.time()
    timeout = 600
    deadline = start_time + timeout - 10  # Safety margin

    # We have 64 tubes.
    # Tube k has i = k // N, j = k % N.
    # Base position (x_k, y_k).
    # At height c, x-interval: [x_k + c*i/N, x_k + c*i/N + frac]
    # At height c, y-interval: [y_k + c*j/N, y_k + c*j/N + frac]
    
    # We want to minimize the volume of the union.
    # Volume = integral_0^1 Area(Union_{k=0}^{63} Square_k(c)) dc.
    
    # Since the evaluator uses Monte Carlo, we can use a fast Monte Carlo estimate
    # or a deterministic quadrature for the objective function.
    # Deterministic quadrature is more stable for optimization.
    # We will use a grid of c values and compute the exact area of the union of 64 rectangles at each c.
    
    # To compute area of union of 64 rectangles efficiently:
    # We can use a sweep line or simply a fine grid if the rectangles are small?
    # No, exact calculation is better.
    # For 64 rectangles, we can find the union area by:
    # 1. Collect all unique x-coordinates and y-coordinates of the rectangle edges.
    # 2. This creates a grid of cells.
    # 3. A cell is in the union if it is covered by at least one rectangle.
    # 4. Sum the areas of covered cells.
    # With 64 rectangles, we have at most 128 x-coords and 128 y-coords.
    # Grid size 128x128 = 16384 cells.
    # Checking 64 rectangles against 16384 cells is 1M operations per c.
    # With 20 c-values, that's 20M operations per objective call.
    # This might be too slow for Differential Evolution which needs thousands of calls.
    
    # Alternative: Use the fact that the rectangles are axis-aligned.
    # Area of union of rectangles can be computed in O(N^2 log N) using sweep line.
    # But implementing a robust sweep line in Python for every objective call might be slow.
    
    # Let's try a faster approximation or a smarter exact method.
    # Actually, for N=8, the number of distinct x-intervals at a given c is at most 64.
    # But many might overlap.
    
    # Let's use a vectorized approach for the area calculation.
    # For a fixed c, we have 64 rectangles.
    # We can create a boolean mask on a fine grid?
    # If the bounding box is large, the grid needs to be large.
    # But we can normalize the coordinates?
    # No, the volume scales with the bounding box size.
    # The score is 1/V. We want V small.
    # V is the volume of the union.
    
    # Let's use a deterministic Monte Carlo for the objective function, but with a fixed seed and many points, 
    # or better, use the exact area calculation via the "grid of edges" method but optimized.
    
    def union_area_rects(x_starts, y_starts, c, width, height):
        # x_starts: (64,)
        # y_starts: (64,)
        # c: scalar
        # width, height: scalar (frac)
        
        # x intervals: [x_starts + c * i/N, x_starts + c * i/N + width]
        # We need i for each tube.
        # This function is called inside the objective, so passing i, j is overhead.
        # Let's precompute i, j.
        
        # We'll pass the full arrays of x_pos, y_pos and compute inside.
        
        # Actually, let's just use a high-resolution grid in the local bounding box.
        # If the positions are "tight", the bounding box is small.
        # If they are "spread", the bounding box is large, but the union area is likely large too.
        # The optimal solution likely has a small bounding box.
        
        # Let's try a different objective:
        # Minimize the bounding box volume first? No, that's what the seed did.
        
        # Let's use the exact area calculation via sweep line in Python, but keep it simple.
        # Since N=8, we can just loop.
        
        # x intervals
        i_vals = np.arange(64) // N
        j_vals = np.arange(64) % N
        
        x_ints = x_starts + c * i_vals / N
        y_ints = y_starts + c * j_vals / N
        
        # Rectangles: [x_ints[k], x_ints[k]+width] x [y_ints[k], y_ints[k]+height]
        
        # To compute area of union:
        # We can use the inclusion-exclusion principle? No, 64 items is too many.
        # We can use the "cutting" method:
        # The union of rectangles can be decomposed into disjoint rectangles.
        # A simpler method for small N:
        # Collect all unique x-coords: x_ints, x_ints+width.
        # Collect all unique y-coords: y_ints, y_ints+height.
        # Sort them.
        # Iterate over cells formed by these coords.
        # A cell [x_a, x_b] x [y_a, y_b] is covered if there exists a rectangle 
        # such that rect_x_start <= x_a and rect_x_end >= x_b and rect_y_start <= y_a and rect_y_end >= y_b.
        # This is O(N^3) or O(N^4) if done naively.
        # With N=64, 64^4 is too slow.
        
        # Better: For each cell, check if it is covered by ANY rectangle.
        # Number of cells: (2*64-1)^2 ~ 16k.
        # For each cell, check 64 rectangles. 16k * 64 = 1M ops.
        # In Python, 1M ops is ~10ms.
        # With 20 c-points, 200ms per objective call.
        # Differential Evolution with popsize=20, maxiter=50 -> 1000 calls -> 200s.
        # This is feasible!
        
        xs = np.unique(np.concatenate([x_ints, x_ints + width]))
        ys = np.unique(np.concatenate([y_ints, y_ints + height]))
        
        # If the number of unique coords is large, this is slow.
        # But if the rectangles overlap, unique coords might be smaller? No, they are distinct floats.
        # So always ~128 unique x and y.
        
        # Optimization: Vectorize the cell check.
        # Create meshgrid of xs, ys.
        # For each rectangle, mark the cells it covers.
        # This is still O(N * C).
        
        # Let's try to be smarter.
        # We can use the fact that we only need the area, not the mask.
        # Area = sum over cells of (dx * dy) * (covered ? 1 : 0).
        
        # Let's implement the mask approach with numpy.
        # xs: (M,), ys: (K,)
        # M, K <= 128.
        
        # For each rect k:
        # x_range: [x_ints[k], x_ints[k]+width]
        # y_range: [y_ints[k], y_ints[k]+height]
        
        # We can find the indices of xs that fall within [x_start, x_end].
        # Since xs are the edges, a cell [xs[m], xs[m+1]] is inside the rect if 
        # xs[m] >= x_start and xs[m+1] <= x_end.
        
        # This requires care with floating point.
        
        # Let's use a simpler approximation if exact is too slow:
        # Use a fixed grid of 100x100 points in the bounding box?
        # No, the bounding box changes.
        
        # Let's stick to the exact cell method but optimize the loop.
        
        # Actually, there is a much faster way for 1D union length, and we can use it for 2D?
        # No, 2D union area is harder.
        
        # Let's try a different strategy:
        # Keep the separable structure but use a GLOBAL optimizer (Differential Evolution)
        # instead of L-BFGS-B. The seed used local optimization.
        # The separable assumption might be suboptimal, but breaking it increases the search space from 14 to 128 dims,
        # which is hard for DE.
        
        # Let's try to optimize the separable case (14 dims) with DE first.
        # If that improves, great. If not, we might need to break separability.
        # The seed score was 1.42.
        
        # Let's define the objective for the SEPARABLE case using exact integration.
        # This is fast and exact.
        
        def union_length_1d(starts):
            # starts: (N,)
            idx = np.argsort(starts)
            s = starts[idx]
            e = s + width
            total = 0.0
            cur_s = s[0]
            cur_e = e[0]
            for k in range(1, N):
                if s[k] > cur_e:
                    total += cur_e - cur_s
                    cur_s = s[k]
                    cur_e = e[k]
                else:
                    if e[k] > cur_e:
                        cur_e = e[k]
            total += cur_e - cur_s
            return total

        def objective_sep(params):
            # params: (14,) -> u[1..7], v[1..7]
            u = np.zeros(N)
            v = np.zeros(N)
            u[1:] = params[:7]
            v[1:] = params[7:]
            
            # Breakpoints for x
            bps_x = [0.0, 1.0]
            for i in range(N):
                for k in range(N):
                    if i == k: continue
                    denom = i - k
                    if denom == 0: continue
                    c_val = N * (u[k] - u[i]) / denom
                    if 0 < c_val < 1:
                        bps_x.append(c_val)
                        
            bps_y = [0.0, 1.0]
            for i in range(N):
                for k in range(N):
                    if i == k: continue
                    denom = i - k
                    if denom == 0: continue
                    c_val = N * (v[k] - v[i]) / denom
                    if 0 < c_val < 1:
                        bps_y.append(c_val)
                        
            all_bps = sorted(set(bps_x + bps_y))
            
            total_vol = 0.0
            for m in range(len(all_bps) - 1):
                c0 = all_bps[m]
                c1 = all_bps[m+1]
                
                # Lx(c) is linear in [c0, c1]
                # Ly(c) is linear in [c0, c1]
                
                # Evaluate at c0 and c1
                lx0 = union_length_1d(u + c0 * np.arange(N) / N)
                lx1 = union_length_1d(u + c1 * np.arange(N) / N)
                ly0 = union_length_1d(v + c0 * np.arange(N) / N)
                ly1 = union_length_1d(v + c1 * np.arange(N) / N)
                
                dx = lx1 - lx0
                dy = ly1 - ly0
                
                # Integral of (lx0 + t*dx)(ly0 + t*dy) dt from 0 to 1
                # = lx0*ly0 + 0.5*(lx0*dy + ly0*dx) + 1/3*dx*dy
                I = lx0 * ly0 + 0.5 * (lx0 * dy + ly0 * dx) + (1.0/3.0) * dx * dy
                total_vol += (c1 - c0) * I
                
            return total_vol

        # Let's use Differential Evolution on the 14-dimensional separable space.
        # Bounds: u, v in [-2, 2]
        bounds = [(-2.0, 2.0)] * 14
        
        # Initial population
        # Use Keich as one of the seeds? DE generates random pops.
        # We can set `init` to a specific array.
        
        # Let's run DE.
        # popsize=10, maxiter=50.
        # 10 * 50 = 500 evaluations.
        # Each evaluation is fast (exact integration).
        
        try:
            res = differential_evolution(
                objective_sep, 
                bounds, 
                seed=42, 
                maxiter=100, 
                popsize=15,
                tol=1e-10,
                polish=False # We'll polish later
            )
            best_params = res.x
            best_val = res.fun
        except Exception as e:
            # Fallback to Keich
            def keich_positions(N):
                pos = int(np.log2(N))
                j_terms = np.arange(1, pos + 1)
                keich = np.empty(N, dtype=np.float64)
                for idx in range(N):
                    eps = np.array(list(str(bin(idx))[2:]), dtype=int)
                    eps = np.pad(eps, (pos - len(eps), 0), "constant")
                    keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
                return keich
            u_init = keich_positions(N)
            v_init = keich_positions(N)
            best_params = np.concatenate([u_init[1:], v_init[1:]])
            best_val = objective_sep(best_params)

        # Polish with L-BFGS-B
        if time.time() < deadline:
            from scipy.optimize import minimize
            try:
                res_polish = minimize(
                    objective_sep, 
                    best_params, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-15}
                )
                if res_polish.fun < best_val:
                    best_params = res_polish.x
                    best_val = res_polish.fun
            except Exception:
                pass

        # Construct the separable positions
        u_final = np.zeros(N)
        v_final = np.zeros(N)
        u_final[1:] = best_params[:7]
        v_final[1:] = best_params[7:]
        
        positions = np.zeros((64, 2))
        for k in range(64):
            i = k // N
            j = k % N
            positions[k, 0] = u_final[i]
            positions[k, 1] = v_final[j]
            
        return positions

    # If we have time, we could try non-separable, but DE on 128 dims is risky.
    # The separable DE might find a better local min than L-BFGS-B.
    
    # Let's add a check: if we have time, try a few random non-separable perturbations
    # and evaluate them with a fast Monte Carlo estimate (5000 points) to see if they beat the separable best.
    # If yes, optimize them? Too complex for the time left.
    # Just return the separable result.
    
    return positions

if __name__ == "__main__":
    pos = kakeya_needle_3d8()
    print(pos.shape)
