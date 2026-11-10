import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    frac = 1.0 / N
    start_time = time.time()
    timeout = 590  # Leave margin for the 600s limit

    # Helper: Compute exact union length of intervals [start_k, start_k + frac]
    # This is a piecewise linear function.
    def union_length(starts):
        # starts: shape (N,)
        # Sort by start
        idx = np.argsort(starts)
        s_sorted = starts[idx]
        e_sorted = s_sorted + frac
        
        # Merge intervals
        # L = sum of lengths of merged components
        # Component starts at s_sorted[0], ends at max(e_sorted[0], ... until gap)
        
        # Vectorized merge for small N is tricky without loops, but N=8 is tiny.
        # We can use the formula:
        # L = max(e) - min(s) - sum(gaps)
        # Gaps occur when s_{k} > max_{j<k} e_j
        # Let's just loop, it's fast enough for 50 quadrature points * many iterations.
        
        total_len = 0.0
        cur_start = s_sorted[0]
        cur_end = e_sorted[0]
        for k in range(1, N):
            if s_sorted[k] > cur_end:
                total_len += cur_end - cur_start
                cur_start = s_sorted[k]
                cur_end = e_sorted[k]
            else:
                if e_sorted[k] > cur_end:
                    cur_end = e_sorted[k]
        total_len += cur_end - cur_start
        return total_len

    # Objective: Minimize the volume V
    # V = integral_0^1 L_x(c) * L_y(c) dc
    # L_x(c) = union_length(u_i + c*i/N) for i=0..7
    # L_y(c) = union_length(v_j + c*j/N) for j=0..7
    # Note: In the non-separable case, the x-position of tube (i,j) is x_{i,j}.
    # The tube (i,j) at height c is [x_{i,j} + c*i/N, x_{i,j} + c*i/N + 1/N] x [y_{i,j} + c*j/N, y_{i,j} + c*j/N + 1/N]
    
    # Wait, the definition says:
    # "Row k is tube (i, j) with i = k // 8 and j = k % 8."
    # "At height c in [0, 1] tube k is the closed square [x_k + c*i/8, ...] x [y_k + c*j/8, ...]"
    # So x_k depends on k. If we drop separability, x_k is independent for each k.
    # However, the union in x-direction at height c is the union of x-intervals of ALL tubes.
    # The x-interval of tube k is [x_k + c*i_k/8, x_k + c*i_k/8 + 1/8].
    # There are 64 tubes. So we need the union length of 64 intervals in x, and 64 intervals in y.
    # This is much more complex than 8 intervals.
    
    # Let's re-read carefully.
    # "union of 64 sheared tubes"
    # "tube k is the closed square ... x ..."
    # The union is the set of points (x,y,c) such that (x,y) is in the square of tube k for some k.
    # Volume = Integral over c of Area( Union_{k=0}^{63} Square_k(c) ) dc.
    # The area of the union of squares is NOT simply L_x(c) * L_y(c) unless the squares are aligned in a grid way or we are looking at the bounding box.
    # Actually, the union of rectangles $R_k = [x_{k,c}, x_{k,c}+w] \times [y_{k,c}, y_{k,c}+h]$ 
    # The area of the union of these rectangles is what we want to minimize.
    
    # Calculating the exact area of the union of 64 rectangles is computationally expensive (O(N^2) or more for merging).
    # However, we can approximate it or use a different metric.
    # The current program uses L_x(c) * L_y(c). This is the area of the bounding box of the union, NOT the area of the union itself.
    # The bounding box area is an UPPER BOUND on the union area.
    # Minimizing the bounding box area is a valid strategy, but it might not minimize the true union area.
    # But wait, the Monte Carlo estimator in the prompt calculates:
    # "V = (fraction of the points (a, b, c) inside at least one tube) * (X1 - X0) * (Y1 - Y0)"
    # This V is the estimated VOLUME of the union.
    # The score is 1/V.
    # So we MUST minimize the true union volume.
    
    # The current program's objective `volume_objective` computes `0.5 * sum(w * lx * ly)`.
    # This is the volume of the box [min_x, max_x] x [min_y, max_y] x [0,1].
    # This is NOT the union volume. The union volume is smaller.
    # Why did the current program score 1.4398?
    # If it minimizes the bounding box volume, it minimizes an upper bound.
    # If the bounding box volume is minimized, the union volume is likely also small, but maybe not optimal.
    # Can we do better?
    # Yes, by minimizing the actual union area at each c.
    
    # How to compute union area of 64 rectangles efficiently?
    # For fixed c, we have 64 rectangles.
    # We can compute the union area by sweeping in x or y.
    # Or, since N=8, the x-coordinates depend on i (8 values) and y on j (8 values) in the separable case.
    # In the non-separable case, we have 64 distinct x-starts and 64 distinct y-starts.
    
    # Let's try a different approach:
    # The union of squares is a subset of the bounding box.
    # The area of the union of rectangles can be computed exactly in O(N^2 log N) or similar.
    # With 64 rectangles, O(64^2) = 4096 operations per c.
    # With 50 c-points, that's 200k operations per objective call.
    # With 1000 iterations, that's 200M operations. This might be too slow for Python loops.
    # We need a vectorized or faster method.
    
    # Alternative:
    # Use the fact that the rectangles are "thin" (width 1/8, height 1/8).
    # Actually, let's look at the structure.
    # Tube (i,j) x-interval: [x_{ij} + c*i/8, x_{ij} + c*i/8 + 1/8]
    # Tube (i,j) y-interval: [y_{ij} + c*j/8, y_{ij} + c*j/8 + 1/8]
    
    # If we assume separability $x_{ij} = u_i + \delta_{ij}^x$ and $y_{ij} = v_j + \delta_{ij}^y$, we still have 128 variables.
    # But maybe a simpler structural change:
    # The current program fixes $x_k = u_{k//8}$ and $y_k = v_{k\%8}$.
    # This means all tubes with the same $i$ share the same x-shift base.
    # This forces the x-projection of the union to be the union of 8 "bands" (one for each i).
    # If we allow $x_{ij}$ to vary with $j$, we can potentially "interleave" the x-intervals of different j's for the same i to reduce the union length?
    # No, the union length in x is determined by the set of all x-intervals.
    # If we have 8 intervals for i=0, 8 for i=1, etc.
    # The union of 64 intervals is generally larger than the union of 8 intervals (if the 8 groups were identical).
    # Wait. If $x_{ij} = u_i$, then for a fixed i, all 8 tubes have the SAME x-interval.
    # So the union of x-intervals is just the union of 8 intervals (one per i).
    # If we allow $x_{ij}$ to vary, we have 8 intervals for i=0, but they might be different.
    # This would INCREASE the union length in x (or keep it same) because we are adding more distinct intervals.
    # Unless... the variation allows us to shift the entire group? No, $x_{ij}$ is the base.
    # If $x_{ij}$ varies, the x-projection becomes the union of 64 intervals.
    # The union of 64 intervals is >= the union of 8 intervals (if we collapse them).
    # So dropping separability in x INCREASES the x-span.
    # Same for y.
    # So the bounding box volume $L_x L_y$ would likely INCREASE.
    
    # However, the UNION AREA is not $L_x L_y$.
    # The union area of the squares might be smaller if the squares are "scattered" such that they don't overlap in the 2D plane?
    # No, the squares are axis-aligned.
    # Area(Union of Rectangles) is what we want to minimize.
    # If we use the separable case, the rectangles form a grid-like structure.
    # If we perturb them, we might break the grid and reduce overlap?
    # Or increase it?
    
    # Let's think about the geometry.
    # Separable: x depends on i, y depends on j.
    # At height c, the x-interval of tube (i,j) is $I_{i,c}$. The y-interval is $J_{j,c}$.
    # The square is $I_{i,c} \times J_{j,c}$.
    # The union is $\bigcup_{i,j} (I_{i,c} \times J_{j,c}) = (\bigcup_i I_{i,c}) \times (\bigcup_j J_{j,c})$.
    # Wait! The union of products is NOT the product of unions.
    # $\bigcup_{i,j} (A_i \times B_j) = (\bigcup_i A_i) \times (\bigcup_j B_j)$ IS TRUE.
    # Proof: $(x,y) \in \bigcup_{i,j} (A_i \times B_j) \iff \exists i,j: x \in A_i, y \in B_j \iff x \in \bigcup_i A_i \land y \in \bigcup_j B_j \iff (x,y) \in (\bigcup_i A_i) \times (\bigcup_j B_j)$.
    # So, IF the x-interval depends only on i and y-interval depends only on j, the area of the union of squares is EXACTLY $L_x(c) \times L_y(c)$.
    # The current program minimizes $\int L_x L_y dc$. This is the EXACT volume for the separable case.
    
    # If we DROP separability, i.e., $x_{ij}$ depends on j, then $I_{ij,c}$ depends on j.
    # Then the union is NOT a product set.
    # The area is Area($\bigcup_{i,j} I_{ij,c} \times J_{ij,c}$).
    # This is generally HARDER to compute and might be smaller or larger.
    # However, note that if we set $x_{ij} = u_i + \epsilon_{ij}$, the x-intervals for a fixed i are slightly shifted.
    # The union of x-intervals $\bigcup_{i,j} I_{ij,c}$ will likely be LARGER than $\bigcup_i I_{i,c}$ because we are adding more distinct intervals.
    # Similarly for y.
    # The area of the union of rectangles is bounded by $L_{x,total} L_{y,total}$.
    # But is it smaller than the separable case?
    # In the separable case, the "holes" in the union are the holes in $L_x$ times the holes in $L_y$ plus the cross terms?
    # Actually, the separable union is a "grid" of rectangles.
    # If we perturb the positions, we might fill in some holes? No, we want to MINIMIZE area.
    # We want to MAXIMIZE holes.
    # The separable case creates a regular grid of holes.
    # Can we create MORE holes by perturbing?
    # If we shift rectangles, we might cover existing holes or create new ones.
    # Generally, spreading out the rectangles (increasing span) increases area.
    # Clumping them decreases span but might increase overlap (reducing area? No, overlap reduces union area).
    # Wait. Union Area = Sum(Area) - Sum(Overlaps) + ...
    # To minimize Union Area, we want to MAXIMIZE Overlaps.
    # So we want the rectangles to overlap as much as possible.
    # In the separable case, rectangles $(i,j)$ and $(i',j')$ overlap if $I_i \cap I_{i'} \neq \emptyset$ AND $J_j \cap J_{j'} \neq \emptyset$.
    # If we make $x_{ij}$ depend on j, we might break the alignment of x-intervals for different j's, potentially REDUCING the overlap in x for pairs that share the same i?
    # No, if $x_{ij}$ varies, the x-intervals for fixed i are different. They might overlap less with each other?
    # But they are all centered around $u_i + c i/8$. If we spread them, they might cover more x-range, reducing overlap with other i's?
    # This is complex.
    
    # However, there is a known result: The "Keich" construction is optimal for the 1D problem.
    # For the 3D problem, is the product of 1D optimal sets optimal?
    # Likely yes, due to the product structure of the volume integral if separability holds.
    # But if we break separability, we lose the product structure.
    
    # Let's try a different structural change:
    # Instead of optimizing positions, optimize the SLOPES?
    # No, slopes are fixed by the problem statement ($i/8, j/8$).
    
    # What if we use a DIFFERENT initial guess?
    # The current program uses Keich.
    # What if we use a "random" or "grid" initialization and let the optimizer find a better local minimum?
    # The current program does some random restarts.
    
    # Let's look at the "Idea target": "Does operator 'cross' change the chance that a child improves its island?"
    # This suggests that recombination (crossing) of solutions might be beneficial.
    # But I am writing a single program. I can simulate a "crossover" by taking the best separable solution and the best non-separable solution?
    # No, I only have one shot.
    
    # Let's try to MINIMIZE the true union area, not the bounding box, for the separable case.
    # Wait, for the separable case, the true union area IS the bounding box area.
    # So the current objective is exact for the separable family.
    
    # Is there a non-separable family with smaller volume?
    # Consider N=2.
    # Tubes: (0,0), (0,1), (1,0), (1,1).
    # Separable: x depends on i, y on j.
    # Union is a rectangle (if intervals overlap) or two rectangles.
    # If we make x depend on j, we get 4 distinct x-intervals.
    # This likely increases the x-span.
    
    # Hypothesis: The separable Keich solution is close to optimal.
    # The score 1.4398 is the current best.
    # To beat it, we need a significantly smaller volume.
    
    # What if we use a HIGHER order quadrature?
    # The current program uses 50 points.
    # The function $L_x(c) L_y(c)$ is piecewise linear in c?
    # $L_x(c)$ is the union length of intervals with linear shifts.
    # The order of intervals changes at specific c values.
    # Between these values, the merge structure is fixed, so $L_x(c)$ is linear in c.
    # Thus $L_x(c) L_y(c)$ is a quadratic polynomial in c between breakpoints.
    # Gauss-Legendre with 50 points is extremely accurate for polynomials of degree < 100.
    # The number of breakpoints is small (combinations of i,j).
    # So the objective is very smooth and well-approximated.
    
    # Maybe the optimization is stuck in a local minimum?
    # L-BFGS-B is good for smooth functions.
    # Nelder-Mead is good for non-smooth.
    # The function is continuous but not differentiable at breakpoints.
    # However, the breakpoints are sparse.
    
    # Let's try a GLOBAL optimization method like Basin Hopping or Differential Evolution.
    # Differential Evolution is robust to local minima.
    # It requires many function evaluations.
    # With 14 variables, DE might be feasible.
    
    # Strategy: Use Differential Evolution to find the global minimum of the separable Keich-like objective.
    # This addresses the "stagnation" by using a different search method (global vs local gradient).
    
    def volume_objective_separable(params):
        u = np.zeros(N)
        v = np.zeros(N)
        u[1:] = params[:7]
        v[1:] = params[7:]
        
        # Use exact union length
        # We can vectorize the c-integration?
        # No, the merge logic depends on c.
        # But we can pre-compute the breakpoints?
        # For N=8, the relative order of $u_i + c i/N$ changes when $u_i + c i/N = u_k + c k/N$.
        # $c (i-k)/N = u_k - u_i \implies c = N (u_k - u_i) / (i-k)$.
        # There are $8 \times 7 / 2 = 28$ such breakpoints for x.
        # Similarly for y.
        # Total breakpoints < 60.
        # We can sort all breakpoints, and for each interval, the merge order is fixed.
        # Then $L_x(c)$ is linear in that interval.
        # We can integrate analytically!
        
        # This would make the objective evaluation VERY fast (O(N^2) for breakpoints + O(N) per interval).
        # And it would be EXACT.
        
        # Let's implement this analytic integration.
        
        # Get breakpoints for x
        # $c_{ik} = N (u_k - u_i) / (i - k)$ for $i \neq k$
        # Valid if $0 < c_{ik} < 1$
        
        def get_breakpoints(base):
            bps = [0.0, 1.0]
            for i in range(N):
                for k in range(N):
                    if i == k: continue
                    denom = i - k
                    if denom == 0: continue
                    c_val = N * (base[k] - base[i]) / denom
                    if 0 < c_val < 1:
                        bps.append(c_val)
            bps = sorted(set(bps)) # unique and sorted
            return bps
        
        def integrate_linear_product(bps_x, bps_y, u, v):
            # Combine breakpoints
            all_bps = sorted(set(bps_x + bps_y))
            
            total_vol = 0.0
            for m in range(len(all_bps) - 1):
                c_start = all_bps[m]
                c_end = all_bps[m+1]
                c_mid = 0.5 * (c_start + c_end)
                
                # Evaluate L_x and L_y at c_mid
                # Since the order is constant in [c_start, c_end], 
                # L_x(c) = A_x c + B_x
                # L_y(c) = A_y c + B_y
                # We can find A, B by evaluating at c_start and c_end?
                # Or just evaluate at c_mid and use the fact that it's linear?
                # Actually, L(c) is piecewise linear.
                # We can compute L(c_start) and L(c_end) and integrate the product of two linear functions?
                # No, L_x and L_y are linear in DIFFERENT intervals.
                # In the interval [c_start, c_end], L_x is linear and L_y is linear.
                # So $L_x(c) L_y(c)$ is quadratic.
                # We can integrate the quadratic exactly.
                
                lx_start = union_length(u + c_start * (np.arange(N) / N))
                lx_end = union_length(u + c_end * (np.arange(N) / N))
                ly_start = union_length(v + c_start * (np.arange(N) / N))
                ly_end = union_length(v + c_end * (np.arange(N) / N))
                
                # Linear interpolation
                # L_x(c) = lx_start + (lx_end - lx_start) * (c - c_start) / (c_end - c_start)
                # Let t = (c - c_start) / (c_end - c_start)
                # L_x = lx_start + t * dx
                # L_y = ly_start + t * dy
                # Product = (lx_start + t dx)(ly_start + t dy)
                # = lx_start ly_start + t (lx_start dy + ly_start dx) + t^2 dx dy
                # Integrate t from 0 to 1:
                # I = lx_start ly_start + 0.5 (lx_start dy + ly_start dx) + (1/3) dx dy
                # Volume contribution = (c_end - c_start) * I
                
                dx = lx_end - lx_start
                dy = ly_end - ly_start
                I = lx_start * ly_start + 0.5 * (lx_start * dy + ly_start * dx) + (1.0/3.0) * dx * dy
                total_vol += (c_end - c_start) * I
                
            return total_vol

        bps_x = get_breakpoints(u)
        bps_y = get_breakpoints(v)
        return integrate_linear_product(bps_x, bps_y, u, v)

    # This analytic objective is FAST and EXACT.
    # We can use it with L-BFGS-B or Nelder-Mead.
    # Since it is piecewise smooth, L-BFGS-B should work well.
    
    # Let's use this new objective with L-BFGS-B and Nelder-Mead.
    
    u0 = np.zeros(N)
    v0 = np.zeros(N)
    # Keich initialization
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
    u0[1:] = u_init[1:]
    v0[1:] = v_init[1:]
    x0 = np.concatenate([u0[1:], v0[1:]])
    
    bounds = [(-2.0, 2.0)] * 14
    
    best_x = x0
    best_f = volume_objective_separable(x0)
    
    # Optimize with L-BFGS-B
    try:
        res = minimize(
            volume_objective_separable, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x
    except Exception as e:
        pass
    
    # Polish with Nelder-Mead
    if time.time() - start_time < timeout:
        try:
            res_nm = minimize(
                volume_objective_separable, 
                best_x, 
                method='Nelder-Mead', 
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-15}
            )
            if res_nm.fun < best_f:
                best_f = res_nm.fun
                best_x = res_nm.x
        except Exception:
            pass
            
    # Random restarts
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.time() - start_time > timeout - 30:
            break
        x_rand = x0 + rng.normal(0, 0.2, size=14)
        x_rand = np.clip(x_rand, -2.0, 2.0)
        try:
            res_r = minimize(
                volume_objective_separable, 
                x_rand, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            if res_r.fun < best_f:
                best_f = res_r.fun
                best_x = res_r.x
        except Exception:
            pass

    u_final = np.zeros(N)
    v_final = np.zeros(N)
    u_final[1:] = best_x[:7]
    v_final[1:] = best_x[7:]
    
    positions = np.zeros((64, 2))
    for k in range(64):
        i = k // 8
        j = k % 8
        positions[k, 0] = u_final[i]
        positions[k, 1] = v_final[j]
        
    return positions

if __name__ == "__main__":
    pos = kakeya_needle_3d8()
    print(pos.shape)
