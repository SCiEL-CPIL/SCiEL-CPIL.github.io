# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _calc_area_perm(perm, x_base, a, y, w):
    """
    Calculate area given a permutation of base positions.
    perm: array of indices 0..n-1 indicating the order of x_base elements.
    x_base: the actual x coordinates.
    """
    n = len(x_base)
    # Reorder x according to permutation
    x_perm = x_base[perm]
    
    # L_i(y) = x_i + a_i * y
    # We need to be careful: a_i is fixed to the triangle index i, not the permutation index.
    # Wait, the problem defines triangle R_i with vertices depending on x_i and a_i = (i+1)/64.
    # So x_i is the position of triangle i.
    # The union is over all i.
    # L_i(y) = x_i + a_i * y.
    # The sorting is on L_i(y) values.
    # If we permute the *labels* of the triangles, we are effectively changing which x_i goes with which a_i.
    # But x_i is the variable. 
    # Actually, the variable is the vector x = [x_0, ..., x_63].
    # The triangle i has slope a_i.
    # So we are assigning positions x_i to slopes a_i.
    # This is equivalent to choosing a permutation P such that x_i = x_base[P[i]]? 
    # No, x_i is just a number.
    # However, since the objective is symmetric with respect to relabeling if we also swap the slopes?
    # No, the slopes a_i are fixed to the index i in the definition of the triangle R_i.
    # But the union area is the area of the set union. The set union does not care about labels.
    # The set of triangles is {R_0, ..., R_63}.
    # R_i is determined by x_i and a_i.
    # If we swap x_i and x_j, we swap the positions of triangle i and j.
    # The set of triangles becomes {R_i', R_j', ...} where R_i' has x_j and a_i, R_j' has x_i and a_j.
    # This is generally a different set of triangles.
    # So permuting x is a valid move in the search space.
    
    # Efficient calculation:
    # L = x_perm[:, None] + a[:, None] * y[None, :]
    # Note: a is fixed. x_perm is the reordered x.
    # But wait. If I swap x_0 and x_1, I am changing the position of triangle 0 to x_1 and triangle 1 to x_0.
    # Triangle 0 still has slope a_0. Triangle 1 still has slope a_1.
    # So L_0(y) = x_1 + a_0 * y.
    # L_1(y) = x_0 + a_1 * y.
    
    # So we just need to compute L for the current x assignment.
    L = x_perm[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L at each y
    L_sorted = np.sort(L, axis=0)
    diffs = np.diff(L_sorted, axis=0)
    added_lengths = np.minimum(diffs, w)
    union_lengths = w + np.sum(added_lengths, axis=0)
    area = np.mean(union_lengths)
    return area

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    w = 1.0 / n
    a = (np.arange(n) + 1) / n
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    
    start_time = time.time()
    timeout = 600.0
    margin = 10.0
    
    # Initial guess: Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    keich_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich_positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(keich_positions, dtype=np.float64)
    
    # We will optimize x directly, but using a strategy that recognizes the structure.
    # Actually, a pure permutation search on a fixed set of x values might be too restrictive.
    # The optimal x values are likely not just permutations of the Keich values.
    # However, the "kinks" in the objective function occur when L_i(y) = L_j(y).
    # L_i(y) - L_j(y) = x_i - x_j + (a_i - a_j)y.
    # This is linear in y.
    
    # Let's try a hybrid: 
    # 1. Use a fast local search (coordinate descent) but with adaptive step sizes.
    # 2. Add a "swap" heuristic: if two triangles are very close in position, swapping them might help 
    #    if their slopes are very different? No, swapping x_i and x_j is just a move in x-space.
    
    # The previous program used L-BFGS-B. Let's try a different optimizer: 
    # Powell's method (derivative-free, good for kinked functions) or 
    # A custom "Subgradient" style method?
    # Or simply, a more aggressive random restart local search.
    
    # Let's implement a fast "Pairwise Swap and Shift" local search.
    # The objective is sum_y min(gaps, w).
    # This function is convex in each x_i? 
    # Union of intervals length is a convex function of the endpoints.
    # Endpoints are linear in x_i. So the objective is convex in x?
    # If the objective is convex, gradient methods should work well.
    # But L-BFGS-B might get stuck or slow.
    
    # Let's try a "Coordinate Descent with Binary Search" for optimal x_i.
    # For a fixed set of other x_j, the objective as a function of x_i is piecewise linear and convex.
    # We can find the minimum of a 1D piecewise linear convex function by checking the "breakpoints".
    # The breakpoints for x_i occur when L_i(y) = L_j(y) for some j and y, or when the min(gap, w) switches.
    # This is complex to compute exactly.
    
    # Alternative: Use scipy.optimize.minimize with method='Nelder-Mead' but with a very tight budget and 
    # a better initial guess? The inspiration 1 used Nelder-Mead and got 1.05.
    # The current program uses L-BFGS-B and got 0.965.
    # Nelder-Mead is often better for non-smooth functions.
    
    # Let's implement a robust Nelder-Mead with adaptive bounds and restarts.
    
    def _calc_area_fast(x):
        L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
        L_sorted = np.sort(L, axis=0)
        diffs = np.diff(L_sorted, axis=0)
        added_lengths = np.minimum(diffs, w)
        union_lengths = w + np.sum(added_lengths, axis=0)
        return np.mean(union_lengths)
    
    best_x = x0.copy()
    best_area = _calc_area_fast(x0)
    
    rng = np.random.default_rng(42)
    
    # Strategy: Multiple restarts of Nelder-Mead.
    # Each run is limited in time.
    
    num_restarts = 20
    time_per_restart = (timeout - margin) / num_restarts
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout - margin:
            break
            
        # Generate initial guess
        if r == 0:
            x_start = x0.copy()
        else:
            # Perturb best solution
            noise_scale = 0.1 * (1.0 + r * 0.1) # Increasing noise
            x_start = best_x + rng.normal(0, noise_scale, n)
            # Clip to reasonable range
            x_start = np.clip(x_start, -1.0, 2.0)
            
        # Run Nelder-Mead
        # To speed up NM, we can limit iterations.
        # scipy's Nelder-Mead can be slow.
        # Let's use a manual implementation or limit maxiter.
        
        try:
            from scipy.optimize import minimize
            res = minimize(_calc_area_fast, x_start, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            if np.all(np.isfinite(res.x)) and res.fun < best_area:
                best_x = res.x
                best_area = res.fun
        except Exception:
            pass
            
    # Final polish with L-BFGS-B on the best found so far
    if time.time() - start_time < timeout - margin - 30:
        try:
            from scipy.optimize import minimize
            res = minimize(_calc_area_fast, best_x, method='L-BFGS-B', 
                           bounds=[(-1.0, 2.0)] * n,
                           options={'maxiter': 200, 'ftol': 1e-8})
            if np.all(np.isfinite(res.x)) and res.fun < best_area:
                best_x = res.x
                best_area = res.fun
        except Exception:
            pass
            
    # One last check: simple coordinate descent with small steps
    if time.time() - start_time < timeout - margin:
        step = 0.01
        improved = True
        while improved and time.time() - start_time < timeout - margin:
            improved = False
            for i in range(n):
                for delta in [step, -step]:
                    trial_x = best_x.copy()
                    trial_x[i] += delta
                    trial_area = _calc_area_fast(trial_x)
                    if trial_area < best_area - 1e-10:
                        best_x = trial_x
                        best_area = trial_area
                        improved = True
                        break
                if time.time() - start_time > timeout - margin:
                    break
            # Shrink step
            step *= 0.8
    
    return best_x

# EVOLVE-BLOCK-END
