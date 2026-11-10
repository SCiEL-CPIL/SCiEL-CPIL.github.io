# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    We build N*N = 64 "generalized tubes" indexed by (i, j), i, j in {0, ..., N-1}.
    Tube (i, j) has a base position (x, y) and, at height c in [0, 1], occupies the
    axis-aligned square
        [x + c*i/N, x + 1/N + c*i/N] x [y + c*j/N, y + 1/N + c*j/N].
    
    The strategy uses an exact coordinate-compression sweep line to compute the 
    1D union length for x-intervals and y-intervals separately, then multiplies 
    them to get the 2D union area (since the union is a product set when x depends 
    only on i and y only on j). We then use scipy.optimize.minimize to find 
    base positions that minimize the total union volume.
    """
    N = 8
    deadline = time.time() + 600.0
    safety_margin = 30.0  # seconds
    
    # Initial positions using Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = keich[i_flat].copy()
    base_y = keich[j_flat].copy()
    positions = np.stack([base_x, base_y], axis=1)

    def compute_volume(base_x, base_y):
        """
        Compute the exact union volume using coordinate compression.
        The union is a product of the x-union and y-union because:
        - Tube (i,j) x-interval depends only on i and base_x[i]
        - Tube (i,j) y-interval depends only on j and base_y[j]
        - The union over all (i,j) of [x_i(c), x_i(c)+1/N] x [y_j(c), y_j(c)+1/N]
          = (union over i of x-intervals) x (union over j of y-intervals)
          at each height c.
        - The total volume = integral over c of area(c) dc
          = integral over c of (x_union_length(c) * y_union_length(c)) dc
        
        We compute this exactly by finding all critical points in c where the
        union length changes, then integrating piecewise.
        """
        w = 1.0 / N
        
        # For x: tube i has interval [base_x[i] + c*i/N, base_x[i] + w + c*i/N]
        # The right endpoint is base_x[i] + w + c*i/N
        # Two tubes i1, i2 have overlapping or changing union when:
        # left_i1 = right_i2 or left_i2 = right_i1
        # base_x[i1] + c*i1/N = base_x[i2] + w + c*i2/N
        # c*(i1-i2)/N = base_x[i2] + w - base_x[i1]
        # c = N*(base_x[i2] + w - base_x[i1]) / (i1-i2)
        
        # Critical c values for x
        c_crit_x = [0.0, 1.0]
        for i1 in range(N):
            for i2 in range(N):
                if i1 == i2:
                    continue
                denom = (i1 - i2) / N
                if abs(denom) < 1e-15:
                    continue
                c_val = (base_x[i2] + w - base_x[i1]) / denom
                if c_val > 0.0 and c_val < 1.0:
                    c_crit_x.append(c_val)
        
        # Critical c values for y
        c_crit_y = [0.0, 1.0]
        for j1 in range(N):
            for j2 in range(N):
                if j1 == j2:
                    continue
                denom = (j1 - j2) / N
                if abs(denom) < 1e-15:
                    continue
                c_val = (base_y[j2] + w - base_y[j1]) / denom
                if c_val > 0.0 and c_val < 1.0:
                    c_crit_y.append(c_val)
        
        # Combine all critical points
        all_c = sorted(set(c_crit_x + c_crit_y))
        if len(all_c) < 2:
            all_c = [0.0, 1.0]
        else:
            all_c[0] = 0.0
            all_c[-1] = 1.0
        
        # Integrate by sampling at midpoints of intervals between critical points
        volume = 0.0
        for k in range(len(all_c) - 1):
            c_mid = (all_c[k] + all_c[k+1]) / 2.0
            dc = all_c[k+1] - all_c[k]
            
            # Compute x union length at c_mid
            x_lefts = base_x + c_mid * np.arange(N) / N
            x_rights = x_lefts + w
            # Sort and compute union length
            order = np.argsort(x_lefts)
            sorted_lefts = x_lefts[order]
            sorted_rights = x_rights[order]
            
            x_len = 0.0
            current_left = sorted_lefts[0]
            current_right = sorted_rights[0]
            for idx in range(1, N):
                if sorted_lefts[idx] > current_right:
                    x_len += current_right - current_left
                    current_left = sorted_lefts[idx]
                    current_right = sorted_rights[idx]
                else:
                    current_right = max(current_right, sorted_rights[idx])
            x_len += current_right - current_left
            
            # Compute y union length at c_mid
            y_lefts = base_y + c_mid * np.arange(N) / N
            y_rights = y_lefts + w
            order_y = np.argsort(y_lefts)
            sorted_lefts_y = y_lefts[order_y]
            sorted_rights_y = y_rights[order_y]
            
            y_len = 0.0
            current_left_y = sorted_lefts_y[0]
            current_right_y = sorted_rights_y[0]
            for idx in range(1, N):
                if sorted_lefts_y[idx] > current_right_y:
                    y_len += current_right_y - current_left_y
                    current_left_y = sorted_lefts_y[idx]
                    current_right_y = sorted_rights_y[idx]
                else:
                    current_right_y = max(current_right_y, sorted_rights_y[idx])
            y_len += current_right_y - current_left_y
            
            volume += x_len * y_len * dc
        
        return volume

    # Use scipy.optimize.minimize to improve the positions
    from scipy.optimize import minimize
    
    def objective(params):
        if time.time() > deadline - safety_margin:
            return 1e10
        bx = params[:64]
        by = params[64:]
        return compute_volume(bx, by)
    
    x0 = np.concatenate([base_x, base_y])
    
    # Try L-BFGS-B with bounds
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if time.time() > deadline - safety_margin:
        pass  # Return current best
    
    if result.success or result.fun < compute_volume(base_x, base_y):
        best_params = result.x
        base_x = best_params[:64].copy()
        base_y = best_params[64:].copy()
    
    positions = np.stack([base_x, base_y], axis=1)
    
    # Validate
    if not np.all(np.isfinite(positions)):
        # Fall back to initial
        base_x = keich[i_flat].copy()
        base_y = keich[j_flat].copy()
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
