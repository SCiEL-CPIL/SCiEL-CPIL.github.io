import numpy as np
import time
import sys
from scipy.optimize import minimize

def compute_3d_cost_and_grad(positions):
    """
    Computes the 3D union volume using a differentiable soft-min approximation.
    positions: shape (16,) -> [x_0..x_7, y_0..y_7]
    
    The volume is the integral over c in [0,1] of the 2D union area of the 8 sheared squares.
    We approximate the indicator function of the union using a soft-min (or soft-max for complement)
    which is differentiable everywhere.
    """
    N = 8
    w = 1.0 / 8.0
    
    # Split positions into x and y components
    x_pos = positions[:N]
    y_pos = positions[N:]
    
    # We will integrate over c using a fixed set of quadrature points.
    # Since the geometry changes linearly with c, the "soft" area is smooth.
    # We use Gauss-Legendre quadrature with enough points to resolve the features.
    # The features (tube boundaries) move with speed up to 7/8 * 1 = 0.875.
    # A moderate number of points (e.g., 32) should suffice for the soft approximation.
    c_vals, wts = np.polynomial.legendre.leggauss(32)
    # Map from [-1, 1] to [0, 1]
    c_vals = 0.5 * (c_vals + 1.0)
    wts = 0.5 * wts
    
    total_cost = 0.0
    total_grad = np.zeros(16)
    
    # Soft-min parameter (sigma). Smaller sigma -> sharper, closer to true min.
    # Larger sigma -> smoother, easier to optimize but less accurate (overestimates volume? 
    # Actually soft-min underestimates min, so soft-max overestimates max.
    # We want to compute Union Area = Area(Box) - Area(Complement).
    # Complement is points NOT in any tube.
    # Indicator of tube k: I_k = 1 if inside, 0 if outside.
    # Union Indicator = max(I_k) ? No, Union is 1 if any I_k is 1.
    # So Union = max(I_0, ..., I_7).
    # We can approximate max using soft-max: max(z) ~ (1/beta) * log(sum(exp(b*z))).
    # But our "value" is binary (0 or 1). 
    # Alternatively, compute the distance to the union boundary.
    
    # Let's use the "soft indicator" approach for the union directly.
    # For a point (a,b) at height c, it is in tube k if:
    # x_k + c*i/8 <= a <= x_k + c*i/8 + w
    # y_k + c*j/8 <= b <= y_k + c*j/8 + w
    # Let d_k(a,b,c) be the "signed distance" to the square k.
    # If d_k > 0, it's inside. 
    # Union is 1 if max(d_k) > 0.
    # This is still non-smooth at the boundary.
    
    # Better: Use the "soft" area formula for the union of axis-aligned rectangles.
    # The area of the union of rectangles R_k is:
    # A = sum_k Area(R_k) - sum_{k<l} Area(R_k cap R_l) + ... (Inclusion-Exclusion)
    # This is exact but combinatorial (2^8 terms). With N=8, 255 terms.
    # This is EXACT and DIFFERENTIABLE if we handle the intersection area of two
    # axis-aligned rectangles smoothly?
    # Intersection of two axis-aligned rectangles is an axis-aligned rectangle (or empty).
    # Its area is max(0, min(r1, r2) - max(l1, l2)) * max(0, min(t1, t2) - max(b1, b2)).
    # The function max(0, z) is non-differentiable at z=0.
    # We can replace max(0, z) with softplus(z) = log(1 + exp(beta*z)) / beta.
    # This makes the inclusion-exclusion sum differentiable.
    
    # Let's implement Inclusion-Exclusion with Soft-Plus.
    # N=8, so 2^8 - 1 = 255 subsets.
    # For each subset S, we compute the intersection of all rectangles in S.
    # The intersection is a rectangle [max(l_k), min(r_k)] x [max(b_k), min(t_k)].
    # Area = softplus(min_r - max_l) * softplus(min_t - max_b) / beta^2 ?
    # Wait, softplus(z) approximates max(0,z).
    # Area of intersection = max(0, w_x) * max(0, w_y).
    # Approx: softplus(beta * w_x) * softplus(beta * w_y) / beta^2.
    
    beta = 50.0  # Sharpness of the approximation
    
    # Precompute indices for inclusion-exclusion
    # It might be faster to just iterate over all non-empty subsets of {0..7}
    # There are 255 subsets.
    # For each c in c_vals (32 points), we have 255 * 32 = 8160 intersection computations.
    # Each involves finding max of 8 lefts, min of 8 rights, etc.
    # This is very fast in numpy if vectorized, but subsets have different sizes.
    # We can precompute the list of subsets and their signs.
    
    subsets = []
    signs = []
    for mask in range(1, 1 << N):
        bits = [i for i in range(N) if (mask >> i) & 1]
        subsets.append(bits)
        # Inclusion-Exclusion sign is (-1)^(|S|+1)
        sign = 1 if len(bits) % 2 == 1 else -1
        signs.append(sign)
        
    # Vectorize over c_vals?
    # c_vals shape (32,)
    # For each c, we need to compute the union area.
    # Let's loop over c_vals (32 iterations) and vectorize over subsets (255 iterations).
    
    for idx_c in range(len(c_vals)):
        c = c_vals[idx_c]
        wt = wts[idx_c]
        
        # Compute rectangle parameters for all 8 tubes at height c
        # Lefts: x_k + c * k / 8
        # Rights: Lefts + w
        # Bottoms: y_k + c * k / 8
        # Tops: Bottoms + w
        # Note: The shear index is k (0..7).
        
        ks = np.arange(N)
        lefts = x_pos + c * ks / 8.0
        rights = lefts + w
        bottoms = y_pos + c * ks / 8.0
        tops = bottoms + w
        
        union_area = 0.0
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        for mask_idx, bits in enumerate(subsets):
            sign = signs[mask_idx]
            # Intersection of rectangles in 'bits'
            # Max Left, Min Right, Max Bottom, Min Top
            max_l = np.max(lefts[bits])
            min_r = np.min(rights[bits])
            max_b = np.max(bottoms[bits])
            min_t = np.min(tops[bits])
            
            w_x = min_r - max_l
            w_y = min_t - max_b
            
            # Soft area
            # softplus(z) = log(1+exp(z))
            # To avoid overflow, use stable softplus
            # But beta is fixed.
            # A_soft = softplus(beta * w_x) * softplus(beta * w_y) / beta^2
            
            # Gradient of softplus(beta * z) w.r.t z is beta * sigmoid(beta * z)
            # d/dw_x (softplus(beta w_x) / beta) = sigmoid(beta w_x)
            # Wait. Let f(w) = softplus(beta w) / beta.
            # f'(w) = (1/beta) * beta * sigmoid(beta w) = sigmoid(beta w).
            # So Area = f(w_x) * f(w_y).
            # dArea/dw_x = f'(w_x) * f(w_y) = sigmoid(beta w_x) * f(w_y).
            # dArea/dw_y = f(w_x) * sigmoid(beta w_y).
            
            sx = 1.0 / (1.0 + np.exp(-beta * w_x))
            sy = 1.0 / (1.0 + np.exp(-beta * w_y))
            
            fx = np.log1p(np.exp(beta * w_x)) / beta
            fy = np.log1p(np.exp(beta * w_y)) / beta
            
            area_term = fx * fy
            
            union_area += sign * area_term
            
            # Gradient contribution
            # w_x = min(rights[bits]) - max(lefts[bits])
            # d w_x / d x_k:
            # If k is in bits:
            #   d min_r / d x_k = 1 if k provides min_r else 0
            #   d max_l / d x_k = 1 if k provides max_l else 0
            #   So d w_x / d x_k = (1 if k is min_r) - (1 if k is max_l)
            # Note: x_k appears in both left_k and right_k (shifted by same amount).
            # left_k = x_k + c*k/8. d left_k / d x_k = 1.
            # right_k = x_k + c*k/8 + w. d right_k / d x_k = 1.
            
            # We need to identify which k in bits provides min_r and max_l.
            # min_r_idx = argmin(rights[bits])
            # max_l_idx = argmax(lefts[bits])
            
            # Since N is small, we can just loop over bits to update gradients.
            # Or vectorize:
            # min_r_val = min_r
            # max_l_val = max_l
            
            # For each k in bits:
            #   d_w_x_dx_k = (1 if rights[k] == min_r) - (1 if lefts[k] == max_l)
            #   d_w_y_dy_k = (1 if tops[k] == min_t) - (1 if bottoms[k] == max_b)
            
            # Note: ties are possible. In case of ties, the subgradient is the average?
            # Or we can just pick one. Since it's a measure zero event, it doesn't matter for optimization.
            # However, for stability, let's use argmin/argmax which picks the first one.
            
            # Vectorized gradient update for this subset
            # Create masks for bits
            mask_bits = np.zeros(N, dtype=bool)
            mask_bits[bits] = True
            
            # d w_x / d x_k
            # min_r is min of rights[bits]
            # If multiple have min_r, we distribute? 
            # Actually, if we use softplus, the gradient is smooth.
            # The derivative of min_r w.r.t x_k is 1 if x_k determines the min, 0 otherwise.
            # This is the "hard" min gradient.
            # But we are using softplus on the *width*, not on the individual coordinates.
            # The width w_x is a piecewise linear function of x_pos.
            # The gradient of the softplus composition is:
            # d/dx_k (softplus(beta * w_x)) = beta * sigmoid(beta * w_x) * d w_x / d x_k
            # So we still need d w_x / d x_k.
            
            # d w_x / d x_k = d(min_r)/dx_k - d(max_l)/dx_k
            # d(min_r)/dx_k = 1 if k is the argmin of rights, 0 else.
            # d(max_l)/dx_k = 1 if k is the argmax of lefts, 0 else.
            
            min_r_idx = np.argmin(rights[bits])
            max_l_idx = np.argmax(lefts[bits])
            min_t_idx = np.argmin(tops[bits])
            max_b_idx = np.argmax(bottoms[bits])
            
            # Map indices back to original indices
            orig_min_r_idx = bits[min_r_idx]
            orig_max_l_idx = bits[max_l_idx]
            orig_min_t_idx = bits[min_t_idx]
            orig_max_b_idx = bits[max_b_idx]
            
            # Gradient w.r.t x_k
            # d w_x / d x_k is non-zero only for k in {orig_min_r_idx, orig_max_l_idx}
            # If orig_min_r_idx == orig_max_l_idx, they cancel (0).
            
            # d Area / d x_k = sign * (d w_x / d x_k) * sigmoid(beta w_x) * fy
            # d Area / d y_k = sign * (d w_y / d y_k) * fx * sigmoid(beta w_y)
            
            # Initialize temp grads
            dx = np.zeros(N)
            dy = np.zeros(N)
            
            # x contribution
            # d w_x / d x_k
            if orig_min_r_idx == orig_max_l_idx:
                # 0
                pass
            else:
                dx[orig_min_r_idx] += 1.0
                dx[orig_max_l_idx] -= 1.0
            
            # y contribution
            if orig_min_t_idx == orig_max_b_idx:
                pass
            else:
                dy[orig_min_t_idx] += 1.0
                dy[orig_max_b_idx] -= 1.0
            
            # Scale by softplus derivatives and signs
            # dArea/dx_k = sign * dx_k * sx * fy
            # dArea/dy_k = sign * dy_k * fx * sy
            
            grad_x += sign * dx * (sx * fy)
            grad_y += sign * dy * (fx * sy)
            
        # Accumulate cost and gradient
        total_cost += wt * union_area
        total_grad[:N] += wt * grad_x
        total_grad[N:] += wt * grad_y
        
    return total_cost, total_grad

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0  # Safety margin
    
    N = 8
    
    # Initial guess: Keich configuration for x and y
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    # The Keich configuration is for 1D. For 2D, we can use it for both x and y.
    # But the shear is different for x and y (same shear factor c/8, but applied to x and y).
    # Actually, the problem statement says:
    # [x_k + c*i/8, ...] x [y_k + c*j/8, ...]
    # So x is sheared by i, y is sheared by j.
    # The Keich config minimizes the 1D union length for intervals sheared by i.
    # So using Keich for x_pos and y_pos is a good starting point.
    
    x0 = np.concatenate([keich, keich])
    
    best_positions = x0.copy()
    best_cost, _ = compute_3d_cost_and_grad(best_positions)
    
    rng = np.random.default_rng(12345)
    
    # Optimization
    # L-BFGS-B is good for box-constrained problems.
    # Bounds: The positions can be anywhere, but to keep the box small,
    # we can constrain them to a reasonable range, e.g., [-1, 2].
    # Actually, the evaluator computes X0, X1, Y0, Y1 based on the positions.
    # So there is no fixed box. The "score" is 1/V.
    # V = fraction * (X1-X0) * (Y1-Y0).
    # If we scale all positions by S, V scales by S^2. So 1/V scales by 1/S^2.
    # Wait, the tubes have fixed width 1/8.
    # If we scale positions, the relative positions change, but the absolute size of the box changes.
    # The union volume V is the physical volume.
    # If we translate, V is unchanged.
    # If we scale, the tube widths DO NOT scale.
    # So scaling positions apart increases the box size, but the tubes stay the same size.
    # This increases V. So we want to pack them tightly.
    # The optimal packing will have the tubes overlapping as much as possible.
    
    # Let's use L-BFGS-B with no bounds, or loose bounds.
    # Since the function is smooth, L-BFGS-B should work well.
    
    num_starts = 20
    max_iter_per_start = 1000
    
    for start_idx in range(num_starts):
        if time.time() - start_time > deadline:
            break
        
        if start_idx == 0:
            x_start = x0.copy()
        else:
            x_start = x0 + rng.normal(0, 0.05, 16)
            
        def objective(x):
            cost, _ = compute_3d_cost_and_grad(x)
            return cost
            
        def gradient(x):
            _, grad = compute_3d_cost_and_grad(x)
            return grad
            
        try:
            res = minimize(objective, x_start, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': max_iter_per_start, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_cost:
                best_cost = res.fun
                best_positions = res.x.copy()
        except Exception:
            pass
            
    # Construct the 2D positions
    # Row k is tube (i, j) with i = k // 8 and j = k % 8.
    # x_k = x_pos[i]
    # y_k = y_pos[j]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_positions[:N][i_flat]
    base_y = best_positions[N:][j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

if __name__ == "__main__":
    res = kakeya_needle_3d8()
    print(res.shape)
