import numpy as np
import scipy.optimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # Initial guess: Keich construction
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    # Precompute constants
    a = (np.arange(n) + 1) / n
    M = 20000
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n  # Shape (M,)
    
    # Precompute A = a[:, None] * y[None, :] for L calculation
    # L = x[:, None] + A
    A = a[:, None] * y[None, :]
    
    def calc_area_and_grad(x_vals):
        # x_vals: shape (n,)
        L = x_vals[:, None] + A  # Shape (n, M)
        
        # Sort L along axis 0
        # I: indices of sorted elements, shape (n, M)
        # L_sorted: sorted values, shape (n, M)
        I = np.argsort(L, axis=0)
        L_sorted = np.take_along_axis(L, I, axis=0)
        
        # Gaps
        d = L_sorted[1:, :] - L_sorted[:-1, :]  # Shape (n-1, M)
        
        # Determine where gaps are less than w (overlap)
        # w is shape (M,), broadcast to (n-1, M)
        overlap_mask = d < w[None, :]
        
        # Gradient calculation
        # For each height m, the gradient of the union length w.r.t L_sorted[k, m] is:
        # -1 if k is the left interval of an overlapping pair
        # +1 if k is the right interval of an overlapping pair
        # 0 otherwise
        
        # grad_L_sorted shape (n, M)
        grad_L_sorted = np.zeros_like(L)
        
        # For each gap k (0 to n-2), if overlap_mask[k, m] is True:
        # grad_L_sorted[k, m] -= 1  (left interval)
        # grad_L_sorted[k+1, m] += 1 (right interval)
        
        for k in range(n - 1):
            mask = overlap_mask[k, :]
            grad_L_sorted[k, mask] -= 1.0
            grad_L_sorted[k + 1, mask] += 1.0
            
        # Now grad_L_sorted contains d(U)/d(L_sorted)
        # We need d(U)/d(x_i).
        # L_i = x_i + A_i. So d(L_i)/d(x_i) = 1.
        # But L_sorted is a permutation of L.
        # So d(U)/d(x_i) = sum over m of d(U)/d(L_i, m)
        # d(U)/d(L_i, m) is the value in grad_L_sorted at the position where L_i was in the sorted order.
        
        # We can use np.add.at to accumulate gradients back to original indices.
        # grad_x shape (n,)
        grad_x = np.zeros(n)
        
        # For each column m, for each k, the gradient grad_L_sorted[k, m] belongs to x_{I[k, m]}
        # So we can do:
        # np.add.at(grad_x, I, grad_L_sorted)
        
        np.add.at(grad_x, I, grad_L_sorted)
        
        # The area is the mean of union lengths.
        # Union length U(y) = w + sum(min(d, w))
        mins = np.minimum(d, w[None, :])
        union_lens = w + np.sum(mins, axis=0)
        area = np.mean(union_lens)
        
        # The gradient of the area is the mean of the gradients of the union lengths.
        # So we need to divide grad_x by M.
        grad_x /= M
        
        return area, grad_x

    start_time = time.time()
    timeout = 600
    safety_margin = 10
    
    # Use L-BFGS-B
    # x0 is the initial guess
    # We need to define the objective and gradient functions for scipy
    def obj_func(x_vals):
        area, _ = calc_area_and_grad(x_vals)
        return area
    
    def grad_func(x_vals):
        _, grad = calc_area_and_grad(x_vals)
        return grad

    # Bounds: x_i should be in a reasonable range.
    # The triangles are in [0, 1] roughly.
    # Let's use bounds (-1, 2) to be safe.
    bounds = [(-1.0, 2.0) for _ in range(n)]
    
    try:
        res = scipy.optimize.minimize(
            obj_func,
            x,
            method='L-BFGS-B',
            jac=grad_func,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if res.success or res.fun < obj_func(x):
            x = res.x
            
    except Exception as e:
        # If optimization fails, keep original x
        pass

    # Check time
    if time.time() - start_time > timeout - safety_margin:
        return x

    # If we have time left, do coordinate descent refinement with the full grid
    # This can escape local minima that L-BFGS-B might get stuck in
    while time.time() - start_time < timeout - safety_margin:
        improved = False
        for i in range(n):
            if time.time() - start_time > timeout - safety_margin:
                break
            
            x_i_old = x[i]
            
            # Objective function for 1D optimization
            def obj_1d(xi):
                x_new = x.copy()
                x_new[i] = xi
                area, _ = calc_area_and_grad(x_new)
                return area
            
            # Optimize xi
            # Search range: The triangles should stay relatively compact.
            # Let's use a reasonable bound.
            res = scipy.optimize.minimize_scalar(
                obj_1d, 
                bounds=(x_i_old - 0.5, x_i_old + 0.5), 
                method='bounded', 
                options={'xatol': 1e-8}
            )
            
            if res.success and res.fun < obj_1d(x_i_old) - 1e-12:
                x[i] = res.x
                improved = True
        
        if not improved:
            break

    return x
