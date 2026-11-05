import numpy as np
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    T = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    if T == 0:
        return y0
        
    # Augment the system to handle affine linear ODEs y' = Ay + b
    # by converting to homogeneous z' = M z where z = [y; 1]
    # M = [[A, b], [0, 0]]
    
    # Initial state for augmented system
    z0 = np.zeros(n + 1)
    z0[:n] = y0
    z0[n] = 1.0
    
    norm_z0 = np.linalg.norm(z0)
    if norm_z0 == 0:
        return y0
        
    v1 = z0 / norm_z0
    
    # Arnoldi parameters
    max_iter = n + 1  # Dimension of augmented system
    tol = 1e-12
    
    V = [v1]
    H = np.zeros((max_iter, max_iter), dtype=float)
    
    for j in range(max_iter):
        # Apply the augmented operator F to V[j]
        # F([y; c]) = [f(t0, y); 0]
        # Note: f(t, y) is the original derivative.
        # For the augmented system, the last component derivative is 0.
        
        y_part = V[j][:n]
        c_part = V[j][n]
        
        # Compute f(t0, y_part)
        # If the system is affine y' = Ay + b, f(t0, y) returns Ay + b.
        # The augmented derivative is [Ay + b*c; 0].
        # Wait, if f(t, y) = Ay + b, then for z=[y; c], 
        # z' = [Ay + b; 0] is only correct if c=1.
        # Generally z' = [Ay + b*c; 0] requires knowing A and b separately 
        # or assuming f is linear in y.
        # If f is linear (affine), f(t, y) = Ay + b.
        # Then f(t, c*y) = c*Ay + b. This is NOT c*(Ay+b).
        # So we cannot simply scale the output of f.
        
        # However, we are probing the operator M.
        # M * [y; c] = [Ay + b*c; 0].
        # We can compute this if we can evaluate A*y and b separately?
        # No, we only have f(t, y).
        
        # If we assume f(t, y) is strictly linear homogeneous (b=0), then f(t, c*y) = c*f(t, y).
        # If b != 0, f(t, c*y) = c*Ay + b.
        # We need M*v.
        # Let v = [y; c].
        # M*v = [Ay + b*c; 0].
        # We know f(t, y) = Ay + b.
        # We know f(t, 0) = b.
        # So Ay = f(t, y) - b.
        # M*v = [f(t, y) - b + b*c; 0] = [f(t, y) + b*(c-1); 0].
        
        # To compute this, we need b = f(t0, 0).
        # We can compute b once.
        
        # Let's pre-compute b
        b_vec = f(t0, np.zeros(n))
        
        # Now compute M * V[j]
        y_v = V[j][:n]
        c_v = V[j][n]
        
        # f(t0, y_v) gives A*y_v + b
        # We need A*y_v + b*c_v = (A*y_v + b) + b*(c_v - 1) = f(t0, y_v) + b*(c_v - 1)
        deriv_y = f(t0, y_v) + b_vec * (c_v - 1.0)
        
        w = np.zeros(n + 1)
        w[:n] = deriv_y
        # w[n] is 0
        
        # Modified Gram-Schmidt
        for i in range(j + 1):
            hij = np.dot(w, V[i])
            H[i, j] = hij
            w -= hij * V[i]
            
        hjj = np.linalg.norm(w)
        H[j, j] = hjj
        
        if hjj < tol:
            m = j + 1
            break
        else:
            V.append(w / hjj)
            m = j + 1
            
    if 'm' not in locals():
        m = max_iter
        
    H_sub = H[:m, :m]
    E = expm(T * H_sub)
    e1 = np.zeros(m)
    e1[0] = 1.0
    z = E @ e1
    V_mat = np.column_stack(V[:m])
    
    z_end = norm_z0 * (V_mat @ z)
    
    return z_end[:n]
