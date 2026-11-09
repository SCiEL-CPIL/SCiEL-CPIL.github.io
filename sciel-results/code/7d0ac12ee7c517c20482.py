<<<<<<< SEARCH
def _conjugate_gradient(pts, max_iter=1000, tol=1e-10, deadline=None):
    x = pts.copy()
    energy, grad = _energy_and_tangent_grad(x)
    p = -grad
    
    dot_p_x = np.sum(p * x, axis=1)
    p = p - dot_p_x[:, None] * x
    
    grad_prev = grad.copy()
    
    for k in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        alpha = 1.0
        c1 = 1e-4
        c2 = 0.9
        rho_shrink = 0.5
        rho_expand = 1.5
        max_alpha = 10.0
        
        dir_deriv = np.sum(grad * p)
        
        if dir_deriv > 0:
            p = -grad
            dot_p_x = np.sum(p * x, axis=1)
            p = p - dot_p_x[:, None] * x
            dir_deriv = np.sum(grad * p)
        
        found_step = False
        while alpha > 1e-12:
            x_new = x + alpha * p
            x_new = _project_to_sphere(x_new)
            e_new, g_new = _energy_and_tangent_grad(x_new)
            
            if e_new <= energy + c1 * alpha * dir_deriv:
                new_dir_deriv = np.sum(g_new * p)
                if new_dir_deriv >= c2 * dir_deriv:
                    found_step = True
                    break
                
                alpha_try = alpha
                while alpha_try < max_alpha:
                    alpha_try *= rho_expand
                    x_try = x + alpha_try * p
                    x_try = _project_to_sphere(x_try)
                    e_try, g_try = _energy_and_tangent_grad(x_try)
                    
                    if e_try <= energy + c1 * alpha_try * dir_deriv:
                        new_dir_deriv_try = np.sum(g_try * p)
                        if new_dir_deriv_try >= c2 * dir_deriv:
                            x_new = x_try
                            e_new = e_try
                            g_new = g_try
                            alpha = alpha_try
                            found_step = True
                            break
                    else:
                        found_step = True
                        break
                
                if not found_step:
                    found_step = True
                break
            else:
                alpha *= rho_shrink
        
        if not found_step:
            break
        
        x = x_new
        energy = e_new
        grad = g_new
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        grad_dot = np.sum(grad * (grad - grad_prev))
        grad_prev_dot = np.sum(grad_prev * grad_prev)
        
        if grad_prev_dot < 1e-16:
            beta = 0.0
        else:
            beta = max(0.0, grad_dot / grad_prev_dot)
            
        p_new = -grad + beta * p
        
        dot_p_new_x = np.sum(p_new * x, axis=1)
        p_new = p_new - dot_p_new_x[:, None] * x
        
        if np.sum(grad * p_new) > 0:
            beta = 0.0
            p_new = -grad
            dot_p_new_x = np.sum(p_new * x, axis=1)
            p_new = p_new - dot_p_new_x[:, None] * x
            
        p = p_new
        grad_prev = grad.copy()
        
    return x
=======
def _adam_manifold(pts, max_iter=2000, deadline=None, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam optimizer adapted for the sphere manifold.
    """
    x = pts.copy()
    energy, grad = _energy_and_tangent_grad(x)
    
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    
    for k in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        t += 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update direction in tangent space
        update = m_hat / (np.sqrt(v_hat) + eps)
        
        # Project update to tangent space to ensure we move along the manifold
        dot_update_x = np.sum(update * x, axis=1)
        update_tan = update - dot_update_x[:, None] * x
        
        # Adaptive step size
        # We can scale lr based on the norm of the gradient or just use fixed lr
        # Let's try a simple fixed lr first, maybe scaled by 1/sqrt(t) for early steps?
        # Standard Adam uses lr / (1 - beta1^t) etc which is already in m_hat/v_hat.
        
        x_new = x - lr * update_tan
        x_new = _project_to_sphere(x_new)
        
        e_new, g_new = _energy_and_tangent_grad(x_new)
        
        # Simple check: if energy increased significantly, reduce step? 
        # For now, just accept if it's finite. 
        # To be safe, we can do a line search if e_new > energy + 0.1 * np.linalg.norm(grad)
        if not np.isfinite(e_new) or e_new > energy + 1.0:
            # Fallback to smaller step
            x_new = x - lr * 0.1 * update_tan
            x_new = _project_to_sphere(x_new)
            e_new, g_new = _energy_and_tangent_grad(x_new)
            if not np.isfinite(e_new):
                break
                
        x = x_new
        energy = e_new
        grad = g_new
        
        if np.linalg.norm(grad) < 1e-10:
            break
            
    return x
>>>>>>> REPLACE

Now, we need to update the `construct` function to use `_adam_manifold` for the main optimization loop, as it might be more robust for finding better local minima. We will keep `_conjugate_gradient` for the final polish if needed, but Adam is often better for the "search" phase.

Actually, let's just replace the `_conjugate_gradient` calls in the main loop with `_adam_manifold`.

