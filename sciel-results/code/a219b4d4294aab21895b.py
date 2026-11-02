<<<<<<< SEARCH
    # Optimization Loop
    max_starts = len(init_bary_list)
    current_start = 0
    
    for bary0 in init_bary_list:
        if not _elapsed_ok(25):
            break
        if current_start >= max_starts:
            break
        current_start += 1
        
        # Annealing schedule for beta - more gradual
        betas = [5.0, 15.0, 50.0, 150.0, 500.0, 1500.0, 5000.0]
        
        current_bary = bary0.copy()
        
        for beta in betas:
            if not _elapsed_ok(15):
                break
            
            def objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_soft_min_area_cart(cart, beta)
            
            x0 = current_bary.flatten()
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                               options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
                current_bary = res.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass
        
        # Polish with Nelder-Mead on the hard minimum (exact min)
        if _elapsed_ok(8):
            nm_start = current_bary.flatten()
            
            def nm_objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_min_triangle_area_cart(cart)
            
            try:
                res_nm = minimize(nm_objective, nm_start, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-11,
                                           'adaptive': True})
                current_bary = res_nm.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass
        
        # Random perturbations and re-optimize
        if _elapsed_ok(12) and current_start < max_starts // 2:
            for _ in range(3):
                if not _elapsed_ok(8):
                    break
                perturbed = current_bary + rng.normal(0, 0.04, size=(N, 2))
                perturbed = _project_bary_to_valid(perturbed)
                
                def quick_obj(x):
                    b = x.reshape(N, 2)
                    b = _project_bary_to_valid(b)
                    cart = _bary_to_cart(b)
                    return -_soft_min_area_cart(cart, 1000.0)
                
                try:
                    res_q = minimize(quick_obj, perturbed.flatten(), method='L-BFGS-B',
                                     options={'maxiter': 80, 'ftol': 1e-14})
                    new_bary = res_q.x.reshape(N, 2)
                    new_bary = _project_bary_to_valid(new_bary)
                    _update_best(new_bary)
                    if new_bary is not None:
                        current_bary = new_bary
                except Exception:
                    pass
=======
    # Optimization Loop
    max_starts = len(init_bary_list)
    current_start = 0
    
    for bary0 in init_bary_list:
        if not _elapsed_ok(20):
            break
        if current_start >= max_starts:
            break
        current_start += 1
        
        # Annealing schedule for beta - more gradual with more stages
        betas = [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
        
        current_bary = bary0.copy()
        
        for beta in betas:
            if not _elapsed_ok(12):
                break
            
            def objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                sm, _ = _soft_min_area_and_grad(cart, beta)
                return -sm
            
            def gradient(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                _, grad_cart = _soft_min_area_and_grad(cart, beta)
                # Chain rule: bary -> cart
                # cart_x = 0.5*v + (1-u-v)*0.5 = 0.5 - 0.5*u
                # cart_y = v*S3H + (1-u-v)*S3H = S3H - u*S3H
                # d(cart_x)/d(u) = -0.5, d(cart_x)/d(v) = 0.5
                # d(cart_y)/d(u) = -S3H, d(cart_y)/d(v) = 0
                # grad_bary[i,0] = -0.5*grad_cart[i,0] - S3H*grad_cart[i,1]
                # grad_bary[i,1] = 0.5*grad_cart[i,0]
                grad_bary = np.zeros_like(b)
                grad_bary[:, 0] = -0.5 * grad_cart[:, 0] - S3H * grad_cart[:, 1]
                grad_bary[:, 1] = 0.5 * grad_cart[:, 0]
                return -grad_bary.flatten()
            
            x0 = current_bary.flatten()
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', jac=gradient,
                               options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13})
                current_bary = res.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass
        
        # Polish with Nelder-Mead on the hard minimum (exact min)
        if _elapsed_ok(6):
            nm_start = current_bary.flatten()
            
            def nm_objective(x):
                b = x.reshape(N, 2)
                b = _project_bary_to_valid(b)
                cart = _bary_to_cart(b)
                return -_min_triangle_area_cart(cart)
            
            try:
                res_nm = minimize(nm_objective, nm_start, method='Nelder-Mead',
                                  options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12,
                                           'adaptive': True})
                current_bary = res_nm.x.reshape(N, 2)
                current_bary = _project_bary_to_valid(current_bary)
                _update_best(current_bary)
            except Exception:
                pass
        
        # Random perturbations and re-optimize
        if _elapsed_ok(10) and current_start < max_starts // 2:
            for _ in range(4):
                if not _elapsed_ok(6):
                    break
                perturbed = current_bary + rng.normal(0, 0.03, size=(N, 2))
                perturbed = _project_bary_to_valid(perturbed)
                
                def quick_obj(x):
                    b = x.reshape(N, 2)
                    b = _project_bary_to_valid(b)
                    cart = _bary_to_cart(b)
                    sm, _ = _soft_min_area_and_grad(cart, 5000.0)
                    return -sm
                
                def quick_grad(x):
                    b = x.reshape(N, 2)
                    b = _project_bary_to_valid(b)
                    cart = _bary_to_cart(b)
                    _, grad_cart = _soft_min_area_and_grad(cart, 5000.0)
                    grad_bary = np.zeros_like(b)
                    grad_bary[:, 0] = -0.5 * grad_cart[:, 0] - S3H * grad_cart[:, 1]
                    grad_bary[:, 1] = 0.5 * grad_cart[:, 0]
                    return -grad_bary.flatten()
                
                try:
                    res_q = minimize(quick_obj, perturbed.flatten(), method='L-BFGS-B', jac=quick_grad,
                                     options={'maxiter': 100, 'ftol': 1e-15})
                    new_bary = res_q.x.reshape(N, 2)
                    new_bary = _project_bary_to_valid(new_bary)
                    _update_best(new_bary)
                    if new_bary is not None:
                        current_bary = new_bary
                except Exception:
                    pass
>>>>>>> REPLACE

