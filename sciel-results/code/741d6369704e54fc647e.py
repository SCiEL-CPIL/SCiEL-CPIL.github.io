def _riemannian_lbfgs(pts, time_limit=10.0):
    """Riemannian L-BFGS on the sphere."""
    start = time.time()
    current = pts.copy()
    energy, grad = _energy_and_grad(current)
    
    m = 10
    s_hist = []
    y_hist = []
    
    d = -grad.copy()
    
    for it in range(500):
        if time.time() - start > time_limit - 0.1:
            break
        
        grad_norm_sq = np.sum(grad * grad)
        if grad_norm_sq < 1e-24:
            break
        
        dir_deriv = np.sum(grad * d)
        if dir_deriv > 0:
            d = -grad.copy()
            dir_deriv = -grad_norm_sq
            
        best_alpha = 0.0
        best_energy = energy
        alpha = 0.1
        c1 = 1e-4
        
        for _ls in range(40):
            alpha_norm = alpha * np.linalg.norm(d, axis=1)
            mask = alpha_norm > 1e-15
            new_pts = current.copy()
            if np.any(mask):
                cos_a = np.cos(alpha_norm)
                sin_a = np.sin(alpha_norm)
                d_hat = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-15)
                new_pts[mask] = (current[mask] * cos_a[mask, np.newaxis] + 
                                d_hat[mask] * sin_a[mask, np.newaxis])
            new_pts = _normalize(new_pts)
            new_energy = _log_energy_fast(new_pts)
            
            if new_energy <= energy + c1 * alpha * dir_deriv:
                best_alpha = alpha
                best_energy = new_energy
                break
            alpha *= 0.5
            
        if best_alpha == 0.0:
            break
            
        # Update
        alpha_norm = best_alpha * np.linalg.norm(d, axis=1)
        mask = alpha_norm > 1e-15
        new_pts = current.copy()
        if np.any(mask):
            cos_a = np.cos(alpha_norm)
            sin_a = np.sin(alpha_norm)
            d_hat = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-15)
            new_pts[mask] = (current[mask] * cos_a[mask, np.newaxis] + 
                            d_hat[mask] * sin_a[mask, np.newaxis])
        current = _normalize(new_pts)
        new_energy, new_grad = _energy_and_grad(current)
        
        s = best_alpha * d
        y = new_grad - grad
        
        # Check curvature condition
        ys = np.sum(y * s)
        if ys > 1e-12:
            s_hist.append(s)
            y_hist.append(y)
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
                
        # L-BFGS update
        q = new_grad.copy()
        rho = [1.0 / np.sum(y_hist[i] * s_hist[i]) for i in range(len(s_hist))]
        alphas = [0.0] * len(s_hist)
        
        for i in range(len(s_hist) - 1, -1, -1):
            alphas[i] = rho[i] * np.sum(s_hist[i] * q)
            q = q - alphas[i] * y_hist[i]
            
        gamma = 1.0
        if s_hist:
            gamma = np.sum(s_hist[-1] * y_hist[-1]) / np.sum(y_hist[-1] * y_hist[-1])
        r = gamma * q
        
        for i in range(len(s_hist)):
            beta = rho[i] * np.sum(y_hist[i] * r)
            r = r + (alphas[i] - beta) * s_hist[i]
            
        d = -r
        dot_prod = np.sum(d * current, axis=1)
        d -= dot_prod[:, np.newaxis] * current
        
        if np.linalg.norm(new_grad) < 1e-12:
            break
            
        grad = new_grad
        energy = new_energy
        
    return current, energy
