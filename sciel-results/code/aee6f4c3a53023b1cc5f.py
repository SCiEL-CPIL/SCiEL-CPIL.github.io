<<<<<<< SEARCH
    def polish(x, w, h):
        """Use gradient-based polish with softplus penalties."""
        if time.time() > deadline:
            return x
        
        def objective(params):
            xs = params[0::3]
            ys = params[1::3]
            rs = params[2::3]
            
            penalty = 0.0
            
            # Boundary penalties (softplus for smoothness)
            for i in range(N):
                if rs[i] <= 0:
                    penalty += (rs[i] + 1e-3)**2
                else:
                    penalty += softplus(-(xs[i] - rs[i]))
                    penalty += softplus(xs[i] + rs[i] - w)
                    penalty += softplus(-(ys[i] - rs[i]))
                    penalty += softplus(ys[i] + rs[i] - h)
            
            # Overlap penalties (softplus for smoothness)
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = rs[i] + rs[j]
                    penalty += softplus(min_dist - dist)
            
            return penalty - np.sum(rs)

        bounds = []
        r_max = min(w, h) / 2.0
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((1e-6, r_max))
            
        try:
            res = minimize(objective, x, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.success and is_valid(res.x, w, h, tol=1e-6):
                if np.sum(res.x[2::3]) >= np.sum(x[2::3]) - 1e-6:
                    return res.x
        except Exception:
            pass
        return x
=======
    def polish(x, w, h):
        """
        Optimize positions and a SINGLE radius for all circles.
        This enforces equal radii, which is often optimal for dense packings.
        """
        if time.time() > deadline:
            return x
        
        # Extract initial positions and a representative radius
        xs_init = x[0::3].copy()
        ys_init = x[1::3].copy()
        r_init = np.median(x[2::3])
        
        # Variables: x_1, y_1, ..., x_N, y_N, R
        # Total 2*N + 1 variables
        def objective(params):
            xs = params[0::2]
            ys = params[1::2]
            R = params[-1]
            
            if R <= 0:
                return 1e6 - R # Penalize non-positive radius heavily, maximize R
            
            penalty = 0.0
            
            # Boundary penalties
            for i in range(N):
                penalty += softplus(-(xs[i] - R))
                penalty += softplus(xs[i] + R - w)
                penalty += softplus(-(ys[i] - R))
                penalty += softplus(ys[i] + R - h)
            
            # Overlap penalties (dist >= 2R)
            for i in range(N):
                for j in range(i+1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    penalty += softplus(2*R - dist)
            
            # Maximize sum of radii = N * R
            # Minimize -N*R + penalty
            return penalty - N * R

        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
        bounds.append((1e-6, min(w, h)/2.0))
        
        # Initial guess
        x0 = np.zeros(2*N + 1)
        x0[0::2] = xs_init
        x0[1::2] = ys_init
        x0[-1] = r_init
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success:
                xs = res.x[0::2]
                ys = res.x[1::2]
                R = res.x[-1]
                
                # Construct result vector
                new_x = np.zeros(3*N)
                for i in range(N):
                    new_x[3*i] = xs[i]
                    new_x[3*i+1] = ys[i]
                    new_x[3*i+2] = R
                
                # Check validity
                if is_valid(new_x, w, h, tol=1e-6):
                    # Stretch radius if possible
                    new_x = stretch_radius(new_x, w, h)
                    return new_x
        except Exception:
            pass
        
        return x

    def stretch_radius(x, w, h):
        """
        Try to increase all radii by a small factor if valid.
        """
        if time.time() > deadline:
            return x
        
        x = x.copy()
        r = x[2::3]
        # Check if all radii are equal (within tolerance)
        if np.max(r) - np.min(r) > 1e-4:
             return x
             
        R = np.mean(r)
        
        # Calculate max possible R given positions
        min_wall = np.inf
        for i in range(N):
            min_wall = min(min_wall, x[3*i], w-x[3*i], x[3*i+1], h-x[3*i+1])
        
        min_neighbor = np.inf
        for i in range(N):
            for j in range(i+1, N):
                dx = x[3*i] - x[3*j]
                dy = x[3*i+1] - x[3*j+1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_neighbor = min(min_neighbor, dist / 2.0)
                
        max_R = min(min_wall, min_neighbor)
        
        if max_R > R + 1e-8:
             for i in range(N):
                 x[3*i+2] = max_R
             if is_valid(x, w, h, tol=1e-6):
                 return x
        return x
>>>>>>> REPLACE

I will also slightly tweak the initialization to be tighter (0.95 instead of 0.85) to give the optimizer a better starting point.

