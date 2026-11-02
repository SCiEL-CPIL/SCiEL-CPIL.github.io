    def local_search(positions_0, phi_0, cx_0, cy_0):
        # params: [x1..x11, y1..y11, cx, cy, phi]
        # 22 + 2 + 1 = 25
        
        def objective(params):
            pts = params[:22].reshape(11, 2)
            cx, cy = params[22], params[23]
            phi = params[24]
            
            # Normals of outer hexagon (fixed at 0)
            n_outer = get_normals(0.0)
            # Normals of inner hexagons (orient phi)
            n_inner = get_normals(phi)
            
            # Calculate S
            max_val = 0.0
            for k in range(6):
                n_k = n_outer[k]
                projs = (pts - np.array([cx, cy])) @ n_k
                h_val = h_unit(n_k, phi)
                vals = np.abs(projs) + h_val
                max_val = max(max_val, float(np.max(vals)))
            S = max_val / APOTHEM
            
            # Calculate Penalty
            penalty = 0.0
            for i in range(11):
                for j in range(i+1, 11):
                    diff = pts[i] - pts[j]
                    # Check overlap with Minkowski hexagon (orient phi)
                    # Normals are n_inner
                    min_margin = float('inf')
                    for k in range(6):
                        n_k = n_inner[k]
                        h_diff = 2.0 * h_unit(n_k, phi) # h_unit is for side 1, diff is side 2
                        # Wait, h_unit(n_k, phi) is support of unit hex.
                        # Support of Minkowski sum is sum of supports.
                        # h_diff = h_unit(n_k, phi) + h_unit(-n_k, phi) = 2 * h_unit(n_k, phi)
                        proj = abs(diff @ n_k)
                        margin = proj - h_diff
                        if margin < min_margin:
                            min_margin = margin
                    if min_margin < 0:
                        penalty += (-min_margin) ** 2
                        
            return S + 10.0 * penalty

        # Initial params
        p0 = []
        for x, y in positions_0:
            p0.append(x)
            p0.append(y)
        p0.append(cx_0)
        p0.append(cy_0)
        p0.append(phi_0)
        
        try:
            res = minimize(objective, p0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            return res.x, res.fun
        except Exception:
            return p0, float('inf')

    # Generate initial guesses from axial candidates
    # ... (loop over candidates, convert to cartesian, run local search)
