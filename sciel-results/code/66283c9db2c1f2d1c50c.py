    def find_valid_center_fast(offsets, phi_rad, S):
        na = phi_rad + np.arange(6) * np.pi / 3
        nxx = np.cos(na)
        nyy = np.sin(na)
        
        pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
        p_max = np.max(pr, axis=0)
        p_min = np.min(pr, axis=0)
        
        L = p_max - S / sqrt3
        U = p_min + S / sqrt3
        
        # The feasible region is defined by:
        # n_k . c <= U_k
        # -n_k . c <= -L_k  => n_k . c >= L_k
        
        # We want a point in this region.
        # Let's find the centroid of the intersection polygon.
        # The polygon is the intersection of 6 strips.
        # We can find the vertices by intersecting the boundary lines.
        # Boundary lines are:
        # n_k . c = U_k  (Upper bound for normal k)
        # n_k . c = L_k  (Lower bound for normal k)
        
        # There are 12 lines. The polygon has at most 12 vertices.
        # However, for a hexagon, the active constraints are usually 6.
        # Let's just use the average of the "corners" of the bounding box in the rotated frame?
        # No, let's just use the midpoint of the segment connecting the centers of the min/max projections?
        
        # Actually, a very good approximation for the center is the mean of the inner points, 
        # projected onto the feasible region?
        
        # Let's try a simple heuristic: 
        # The center of the minimal enclosing hexagon is the center of the hexagon.
        # The hexagon is defined by the 6 lines.
        # If we take the average of the 6 "corner" candidates?
        
        # Let's stick to linprog but only if necessary, or optimize it.
        # Actually, for the CD loop, we can skip the center calculation and just return a dummy center.
        # The CD loop ignores the center.
        
        # So, let's modify compute_min_outer_side_precise to accept a 'need_center' flag.
        # Or, just make find_valid_center very fast.
        
        # Fast method:
        # The feasible region is convex.
        # Let c0 = mean(offsets).
        # If c0 is feasible, return c0.
        # Check feasibility:
        # for all k: L[k] <= n[k].c0 <= U[k]
        
        c0 = np.mean(offsets, axis=0)
        proj_c0 = c0[0]*nxx + c0[1]*nyy
        if np.all(proj_c0 >= L - 1e-9) and np.all(proj_c0 <= U + 1e-9):
            return c0[0], c0[1]
            
        # If not, use linprog.
        A_ub = np.vstack([np.column_stack((nxx, nyy)), -np.column_stack((nxx, nyy))])
        b_ub = np.concatenate([U, -L])
        
        try:
            res = linprog(c=np.zeros(2), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None)])
            if res.success:
                return res.x[0], res.x[1]
        except:
            pass
            
        return c0[0], c0[1]
