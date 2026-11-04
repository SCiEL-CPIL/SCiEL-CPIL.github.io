<<<<<<< SEARCH
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        # Re-center to remove translation invariance and improve gradient
        centers = xys_rots[:, :2]
        mean_center = np.mean(centers, axis=0)
        xys_rots_rc = xys_rots.copy()
        xys_rots_rc[:, :2] -= mean_center
        
        # Use smooth penalty for better gradient information
        penalty = smooth_overlap_penalty(xys_rots_rc, beta=50.0)
        
        # Compute outer side length
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots_rc])
        L = outer_hex_side(all_points, outer_rot)
        
        return L + 100.0 * penalty
=======
    def objective(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e6
        
        xys_rots = params[:3*N].reshape(N, 3)
        outer_rot = params[3*N]
        
        # Re-center to remove translation invariance and improve gradient
        centers = xys_rots[:, :2]
        mean_center = np.mean(centers, axis=0)
        xys_rots_rc = xys_rots.copy()
        xys_rots_rc[:, :2] -= mean_center
        
        # Adaptive beta: smaller beta for sharper penalty when close to feasible
        L_est = outer_hex_side(np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots_rc]), outer_rot)
        # Use beta that scales with how tight the packing is
        beta = max(10.0, 50.0 - 10.0 * (12.0 - L_est))
        
        # Use smooth penalty for better gradient information
        penalty = smooth_overlap_penalty(xys_rots_rc, beta=beta)
        
        # Compute outer side length
        all_points = np.vstack([get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots_rc])
        L = outer_hex_side(all_points, outer_rot)
        
        return L + 100.0 * penalty
>>>>>>> REPLACE

