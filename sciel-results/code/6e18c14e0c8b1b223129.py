<<<<<<< SEARCH
    # Phase 2: Aggressive Facet Refinement Loop
    # This is the key improvement: repeatedly find the worst facet and fix it.
    if best_points is not None and time.time() < deadline - 20.0:
        points = best_points.copy()
        # Run many iterations of facet refinement
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            rad, worst_face, c_center = compute_covering_radius_exact(points)
            if rad < best_rad - 1e-9:
                best_rad = rad
                best_points = points.copy()
            
            # Move vertices of worst face towards center
            i, j, k = worst_face[0], worst_face[1], worst_face[2]
            step_size = 0.02
            
            for idx in [i, j, k]:
                p = points[idx]
                dot = np.dot(p, c_center)
                dot = np.clip(dot, -1.0, 1.0)
                angle = np.arccos(dot)
                if angle < 1e-6:
                    continue
                s = np.sin(angle)
                if s < 1e-6:
                    continue
                new_p = (np.sin((1-step_size)*angle)/s)*p + (np.sin(step_size*angle)/s)*c_center
                new_p /= np.linalg.norm(new_p)
                points[idx] = new_p
            
            # Check for improvement
            new_rad, _, _ = compute_covering_radius_exact(points)
            if new_rad > rad + 1e-7:
                # Revert
                # We need to store old points. 
                # To save memory/time, we'll just accept the risk or do a simpler check.
                # Actually, let's just stop if it gets worse significantly.
                pass
=======
    # Phase 2: Aggressive Facet Refinement Loop
    # This is the key improvement: repeatedly find the worst facet and fix it.
    if best_points is not None and time.time() < deadline - 20.0:
        points = best_points.copy()
        # Run many iterations of facet refinement
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            rad, worst_face, c_center = compute_covering_radius_exact(points)
            if rad < best_rad - 1e-9:
                best_rad = rad
                best_points = points.copy()
            
            # Move vertices of worst face towards center
            if worst_face is None or len(worst_face) < 3:
                continue
            i, j, k = worst_face[0], worst_face[1], worst_face[2]
            step_size = 0.02
            
            for idx in [i, j, k]:
                p = points[idx]
                dot = np.dot(p, c_center)
                dot = np.clip(dot, -1.0, 1.0)
                angle = np.arccos(dot)
                if angle < 1e-6:
                    continue
                s = np.sin(angle)
                if s < 1e-6:
                    continue
                new_p = (np.sin((1-step_size)*angle)/s)*p + (np.sin(step_size*angle)/s)*c_center
                norm_new = np.linalg.norm(new_p)
                if norm_new < 1e-10:
                    continue
                new_p = new_p / norm_new
                points[idx] = new_p
            
            # Check for improvement
            new_rad, _, _ = compute_covering_radius_exact(points)
            if new_rad > rad + 1e-7:
                # Revert
                # We need to store old points. 
                # To save memory/time, we'll just accept the risk or do a simpler check.
                # Actually, let's just stop if it gets worse significantly.
                pass
>>>>>>> REPLACE
