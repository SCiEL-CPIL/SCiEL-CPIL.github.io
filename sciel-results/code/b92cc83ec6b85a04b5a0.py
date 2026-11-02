<<<<<<< SEARCH
    def epigraph_objective(flat_params, beta=80.0):
        """
        Epigraph formulation: minimize t subject to all facet radii <= t.
        Uses soft constraints via log-sum-exp for smoothness.
        flat_params = [point_1_x, point_1_y, point_1_z, ..., point_50_x, point_50_y, point_50_z, t]
        """
        pts = flat_params[:N*3].reshape(N, 3)
        t = flat_params[N*3]
        
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms

        facets = compute_facet_circumcenters(pts)
        if not facets or len(facets) < 2:
            return t + 10.0

        radii = np.array([r for r, _, _ in facets])
        
        # Soft constraint: sum of exp(beta * (radius - t)) should be small
        # This is smooth and differentiable
        diff = radii - t
        max_diff = np.max(diff)
        if max_diff > 0:
            log_sum = max_diff + np.log(np.sum(np.exp((diff - max_diff) * beta)))
            penalty = log_sum / beta
        else:
            penalty = 0.0
        
        # Objective: t + penalty (penalty pushes t up if any radius > t)
        return t + penalty
=======
    def epigraph_objective(flat_params, beta=100.0):
        """
        Epigraph formulation: minimize t subject to all facet radii <= t.
        Uses soft constraints via log-sum-exp for smoothness.
        flat_params = [point_1_x, point_1_y, point_1_z, ..., point_50_x, point_50_y, point_50_z, t]
        """
        pts = flat_params[:N*3].reshape(N, 3)
        t = flat_params[N*3]
        
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms

        facets = compute_facet_circumcenters(pts)
        if not facets or len(facets) < 2:
            return t + 10.0

        radii = np.array([r for r, _, _ in facets])
        
        # Soft constraint: sum of exp(beta * (radius - t)) should be small
        # This is smooth and differentiable
        diff = radii - t
        max_diff = np.max(diff)
        if max_diff > 0:
            log_sum = max_diff + np.log(np.sum(np.exp((diff - max_diff) * beta)))
            penalty = log_sum / beta
        else:
            penalty = 0.0
        
        # Objective: t + penalty (penalty pushes t up if any radius > t)
        return t + penalty
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Phase 3: L-BFGS-B optimization using epigraph formulation
    if time.time() < deadline - 25:
        # Initialize with current best points and t = current covering radius + small margin
        initial_radius = compute_covering_radius(best)[0]
        t_init = initial_radius + 0.01
        flat_initial = np.concatenate([best.flatten(), [t_init]])
        
        try:
            result = minimize(
                epigraph_objective,
                flat_initial,
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            if result.success and result.x is not None:
                current = result.x[:N*3].reshape(N, 3)
                # Normalize to ensure valid output
                current = current / np.linalg.norm(current, axis=1, keepdims=True)
                radius, _ = compute_covering_radius(current)
                if radius < best_radius:
                    best_radius = radius
                    best = current.copy()
        except:
            pass
=======
    # Phase 3: L-BFGS-B optimization using epigraph formulation
    if time.time() < deadline - 25:
        # Initialize with current best points and t = current covering radius + small margin
        initial_radius = compute_covering_radius(best)[0]
        t_init = initial_radius + 0.01
        flat_initial = np.concatenate([best.flatten(), [t_init]])
        
        try:
            result = minimize(
                epigraph_objective,
                flat_initial,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
            )
            if result.success and result.x is not None:
                current = result.x[:N*3].reshape(N, 3)
                # Normalize to ensure valid output
                current = current / np.linalg.norm(current, axis=1, keepdims=True)
                radius, _ = compute_covering_radius(current)
                if radius < best_radius:
                    best_radius = radius
                    best = current.copy()
        except:
            pass
>>>>>>> REPLACE
<<<<<<< SEARCH
            if time.time() < deadline - 20:
                # Use epigraph formulation for restarts too
                pert_radius = compute_covering_radius(perturbed)[0]
                t_pert = pert_radius + 0.01
                flat_pert = np.concatenate([perturbed.flatten(), [t_pert]])
                try:
                    result = minimize(
                        epigraph_objective,
                        flat_pert,
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 15}
                    )
                    if result.success and result.x is not None:
                        perturbed = result.x[:N*3].reshape(N, 3)
                        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
                except:
                    pass
=======
            if time.time() < deadline - 20:
                # Use epigraph formulation for restarts too
                pert_radius = compute_covering_radius(perturbed)[0]
                t_pert = pert_radius + 0.01
                flat_pert = np.concatenate([perturbed.flatten(), [t_pert]])
                try:
                    result = minimize(
                        epigraph_objective,
                        flat_pert,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
                    )
                    if result.success and result.x is not None:
                        perturbed = result.x[:N*3].reshape(N, 3)
                        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
                except:
                    pass
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Phase 6: Final fine-tuning with high-precision epigraph polish
    if time.time() < deadline - 5:
        # First do a few steps of worst-facet local search
        final = best.copy()
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            radius, worst_center = compute_covering_radius(final)

            try:
                hull = ConvexHull(final)
            except:
                break

            target_indices = None
            for simplex in hull.simplices:
                v1 = final[simplex[0]]
                v2 = final[simplex[1]]
                v3 = final[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                if np.dot(c, worst_center) > 0.9999:
                    target_indices = simplex
                    break

            if target_indices is None:
                break

            for idx in target_indices:
                v = final[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    final[idx] = v + 0.002 * dir_vec
                    final[idx] /= np.linalg.norm(final[idx])

        # Then do a high-precision epigraph polish
        if time.time() < deadline - 3:
            final_radius = compute_covering_radius(final)[0]
            t_final = final_radius + 0.005
            flat_final = np.concatenate([final.flatten(), [t_final]])
            try:
                result = minimize(
                    epigraph_objective,
                    flat_final,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30}
                )
                if result.success and result.x is not None:
                    final = result.x[:N*3].reshape(N, 3)
                    final = final / np.linalg.norm(final, axis=1, keepdims=True)
            except:
                pass

        radius, _ = compute_covering_radius(final)
        if radius < best_radius:
            best = final
=======
    # Phase 6: Final fine-tuning with high-precision epigraph polish
    if time.time() < deadline - 5:
        # First do a few steps of worst-facet local search
        final = best.copy()
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            radius, worst_center = compute_covering_radius(final)

            try:
                hull = ConvexHull(final)
            except:
                break

            target_indices = None
            for simplex in hull.simplices:
                v1 = final[simplex[0]]
                v2 = final[simplex[1]]
                v3 = final[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                if np.dot(c, worst_center) > 0.9999:
                    target_indices = simplex
                    break

            if target_indices is None:
                break

            for idx in target_indices:
                v = final[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    final[idx] = v + 0.001 * dir_vec
                    final[idx] /= np.linalg.norm(final[idx])

        # Then do a high-precision epigraph polish
        if time.time() < deadline - 3:
            final_radius = compute_covering_radius(final)[0]
            t_final = final_radius + 0.005
            flat_final = np.concatenate([final.flatten(), [t_final]])
            try:
                result = minimize(
                    epigraph_objective,
                    flat_final,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 40}
                )
                if result.success and result.x is not None:
                    final = result.x[:N*3].reshape(N, 3)
                    final = final / np.linalg.norm(final, axis=1, keepdims=True)
            except:
                pass

        radius, _ = compute_covering_radius(final)
        if radius < best_radius:
            best = final
>>>>>>> REPLACE

BET: {mechanism: "Tighter tolerances and more iterations in L-BFGS-B allow deeper convergence in the epigraph formulation, while smaller step sizes in the final local search enable finer adjustments to the worst facet, potentially escaping shallow local minima that coarser steps miss", q: 0.7, delta: [0.01, 0.05]}
