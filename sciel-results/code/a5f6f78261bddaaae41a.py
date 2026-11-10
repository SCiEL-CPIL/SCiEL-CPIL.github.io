<<<<<<< SEARCH
    # Phase 2: Fine-tuning with gradient-based approach
    if time.time() < deadline - 15:
        points = best_points.copy()
        polish_step = 0.005
        polish_decay = 0.999
        polish_deadline = deadline - 5

        for step in range(5000):
            if time.time() > polish_deadline:
                break

            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)

            min_pair_dist = np.min(dists)
            norms = np.linalg.norm(points, axis=1)
            min_boundary_dist = np.min(1.0 - norms)

            current_score = min(min_pair_dist / 2.0, min_boundary_dist)

            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()

            # Use top 20 closest pairs for precision
            flat_dists = dists.flatten()
            k = min(20, N * (N - 1) // 2)
            part_idx = np.argpartition(flat_dists, k)[:k]

            pair_grad = np.zeros_like(points)
            for idx in part_idx:
                i = int(idx // N)
                j = int(idx % N)
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-6:
                    continue
                vec = (points[i] - points[j]) / d
                pair_grad[i] += vec
                pair_grad[j] -= vec

            # Boundary gradient for points close to boundary
            boundary_grad = np.zeros_like(points)
            close_to_boundary = norms > (1.0 - current_score - 0.002)
            for i in np.where(close_to_boundary)[0]:
                if norms[i] > 1e-6:
                    boundary_grad[i] = -points[i] / norms[i]

            # Combine based on which constraint is tighter
            pair_score = min_pair_dist / 2.0
            bound_score = min_boundary_dist

            if pair_score <= bound_score:
                w_pair = 1.0
                w_bound = 0.2
            else:
                w_pair = 0.2
                w_bound = 1.0

            grad = w_pair * pair_grad + w_bound * boundary_grad

            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-6:
                grad = grad / g_norm

            points += polish_step * grad

            # Keep inside
            norms_new = np.linalg.norm(points, axis=1)
            outside = norms_new > 0.999
            if np.any(outside):
                points[outside] = points[outside] * (0.999 / norms_new[outside, np.newaxis])

            polish_step = max(1e-5, polish_step * polish_decay)
=======
    # Phase 2: Fine-tuning with gradient-based approach using KDTree
    if time.time() < deadline - 15:
        points = best_points.copy()
        polish_step = 0.005
        polish_decay = 0.999
        polish_deadline = deadline - 5
        r_est = best_score

        for step in range(8000):
            if time.time() > polish_deadline:
                break

            n = len(points)
            tree = cKDTree(points)
            search_r = 2.5 * r_est + 0.02
            pairs = tree.query_pairs(search_r)

            norms = np.linalg.norm(points, axis=1)
            min_boundary_dist = np.min(1.0 - norms)

            # Find minimum pair distance among close pairs
            min_pair_dist = np.inf
            pair_list = []
            for i, j in pairs:
                d = np.linalg.norm(points[i] - points[j])
                if d < min_pair_dist:
                    min_pair_dist = d
                pair_list.append((i, j, d))

            # Sort by distance to get closest pairs
            pair_list.sort(key=lambda x: x[2])
            closest_pairs = pair_list[:30]

            current_score = min(min_pair_dist / 2.0, min_boundary_dist)

            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
                r_est = current_score

            # Gradient from closest pairs
            pair_grad = np.zeros_like(points)
            for i, j, d in closest_pairs:
                if d < 1e-6:
                    continue
                vec = (points[i] - points[j]) / d
                pair_grad[i] += vec
                pair_grad[j] -= vec

            # Boundary gradient for points close to boundary
            boundary_grad = np.zeros_like(points)
            close_to_boundary = norms > (1.0 - current_score - 0.002)
            for i in np.where(close_to_boundary)[0]:
                if norms[i] > 1e-6:
                    boundary_grad[i] = -points[i] / norms[i]

            # Combine based on which constraint is tighter
            pair_score = min_pair_dist / 2.0
            bound_score = min_boundary_dist

            if pair_score <= bound_score:
                w_pair = 1.0
                w_bound = 0.2
            else:
                w_pair = 0.2
                w_bound = 1.0

            grad = w_pair * pair_grad + w_bound * boundary_grad

            g_norm = np.linalg.norm(grad)
            if g_norm > 1e-6:
                grad = grad / g_norm

            points += polish_step * grad

            # Keep inside
            norms_new = np.linalg.norm(points, axis=1)
            outside = norms_new > 0.999
            if np.any(outside):
                points[outside] = points[outside] * (0.999 / norms_new[outside, np.newaxis])

            polish_step = max(1e-5, polish_step * polish_decay)

        # Final exact score
        final_score = compute_score(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()
>>>>>>> REPLACE

