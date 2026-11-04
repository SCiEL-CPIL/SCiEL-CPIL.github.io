<<<<<<< SEARCH
    def local_search(x0, max_iter, time_limit):
        current_x = x0.copy()
        blocked = compute_blocked(current_x)
        current_x, blocked = greedy_fill(current_x, blocked)
        current_score = int(np.sum(current_x))

        best_x = current_x.copy()
        best_score = current_score

        for it in range(max_iter):
            if time.time() > time_limit:
                break

            r = rng.random()

            if r < 0.30:
                # Remove 1-3 nodes, refill
                in_set = np.where(current_x)[0]
                if len(in_set) < 1:
                    continue
                n_remove = 1 + rng.randint(0, 2)
                n_remove = min(n_remove, len(in_set))
                to_remove = in_set[rng.choice(len(in_set), size=n_remove, replace=False)]
                for v in to_remove:
                    current_x[v] = False
                blocked = compute_blocked(current_x)
                current_x, blocked = greedy_fill(current_x, blocked)
                current_score = int(np.sum(current_x))
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x.copy()

            elif r < 0.55:
                # Swap: remove one, add one
                in_set = np.where(current_x)[0]
                if len(in_set) == 0:
                    continue
                u = in_set[rng.randint(len(in_set))]
                temp_x = current_x.copy()
                temp_x[u] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) > 0:
                    v = candidates[rng.randint(len(candidates))]
                    current_x[u] = False
                    current_x[v] = True
                    blocked = compute_blocked(current_x)
                    current_x, blocked = greedy_fill(current_x, blocked)
                    current_score = int(np.sum(current_x))
                    if current_score > best_score:
                        best_score = current_score
                        best_x = current_x.copy()

            elif r < 0.80:
                # Double swap
                in_set = np.where(current_x)[0]
                if len(in_set) < 2:
                    continue
                u1_idx, u2_idx = rng.choice(len(in_set), size=2, replace=False)
                u1, u2 = in_set[u1_idx], in_set[u2_idx]
                temp_x = current_x.copy()
                temp_x[u1] = False
                temp_x[u2] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) >= 2:
                    v1_idx, v2_idx = rng.choice(len(candidates), size=2, replace=False)
                    v1, v2 = candidates[v1_idx], candidates[v2_idx]
                    if not conflict[v1, v2]:
                        current_x[u1] = False
                        current_x[u2] = False
                        current_x[v1] = True
                        current_x[v2] = True
                        blocked = compute_blocked(current_x)
                        current_x, blocked = greedy_fill(current_x, blocked)
                        current_score = int(np.sum(current_x))
                        if current_score > best_score:
                            best_score = current_score
                            best_x = current_x.copy()

            elif r < 0.92:
                # Triple swap
                in_set = np.where(current_x)[0]
                if len(in_set) < 3:
                    continue
                u_indices = rng.choice(len(in_set), size=3, replace=False)
                u_list = in_set[u_indices]
                temp_x = current_x.copy()
                for u in u_list:
                    temp_x[u] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) >= 3:
                    v_indices = rng.choice(len(candidates), size=3, replace=False)
                    v_list = candidates[v_indices]
                    # Check all pairs among v_list are non-conflicting
                    ok = True
                    for a in range(3):
                        for b in range(a + 1, 3):
                            if conflict[v_list[a], v_list[b]]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        for u in u_list:
                            current_x[u] = False
                        for v in v_list:
                            current_x[v] = True
                        blocked = compute_blocked(current_x)
                        current_x, blocked = greedy_fill(current_x, blocked)
                        current_score = int(np.sum(current_x))
                        if current_score > best_score:
                            best_score = current_score
                            best_x = current_x.copy()

            else:
                # Kick: remove 4-6 nodes
                in_set = np.where(current_x)[0]
                if len(in_set) < 4:
                    continue
                n_remove = 4 + rng.randint(0, 2)
                n_remove = min(n_remove, len(in_set))
                to_remove = in_set[rng.choice(len(in_set), size=n_remove, replace=False)]
                for v in to_remove:
                    current_x[v] = False
                blocked = compute_blocked(current_x)
                current_x, blocked = greedy_fill(current_x, blocked)
                current_score = int(np.sum(current_x))
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x.copy()

        return best_x, best_score
=======
    def vectorized_greedy_set_cover(x0, time_limit):
        """Vectorized greedy heuristic: iteratively select support maximizing marginal gain/penalty ratio."""
        x = x0.copy()
        blocked = compute_blocked(x)
        x, blocked = greedy_fill(x, blocked)
        current_score = int(np.sum(x))
        
        best_x = x.copy()
        best_score = current_score
        
        # Precompute adjacency as boolean matrix for vectorized operations
        adj_mat = np.zeros((N_S, N_S), dtype=bool)
        for i in range(N_S):
            if len(adj_arrays[i]) > 0:
                adj_mat[i, adj_arrays[i]] = True
        
        for it in range(50000):
            if time.time() > time_limit:
                break
            
            in_set = np.where(x)[0]
            not_in_set = np.where(~x)[0]
            
            if len(in_set) == 0 or len(not_in_set) == 0:
                break
            
            # For each candidate not in set, compute:
            # gain = number of currently unselected supports it conflicts with (that would become selectable)
            # penalty = number of currently selected supports it conflicts with (that would need removal)
            
            # Vectorized computation
            # For candidates: conflict with selected nodes
            cand_conflict_sel = adj_mat[not_in_set][:, in_set]  # shape (n_cand, n_sel)
            penalty = cand_conflict_sel.sum(axis=1)
            
            # Only consider candidates with penalty == 0 (can be added without removing anything)
            zero_penalty_mask = penalty == 0
            
            if not np.any(zero_penalty_mask):
                # No zero-penalty candidates; try swap-based moves
                # Find candidates with minimal penalty
                min_penalty = penalty.min()
                if min_penalty >= 3:
                    continue
                low_penalty_mask = penalty <= min_penalty
                candidates = not_in_set[low_penalty_mask]
                if len(candidates) == 0:
                    continue
                v = candidates[rng.randint(len(candidates))]
                # Remove conflicting selected nodes
                to_remove = in_set[adj_mat[v, in_set]]
                for u in to_remove:
                    x[u] = False
                x[v] = True
                blocked = compute_blocked(x)
                x, blocked = greedy_fill(x, blocked)
                current_score = int(np.sum(x))
                if current_score > best_score:
                    best_score = current_score
                    best_x = x.copy()
                continue
            
            zero_penalty_candidates = not_in_set[zero_penalty_mask]
            
            # Among zero-penalty candidates, prefer those with high degree to unselected (more future flexibility)
            cand_conflict_unsel = adj_mat[zero_penalty_candidates][:, not_in_set]
            # Count conflicts with unselected nodes (higher = more connected to remaining pool)
            connectivity = cand_conflict_unsel.sum(axis=1)
            
            # Also consider: how many of its neighbors are already blocked?
            # Prefer candidates whose addition unblocks the most nodes
            v_idx = zero_penalty_candidates[rng.randint(len(zero_penalty_candidates))]
            
            x[v_idx] = True
            neighbors = adj_arrays[v_idx]
            if len(neighbors) > 0:
                blocked[neighbors] = True
            # Re-fill with any newly available nodes
            changed = True
            while changed:
                changed = False
                new_candidates = np.where(~x & ~blocked)[0]
                for v in new_candidates:
                    if not x[v] and not blocked[v]:
                        x[v] = True
                        nbrs = adj_arrays[v]
                        if len(nbrs) > 0:
                            blocked[nbrs] = True
                        changed = True
            
            current_score = int(np.sum(x))
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
            
            # Occasionally do a kick to escape local optima
            if it % 500 == 0 and len(in_set) > 5:
                n_remove = rng.randint(2, 5)
                n_remove = min(n_remove, len(in_set))
                to_remove = in_set[rng.choice(len(in_set), size=n_remove, replace=False)]
                for v in to_remove:
                    x[v] = False
                blocked = compute_blocked(x)
                x, blocked = greedy_fill(x, blocked)
                current_score = int(np.sum(x))
        
        return best_x, best_score

    def local_search(x0, max_iter, time_limit):
        current_x = x0.copy()
        blocked = compute_blocked(current_x)
        current_x, blocked = greedy_fill(current_x, blocked)
        current_score = int(np.sum(current_x))

        best_x = current_x.copy()
        best_score = current_score

        # First pass: vectorized greedy set cover
        if time.time() < time_limit - 5.0:
            vc_best, vc_score = vectorized_greedy_set_cover(current_x, time_limit - 30.0)
            if vc_score > best_score:
                best_score = vc_score
                best_x = vc_best.copy()
                current_x = vc_best.copy()
                current_score = vc_score

        for it in range(max_iter):
            if time.time() > time_limit:
                break

            r = rng.random()

            if r < 0.30:
                # Remove 1-3 nodes, refill
                in_set = np.where(current_x)[0]
                if len(in_set) < 1:
                    continue
                n_remove = 1 + rng.randint(0, 2)
                n_remove = min(n_remove, len(in_set))
                to_remove = in_set[rng.choice(len(in_set), size=n_remove, replace=False)]
                for v in to_remove:
                    current_x[v] = False
                blocked = compute_blocked(current_x)
                current_x, blocked = greedy_fill(current_x, blocked)
                current_score = int(np.sum(current_x))
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x.copy()

            elif r < 0.55:
                # Swap: remove one, add one
                in_set = np.where(current_x)[0]
                if len(in_set) == 0:
                    continue
                u = in_set[rng.randint(len(in_set))]
                temp_x = current_x.copy()
                temp_x[u] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) > 0:
                    v = candidates[rng.randint(len(candidates))]
                    current_x[u] = False
                    current_x[v] = True
                    blocked = compute_blocked(current_x)
                    current_x, blocked = greedy_fill(current_x, blocked)
                    current_score = int(np.sum(current_x))
                    if current_score > best_score:
                        best_score = current_score
                        best_x = current_x.copy()

            elif r < 0.80:
                # Double swap
                in_set = np.where(current_x)[0]
                if len(in_set) < 2:
                    continue
                u1_idx, u2_idx = rng.choice(len(in_set), size=2, replace=False)
                u1, u2 = in_set[u1_idx], in_set[u2_idx]
                temp_x = current_x.copy()
                temp_x[u1] = False
                temp_x[u2] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) >= 2:
                    v1_idx, v2_idx = rng.choice(len(candidates), size=2, replace=False)
                    v1, v2 = candidates[v1_idx], candidates[v2_idx]
                    if not conflict[v1, v2]:
                        current_x[u1] = False
                        current_x[u2] = False
                        current_x[v1] = True
                        current_x[v2] = True
                        blocked = compute_blocked(current_x)
                        current_x, blocked = greedy_fill(current_x, blocked)
                        current_score = int(np.sum(current_x))
                        if current_score > best_score:
                            best_score = current_score
                            best_x = current_x.copy()

            elif r < 0.92:
                # Triple swap
                in_set = np.where(current_x)[0]
                if len(in_set) < 3:
                    continue
                u_indices = rng.choice(len(in_set), size=3, replace=False)
                u_list = in_set[u_indices]
                temp_x = current_x.copy()
                for u in u_list:
                    temp_x[u] = False
                temp_blocked = compute_blocked(temp_x)
                candidates = np.where(~current_x & ~temp_blocked)[0]
                if len(candidates) >= 3:
                    v_indices = rng.choice(len(candidates), size=3, replace=False)
                    v_list = candidates[v_indices]
                    ok = True
                    for a in range(3):
                        for b in range(a + 1, 3):
                            if conflict[v_list[a], v_list[b]]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        for u in u_list:
                            current_x[u] = False
                        for v in v_list:
                            current_x[v] = True
                        blocked = compute_blocked(current_x)
                        current_x, blocked = greedy_fill(current_x, blocked)
                        current_score = int(np.sum(current_x))
                        if current_score > best_score:
                            best_score = current_score
                            best_x = current_x.copy()

            else:
                # Kick: remove 4-6 nodes
                in_set = np.where(current_x)[0]
                if len(in_set) < 4:
                    continue
                n_remove = 4 + rng.randint(0, 2)
                n_remove = min(n_remove, len(in_set))
                to_remove = in_set[rng.choice(len(in_set), size=n_remove, replace=False)]
                for v in to_remove:
                    current_x[v] = False
                blocked = compute_blocked(current_x)
                current_x, blocked = greedy_fill(current_x, blocked)
                current_score = int(np.sum(current_x))
                if current_score > best_score:
                    best_score = current_score
                    best_x = current_x.copy()

        return best_x, best_score
>>>>>>> REPLACE

