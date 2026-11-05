<<<<<<< SEARCH
        # Simulated annealing with adaptive temperature
        max_iters = 40000
        start_temp = 4.0
        end_temp = 0.05
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        temp = start_temp
        iters_done = 0

        # Track for adaptive reheat
        no_improve_count = 0
        reheat_threshold = 3000
        reheat_temp = 2.0

        while iters_done < max_iters:
            if time.time() - t0 > DEADLINE:
                break

            # Targeted search: find worst triangle
            worst_tri, worst_sum = _find_worst_triangle(adj, deg)

            # Decide which edge to remove
            if worst_tri and rng.random() < 0.6:
                i, j, k = worst_tri
                candidates = [(i, j), (i, k), (j, k)]
                e_remove = candidates[rng.randrange(3)]
            else:
                e_remove = edges[rng.randrange(len(edges))]

            e_add = nonedges[rng.randrange(len(nonedges))]

            u, v = e_remove
            x, y = e_add

            if (u == x and v == y) or (u == y and v == x):
                iters_done += 1
                temp *= cooling_rate
                continue

            # Perform swap
            _apply_swap(adj, deg, e_remove, e_add)
            new_score = _score(adj, deg)
            new_soft = _soft_score(adj, deg)

            # Acceptance: use soft-max for smooth guidance, hard max for tracking
            if new_score < cur_score:
                accept = True
                no_improve_count = 0
            elif new_score == cur_score:
                # Accept with moderate probability to explore
                accept = rng.random() < 0.4
                no_improve_count += 1
            else:
                # Use soft-max delta for smoother acceptance
                delta_soft = new_soft - cur_soft
                delta_hard = new_score - cur_score
                # Blend: use soft delta if it's informative, else hard
                if delta_soft < 0.01:
                    eff_delta = delta_soft
                else:
                    eff_delta = delta_hard
                if temp < 1e-9:
                    accept = False
                else:
                    accept = math.exp(-eff_delta / temp) > rng.random()
                no_improve_count += 1

            if accept:
                cur_score = new_score
                cur_soft = new_soft
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    no_improve_count = 0
                remove_edge_from_list(e_remove)
                add_edge_to_list(e_add)
            else:
                _undo_swap(adj, deg, e_remove, e_add)

            iters_done += 1
            temp *= cooling_rate

            # Adaptive reheat: if stuck, boost temperature
            if no_improve_count > reheat_threshold:
                temp = max(temp, reheat_temp)
                no_improve_count = 0
                reheat_threshold = int(reheat_threshold * 1.3)
                reheat_temp = max(0.5, reheat_temp * 0.7)
=======
        # Simulated annealing with adaptive temperature
        max_iters = 40000
        start_temp = 4.0
        end_temp = 0.05
        cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
        temp = start_temp
        iters_done = 0

        # Track for adaptive reheat
        no_improve_count = 0
        reheat_threshold = 3000
        reheat_temp = 2.0

        while iters_done < max_iters:
            if time.time() - t0 > DEADLINE:
                break

            # Targeted search: find worst triangle
            worst_tri, worst_sum = _find_worst_triangle(adj_np, deg)

            # Decide which edge to remove
            if worst_tri and rng.random() < 0.6:
                i, j, k = worst_tri
                candidates = [(i, j), (i, k), (j, k)]
                e_remove = candidates[rng.randrange(3)]
            else:
                e_remove = edges[rng.randrange(len(edges))]

            # Bias e_add selection: prefer non-edges with lower degree sum
            # to avoid creating new high-sum triangles
            if worst_tri:
                i, j, k = worst_tri
                # Filter nonedges that don't involve i,j,k if possible?
                # Or just pick randomly from a subset?
                # Let's stick to random for now but maybe bias towards low degree
                # Simple bias: pick 10 random candidates, choose one with min deg[x]+deg[y]
                candidates_add = [nonedges[rng.randrange(len(nonedges))] for _ in range(10)]
                e_add = min(candidates_add, key=lambda e: deg[e[0]] + deg[e[1]])
            else:
                e_add = nonedges[rng.randrange(len(nonedges))]

            u, v = e_remove
            x, y = e_add

            if (u == x and v == y) or (u == y and v == x):
                iters_done += 1
                temp *= cooling_rate
                continue

            # Perform swap
            _apply_swap(adj, adj_np, deg, e_remove, e_add)
            new_score = _score(adj_np, deg)
            new_soft = _soft_score(adj_np, deg)

            # Acceptance: use soft-max for smooth guidance, hard max for tracking
            if new_score < cur_score:
                accept = True
                no_improve_count = 0
            elif new_score == cur_score:
                # Accept with moderate probability to explore
                accept = rng.random() < 0.4
                no_improve_count += 1
            else:
                # Use soft-max delta for smoother acceptance
                delta_soft = new_soft - cur_soft
                delta_hard = new_score - cur_score
                # Blend: use soft delta if it's informative, else hard
                if delta_soft < 0.01:
                    eff_delta = delta_soft
                else:
                    eff_delta = delta_hard
                if temp < 1e-9:
                    accept = False
                else:
                    accept = math.exp(-eff_delta / temp) > rng.random()
                no_improve_count += 1

            if accept:
                cur_score = new_score
                cur_soft = new_soft
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    no_improve_count = 0
                remove_edge_from_list(e_remove)
                add_edge_to_list(e_add)
            else:
                _undo_swap(adj, adj_np, deg, e_remove, e_add)

            iters_done += 1
            temp *= cooling_rate

            # Adaptive reheat: if stuck, boost temperature
            if no_improve_count > reheat_threshold:
                temp = max(temp, reheat_temp)
                no_improve_count = 0
                reheat_threshold = int(reheat_threshold * 1.3)
                reheat_temp = max(0.5, reheat_temp * 0.7)
>>>>>>> REPLACE

Update the final polish section to use `adj_np` and the new swap functions.

